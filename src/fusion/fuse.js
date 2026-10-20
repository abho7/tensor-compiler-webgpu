// The fusion pass (DESIGN.md amendments 29 and 30): an IR-to-IR transform that
// replaces groups of nodes by one `fused` node each.
//
// A fused node's attributes hold its body, an ordinary graph of existing ops
// whose inputs a0, a1, ... are the fused node's operands. The interpreter
// evaluates a fused node by evaluating its body, so a fused graph must give
// bitwise the same float64 values as the original at every node that survives
// (checkSemantics, below). Pure JavaScript: nothing here touches a GPU.
//
// Kinds:
//   elementwise  elementwise ops of one element count, with indexed loads
//                through broadcast / reshape / transpose of outside values
//   epilogue     a matmul whose only consumer is such a group, of its shape
//   rowwise      a group over [R, C] with sum / max over the last axis whose
//                results come back through broadcast (LayerNorm, softmax, ...)

import { Graph } from '../ir/graph.js';
import { size, sameShape } from '../ir/shapes.js';
import { evaluate, KERNELS } from '../interp.js';

export const ELEMENTWISE = new Set(['add', 'sub', 'mul', 'div', 'neg', 'exp', 'log', 'relu', 'tanh', 'sqrt', 'step', 'stopGradient']);
export const MOVES = new Set(['broadcast', 'reshape', 'transpose']);
export const MAX_OPERANDS = 6;
export const PATTERNS = ['elementwise', 'epilogue', 'rowwise'];

/** A scalar constant: one value, whatever its shape. Inlined into WGSL as its float32 bits. */
export function scalarConst(node) {
  if (node.op !== 'const') return false;
  const v = node.attrs.value;
  const first = typeof v === 'number' ? v : v[0];
  // Only finite float32 values: WGSL rejects a constant expression that overflows.
  if (!Number.isFinite(Math.fround(first))) return false;
  return typeof v === 'number' || v.every((x) => Object.is(x, first));
}
export const constValue = (node) => (typeof node.attrs.value === 'number' ? node.attrs.value : node.attrs.value[0]);

function consumersOf(graph) {
  const c = graph.nodes.map(() => []);
  for (const n of graph.nodes) for (const i of new Set(n.inputs)) c[i].push(n.id);
  return c;
}

/** True for a sum or max over the last axis with keepDims. */
const lastAxisReduce = (graph, n) => (n.op === 'sum' || n.op === 'max') && n.attrs.keepDims
  && n.attrs.axis === graph.nodes[n.inputs[0]].shape.length - 1;

/** The row-tier shape of an element-tier shape S: S with its last dim set to 1. */
const rowShape = (s) => [...s.slice(0, -1), 1];

/**
 * Grow one group from `sink`. Returns { members: Set, kind, root } or null.
 * `taken` holds nodes already in a group (exclusive unless `recompute`).
 */
function grow(graph, sink, consumers, taken, keep, { patterns, recompute, maxOperands, sabotage }) {
  const nodes = graph.nodes;
  const S = nodes[sink].shape;
  const n = size(S);
  const members = new Set([sink]);
  let kind = 'elementwise', root = null;
  // Element tier for rowwise groups: shape S; row tier: rowShape(S).
  const C = S.at(-1);
  const isRowShape = (s) => s.length === S.length && sameShape(s, rowShape(S));

  const free = (id) => !taken.has(id) || recompute;
  const allIn = (id, extra = new Set()) => consumers[id].every((c) => members.has(c) || extra.has(c));
  // Sabotage 5 (amendment 31): the pass absorbs a producer that is still used outside.
  const exclusive = (id, extra) => !keep.has(id) && (recompute || allIn(id, extra) || sabotage === 'fusion-absorbs-shared');

  const operandsOf = (set) => {
    const ext = new Set();
    for (const m of set) {
      const node = nodes[m];
      if (node.op === 'const') continue;
      for (const i of node.inputs) if (!set.has(i)) ext.add(i);
    }
    return ext;
  };
  const fits = (set) => operandsOf(set).size <= maxOperands;

  // The row region feeding broadcast `b` (row tier -> element tier): row-tier
  // elementwise ops, scalar constants, and last-axis reductions of element-tier
  // values. Null if it holds no reduction or cannot be absorbed whole.
  const rowRegion = (b) => {
    const region = new Set();
    let reductions = 0;
    const stack = [nodes[b].inputs[0]];
    while (stack.length) {
      const id = stack.pop();
      if (region.has(id) || members.has(id)) continue;
      const node = nodes[id];
      if (scalarConst(node)) { region.add(id); continue; }
      if (node.op === 'broadcast' && scalarConst(nodes[node.inputs[0]])) { region.add(id); region.add(node.inputs[0]); continue; }
      if (!isRowShape(node.shape) || !free(id)) continue; // stays an operand, loaded at the row index
      if (lastAxisReduce(graph, node) && sameShape(nodes[node.inputs[0]].shape, S)) {
        region.add(id); reductions++;
        continue;
      }
      if (ELEMENTWISE.has(node.op) || node.op === 'stopGradient') {
        region.add(id);
        stack.push(...node.inputs);
      }
    }
    if (!reductions) return null;
    const extra = new Set([...region, b]);
    for (const id of region) if (!scalarConst(nodes[id]) && nodes[id].op !== 'broadcast' && !exclusive(id, extra)) return null;
    return region;
  };

  const tryAdd = (set) => {
    const next = new Set([...members, ...set]);
    if (!fits(next)) return false;
    for (const id of set) members.add(id);
    return true;
  };
  // Grow to a fixpoint: a producer refused because one of its consumers was
  // not in the group yet may be accepted once that consumer joins.
  let changed = true;
  while (changed) {
    changed = false;
    for (const m of [...members]) {
      const mNode = nodes[m];
      // A broadcast or transpose member is a load of its source, or (recompute, or a
      // rowwise row broadcast) its source joined with it: either way it is done.
      if (mNode.op === 'const' || m === root || mNode.op === 'broadcast' || mNode.op === 'transpose') continue;
      // A reduction reads the element tier; a row-tier member of a rowwise group reads the row tier.
      const rowTier = kind === 'rowwise' && !lastAxisReduce(graph, mNode) && mNode.op !== 'broadcast' && isRowShape(mNode.shape);
      for (const p of new Set(mNode.inputs)) {
        if (members.has(p)) continue;
        // A matmul epilogue's own operands are read at arbitrary indices, so they
        // must stay stored values, never members (found by the GPU fuzzer: run 1).
        if (root !== null && nodes[root].inputs.includes(p)) continue;
        const pn = nodes[p];
        if (scalarConst(pn)) { changed = tryAdd([p]) || changed; continue; }
        if (pn.op === 'broadcast' && scalarConst(nodes[pn.inputs[0]])) { changed = tryAdd([p, pn.inputs[0]]) || changed; continue; }
        if (!free(p) || !exclusive(p)) continue;
        if (ELEMENTWISE.has(pn.op) || pn.op === 'reshape') {
          // Same element count by the IR's typing; reshape is free under flat indexing.
          changed = tryAdd([p]) || changed;
          continue;
        }
        if ((pn.op === 'broadcast' || pn.op === 'transpose') && mNode.op !== 'broadcast' && mNode.op !== 'transpose') {
          const src = nodes[pn.inputs[0]];
          // A row-tier value broadcast back over the row: the rowwise pattern.
          if (pn.op === 'broadcast' && patterns.includes('rowwise') && root === null && !rowTier
            && sameShape(pn.shape, S) && isRowShape(src.shape) && C > 1) {
            const region = rowRegion(p);
            if (region && tryAdd([p, ...region])) { kind = 'rowwise'; changed = true; continue; }
          }
          // Recompute across a broadcast or transpose: the producer joins, evaluated at the source index.
          if (recompute && kind !== 'rowwise' && ELEMENTWISE.has(src.op) && free(src.id) && !keep.has(src.id)) {
            if (tryAdd([p, src.id])) { changed = true; continue; }
          }
          // Otherwise an indexed load of the source.
          changed = tryAdd([p]) || changed;
          continue;
        }
        if (pn.op === 'matmul' && patterns.includes('epilogue') && kind === 'elementwise' && root === null
          && size(pn.shape) === n && !rowTier && !pn.inputs.some((i) => members.has(i))) {
          if (tryAdd([p])) { kind = 'epilogue'; root = p; changed = true; }
          continue;
        }
      }
    }
  }
  if (kind === 'elementwise' && !patterns.includes('elementwise')) return null;
  const real = [...members].filter((id) => nodes[id].op !== 'const');
  if (real.length < 2) return null;
  return { members, kind, root };
}

/** The label a rowwise group is reported under (amendment 30). */
function label(graph, members, kind) {
  if (kind !== 'rowwise') return kind === 'epilogue' ? 'matmul-epilogue' : 'elementwise';
  const ops = [...members].map((id) => graph.nodes[id].op);
  const count = (op) => ops.filter((o) => o === op).length;
  if (count('sum') >= 2 && count('sqrt') >= 1) return 'layernorm';
  if (count('max') >= 1 && count('exp') >= 1 && count('sum') >= 1 && count('log') >= 1) return 'log-softmax';
  if (count('max') >= 1 && count('exp') >= 1 && count('sum') >= 1 && count('div') >= 1) return 'softmax';
  return 'rowwise';
}

/** Build the body graph of a group: inputs a0.. for its operands, then its members in order. */
function body(graph, members, sink) {
  const g = new Graph();
  const order = [...members].sort((a, b) => a - b);
  const operands = [];
  const map = new Map();
  for (const id of order) {
    const node = graph.nodes[id];
    for (const i of node.inputs) {
      if (!members.has(i) && !map.has(i)) {
        map.set(i, g.input(`a${operands.length}`, graph.nodes[i].shape));
        operands.push(i);
      }
    }
    map.set(id, g.emit(node.op, node.inputs.map((i) => map.get(i)), node.attrs));
  }
  return { graph: g, operands, output: map.get(sink) };
}

const bodies = new WeakMap();
/** The body of a fused node as a Graph (cached). */
export function bodyOf(node) {
  let b = bodies.get(node);
  if (!b) { b = Graph.fromJSON(node.attrs.body); bodies.set(node, b); }
  return b;
}

/**
 * Fuse `graph`. Options: patterns (default all three), recompute (default
 * false), keep (node ids that must survive, e.g. a step's outputs), sabotage.
 * Returns { graph, origin, index, groups }: origin[newId] = original id,
 * index.get(originalId) = new id for every surviving node.
 */
export function fuse(graph, { patterns = PATTERNS, recompute = false, keep = [], maxOperands = MAX_OPERANDS, sabotage } = {}) {
  const nodes = graph.nodes;
  const consumers = consumersOf(graph);
  const keepSet = new Set(keep);
  const taken = new Set();
  const groups = [];
  const sinkOf = new Map();
  for (let id = nodes.length - 1; id >= 0; id--) {
    const node = nodes[id];
    if (taken.has(id) || !ELEMENTWISE.has(node.op)) continue;
    const gr = grow(graph, id, consumers, taken, keepSet, { patterns, recompute, maxOperands, sabotage });
    if (!gr) continue;
    for (const m of gr.members) if (nodes[m].op !== 'const' && !(nodes[m].op === 'broadcast' && scalarConst(nodes[nodes[m].inputs[0]]))) taken.add(m);
    const group = { sink: id, members: gr.members, kind: gr.kind, root: gr.root, pattern: label(graph, gr.members, gr.kind) };
    groups.push(group);
    sinkOf.set(id, group);
  }

  // Sabotage 5: a member still used outside its group is treated as gone, and
  // that outside consumer is rewired to the group's output.
  const rewire = new Map();
  if (sabotage === 'fusion-absorbs-shared') {
    for (const gr of groups) {
      for (const m of gr.members) {
        if (m === gr.sink || nodes[m].op === 'const') continue;
        if (consumers[m].some((c) => !gr.members.has(c))) rewire.set(m, gr.sink);
      }
    }
  }
  const via = (id) => rewire.get(id) ?? id;

  // Which nodes survive: walk back from the graph's outputs (nodes nobody
  // consumes, and `keep`), through plain nodes' operands and fused groups' operands.
  const needed = new Set();
  const built = new Map();
  for (let id = nodes.length - 1; id >= 0; id--) {
    const node = nodes[id];
    if (consumers[id].length === 0 || keepSet.has(id)) needed.add(id);
    if (!needed.has(id)) continue;
    if (sinkOf.has(id)) {
      const b = body(graph, sinkOf.get(id).members, id);
      built.set(id, b);
      for (const o of b.operands) needed.add(via(o));
    } else {
      for (const i of node.inputs) needed.add(via(i));
    }
  }

  const out = new Graph();
  const index = new Map();
  const origin = [];
  const ref = (id) => {
    const r = via(id);
    if (!index.has(r)) throw new Error(`fuse: node ${r} used before it is built`);
    return index.get(r);
  };
  for (const node of nodes) {
    if (!needed.has(node.id)) continue;
    let id;
    if (built.has(node.id)) {
      const gr = sinkOf.get(node.id);
      const b = built.get(node.id);
      id = out.nodes.length;
      out.nodes.push({ id, op: 'fused', inputs: b.operands.map(ref),
        attrs: { kind: gr.kind, pattern: gr.pattern, body: b.graph.toJSON(), output: b.output }, shape: [...node.shape] });
    } else if (rewire.size && node.op !== 'input' && node.op !== 'const') {
      id = out.nodes.length;
      const inputs = node.inputs.map(ref);
      // A rewired operand may not have the original shape; the sabotaged pass does not check.
      out.nodes.push({ id, op: node.op, inputs, attrs: structuredClone(node.attrs), shape: [...node.shape] });
    } else {
      id = out.emit(node.op, node.inputs.map(ref), node.attrs);
    }
    index.set(node.id, id);
    origin[id] = node.id;
  }
  return { graph: out, origin, index, groups };
}

/** KERNELS plus `fused`, which evaluates the node's body. */
export const FUSED_KERNELS = {
  ...KERNELS,
  fused(node, args) {
    const b = bodyOf(node);
    const values = evaluate(b, Object.fromEntries(args.map((a, k) => [`a${k}`, a])));
    return values[node.attrs.output];
  },
};

/** Evaluate a fused graph on the float64 interpreter. */
export const evaluateFused = (graph, inputs, options = {}) => evaluate(graph, inputs, { ...options, kernels: FUSED_KERNELS });

/**
 * Amendment 29: the fused graph must give bitwise the same float64 values as
 * the original at every surviving node. Returns { identical, compared, first }.
 */
export function checkSemantics(original, fused, inputs) {
  const a = evaluate(original, inputs);
  let b;
  try {
    b = evaluateFused(fused.graph, inputs);
  } catch (error) {
    return { identical: false, compared: 0, first: { error: String(error.message ?? error) } };
  }
  let compared = 0;
  for (let id = 0; id < fused.graph.nodes.length; id++) {
    const o = fused.origin[id];
    const x = a[o], y = b[id];
    if (x.length !== y.length) return { identical: false, compared, first: { node: id, origin: o, reason: 'length' } };
    for (let i = 0; i < x.length; i++) {
      if (!Object.is(x[i], y[i])) return { identical: false, compared, first: { node: id, origin: o, index: i, want: x[i], got: y[i] } };
    }
    compared++;
  }
  return { identical: true, compared, first: null };
}

/** Counts of fused nodes by pattern, and how many launches the fused graph needs. */
export function fusionSummary(original, fused) {
  const launches = (g) => g.nodes.filter((n) => n.op !== 'input' && n.op !== 'const').length;
  const byPattern = {};
  for (const n of fused.graph.nodes) if (n.op === 'fused') byPattern[n.attrs.pattern] = (byPattern[n.attrs.pattern] ?? 0) + 1;
  return { launchesBefore: launches(original), launchesAfter: launches(fused.graph), byPattern };
}
