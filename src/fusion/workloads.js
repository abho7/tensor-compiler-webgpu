// Milestone 4's timed workloads (DESIGN.md amendments 34 and 37, revisions R5
// and R6), and what each must move. Pure JavaScript: building a workload's
// graphs, lowering them, counting their bytes and computing their reference
// needs no GPU.
//
// A workload is { id, family, params, variants, gate }. Each variant is a
// program: buffers (with initial data), dispatches (WGSL, buffer bindings,
// grid) and the buffers that hold its outputs. Graph variants come from the
// compiler (unfused, fused, recompute, matmul-only); the rest are
// llm-inference-webgpu's hand-written kernels.

import { Graph } from '../ir/graph.js';
import { size } from '../ir/shapes.js';
import { Rng } from '../testing/rng.js';
import { referenceWithBounds, S, F32_MAX } from '../gpu/bounds.js';
import { lower } from '../gpu/lower.js';
import { fuse } from './fuse.js';
import { lowerFused } from './lower-fused.js';
import { layerNormGraph, geluGraph, unaryChain } from './direct-cases-m4.js';
import { buildForward, initParams } from '../m3b/model.js';
import { buildStep3b, paramShapes } from '../m3b/step.js';
import { adamInitialState } from '../m3b/adam.js';
import { ADD, SWIGLU, RMSNORM, MATVEC, GROUP } from './llm-kernels.js';

export const CHAIN_LENGTHS = [1, 2, 4, 8, 16];
export const CHAIN_SIZES = [1000, 65537, 1048573, 4194301];
export const BINARY_LENGTHS = [1, 2, 4, 8];
export const BINARY_SIZES = [65537, 4194301];
export const MATMUL_SHAPES = [[512, 64, 64], [512, 64, 256], [512, 256, 64], [1000, 63, 257], [4096, 64, 256], [128, 1024, 128], [2048, 512, 512]];
export const ROW_C = [1, 7, 64, 65, 256, 1000, 4096];
export const ROW_R = [1, 64, 4096, 65536];
/** Revision R6: rows are capped at 2^22 elements (amendment 34 said 2^26). */
export const ROW_CAP = 2 ** 22;
export const BROADCAST_R = [16, 256, 4096];
export const REUSE_SIZE = 1048573;

// ---- graphs ---------------------------------------------------------------------

function drawInputs(graph, seed, ranges = {}) {
  const rng = new Rng(seed);
  const out = {};
  for (const n of graph.inputs()) {
    const [lo, hi] = ranges[n.attrs.name] ?? [-2, 2];
    out[n.attrs.name] = Float32Array.from(rng.array(size(n.shape), lo, hi));
  }
  return out;
}

/** A graph workload: the variants are lowered from the graph and its fusions. */
function graphWorkload(id, family, params, build, { variants = ['unfused', 'fused'], matmul = 'naive', ranges, seed = 1, judge = 'composed' } = {}) {
  return { id, family, params, kind: 'graph', matmul, variants, judge,
    make() {
      const g = new Graph();
      const outputs = [].concat(build(g));
      const inputs = drawInputs(g, seed, ranges);
      return { graph: g, outputs, inputs };
    } };
}

const binaryChain = (g, n, L) => {
  let y = g.input('x0', [n]);
  const ops = ['add', 'mul', 'sub'];
  for (let k = 1; k <= L; k++) y = g[ops[(k - 1) % 3]](y, g.input(`x${k}`, [n]));
  return y;
};

const rmsNormGraph = (g, x, gain, eps = 1e-6) => {
  const d = g.shape(x).at(-1);
  const rms = g.sqrt(g.add(g.div(g.sum(g.mul(x, x), 1, true), g.const(d)), g.const(eps)));
  return g.mul(g.div(x, rms), gain);
};

const swigluGraph = (g, gate, up) => g.mul(g.div(gate, g.add(g.const(1), g.exp(g.neg(gate)))), up);

/** Every workload of the sweep, in the order they are timed. */
export function workloads() {
  const w = [];
  for (const n of CHAIN_SIZES) for (const L of CHAIN_LENGTHS) {
    w.push(graphWorkload(`chain-unary-L${L}-n${n}`, 'chain-unary', { L, n }, (g) => unaryChain(g, g.input('x', [n]), L)));
  }
  for (const n of BINARY_SIZES) for (const L of BINARY_LENGTHS) {
    w.push(graphWorkload(`chain-binary-L${L}-n${n}`, 'chain-binary', { L, n }, (g) => binaryChain(g, n, L)));
  }
  for (const matmul of ['naive', 'tiled']) for (const [M, K, N] of MATMUL_SHAPES) for (const act of ['relu', 'gelu']) {
    w.push(graphWorkload(`matmul-${matmul}-${act}-${M}x${K}x${N}`, 'matmul-epilogue', { M, K, N, act, matmul }, (g) => {
      const y = g.add(g.matmul(g.input('a', [M, K]), g.input('w', [K, N])), g.input('bias', [N]));
      return act === 'relu' ? g.relu(y) : geluGraph(g, y);
    }, { matmul, ranges: { w: [-1 / Math.sqrt(K), 1 / Math.sqrt(K)] } }));
  }
  for (const pattern of ['layernorm', 'softmax']) for (const C of ROW_C) for (const R of ROW_R) {
    if (R * C > ROW_CAP) continue;
    w.push(graphWorkload(`${pattern}-R${R}-C${C}`, pattern, { R, C }, (g) => (pattern === 'layernorm'
      ? layerNormGraph(g, g.input('x', [R, C]), g.input('g', [C]), g.input('b', [C]))
      : g.softmax(g.input('x', [R, C]), 1))));
  }
  for (const R of BROADCAST_R) {
    w.push(graphWorkload(`broadcast-recompute-R${R}`, 'broadcast-recompute', { R, C: 256, ops: 16 },
      (g) => g.add(g.input('y', [R, 256]), unaryChain(g, g.input('p', [256]), 16)), { variants: ['unfused', 'fused', 'recompute'] }));
  }
  for (const ops of [1, 16]) {
    w.push(graphWorkload(`lost-reuse-${ops}op`, 'lost-reuse', { ops, consumers: 4, n: REUSE_SIZE }, (g) => {
      const t = unaryChain(g, g.input('x', [REUSE_SIZE]), ops);
      return [g.mul(t, g.const(2)), g.add(t, g.const(1)), g.tanh(t), g.exp(g.neg(t))];
    }, { variants: ['unfused', 'fused', 'recompute'] }));
  }
  // Revision R5: the forward pass, which holds the full LayerNorm, softmax and GELU kernels.
  for (const [V, B] of [[16, 8], [75, 16]]) w.push(modelWorkload('forward', V, B));
  for (const [V, B] of [[16, 8], [16, 16], [75, 16]]) w.push(modelWorkload('step', V, B));
  // Amendment 37: the shapes llm-inference-webgpu's hand-written kernels share with the compiler.
  for (const n of [896, 52 * 896]) {
    w.push({ ...graphWorkload(`baseline-add-n${n}`, 'baseline-add', { n }, (g) => g.add(g.input('x', [n]), g.input('delta', [n])), { variants: ['unfused', 'hand'] }), hand: 'add' });
  }
  for (const n of [4864, 52 * 4864]) {
    w.push({ ...graphWorkload(`baseline-swiglu-n${n}`, 'baseline-swiglu', { n }, (g) => swigluGraph(g, g.input('gate', [n]), g.input('up', [n])), { variants: ['unfused', 'fused', 'hand'] }), hand: 'swiglu' });
  }
  for (const rows of [1, 52]) {
    w.push({ ...graphWorkload(`baseline-rmsnorm-rows${rows}`, 'baseline-rmsnorm', { rows, cols: 896 }, (g) => rmsNormGraph(g, g.input('x', [rows, 896]), g.input('gain', [896])),
      { variants: ['unfused', 'fused', 'hand'], ranges: { gain: [0.5, 2] } }), hand: 'rmsnorm' });
  }
  for (const rows of [896, 4864]) for (const matmul of ['naive', 'tiled']) {
    const cols = 896;
    w.push({ ...graphWorkload(`baseline-matvec-${rows}x${cols}-${matmul}`, 'baseline-matvec', { rows, cols, matmul }, (g) => g.matmul(g.input('x', [1, cols]), g.input('wt', [cols, rows])),
      { matmul, variants: ['unfused', 'hand', 'hand-f32'], ranges: { wt: [-0.1, 0.1] } }), hand: 'matvec' });
  }
  return w;
}

function modelWorkload(what, V, B) {
  return { id: `${what}-V${V}-B${B}`, family: what === 'step' ? 'training-step' : 'forward', params: { V, B }, kind: 'graph', matmul: 'naive',
    variants: what === 'step' ? ['unfused', 'fused', 'matmul-only'] : ['unfused', 'fused'], judge: 'compare',
    make() {
      const rng = new Rng(V * 1000 + B);
      const tokens = Array.from({ length: B * 64 }, () => rng.int(0, V - 1));
      const targets = Array.from({ length: B * 64 }, () => rng.int(0, V - 1));
      const onehot = (ids) => { const a = new Float32Array(ids.length * V); ids.forEach((t, i) => { a[i * V + t] = 1; }); return a; };
      const params = Object.fromEntries(Object.entries(initParams(V, 1)).map(([k, v]) => [k, Float32Array.from(v)]));
      if (what === 'forward') {
        const f = buildForward(V, B);
        return { graph: f.graph, outputs: [f.logits, f.loss], inputs: { ...params, x: onehot(tokens), y: onehot(targets) } };
      }
      const s = buildStep3b(V, B);
      const state = adamInitialState(s.names, paramShapes(V));
      return { graph: s.graph, outputs: [s.loss, ...Object.values(s.outputs)],
        inputs: { ...params, ...state, x: onehot(tokens), y: onehot(targets), [s.cotangent]: new Float32Array([1]) } };
    } };
}

// ---- programs ----------------------------------------------------------------------

/**
 * A lowered graph as a program. Buffers: one per node (inputs and constants
 * carry data), then one parameter block per launch. `only` keeps a subset of
 * the launches (the matmul-only variant).
 */
function graphProgram(graph, lowered, inputs, outputs, { only } = {}) {
  const buffers = graph.nodes.map((n) => {
    const count = Math.max(1, size(n.shape));
    let data;
    if (n.op === 'input') data = Float32Array.from(inputs[n.attrs.name]);
    else if (n.op === 'const') data = typeof n.attrs.value === 'number' ? new Float32Array(count).fill(n.attrs.value) : Float32Array.from(n.attrs.value);
    return { bytes: count * 4, data };
  });
  const dispatches = [];
  let bytes = 0;
  for (const l of lowered.launches) {
    if (only && !only(l)) continue;
    const p = buffers.length;
    buffers.push({ bytes: Math.max(1, l.params.length) * 4, data: l.params });
    dispatches.push({ code: l.code, bindings: [...l.inputs, l.node, p], grid: l.grid, label: l.pattern ?? l.kind });
    // Amendment 33: each operand buffer read once, at its own size, plus the output.
    for (const id of new Set(l.inputs)) bytes += size(graph.nodes[id].shape) * 4;
    bytes += size(graph.nodes[l.node].shape) * 4;
  }
  return { buffers, dispatches, outputs, bytes, launches: dispatches.length };
}

const f16 = (values) => {
  // Round-to-nearest-even float32 -> float16 bits, as the hand-written kernels store weights.
  const out = new Uint16Array(values.length);
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  for (let i = 0; i < values.length; i++) {
    f[0] = values[i];
    const x = u[0], sign = (x >>> 16) & 0x8000, exp = ((x >>> 23) & 0xff) - 127 + 15, mant = x & 0x7fffff;
    if (exp >= 31) { out[i] = sign | 0x7c00; continue; }
    if (exp <= 0) {
      if (exp < -10) { out[i] = sign; continue; }
      const m = mant | 0x800000, shift = 14 - exp;
      let h = m >>> shift; const rem = m & ((1 << shift) - 1), half = 1 << (shift - 1);
      if (rem > half || (rem === half && (h & 1))) h++;
      out[i] = sign | h; continue;
    }
    let h = sign | (exp << 10) | (mant >>> 13);
    const rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
    out[i] = h;
  }
  return out;
};
export const f16ToNumber = (h) => {
  const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
};
export { f16 as toF16 };

const u32 = (...v) => Uint32Array.from(v);

/** A hand-written kernel as a program, on the workload's own inputs (f16 where the kernel stores f16). */
function handProgram(w, inputs, variant) {
  if (w.hand === 'add') {
    const n = w.params.n;
    return { buffers: [{ bytes: n * 4, data: inputs.x }, { bytes: n * 4, data: inputs.delta }, { bytes: 16, data: u32(n, 0, 0, 0), uniform: true }],
      dispatches: [{ code: ADD, bindings: [0, 1, 2], grid: [Math.ceil(n / GROUP), 1, 1], label: 'ADD' }], outputs: [0], bytes: 3 * n * 4, launches: 1, inPlace: true };
  }
  if (w.hand === 'swiglu') {
    const n = w.params.n;
    return { buffers: [{ bytes: n * 4, data: inputs.gate }, { bytes: n * 4, data: inputs.up }, { bytes: 16, data: u32(n, 0, 0, 0), uniform: true }],
      dispatches: [{ code: SWIGLU, bindings: [0, 1, 2], grid: [Math.ceil(n / GROUP), 1, 1], label: 'SWIGLU' }], outputs: [0], bytes: 3 * n * 4, launches: 1, inPlace: true };
  }
  if (w.hand === 'rmsnorm') {
    const { rows, cols } = w.params;
    const params = new ArrayBuffer(16);
    new Uint32Array(params, 0, 2).set([rows, cols]);
    new Float32Array(params, 8, 1)[0] = 1e-6;
    return { buffers: [{ bytes: rows * cols * 4, data: inputs.x }, { bytes: cols * 2, data: f16(inputs.gain) }, { bytes: rows * cols * 4 }, { bytes: 16, data: new Uint32Array(params), uniform: true }],
      dispatches: [{ code: RMSNORM, bindings: [0, 1, 2, 3], grid: [rows, 1, 1], label: 'RMSNORM' }], outputs: [2], bytes: (2 * rows * cols * 4) + cols * 2, launches: 1, f16Inputs: ['gain'] };
  }
  if (w.hand === 'matvec') {
    // y = W x with W [rows, cols]; the compiler's version multiplies x [1, cols] by wt = Wᵀ [cols, rows].
    const { rows, cols } = w.params;
    const W = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) W[r * cols + c] = inputs.wt[c * rows + r];
    const asF32 = variant === 'hand-f32';
    const code = asF32 ? MATVEC.replace('enable f16;\n', '').replace('array<f16>', 'array<f32>') : MATVEC;
    return { buffers: [{ bytes: rows * cols * (asF32 ? 4 : 2), data: asF32 ? W : f16(W) }, { bytes: cols * 4, data: inputs.x }, { bytes: 4, data: new Float32Array(1) },
      { bytes: rows * 4 }, { bytes: 16, data: u32(rows, cols, 0, 32768), uniform: true }],
    dispatches: [{ code, bindings: [0, 1, 2, 3, 4], grid: [Math.min(rows, 32768), Math.ceil(rows / 32768), 1], label: asF32 ? 'MATVEC (f32 weights)' : 'MATVEC' }],
    outputs: [3], bytes: rows * cols * (asF32 ? 4 : 2) + cols * 4 + rows * 4, launches: 1, f16Inputs: asF32 ? [] : ['wt'] };
  }
  throw new Error(`no hand-written kernel '${w.hand}'`);
}

/**
 * Build every variant of a workload: { graph, outputs, inputs, programs } where
 * programs[variant] is a program and, for graph variants, `outputIds` maps
 * each of the workload's outputs to its buffer.
 */
export function buildVariants(w) {
  const { graph, outputs, inputs } = w.make();
  const programs = {};
  for (const v of w.variants) {
    if (v === 'unfused' || v === 'matmul-only') {
      const lowered = lower(graph, { matmul: w.matmul });
      programs[v] = graphProgram(graph, lowered, inputs, outputs, v === 'matmul-only' ? { only: (l) => l.op === 'matmul' } : {});
    } else if (v === 'fused' || v === 'recompute') {
      const f = fuse(graph, { keep: outputs, recompute: v === 'recompute' });
      programs[v] = { ...graphProgram(f.graph, lowerFused(f.graph, { matmul: w.matmul }), inputs, outputs.map((o) => f.index.get(o))),
        fusedNodes: f.graph.nodes.filter((n) => n.op === 'fused').length };
    } else {
      programs[v] = handProgram(w, inputs, v);
    }
  }
  return { graph, outputs, inputs, programs };
}

// ---- the correctness gate (amendment 32, revision R6) -------------------------------------------

/**
 * Amendment 8's composed bound for `outputs` of `graph`, computed one node at
 * a time so that only live values are held: each node is a one-op graph whose
 * operands are carried in with their value and bound (bounds.js's `carried`).
 * Gives the same values and bounds as referenceWithBounds on the whole graph.
 */
export function streamingReference(graph, inputs, outputs) {
  const lastUse = graph.nodes.map(() => -1);
  for (const n of graph.nodes) for (const i of n.inputs) lastUse[i] = n.id;
  const keep = new Set(outputs);
  const live = new Map();
  const result = new Map();
  for (const node of graph.nodes) {
    let value, bound;
    if (node.op === 'input' || node.op === 'const') {
      const g = new Graph();
      const id = node.op === 'input' ? g.input(node.attrs.name, node.shape) : g.const(node.attrs.value, node.shape);
      const r = referenceWithBounds(g, node.op === 'input' ? { [node.attrs.name]: inputs[node.attrs.name] } : {});
      value = r.values[id]; bound = r.bounds[id];
    } else {
      const g = new Graph();
      const carried = {};
      const ins = node.inputs.map((id, k) => { carried[`a${k}`] = live.get(id); return g.input(`a${k}`, graph.nodes[id].shape); });
      const out = g.emit(node.op, ins, node.attrs);
      const r = referenceWithBounds(g, {}, { carried });
      value = r.values[out]; bound = r.bounds[out];
    }
    live.set(node.id, { value, bound });
    if (keep.has(node.id)) result.set(node.id, { value, bound });
    for (const i of node.inputs) if (lastUse[i] === node.id && !keep.has(i)) live.delete(i);
  }
  return result;
}

// ---- the bandwidth ceiling (amendment 33) ---------------------------------------------

export const CEILING_BYTES = 128 * 2 ** 20;

/**
 * llm-inference-webgpu's pure-read kernel (decode-profile.html), and a copy
 * kernel of the same shape that also writes, over a 128 MiB buffer: u32, vec2
 * and vec4 loads, 256, 1,024 and 4,096 workgroups of 256 threads, grid-strided.
 */
export function ceilingPrograms() {
  const out = [];
  for (const mode of ['read', 'copy']) for (const width of ['u32', 'vec2', 'vec4']) for (const groups of [256, 1024, 4096]) {
    const T = width === 'vec4' ? 'vec4<u32>' : width === 'vec2' ? 'vec2<u32>' : 'u32';
    const fold = width === 'vec4' ? 'acc.x ^ acc.y ^ acc.z ^ acc.w' : width === 'vec2' ? 'acc.x ^ acc.y' : 'acc';
    const zero = width === 'u32' ? '0u' : `${T}(0u)`;
    const code = mode === 'read' ? `
@group(0) @binding(0) var<storage, read> src : array<${T}>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(num_workgroups) nw : vec3<u32>) {
  let n = arrayLength(&src);
  let stride = nw.x * 256u;
  var acc = ${zero};
  var i = g.x;
  loop { if (i >= n) { break; } acc = acc ^ src[i]; i = i + stride; }
  dst[g.x] = ${fold};
}` : `
@group(0) @binding(0) var<storage, read> src : array<${T}>;
@group(0) @binding(1) var<storage, read_write> dst : array<${T}>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g : vec3<u32>, @builtin(num_workgroups) nw : vec3<u32>) {
  let n = arrayLength(&src);
  let stride = nw.x * 256u;
  var i = g.x;
  loop { if (i >= n) { break; } dst[i] = src[i]; i = i + stride; }
}`;
    out.push({ id: `${mode}-${width}-x${groups}`, mode, width, groups,
      program: { buffers: [{ bytes: CEILING_BYTES }, { bytes: mode === 'read' ? groups * 256 * 4 : CEILING_BYTES }],
        dispatches: [{ code, bindings: [0, 1], grid: [groups, 1, 1], label: `${mode} ${width}` }], outputs: [],
        bytes: mode === 'read' ? CEILING_BYTES : 2 * CEILING_BYTES, launches: 1 } });
  }
  return out;
}

// ---- judging a variant's outputs -------------------------------------------------------


/** Outputs against streamingReference(): amendment 8's budget, S = 2, with amendment 22's 1% rule. */
export function judgeOutputs(ref, outputs, gpu) {
  let worst = 0, first = null, unjudgeableCount = 0, elements = 0;
  outputs.forEach((id, k) => {
    const { value, bound } = ref.get(id);
    const g = gpu[k];
    for (let i = 0; i < value.length; i++) {
      elements++;
      if (!Number.isFinite(value[i]) || Math.abs(value[i]) > F32_MAX || !Number.isFinite(bound[i])) { unjudgeableCount++; continue; }
      const d = Math.abs(g[i] - value[i]);
      const ratio = d === 0 ? 0 : (Number.isNaN(d) ? Infinity : d / (S * bound[i]));
      if (ratio > worst) worst = ratio;
      if (ratio > 1 && !first) first = { output: k, index: i, gpu: g[i], ref: value[i], bound: bound[i], ratio };
    }
  });
  const unjudged = unjudgeableCount > 0.01 * elements;
  return { pass: !first && !unjudged, unjudged, worst, unjudgeable: unjudgeableCount, elements, first };
}

/** Revision R6's gate for the model workloads: the largest difference between two variants' outputs, per output. */
export function compareOutputs(a, b) {
  const maxAbs = a.map((x, k) => { let m = 0; for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i] - b[k][i])); return m; });
  const finite = a.every((x) => x.every(Number.isFinite)) && b.every((x) => x.every(Number.isFinite));
  return { finite, maxAbs: Math.max(...maxAbs), lossDiff: maxAbs[0] };
}
