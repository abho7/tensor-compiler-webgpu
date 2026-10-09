// The float64 reference interpreter: the definition of what a graph means.
//
// Deliberately plain. Every value is a fresh Float64Array, every loop runs in
// index order, and every reduction sums its terms first to last. Later
// backends are judged against this, and this is judged against PyTorch, with
// which it shares no code.

import { size } from './ir/shapes.js';

export function strides(shape) {
  const s = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
  return s;
}

const map1 = (f) => (node, [x]) => {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = f(x[i]);
  return out;
};

const map2 = (f) => (node, [a, b]) => {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = f(a[i], b[i]);
  return out;
};

/** [outer, n, inner] for reducing `shape` along `axis`. */
function reduceDims(shape, axis) {
  return [size(shape.slice(0, axis)), shape[axis], size(shape.slice(axis + 1))];
}

const reduce = (init, f) => (node, [x], g) => {
  const [outer, n, inner] = reduceDims(g.nodes[node.inputs[0]].shape, node.attrs.axis);
  const out = new Float64Array(outer * inner);
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      let acc = init;
      for (let k = 0; k < n; k++) acc = f(acc, x[(o * n + k) * inner + i]);
      out[o * inner + i] = acc;
    }
  }
  return out;
};

/** The op table. Each entry maps (node, input arrays, graph) to a new Float64Array. */
export const KERNELS = {
  const(node) {
    const n = size(node.shape);
    const v = node.attrs.value;
    return typeof v === 'number' ? new Float64Array(n).fill(v) : Float64Array.from(v);
  },
  add: map2((a, b) => a + b),
  sub: map2((a, b) => a - b),
  mul: map2((a, b) => a * b),
  div: map2((a, b) => a / b),
  neg: map1((x) => -x),
  exp: map1(Math.exp),
  log: map1(Math.log),
  relu: map1((x) => (x > 0 ? x : 0)),
  tanh: map1(Math.tanh),
  step: map1((x) => (x > 0 ? 1 : 0)),
  stopGradient: map1((x) => x),

  matmul(node, [a, b], g) {
    const as = g.nodes[node.inputs[0]].shape;
    const bs = g.nodes[node.inputs[1]].shape;
    const m = as.at(-2), k = as.at(-1), n = bs.at(-1);
    const batch = size(as.slice(0, -2));
    const out = new Float64Array(batch * m * n);
    for (let t = 0; t < batch; t++) {
      const ao = t * m * k, bo = t * k * n, oo = t * m * n;
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let acc = 0;
          for (let p = 0; p < k; p++) acc += a[ao + i * k + p] * b[bo + p * n + j];
          out[oo + i * n + j] = acc;
        }
      }
    }
    return out;
  },

  sum: reduce(0, (acc, v) => acc + v),
  max: reduce(-Infinity, (acc, v) => (v > acc ? v : acc)),

  broadcast(node, [x], g) {
    const from = g.nodes[node.inputs[0]].shape;
    const to = node.shape;
    const lead = to.length - from.length;
    const fs = strides(from);
    const ts = strides(to);
    const out = new Float64Array(size(to));
    for (let i = 0; i < out.length; i++) {
      let src = 0;
      for (let d = 0; d < from.length; d++) {
        if (from[d] === 1) continue;
        const coord = Math.floor(i / ts[lead + d]) % to[lead + d];
        src += coord * fs[d];
      }
      out[i] = x[src];
    }
    return out;
  },

  reshape: (node, [x]) => Float64Array.from(x),

  transpose(node, [x], g) {
    const from = g.nodes[node.inputs[0]].shape;
    const perm = node.attrs.perm;
    const fs = strides(from);
    const ts = strides(node.shape);
    const out = new Float64Array(x.length);
    for (let i = 0; i < out.length; i++) {
      // Output axis d is input axis perm[d].
      let src = 0;
      for (let d = 0; d < perm.length; d++) {
        const coord = Math.floor(i / ts[d]) % node.shape[d];
        src += coord * fs[perm[d]];
      }
      out[i] = x[src];
    }
    return out;
  },
};

/**
 * Evaluate every node of `graph`. `inputs` maps input names to arrays of
 * numbers in row-major order. Returns an array of Float64Arrays, one per node.
 *
 * `kernels` replaces the op table (the interpreter's own sabotage tests use
 * it); `onNode(node, inputValues, value)` sees every node as it is computed.
 */
export function evaluate(graph, inputs, { kernels = KERNELS, onNode } = {}) {
  const values = new Array(graph.nodes.length);
  for (const node of graph.nodes) {
    let value;
    if (node.op === 'input') {
      const v = inputs[node.attrs.name];
      if (v === undefined) throw new Error(`no value for input '${node.attrs.name}'`);
      if (v.length !== size(node.shape)) {
        throw new Error(`input '${node.attrs.name}': expected ${size(node.shape)} numbers, got ${v.length}`);
      }
      value = Float64Array.from(v);
    } else {
      const kernel = kernels[node.op];
      if (!kernel) throw new Error(`no kernel for '${node.op}'`);
      const args = node.inputs.map((id) => values[id]);
      value = kernel(node, args, graph);
      if (onNode) onNode(node, args, value);
    }
    values[node.id] = value;
  }
  return values;
}

/** Evaluate and return only the values of `outputs` (node ids). */
export function run(graph, inputs, outputs, options) {
  const values = evaluate(graph, inputs, options);
  return outputs.map((id) => values[id]);
}
