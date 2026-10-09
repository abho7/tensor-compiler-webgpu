// Reverse-mode automatic differentiation, as a graph-to-graph transform.
//
// grad() copies the forward graph, adds one `input` per output to carry its
// cotangent, then walks the forward nodes last to first, applying each op's
// rule to the cotangent accumulated so far. The rules build their results
// with the ordinary builder, so the gradient graph is an ordinary graph: the
// same interpreter runs it, the same code generator will lower it, and it can
// be differentiated again.
//
// The rules live in one table, and grad() accepts a replacement, so a test
// can swap a single rule for a wrong one without editing this file.

import { Graph } from './ir/graph.js';
import { sameShape } from './ir/shapes.js';

export class GradientError extends Error {
  constructor(message) { super(message); this.name = 'GradientError'; }
}

/**
 * Each rule takes (G, node, gy), where G is the graph being built, `node` the
 * forward node and `gy` the id of its cotangent, and returns one entry per
 * input of `node`: the id of that input's contribution, or null for none.
 */
export const RULES = {
  add: (G, n, gy) => [gy, gy],
  sub: (G, n, gy) => [gy, G.neg(gy)],
  mul: (G, n, gy) => {
    const [a, b] = n.inputs;
    return [G.mul(gy, b), G.mul(gy, a)];
  },
  div: (G, n, gy) => {
    const [a, b] = n.inputs;
    return [G.div(gy, b), G.neg(G.div(G.mul(gy, a), G.mul(b, b)))];
  },
  neg: (G, n, gy) => [G.neg(gy)],
  exp: (G, n, gy) => [G.mul(gy, n.id)],
  log: (G, n, gy) => [G.div(gy, n.inputs[0])],
  relu: (G, n, gy) => [G.mul(gy, G.step(n.inputs[0]))],
  tanh: (G, n, gy) => [G.mul(gy, G.sub(G.const(1), G.mul(n.id, n.id)))],
  matmul: (G, n, gy) => {
    const [a, b] = n.inputs;
    return [G.matmul(gy, G.transposeLast2(b)), G.matmul(G.transposeLast2(a), gy)];
  },
  sum: (G, n, gy) => {
    const x = n.inputs[0];
    const xs = G.shape(x);
    const kept = xs.map((d, i) => (i === n.attrs.axis ? 1 : d));
    return [G.broadcast(G.reshape(gy, kept), xs)];
  },
  broadcast: (G, n, gy) => {
    const xs = G.shape(n.inputs[0]);
    const lead = n.shape.length - xs.length;
    let g = gy;
    // Axes the broadcast added in front: sum them away.
    for (let i = 0; i < lead; i++) g = G.sum(g, 0, false);
    // Axes it stretched from 1: sum them, keeping the axis.
    xs.forEach((d, i) => {
      if (d === 1 && n.shape[lead + i] !== 1) g = G.sum(g, i, true);
    });
    return [G.reshape(g, xs)];
  },
  reshape: (G, n, gy) => [G.reshape(gy, G.shape(n.inputs[0]))],
  transpose: (G, n, gy) => {
    const inverse = [];
    n.attrs.perm.forEach((p, i) => { inverse[p] = i; });
    return [G.transpose(gy, inverse)];
  },
  step: () => [null],
  stopGradient: () => [null],
  max: () => {
    throw new GradientError('max has no gradient; wrap it in stopGradient where its gradient is zero (as softmax does)');
  },
};

/**
 * The gradient graph of `graph`.
 *
 * `outputs` are node ids; `wrt` are node ids to differentiate with respect to.
 * Returns { graph, cotangents, grads }: the new graph (the forward nodes keep
 * their ids), the names of the inputs to feed each output's cotangent, and the
 * id of each gradient, in the order of `wrt`. A `wrt` the outputs do not
 * depend on gets a constant zero.
 */
export function grad(graph, outputs, wrt, { rules = RULES, prefix = 'd' } = {}) {
  const G = graph.clone();
  const forwardCount = graph.nodes.length;

  // Only nodes that depend on some `wrt` need a gradient; the rest are
  // constants as far as this transform is concerned.
  const dependsOnWrt = new Array(forwardCount).fill(false);
  for (const id of wrt) dependsOnWrt[id] = true;
  for (const node of graph.nodes) {
    if (node.inputs.some((i) => dependsOnWrt[i])) dependsOnWrt[node.id] = true;
  }

  const cot = new Array(forwardCount).fill(null);
  const accumulate = (id, contribution) => {
    if (!sameShape(G.shape(contribution), graph.nodes[id].shape)) {
      throw new GradientError(`gradient for node ${id} has shape [${G.shape(contribution)}], value has [${graph.nodes[id].shape}]`);
    }
    cot[id] = cot[id] === null ? contribution : G.add(cot[id], contribution);
  };

  const cotangents = outputs.map((id, i) => {
    const name = `${prefix}${i}`;
    accumulate(id, G.input(name, graph.nodes[id].shape));
    return name;
  });

  for (let id = forwardCount - 1; id >= 0; id--) {
    const node = graph.nodes[id];
    if (cot[id] === null || !dependsOnWrt[id] || node.inputs.length === 0) continue;
    if (!node.inputs.some((i) => dependsOnWrt[i])) continue;
    const rule = rules[node.op];
    if (!rule) throw new GradientError(`no gradient rule for '${node.op}'`);
    const contributions = rule(G, node, cot[id]);
    node.inputs.forEach((input, k) => {
      if (contributions[k] !== null && dependsOnWrt[input]) accumulate(input, contributions[k]);
    });
  }

  const grads = wrt.map((id) => cot[id] ?? G.const(0, [...graph.nodes[id].shape]));
  return { graph: G, cotangents, grads };
}

export { Graph };
