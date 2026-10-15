// A training step as one IR graph (DESIGN.md amendment 14): the MLP's forward
// pass, the mean softmax cross-entropy, autodiff's backward pass, and the
// optimizer's update, all built from IR ops, so the same interpreter, lowering
// and kernels run all of it. Nothing about an optimizer is special-cased in a
// runtime.
//
// Inputs:  W1 b1 W2 b2 (parameters), the optimizer's state, x, y (one-hot),
//          and d0, the loss's cotangent, always 1.
// Outputs: loss (from the parameters before the update), and the new value of
//          every parameter and state tensor.

import { Graph } from '../ir/graph.js';
import { grad } from '../autodiff.js';

export const PARAMS = ['W1', 'b1', 'W2', 'b2'];

/** Optimizer bugs for the sabotage tests (amendment 18), injected in the graph builder. */
export const OPTIMIZER_SABOTAGE = {
  'adam-bias-correction': 'Adam second-moment bias correction computed with beta1',
  'momentum-off-by-one': 'SGD parameter updated with the previous step\'s momentum buffer',
  'weight-decay-sign': 'weight-decay term subtracted from the gradient instead of added',
};

export const SGD = { kind: 'sgd', lr: 0.1, momentum: 0.9, weightDecay: 5e-4 };
export const ADAM = { kind: 'adam', lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 1e-4 };

const shapes = ({ inputs, hidden, classes }) => ({ W1: [inputs, hidden], b1: [hidden], W2: [hidden, classes], b2: [classes] });

/** Names and shapes of the optimizer state for `opt`. */
export function stateSpec(opt, dims) {
  const s = shapes(dims);
  if (opt.kind === 'sgd') return Object.fromEntries(PARAMS.map((p) => [`${p}.buf`, s[p]]));
  return {
    ...Object.fromEntries(PARAMS.flatMap((p) => [[`${p}.m`, s[p]], [`${p}.v`, s[p]]])),
    beta1Pow: [], beta2Pow: [],
  };
}

/** Initial optimizer state: buffers and moments at 0, the beta powers at 1. */
export function initialState(opt, dims) {
  const spec = stateSpec(opt, dims);
  return Object.fromEntries(Object.entries(spec).map(([name, shape]) => {
    const n = shape.reduce((a, d) => a * d, 1);
    return [name, new Array(n).fill(name.endsWith('Pow') ? 1 : 0)];
  }));
}

/** Logits of the MLP, x -> relu(x W1 + b1) W2 + b2. */
export function mlpLogits(g, ids) {
  const h = g.relu(g.add(g.matmul(ids.x, ids.W1), ids.b1));
  return g.add(g.matmul(h, ids.W2), ids.b2);
}

/**
 * The training step graph for an MLP of `dims` ({ inputs, hidden, classes }),
 * a batch of `batch` rows and optimizer `opt`. Returns
 * { graph, inputs: { name: id }, loss, outputs: { name: id } } where outputs
 * maps each parameter and state name to the id of its new value.
 */
export function buildStep(dims, batch, opt, { sabotage } = {}) {
  const g = new Graph();
  const s = shapes(dims);
  const ids = {};
  for (const p of PARAMS) ids[p] = g.input(p, s[p]);
  ids.x = g.input('x', [batch, dims.inputs]);
  ids.y = g.input('y', [batch, dims.classes]);
  const logp = g.logSoftmax(mlpLogits(g, ids), 1);
  // Mean negative log-likelihood of the true class.
  const loss = g.div(g.neg(g.sum(g.sum(g.mul(ids.y, logp), 1), 0)), g.const(batch));

  const G = grad(g, [loss], PARAMS.map((p) => ids[p]));
  const B = G.graph;
  const grads = Object.fromEntries(PARAMS.map((p, k) => [p, G.grads[k]]));
  const state = {};
  for (const [name, shape] of Object.entries(stateSpec(opt, dims))) state[name] = B.input(name, shape);
  const c = (v) => B.const(v);
  const outputs = {};

  // g' = g + wd * p (L2 weight decay, as both PyTorch optimizers apply it).
  const decayed = (p) => (sabotage === 'weight-decay-sign'
    ? B.sub(grads[p], B.mul(c(opt.weightDecay), ids[p]))
    : B.add(grads[p], B.mul(c(opt.weightDecay), ids[p])));

  if (opt.kind === 'sgd') {
    for (const p of PARAMS) {
      const buf = state[`${p}.buf`];
      const next = B.add(B.mul(c(opt.momentum), buf), decayed(p));
      const used = sabotage === 'momentum-off-by-one' ? buf : next;
      outputs[p] = B.sub(ids[p], B.mul(c(opt.lr), used));
      outputs[`${p}.buf`] = next;
    }
  } else {
    // The beta powers are state: beta^t after t steps, so no pow op is needed.
    const b1 = B.mul(state.beta1Pow, c(opt.beta1));
    const b2 = B.mul(state.beta2Pow, c(opt.beta2));
    outputs.beta1Pow = b1;
    outputs.beta2Pow = b2;
    const bc1 = B.sub(c(1), b1);
    const bc2 = B.sub(c(1), sabotage === 'adam-bias-correction' ? b1 : b2);
    // In PyTorch's order: step = lr / bc1; denom = sqrt(v) / sqrt(bc2) + eps.
    const stepSize = B.div(c(opt.lr), bc1);
    const sqrtBc2 = B.sqrt(bc2);
    for (const p of PARAMS) {
      const gp = decayed(p);
      const m = B.add(B.mul(c(opt.beta1), state[`${p}.m`]), B.mul(c(1 - opt.beta1), gp));
      const v = B.add(B.mul(c(opt.beta2), state[`${p}.v`]), B.mul(c(1 - opt.beta2), B.mul(gp, gp)));
      const denom = B.add(B.div(B.sqrt(v), sqrtBc2), c(opt.eps));
      outputs[p] = B.sub(ids[p], B.mul(stepSize, B.div(m, denom)));
      outputs[`${p}.m`] = m;
      outputs[`${p}.v`] = v;
    }
  }
  const inputs = { ...Object.fromEntries(PARAMS.map((p) => [p, ids[p]])), x: ids.x, y: ids.y, ...state };
  return { graph: B, inputs, loss, outputs, cotangent: G.cotangents[0] };
}

/** A graph for evaluation: log-probabilities of `rows` rows, for loss and accuracy. */
export function buildEval(dims, rows) {
  const g = new Graph();
  const s = shapes(dims);
  const ids = {};
  for (const p of PARAMS) ids[p] = g.input(p, s[p]);
  ids.x = g.input('x', [rows, dims.inputs]);
  const logp = g.logSoftmax(mlpLogits(g, ids), 1);
  return { graph: g, logp };
}
