// Plausible wrong versions of every gradient rule, and of a few interpreter
// kernels, for testing the tests.
//
// Each is the kind of mistake that gets written: a dropped sign, the result
// used where the input belongs, a missing square, a reshape where a transpose
// belongs, a mean where a sum belongs. A checker that cannot tell one of these
// from the real rule is not checking that rule.

import { RULES } from '../autodiff.js';
import { KERNELS } from '../interp.js';

/** rule name -> { bug, rule }: a description and the wrong rule. */
export const RULE_SABOTAGE = {
  add: {
    bug: 'second operand receives -g',
    rule: (G, n, gy) => [gy, G.neg(gy)],
  },
  sub: {
    bug: 'second operand receives +g (sign dropped)',
    rule: (G, n, gy) => [gy, gy],
  },
  mul: {
    bug: 'each operand multiplied by itself instead of the other',
    rule: (G, n, gy) => [G.mul(gy, n.inputs[0]), G.mul(gy, n.inputs[1])],
  },
  div: {
    bug: 'denominator gradient missing its minus sign',
    rule: (G, n, gy) => {
      const [a, b] = n.inputs;
      return [G.div(gy, b), G.div(G.mul(gy, a), G.mul(b, b))];
    },
  },
  neg: {
    bug: 'passes g through unnegated',
    rule: (G, n, gy) => [gy],
  },
  exp: {
    bug: 'multiplies by the input instead of the result',
    rule: (G, n, gy) => [G.mul(gy, n.inputs[0])],
  },
  log: {
    bug: 'divides by the result instead of the input',
    rule: (G, n, gy) => [G.div(gy, n.id)],
  },
  relu: {
    bug: 'no mask: passes g through everywhere',
    rule: (G, n, gy) => [gy],
  },
  tanh: {
    bug: '1 - y instead of 1 - y²',
    rule: (G, n, gy) => [G.mul(gy, G.sub(G.const(1), n.id))],
  },
  matmul: {
    bug: 'aᵀ formed by reshape instead of transpose',
    rule: (G, n, gy) => {
      const [a, b] = n.inputs;
      const as = G.shape(a);
      const wrongT = G.reshape(a, [...as.slice(0, -2), as.at(-1), as.at(-2)]);
      return [G.matmul(gy, G.transposeLast2(b)), G.matmul(wrongT, gy)];
    },
  },
  sum: {
    bug: 'mean instead of sum: g divided by the axis length',
    rule: (G, n, gy) => {
      const [contribution] = RULES.sum(G, n, gy);
      const len = G.shape(n.inputs[0])[n.attrs.axis];
      return [G.div(contribution, G.const(len))];
    },
  },
  broadcast: {
    bug: 'mean instead of sum over the expanded axes',
    rule: (G, n, gy) => {
      const [contribution] = RULES.broadcast(G, n, gy);
      const copies = n.shape.reduce((p, d) => p * d, 1) / G.shape(n.inputs[0]).reduce((p, d) => p * d, 1);
      return [G.div(contribution, G.const(copies))];
    },
  },
  reshape: {
    bug: 'last two axes swapped before reshaping back (a layout mix-up)',
    rule: (G, n, gy) => {
      const g = G.shape(gy).length >= 2 ? G.transposeLast2(gy) : gy;
      return [G.reshape(g, G.shape(n.inputs[0]))];
    },
  },
  transpose: {
    bug: 'applies perm again instead of its inverse',
    rule: (G, n, gy) => {
      const xs = G.shape(n.inputs[0]);
      const again = G.transpose(gy, n.attrs.perm);
      // Only shape-correct when perm is its own inverse; otherwise reshape so
      // the mistake shows up as wrong values rather than a shape error.
      return [G.reshape(again, xs)];
    },
  },
  stopGradient: {
    bug: 'lets the gradient through, like identity',
    rule: (G, n, gy) => [gy],
  },
  // Milestone 3a (DESIGN.md amendment 18).
  sqrt: {
    bug: 'missing the factor of one half: g / y',
    rule: (G, n, gy) => [G.div(gy, n.id)],
  },
};

/**
 * The rules milestones 1 and 1.5 sabotaged, frozen in their order: their
 * records iterate this list, so a rule added later (sqrt, milestone 3a) does
 * not change them.
 */
export const M1_SABOTAGED_RULES = ['add', 'sub', 'mul', 'div', 'neg', 'exp', 'log', 'relu', 'tanh',
  'matmul', 'sum', 'broadcast', 'reshape', 'transpose', 'stopGradient'];

/** Rules with the one named rule replaced by its sabotaged version. */
export function sabotagedRules(name) {
  return { ...RULES, [name]: RULE_SABOTAGE[name].rule };
}

/**
 * Wrong interpreter kernels. These make the forward pass itself wrong, which
 * finite differences cannot see (they differentiate whatever the interpreter
 * computes); the PyTorch golden files must.
 */
export const KERNEL_SABOTAGE = {
  tanh: { bug: 'tanh computed as x / (1 + |x|)', kernel: (n, [x]) => x.map((v) => v / (1 + Math.abs(v))) },
  transpose: {
    bug: 'indexes with the inverse permutation',
    kernel(node, args, g) {
      const inverse = [];
      node.attrs.perm.forEach((p, i) => { inverse[p] = i; });
      return KERNELS.transpose({ ...node, attrs: { perm: inverse } }, args, g);
    },
  },
  sum: {
    bug: 'drops the last term of every reduction',
    kernel(node, [x], g) {
      const shape = g.nodes[node.inputs[0]].shape;
      const axis = node.attrs.axis;
      const trimmed = shape.map((d, i) => (i === axis && d > 1 ? d - 1 : d));
      if (trimmed[axis] === shape[axis]) return KERNELS.sum(node, [x], g);
      const outer = shape.slice(0, axis).reduce((p, d) => p * d, 1);
      const inner = shape.slice(axis + 1).reduce((p, d) => p * d, 1);
      const n = shape[axis];
      const out = new Float64Array(outer * inner);
      for (let o = 0; o < outer; o++) for (let i = 0; i < inner; i++) {
        let acc = 0;
        for (let k = 0; k < n - 1; k++) acc += x[(o * n + k) * inner + i];
        out[o * inner + i] = acc;
      }
      return out;
    },
  },
  exp: { bug: 'exp with a relative error of 1e-9, far below what finite differences resolve', kernel: (n, [x]) => x.map((v) => Math.exp(v) * (1 + 1e-9)) },
};

export function sabotagedKernels(name) {
  return { ...KERNELS, [name]: KERNEL_SABOTAGE[name].kernel };
}
