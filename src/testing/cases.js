// The cases exported for the PyTorch oracle: seeded random programs, the
// softmax composites (checked against torch.softmax itself), and two
// hand-built graphs shaped like the models milestone 3 will train.
//
// Fully deterministic, so CI can rebuild them and confirm that the committed
// golden/cases.json is what this code produces today.

import { Graph } from '../ir/graph.js';
import { size } from '../ir/shapes.js';
import { evaluate } from '../interp.js';
import { randomProgram } from './random-graph-v1.js';
import { randomProgram as randomProgramV2, TARGETS } from './random-graph.js';
import { Rng } from './rng.js';

const RANDOM_CASES = 60;
const DELTA = 1e-3;
const LIMIT = 1e4;

function usable(graph, inputs) {
  let ok = true;
  evaluate(graph, inputs, {
    onNode(node, args, value) {
      for (const v of value) if (!Number.isFinite(v) || Math.abs(v) > LIMIT) ok = false;
      if (node.op === 'log') for (const v of args[0]) if (v < DELTA) ok = false;
      if (node.op === 'div') for (const v of args[1]) if (Math.abs(v) < DELTA) ok = false;
      if (node.op === 'relu' || node.op === 'step') for (const v of args[0]) if (Math.abs(v) < DELTA) ok = false;
    },
  });
  return ok;
}

function draw(rng, graph, outputs, lo = -2, hi = 2) {
  const inputs = {};
  for (const n of graph.inputs()) inputs[n.attrs.name] = rng.array(size(n.shape), lo, hi);
  const cotangents = outputs.map((id) => rng.array(size(graph.nodes[id].shape), -1, 1));
  return { inputs, cotangents };
}

const pack = (name, kind, graph, outputs, point, extra = {}) =>
  ({ name, kind, graph: graph.toJSON(), outputs, ...point, ...extra });

export function buildCases() {
  const cases = [];

  for (let seed = 1; cases.length < RANDOM_CASES; seed++) {
    const p = randomProgram(seed);
    const rng = new Rng(seed + 7919);
    for (let t = 0; t < 20; t++) {
      const point = draw(rng, p.graph, p.outputs);
      if (usable(p.graph, point.inputs)) { cases.push(pack(`random-${seed}`, 'random', p.graph, p.outputs, point)); break; }
    }
  }

  const rng = new Rng(20261009);
  const composites = [
    ['softmax', [5], 0, 3], ['softmax', [3, 4], 1, 3], ['softmax', [3, 4], 0, 3], ['softmax', [2, 3, 4], 1, 3],
    ['logSoftmax', [5], 0, 3], ['logSoftmax', [3, 4], 1, 3], ['logSoftmax', [2, 3, 4], 2, 3],
    // Logits near 800: exp overflows float64 at 709.8, so these pass only
    // because the maximum is subtracted first.
    ['softmax', [2, 6], 1, 800], ['logSoftmax', [2, 6], 1, 800],
  ];
  for (const [fn, shape, axis, scale] of composites) {
    const g = new Graph();
    const x = g.input('x', shape);
    const y = g[fn](x, axis);
    const point = draw(rng, g, [y], scale === 800 ? 790 : -scale, scale === 800 ? 810 : scale);
    cases.push(pack(`${fn}-${shape.join('x')}-axis${axis}${scale === 800 ? '-large' : ''}`, 'composite', g, [y], point,
      { native: { fn, input: 'x', axis } }));
  }

  {
    // A two-layer MLP with a mean negative log-likelihood loss.
    const g = new Graph();
    const x = g.input('x', [4, 3]);
    const w1 = g.input('w1', [3, 5]);
    const b1 = g.input('b1', [5]);
    const w2 = g.input('w2', [5, 3]);
    const b2 = g.input('b2', [3]);
    const onehot = g.const([1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0], [4, 3]);
    const h = g.relu(g.add(g.matmul(x, w1), b1));
    const logp = g.logSoftmax(g.add(g.matmul(h, w2), b2), 1);
    const loss = g.div(g.neg(g.sum(g.sum(g.mul(onehot, logp), 1), 0)), g.const(4));
    for (let t = 0; t < 20; t++) {
      const point = draw(rng, g, [loss]);
      if (usable(g, point.inputs)) { cases.push(pack('mlp-nll', 'model', g, [loss], point)); break; }
    }
  }

  {
    // Single-head attention over a batch of two, sequence 4, width 3.
    const g = new Graph();
    const q = g.input('q', [2, 4, 3]);
    const k = g.input('k', [2, 4, 3]);
    const v = g.input('v', [2, 4, 3]);
    const scores = g.mul(g.matmul(q, g.transposeLast2(k)), g.const(1 / Math.sqrt(3)));
    const out = g.matmul(g.softmax(scores, -1), v);
    cases.push(pack('attention', 'model', g, [out], draw(rng, g, [out])));
  }

  {
    // Axis permutations that are not their own inverse. The random programs
    // rarely produce one (a 3-cycle needs rank 3), and with only those, a
    // transpose kernel indexing with the inverse permutation failed a single
    // case.
    const g = new Graph();
    const x = g.input('x', [2, 3, 4]);
    const w = g.input('w', [2, 4, 3]);
    const t = g.tanh(g.transpose(x, [2, 0, 1]));
    const back = g.transpose(t, [1, 2, 0]);
    const out = g.matmul(back, w);
    cases.push(pack('transpose-cycles', 'model', g, [out], draw(rng, g, [out])));
  }

  // Milestone 1.5: cases from the v2 generator (larger, non-square shapes,
  // size-1 dims, rank 3 and 4, every op live), tagged so the M1 set and its
  // record are untouched.
  const v2 = [
    ...Array.from({ length: 25 }, (_, i) => [`v2-random-${i + 1}`, randomProgramV2(i + 1)]),
    ...TARGETS.flatMap((op) => [1, 2, 3].map((i) => [`v2-${op}-${i}`, randomProgramV2(i, { focus: op })])),
  ];
  for (const [name, p] of v2) {
    const r = new Rng(p.seed * 7 + name.length);
    for (let t = 0; t < 20; t++) {
      const point = draw(r, p.graph, p.outputs);
      if (usable(p.graph, point.inputs)) { cases.push({ ...pack(name, p.focus ? 'targeted' : 'random', p.graph, p.outputs, point), set: 'm1.5' }); break; }
    }
  }

  return cases;
}
