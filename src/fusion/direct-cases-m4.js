// Direct adversarial tests for the fused kernels (DESIGN.md amendment 31):
// small graphs that fuse into the intended kind, on the inputs most likely to
// break it. They carry the correctness claim per fused kernel kind.
//
// Each case: { name, kind, pattern, graph, inputs, matmul? }. The graph is the
// unfused original; the harness fuses it and checks that the fused node it
// expects is there.

import { Graph } from '../ir/graph.js';
import { size } from '../ir/shapes.js';
import { Rng } from '../testing/rng.js';

const draw = (rng, n, [lo, hi]) => Array.from({ length: n }, () => {
  if (lo < 0 && hi > 0 && hi > 1e3) { const m = Math.exp(rng.uniform(Math.log(hi) - 20, Math.log(hi))); return rng.chance(0.5) ? m : -m; }
  if (lo > 0 && hi / lo > 100) return Math.exp(rng.uniform(Math.log(lo), Math.log(hi)));
  return rng.uniform(lo, hi);
});

/** Milestone 3b's LayerNorm, as model.js builds it. */
export function layerNormGraph(g, x, gain, bias, eps = 1e-5) {
  const d = g.shape(x).at(-1);
  const axis = g.shape(x).length - 1;
  const mean = g.div(g.sum(x, axis, true), g.const(d));
  const xc = g.sub(x, mean);
  const variance = g.div(g.sum(g.mul(xc, xc), axis, true), g.const(d));
  const y = g.div(xc, g.sqrt(g.add(variance, g.const(eps))));
  return g.add(g.mul(y, gain), bias);
}

/** GELU, tanh form, as model.js builds it. */
export function geluGraph(g, x) {
  const cube = g.mul(x, g.mul(x, x));
  const inner = g.mul(g.const(Math.sqrt(2 / Math.PI)), g.add(x, g.mul(g.const(0.044715), cube)));
  return g.mul(g.mul(g.const(0.5), x), g.add(g.const(1), g.tanh(inner)));
}

/** A unary chain of length L: the ops cycle so every value stays finite and in range. */
export function unaryChain(g, x, L) {
  // tanh -> (-1, 1), exp -> (0.37, 2.7), neg, exp -> (0.07, 0.69), sqrt, log -> (-1.4, -0.2), and round again.
  const cycle = [(v) => g.tanh(v), (v) => g.exp(v), (v) => g.neg(v), (v) => g.exp(v), (v) => g.sqrt(v), (v) => g.log(v)];
  let y = x;
  for (let k = 0; k < L; k++) y = cycle[k % cycle.length](y);
  return y;
}

function one(name, kind, pattern, build, inputs, extra = {}) {
  const g = new Graph();
  build(g);
  return { name, kind, pattern, graph: g, inputs, ...extra };
}

export function fusedDirectCases() {
  const rng = new Rng(20261019);
  const cases = [];
  const SIZES = [[1], [63], [64], [65], [1000], [100003], [3, 5, 7], [1, 7, 1]];

  // Elementwise: a unary chain, a binary chain, and loads through broadcast and transpose.
  for (const shape of SIZES) {
    const n = size(shape);
    cases.push(one(`chain of 6 [${shape}]`, 'elementwise', 'elementwise', (g) => unaryChain(g, g.input('x', shape), 6), { x: draw(rng, n, [-2, 2]) }));
    cases.push(one(`a*b+c-d/e [${shape}]`, 'elementwise', 'elementwise',
      (g) => g.sub(g.add(g.mul(g.input('a', shape), g.input('b', shape)), g.input('c', shape)), g.div(g.input('d', shape), g.input('e', shape))),
      { a: draw(rng, n, [-2, 2]), b: draw(rng, n, [-2, 2]), c: draw(rng, n, [-2, 2]), d: draw(rng, n, [-2, 2]), e: draw(rng, n, [0.5, 2]) }));
  }
  for (const [cls, range] of [['large', [-1e18, 1e18]], ['small', [-1e-18, 1e-18]], ['subnormal', [-1e-40, 1e-40]]]) {
    cases.push(one(`a*b+c ${cls} [1000]`, 'elementwise', 'elementwise',
      (g) => g.add(g.mul(g.input('a', [1000]), g.input('b', [1000])), g.input('c', [1000])),
      { a: draw(rng, 1000, range), b: draw(rng, 1000, [-2, 2]), c: draw(rng, 1000, range) }));
  }
  cases.push(one('relu(x)*2 exact zeros [65]', 'elementwise', 'elementwise', (g) => g.mul(g.relu(g.input('x', [65])), g.const(2)), { x: new Array(65).fill(0) }));
  cases.push(one('x + broadcast(b) [3,65] <- [65]', 'elementwise', 'elementwise', (g) => g.tanh(g.add(g.input('x', [3, 65]), g.input('b', [65]))),
    { x: draw(rng, 195, [-2, 2]), b: draw(rng, 65, [-2, 2]) }));
  cases.push(one('x + broadcast(b) [2,4,3,5] <- [2,1,3,1]', 'elementwise', 'elementwise', (g) => g.exp(g.sub(g.input('x', [2, 4, 3, 5]), g.input('b', [2, 1, 3, 1]))),
    { x: draw(rng, 120, [-2, 2]), b: draw(rng, 6, [-2, 2]) }));
  cases.push(one('transpose(x) * y [5,7] <- [7,5]', 'elementwise', 'elementwise', (g) => g.mul(g.transpose(g.input('x', [7, 5]), [1, 0]), g.neg(g.input('y', [5, 7]))),
    { x: draw(rng, 35, [-2, 2]), y: draw(rng, 35, [-2, 2]) }));
  cases.push(one('transpose(x) + y [2,3,4] perm [2,0,1]', 'elementwise', 'elementwise', (g) => g.add(g.transpose(g.input('x', [3, 4, 2]), [2, 0, 1]), g.exp(g.input('y', [2, 3, 4]))),
    { x: draw(rng, 24, [-2, 2]), y: draw(rng, 24, [-2, 2]) }));

  // Matmul epilogues, naive and tiled, on shapes off the 16-wide tile.
  const MM = [[[1, 1], [1, 1]], [[17, 33], [33, 19]], [[15, 16], [16, 17]], [[16, 16], [16, 16]], [[1, 64], [64, 1]], [[64, 1], [1, 64]], [[3, 5, 7], [3, 7, 2]], [[2, 17, 19], [2, 19, 33]], [[3, 4096], [4096, 5]]];
  for (const matmul of ['naive', 'tiled']) {
    for (const [as, bs] of MM) {
      const N = bs.at(-1);
      for (const act of ['relu', 'gelu']) {
        cases.push(one(`matmul ${matmul} [${as}] x [${bs}] + bias, ${act}`, 'epilogue', 'matmul-epilogue', (g) => {
          const y = g.add(g.matmul(g.input('a', as), g.input('w', bs)), g.input('bias', [N]));
          return act === 'relu' ? g.relu(y) : geluGraph(g, y);
        }, { a: draw(rng, size(as), [-2, 2]), w: draw(rng, size(bs), [-2, 2]), bias: draw(rng, N, [-2, 2]) }, { matmul }));
      }
    }
    cases.push(one(`matmul ${matmul} [17,33] x [33,19] + bias, relu, large`, 'epilogue', 'matmul-epilogue',
      (g) => g.relu(g.add(g.matmul(g.input('a', [17, 33]), g.input('w', [33, 19])), g.input('bias', [19]))),
      { a: draw(rng, 561, [-1e15, 1e15]), w: draw(rng, 627, [-1e15, 1e15]), bias: draw(rng, 19, [-1e30, 1e30]) }, { matmul }));
    cases.push(one(`matmul ${matmul} [17,33] x [33,19] + residual`, 'epilogue', 'matmul-epilogue',
      (g) => g.add(g.add(g.matmul(g.input('a', [17, 33]), g.input('w', [33, 19])), g.input('bias', [19])), g.input('res', [17, 19])),
      { a: draw(rng, 561, [-2, 2]), w: draw(rng, 627, [-2, 2]), bias: draw(rng, 19, [-2, 2]), res: draw(rng, 323, [-2, 2]) }, { matmul }));
  }

  // Row-wise: LayerNorm, softmax, log-softmax over rows of length C.
  const ROWS = [[1, 1], [3, 7], [65, 63], [2, 64], [3, 65], [2, 257], [3, 1000], [1, 4097]];
  for (const [R, C] of ROWS) {
    const ln = (g) => layerNormGraph(g, g.input('x', [R, C]), g.input('g', [C]), g.input('b', [C]));
    const gb = () => ({ g: draw(rng, C, [0.5, 2]), b: draw(rng, C, [-1, 1]) });
    if (C > 1) {
      cases.push(one(`layernorm [${R},${C}]`, 'rowwise', 'layernorm', ln, { x: draw(rng, R * C, [-2, 2]), ...gb() }));
      cases.push(one(`softmax [${R},${C}]`, 'rowwise', 'softmax', (g) => g.softmax(g.input('x', [R, C]), 1), { x: draw(rng, R * C, [-2, 2]) }));
      cases.push(one(`log-softmax [${R},${C}]`, 'rowwise', 'log-softmax', (g) => g.logSoftmax(g.input('x', [R, C]), 1), { x: draw(rng, R * C, [-2, 2]) }));
    }
  }
  cases.push(one('layernorm [4,64] constant rows', 'rowwise', 'layernorm', (g) => layerNormGraph(g, g.input('x', [4, 64]), g.input('g', [64]), g.input('b', [64])),
    { x: Array.from({ length: 256 }, (_, i) => [0, 1, -3, 10][i >> 6]), g: draw(rng, 64, [0.5, 2]), b: draw(rng, 64, [-1, 1]) }));
  cases.push(one('layernorm [3,65] far from zero', 'rowwise', 'layernorm', (g) => layerNormGraph(g, g.input('x', [3, 65]), g.input('g', [65]), g.input('b', [65])),
    { x: draw(rng, 195, [1000, 1001]), g: draw(rng, 65, [0.5, 2]), b: draw(rng, 65, [-1, 1]) }));
  cases.push(one('layernorm [2,3,64] rank 3', 'rowwise', 'layernorm', (g) => layerNormGraph(g, g.input('x', [2, 3, 64]), g.input('g', [64]), g.input('b', [64])),
    { x: draw(rng, 384, [-2, 2]), g: draw(rng, 64, [0.5, 2]), b: draw(rng, 64, [-1, 1]) }));
  for (const op of ['softmax', 'logSoftmax']) {
    const pattern = op === 'softmax' ? 'softmax' : 'log-softmax';
    const sm = (shape) => (g) => g[op](g.input('x', shape), shape.length - 1);
    for (const [label, range] of [['large', [400, 600]], ['large negative', [-600, -400]], ['small', [-1e-30, 1e-30]]]) {
      cases.push(one(`${pattern} ${label} [3,65]`, 'rowwise', pattern, sm([3, 65]), { x: draw(rng, 195, range) }));
    }
    cases.push(one(`${pattern} exact zeros [3,65]`, 'rowwise', pattern, sm([3, 65]), { x: new Array(195).fill(0) }));
    // Milestone 3b's masked attention rows: scores plus a -1e4 mask above the diagonal.
    cases.push(one(`${pattern} masked [2,64,64]`, 'rowwise', pattern, (g) => {
      const mask = []; for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) mask.push(j <= i ? 0 : -1e4);
      return g[op](g.add(g.input('x', [2, 64, 64]), g.const(mask, [64, 64])), 2);
    }, { x: draw(rng, 8192, [-13, 13]) }));
    cases.push(one(`${pattern} [2,4,8,65] rank 4`, 'rowwise', pattern, sm([2, 4, 8, 65]), { x: draw(rng, 4160, [-2, 2]) }));
  }
  return cases;
}
