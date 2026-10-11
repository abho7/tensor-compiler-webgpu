// The direct per-op adversarial tests: for every kernel, single-op programs
// (or, for softmax and log-softmax, the composite) on the inputs most likely to
// break it. These carry the correctness claim for each kernel; the fuzzer is
// supporting evidence.
//
// Each case: { name, op, kernel, graph, outputs, inputs, matmul? }. Built
// deterministically, in plain JavaScript.

import { Graph } from '../ir/graph.js';
import { size } from '../ir/shapes.js';
import { Rng } from '../testing/rng.js';

/** Sizes that are not multiples of the 64-thread workgroup, and some that are. */
const SIZES = [[1], [63], [64], [65], [1000], [100003], [3, 5, 7], [1, 7, 1]];

const UNARY = ['neg', 'exp', 'log', 'relu', 'tanh', 'step', 'stopGradient'];
const BINARY = ['add', 'sub', 'mul', 'div'];

/** Values for an op's operand: a range per value class. */
const RANGES = {
  neg: { normal: [-2, 2], large: [-1e30, 1e30], small: [-1e-30, 1e-30], subnormal: [-1e-40, 1e-40] },
  exp: { normal: [-2, 2], large: [80, 88], small: [-1e-30, 1e-30] },
  log: { normal: [0.1, 4], large: [1e30, 3e38], small: [1e-37, 1e-30] },
  relu: { normal: [-2, 2], large: [-1e30, 1e30], small: [-1e-30, 1e-30], subnormal: [-1e-40, 1e-40] },
  tanh: { normal: [-2, 2], large: [-1000, 1000], small: [-1e-30, 1e-30], mid: [-20, 20] },
  step: { normal: [-2, 2], large: [-1e30, 1e30], small: [-1e-30, 1e-30], subnormal: [-1e-40, 1e-40] },
  stopGradient: { normal: [-2, 2], large: [-1e30, 1e30], small: [-1e-30, 1e-30], subnormal: [-1e-40, 1e-40] },
  add: { normal: [[-2, 2], [-2, 2]], large: [[-1e30, 1e30], [-1e30, 1e30]], small: [[-1e-30, 1e-30], [-1e-30, 1e-30]], subnormal: [[-1e-40, 1e-40], [-1e-40, 1e-40]] },
  sub: { normal: [[-2, 2], [-2, 2]], large: [[-1e30, 1e30], [-1e30, 1e30]], small: [[-1e-30, 1e-30], [-1e-30, 1e-30]] },
  mul: { normal: [[-2, 2], [-2, 2]], large: [[-1e18, 1e18], [-1e18, 1e18]], small: [[-1e-19, 1e-19], [-1e-19, 1e-19]] },
  div: { normal: [[-2, 2], [0.5, 2]], large: [[-1e30, 1e30], [1e-7, 1e-6]], small: [[-1e-30, 1e-30], [1e6, 1e7]] },
};

/** Exact zeros: inputs whose correct result is exactly 0 (or, for exp, exactly 1). */
const ZEROS = {
  neg: (n) => [new Array(n).fill(0)],
  exp: (n) => [new Array(n).fill(0)],
  log: (n) => [new Array(n).fill(1)],
  relu: (n) => [new Array(n).fill(0)],
  tanh: (n) => [new Array(n).fill(0)],
  step: (n) => [new Array(n).fill(0)],
  stopGradient: (n) => [new Array(n).fill(0)],
  add: (n, rng) => { const a = rng.array(n, -2, 2); return [a, a.map((v) => -v)]; },
  sub: (n, rng) => { const a = rng.array(n, -2, 2); return [a, a]; },
  mul: (n, rng) => [rng.array(n, -2, 2), new Array(n).fill(0)],
  div: (n, rng) => [new Array(n).fill(0), rng.array(n, 0.5, 2)],
};

function draw(rng, n, [lo, hi]) {
  // Magnitudes spread across the range: half drawn log-uniformly when it spans
  // orders of magnitude, so small values are not swamped by large ones.
  return Array.from({ length: n }, () => {
    if (lo > 0 && hi / lo > 100) return Math.exp(rng.uniform(Math.log(lo), Math.log(hi)));
    if (lo < 0 && hi > 0 && hi > 1e3) {
      const m = Math.exp(rng.uniform(Math.log(hi) - 20, Math.log(hi)));
      return rng.chance(0.5) ? m : -m;
    }
    return rng.uniform(lo, hi);
  });
}

function one(name, op, kernel, build, inputs, extra = {}) {
  const g = new Graph();
  const out = build(g);
  return { name, op, kernel, graph: g, outputs: [out], inputs, ...extra };
}

export function directCases() {
  const rng = new Rng(20261011);
  const cases = [];

  for (const op of UNARY) {
    const kernel = op === 'stopGradient' ? 'copy' : op;
    const build = (shape) => (g) => g[op](g.input('x', shape));
    for (const shape of SIZES) {
      cases.push(one(`${op} [${shape}]`, op, kernel, build(shape), { x: draw(rng, size(shape), RANGES[op].normal) }));
    }
    for (const [cls, range] of Object.entries(RANGES[op])) {
      if (cls === 'normal') continue;
      cases.push(one(`${op} ${cls} [1000]`, op, kernel, build([1000]), { x: draw(rng, 1000, range) }));
    }
    cases.push(one(`${op} exact zeros [65]`, op, kernel, build([65]), { x: ZEROS[op](65, rng)[0] }));
  }

  for (const op of BINARY) {
    const build = (shape) => (g) => g[op](g.input('a', shape), g.input('b', shape));
    for (const shape of SIZES) {
      const [ra, rb] = RANGES[op].normal;
      cases.push(one(`${op} [${shape}]`, op, op, build(shape), { a: draw(rng, size(shape), ra), b: draw(rng, size(shape), rb) }));
    }
    for (const [cls, [ra, rb]] of Object.entries(RANGES[op])) {
      if (cls === 'normal') continue;
      cases.push(one(`${op} ${cls} [1000]`, op, op, build([1000]), { a: draw(rng, 1000, ra), b: draw(rng, 1000, rb) }));
    }
    const [a, b] = ZEROS[op](65, rng);
    cases.push(one(`${op} exact zeros [65]`, op, op, build([65]), { a, b }));
  }

  for (const op of ['sum', 'max']) {
    const shapes = [[[1], 0], [[65], 0], [[3, 1, 7], 1], [[4, 5, 6], 0], [[4, 5, 6], 1], [[4, 5, 6], 2],
      [[7, 300], 1], [[300, 7], 0], [[100003], 0], [[262144], 0]];
    for (const [shape, axis] of shapes) {
      cases.push(one(`${op} [${shape}] axis ${axis}${size(shape) >= 100000 ? ' (long reduction)' : ''}`, op, op,
        (g) => g[op](g.input('x', shape), axis), { x: draw(rng, size(shape), [-2, 2]) }));
    }
    cases.push(one(`${op} large [1000]`, op, op, (g) => g[op](g.input('x', [1000]), 0), { x: draw(rng, 1000, [-1e30, 1e30]) }));
    cases.push(one(`${op} small [1000]`, op, op, (g) => g[op](g.input('x', [1000]), 0), { x: draw(rng, 1000, [-1e-30, 1e-30]) }));
    const pairs = rng.array(32, -2, 2).flatMap((v) => [v, -v]);
    cases.push(one(`${op} exact zeros [64]`, op, op, (g) => g[op](g.input('x', [64]), 0), { x: op === 'sum' ? pairs : new Array(64).fill(0) }));
    cases.push(one(`${op} cancellation [3]`, op, op, (g) => g[op](g.input('x', [3]), 0), { x: [1e20, 1, -1e20] }));
  }

  const gathers = [
    ['broadcast', [1], (g, x) => g.broadcast(x, [100003])],
    ['broadcast', [3, 1], (g, x) => g.broadcast(x, [2, 3, 65])],
    ['broadcast', [1, 5, 1], (g, x) => g.broadcast(x, [4, 5, 6])],
    ['broadcast', [], (g, x) => g.broadcast(x, [7])],
    ['broadcast', [2, 1, 3, 1], (g, x) => g.broadcast(x, [2, 4, 3, 5])],
    ['transpose', [65, 3], (g, x) => g.transpose(x, [1, 0])],
    ['transpose', [1, 64], (g, x) => g.transpose(x, [1, 0])],
    ...[[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]].map((perm) => ['transpose', [2, 3, 4], (g, x) => g.transpose(x, perm)]),
    ['transpose', [2, 3, 1, 5], (g, x) => g.transpose(x, [3, 1, 0, 2])],
    ['reshape', [6, 5], (g, x) => g.reshape(x, [5, 6])],
    ['reshape', [2, 3, 4], (g, x) => g.reshape(x, [4, 6])],
    ['reshape', [3, 5, 7], (g, x) => g.reshape(x, [105])],
    ['reshape', [65, 3], (g, x) => g.reshape(x, [3, 65])],
    ['reshape', [1, 7, 1], (g, x) => g.reshape(x, [7])],
    ['reshape', [105], (g, x) => g.reshape(x, [3, 5, 7])],
  ];
  for (const [op, shape, f] of gathers) {
    let k = 0;
    for (const range of [[-2, 2], [-1e30, 1e30], [-1e-30, 1e-30]]) {
      const label = ['', ' large', ' small'][k++];
      const c = one(`${op}${label} [${shape}]`, op, 'gather', (g) => f(g, g.input('x', shape)), { x: draw(rng, size(shape), range) });
      c.name = `${op}${label} [${shape}] -> [${c.graph.shape(c.outputs[0])}]`;
      cases.push(c);
    }
  }

  const matmuls = [
    [[1, 1], [1, 1]], [[17, 33], [33, 19]], [[15, 16], [16, 17]], [[16, 16], [16, 16]], [[32, 48], [48, 16]],
    [[1, 64], [64, 1]], [[64, 1], [1, 64]], [[3, 5, 7], [3, 7, 2]], [[2, 17, 19], [2, 19, 33]], [[3, 4096], [4096, 5]],
  ];
  for (const variant of ['naive', 'tiled']) {
    const kernel = `matmul-${variant}`;
    const mm = (as, bs) => (g) => g.matmul(g.input('a', as), g.input('b', bs));
    for (const [as, bs] of matmuls) {
      cases.push(one(`matmul ${variant} [${as}] x [${bs}]${as.at(-1) >= 4096 ? ' (long reduction)' : ''}`, 'matmul', kernel, mm(as, bs),
        { a: draw(rng, size(as), [-2, 2]), b: draw(rng, size(bs), [-2, 2]) }, { matmul: variant }));
    }
    const [as, bs] = [[17, 33], [33, 19]];
    cases.push(one(`matmul ${variant} large`, 'matmul', kernel, mm(as, bs), { a: draw(rng, size(as), [-1e15, 1e15]), b: draw(rng, size(bs), [-1e15, 1e15]) }, { matmul: variant }));
    cases.push(one(`matmul ${variant} small`, 'matmul', kernel, mm(as, bs), { a: draw(rng, size(as), [-1e-15, 1e-15]), b: draw(rng, size(bs), [-1e-15, 1e-15]) }, { matmul: variant }));
    cases.push(one(`matmul ${variant} exact zeros`, 'matmul', kernel, mm(as, bs), { a: new Array(size(as)).fill(0), b: draw(rng, size(bs), [-2, 2]) }, { matmul: variant }));
  }

  for (const op of ['softmax', 'logSoftmax']) {
    const sm = (shape, axis) => (g) => g[op](g.input('x', shape), axis);
    for (const [shape, axis, range, label] of [
      [[3, 65], 1, [-2, 2], ''], [[4, 5, 6], 0, [-2, 2], ''], [[1, 7], 0, [-2, 2], ' (size-1 axis)'],
      [[3, 65], 1, [400, 600], ' large'], [[3, 65], 1, [-600, -400], ' large negative'],
      [[3, 65], 1, [-1e-30, 1e-30], ' small'],
    ]) {
      cases.push(one(`${op}${label} [${shape}] axis ${axis}`, op, 'composite', sm(shape, axis), { x: draw(rng, size(shape), range) }));
    }
    cases.push(one(`${op} exact zeros [3, 65] axis 1`, op, 'composite', sm([3, 65], 1), { x: new Array(195).fill(0) }));
  }

  return cases;
}
