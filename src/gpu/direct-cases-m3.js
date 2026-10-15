// Direct adversarial tests for milestone 3a's new kernel, sqrt (DESIGN.md
// amendment 13): the same classes as milestone 2's direct tests, plus Adam's
// update with some second moments exactly zero, as a dead ReLU unit gives.

import { Graph } from '../ir/graph.js';
import { size } from '../ir/shapes.js';
import { Rng } from '../testing/rng.js';

const SIZES = [[1], [63], [64], [65], [1000], [100003], [3, 5, 7], [1, 7, 1]];

function draw(rng, n, [lo, hi]) {
  if (lo > 0 && hi / lo > 100) return Array.from({ length: n }, () => Math.exp(rng.uniform(Math.log(lo), Math.log(hi))));
  return rng.array(n, lo, hi);
}

function one(name, kernel, build, inputs) {
  const g = new Graph();
  const out = build(g);
  return { name, op: 'sqrt', kernel, graph: g, outputs: [out], inputs };
}

export function sqrtDirectCases() {
  const rng = new Rng(20261015);
  const cases = [];
  const sq = (shape) => (g) => g.sqrt(g.input('x', shape));
  for (const shape of SIZES) cases.push(one(`sqrt [${shape}]`, 'sqrt', sq(shape), { x: draw(rng, size(shape), [0, 4]) }));
  cases.push(one('sqrt large [1000]', 'sqrt', sq([1000]), { x: draw(rng, 1000, [1e30, 3e38]) }));
  cases.push(one('sqrt small [1000]', 'sqrt', sq([1000]), { x: draw(rng, 1000, [1e-37, 1e-30]) }));
  cases.push(one('sqrt subnormal [1000]', 'sqrt', sq([1000]), { x: draw(rng, 1000, [1e-45, 1e-38]) }));
  cases.push(one('sqrt exact zeros [65]', 'sqrt', sq([65]), { x: new Array(65).fill(0) }));
  cases.push(one('sqrt zeros and squares [65]', 'sqrt', sq([65]), { x: Array.from({ length: 65 }, (_, i) => (i % 3 === 0 ? 0 : (i % 13) ** 2)) }));
  // Adam's update, m / (sqrt(v) + eps), with every third second moment exactly 0.
  const adam = (n) => (g) => g.div(g.input('m', [n]), g.add(g.sqrt(g.input('v', [n])), g.const(1e-8)));
  for (const n of [65, 1000]) {
    const v = draw(rng, n, [1e-12, 1]).map((x, i) => (i % 3 === 0 ? 0 : x));
    const m = rng.array(n, -1, 1).map((x, i) => (i % 3 === 0 ? 0 : x));
    cases.push(one(`adam update, zero moments [${n}]`, 'composite', adam(n), { m, v }));
  }
  return cases;
}
