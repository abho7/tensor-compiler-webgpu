import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Graph } from '../src/ir/graph.js';
import { grad } from '../src/autodiff.js';
import { referenceWithBounds, judge, unjudgeable, U as U_MODULE, EPS as EPS_MODULE, TAU as TAU_MODULE } from '../src/gpu/bounds.js';

// Independent of the module under test: a wrong constant there must not pass
// a test that compares the module against itself.
const U = 2 ** -24, EPS = 2 ** -23, TAU = 2 ** -126;
const gamma = (n) => (n * U) / (1 - n * U);

test('the float32 constants are the ones amendment 8 states', () => {
  assert.equal(U_MODULE, U);
  assert.equal(EPS_MODULE, EPS);
  assert.equal(TAU_MODULE, TAU);
});

/** The bound of a one-op graph's result, element by element. */
function single(build, inputs) {
  const g = new Graph();
  const out = build(g);
  const r = referenceWithBounds(g, inputs);
  return { value: Array.from(r.values[out]), bound: Array.from(r.bounds[out]) };
}

test('inputs and constants are rounded to float32 before the reference runs', () => {
  const g = new Graph();
  const x = g.input('x', [1]);
  const y = g.add(x, g.const(0.1));
  const r = referenceWithBounds(g, { x: [0.1] });
  assert.equal(r.inputs.x[0], Math.fround(0.1));
  assert.equal(r.values[y][0], Math.fround(0.1) + Math.fround(0.1));
});

test('single-op bounds are the per-op constants of amendment 8', () => {
  const close = (a, b) => assert.ok(Math.abs(a - b) <= 1e-12 * Math.abs(b), `${a} vs ${b}`);
  let s = single((g) => g.add(g.input('a', [1]), g.input('b', [1])), { a: [1.5], b: [2.25] });
  close(s.bound[0], U * 3.75);
  s = single((g) => g.div(g.input('a', [1]), g.input('b', [1])), { a: [1], b: [3] });
  close(s.bound[0], 2.5 * EPS * Math.abs(s.value[0]));
  s = single((g) => g.exp(g.input('x', [1])), { x: [2] });
  close(s.bound[0], (3 + 2 * 2) * EPS * Math.exp(2));
  s = single((g) => g.log(g.input('x', [2])), { x: [1.5, 100] });
  close(s.bound[0], 2 ** -21);
  close(s.bound[1], 3 * EPS * Math.abs(s.value[1]));
  s = single((g) => g.tanh(g.input('x', [2])), { x: [0.5, 30] });
  close(s.bound[0], 1e-5);
  close(s.bound[1], (9.5 + 4 * 30) * EPS);
  s = single((g) => g.sum(g.input('x', [4]), 0), { x: [1, -2, 3, -4] });
  close(s.bound[0], gamma(4) * 10);
  s = single((g) => g.matmul(g.input('a', [1, 2]), g.input('b', [2, 1])), { a: [1, -2], b: [3, 4] });
  close(s.bound[0], gamma(2) * (3 + 8));
  s = single((g) => g.transpose(g.input('x', [2, 3])), { x: [1, 2, 3, 4, 5, 6] });
  assert.deepEqual(s.bound, [0, 0, 0, 0, 0, 0]);
  s = single((g) => g.max(g.input('x', [3]), 0), { x: [1, 7, 3] });
  assert.deepEqual(s.bound, [0]);
});

test('a subnormal may be flushed: its bound is at least the smallest normal', () => {
  const s = single((g) => g.mul(g.input('a', [1]), g.input('b', [1])), { a: [1e-20], b: [1e-20] });
  assert.ok(s.value[0] !== 0 && Math.abs(s.value[0]) < TAU);
  assert.ok(s.bound[0] >= TAU);
});

test('random-9 under float32: an exact zero judged by the terms that cancel', () => {
  // x / x has gradient exactly zero; the bound comes from g/x and -g*x/(x*x).
  const g = new Graph();
  const x = g.input('x', [3]);
  const G = grad(g, [g.div(x, x)], [x]);
  const xs = [0.3, -1.7, 1.2], gs = [0.5, -0.25, 1];
  const r = referenceWithBounds(G.graph, { x: xs, d0: gs });
  const values = Array.from(r.values[G.grads[0]]);
  const bounds = Array.from(r.bounds[G.grads[0]]);
  assert.deepEqual(values, [0, 0, 0]);
  bounds.forEach((b, i) => {
    const term = Math.abs(Math.fround(gs[i]) / Math.fround(xs[i]));
    assert.ok(b > 3 * EPS * term && b < 12 * EPS * term, `bound ${b} for |g/x| ${term}`);
  });
});

test('a relu gradient may flip where the value is within its error of zero', () => {
  const g = new Graph();
  const x = g.input('x', [2]);
  const y = g.sub(g.add(x, g.const(1e-8)), g.const(1e-8));   // carries rounding error
  const s = g.step(y);
  // At 1e-17 the value is smaller than the rounding error of x + 1e-8 (about
  // 1e-16), so the sign could come out either way.
  const r = referenceWithBounds(g, { x: [1e-17, 1] });
  assert.equal(r.bounds[s][0], 1, 'a near-zero step input should be allowed to flip');
  assert.equal(r.bounds[s][1], 0);
});

test('division by a value its error can reach is unbounded, and counted', () => {
  const g = new Graph();
  const a = g.input('a', [1]);
  const b = g.sub(g.input('b', [1]), g.const(1));
  const q = g.div(a, b);
  const r = referenceWithBounds(g, { a: [1], b: [1] });   // b - 1 = 0 exactly
  assert.equal(r.bounds[q][0], Infinity);
  assert.ok(unjudgeable(r.values, r.bounds) >= 1);
});

test('judge: exact where the bound is zero, ratio elsewhere', () => {
  const g = new Graph();
  const x = g.input('x', [2]);
  const y = g.add(x, x);
  const r = referenceWithBounds(g, { x: [1, 2] });
  const ok = judge(g, [r.values[0], Float32Array.from(r.values[1])], r.values, r.bounds);
  assert.equal(ok.pass, true);
  const moved = Float64Array.from(r.values[0]); moved[1] += 1e-12;
  const bad = judge(g, [moved, r.values[1]], r.values, r.bounds);
  assert.equal(bad.pass, false, 'an input changed by any amount should fail: its bound is zero');
  assert.equal(bad.first.node, 0);
});
