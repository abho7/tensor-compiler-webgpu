import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Graph } from '../src/ir/graph.js';
import { run } from '../src/interp.js';

const values = (g, inputs, out) => Array.from(run(g, inputs, [out])[0]);

test('matmul, by hand', () => {
  const g = new Graph();
  const c = g.matmul(g.input('a', [2, 3]), g.input('b', [3, 2]));
  assert.deepEqual(values(g, { a: [1, 2, 3, 4, 5, 6], b: [7, 8, 9, 10, 11, 12] }, c), [58, 64, 139, 154]);
});

test('batched matmul keeps batches apart', () => {
  const g = new Graph();
  const c = g.matmul(g.input('a', [2, 1, 2]), g.input('b', [2, 2, 1]));
  assert.deepEqual(values(g, { a: [1, 2, 3, 4], b: [5, 6, 7, 8] }, c), [17, 53]);
});

test('sum and max along each axis', () => {
  const g = new Graph();
  const x = g.input('x', [2, 3]);
  const data = { x: [1, 2, 3, 4, 5, 6] };
  assert.deepEqual(values(g, data, g.sum(x, 0)), [5, 7, 9]);
  assert.deepEqual(values(g, data, g.sum(x, 1)), [6, 15]);
  assert.deepEqual(values(g, data, g.max(x, 1)), [3, 6]);
});

test('broadcast expands stretched and leading axes', () => {
  const g = new Graph();
  const y = g.broadcast(g.input('x', [2, 1]), [2, 2, 3]);
  assert.deepEqual(values(g, { x: [1, 2] }, y), [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]);
});

test('transpose with a 3-cycle', () => {
  const g = new Graph();
  const y = g.transpose(g.input('x', [2, 3, 4]), [2, 0, 1]);
  assert.deepEqual(g.shape(y), [4, 2, 3]);
  const x = Array.from({ length: 24 }, (_, i) => i);
  const out = values(g, { x }, y);
  // out[a][b][c] = x[b][c][a]
  for (let a = 0; a < 4; a++) for (let b = 0; b < 2; b++) for (let c = 0; c < 3; c++) {
    assert.equal(out[a * 6 + b * 3 + c], x[b * 12 + c * 4 + a]);
  }
});

test('softmax stays finite where an unshifted one overflows float64', () => {
  const g = new Graph();
  const x = g.input('x', [3]);
  const s = g.softmax(x, 0);
  const naive = g.div(g.exp(x), g.sum(g.exp(x), 0, true));
  const data = { x: [800, 801, 802] };
  const out = values(g, data, s);
  assert.ok(out.every(Number.isFinite));
  assert.ok(Math.abs(out.reduce((p, v) => p + v) - 1) < 1e-15);
  assert.ok(values(g, data, naive).some(Number.isNaN), 'the unshifted softmax was expected to overflow');
});
