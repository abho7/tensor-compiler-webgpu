import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Graph } from '../src/ir/graph.js';
import { evaluate } from '../src/interp.js';
import { grad, GradientError } from '../src/autodiff.js';

function gradients(g, outputs, wrtNames, inputs, cotangents) {
  const G = grad(g, outputs, wrtNames.map((n) => g.inputId(n)));
  const feed = { ...inputs };
  G.cotangents.forEach((name, i) => { feed[name] = cotangents[i]; });
  const values = evaluate(G.graph, feed);
  return G.grads.map((id) => Array.from(values[id]));
}

test('d(x*x)/dx = 2x, with both uses of x counted', () => {
  const g = new Graph();
  const x = g.input('x', [3]);
  const [dx] = gradients(g, [g.mul(x, x)], ['x'], { x: [1, -2, 3] }, [[1, 1, 1]]);
  assert.deepEqual(dx, [2, -4, 6]);
});

test('a broadcast operand gets its gradient summed back down', () => {
  const g = new Graph();
  const y = g.add(g.input('a', [2, 3]), g.input('b', [3]));
  const [, db] = gradients(g, [y], ['a', 'b'], { a: [0, 0, 0, 0, 0, 0], b: [0, 0, 0] }, [[1, 2, 3, 4, 5, 6]]);
  assert.deepEqual(db, [5, 7, 9]);
});

test('matmul gradients by hand', () => {
  const g = new Graph();
  const c = g.matmul(g.input('a', [1, 2]), g.input('b', [2, 1]));
  const [da, db] = gradients(g, [c], ['a', 'b'], { a: [3, 4], b: [5, 6] }, [[1]]);
  assert.deepEqual(da, [5, 6]);
  assert.deepEqual(db, [3, 4]);
});

test('a gradient reaching max is refused, not treated as zero', () => {
  const g = new Graph();
  const x = g.input('x', [3]);
  assert.throws(() => grad(g, [g.max(x, 0)], [x]), GradientError);
  // Inside softmax, stopGradient keeps it out of reach.
  const h = new Graph();
  const y = h.input('y', [3]);
  assert.doesNotThrow(() => grad(h, [h.softmax(y, 0)], [y]));
});

test('an input the outputs do not depend on gets zeros', () => {
  const g = new Graph();
  const x = g.input('x', [2]);
  g.input('unused', [3]);
  const [, du] = gradients(g, [g.exp(x)], ['x', 'unused'], { x: [0, 0], unused: [1, 2, 3] }, [[1, 1]]);
  assert.deepEqual(du, [0, 0, 0]);
});

test('the gradient graph is an ordinary graph: it can be differentiated again', () => {
  // d²/dx² of x³ is 6x.
  const g = new Graph();
  const x = g.input('x', [2]);
  const G = grad(g, [g.mul(g.mul(x, x), x)], [x]);
  const H = grad(G.graph, [G.grads[0]], [x], { prefix: 'h' });
  const values = evaluate(H.graph, { x: [2, -1], d0: [1, 1], h0: [1, 1] });
  assert.deepEqual(Array.from(values[H.grads[0]]), [12, -6]);
});
