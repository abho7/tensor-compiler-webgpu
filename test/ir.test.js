import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Graph } from '../src/ir/graph.js';
import { ShapeError, broadcastShapes } from '../src/ir/shapes.js';

test('binary ops in the IR need identical shapes; the builder inserts broadcasts', () => {
  const g = new Graph();
  const a = g.input('a', [2, 3]);
  const b = g.input('b', [3]);
  assert.throws(() => g.emit('add', [a, b]), ShapeError);
  const c = g.add(a, b);
  assert.deepEqual(g.shape(c), [2, 3]);
  const bcast = g.nodes.find((n) => n.op === 'broadcast');
  assert.deepEqual(bcast.inputs, [b]);
  assert.deepEqual(g.nodes[c].inputs, [a, bcast.id]);
});

test('node ids are never taken for scalar literals', () => {
  // The builder once accepted numbers as literals, so add(x, y) added two
  // constants equal to the ids. Every op must reference its operands.
  const g = new Graph();
  const x = g.input('x', [2]);
  const y = g.input('y', [2]);
  const s = g.add(x, y);
  assert.deepEqual(g.nodes[s].inputs, [x, y]);
  assert.equal(g.nodes.filter((n) => n.op === 'const').length, 0);
});

test('shape rules reject what they should', () => {
  const g = new Graph();
  const a = g.input('a', [2, 3]);
  const b = g.input('b', [2, 3]);
  assert.throws(() => g.matmul(a, b), /inner dims/);
  assert.throws(() => g.reshape(a, [4]), /elements/);
  assert.throws(() => g.transpose(a, [0, 0]), /permutation/);
  assert.throws(() => g.broadcast(a, [3, 3]), /does not broadcast/);
  assert.throws(() => g.sum(a, 2), /out of range/);
  assert.throws(() => g.add(a, g.input('c', [3, 2])), /do not broadcast/);
  const m = g.matmul(g.input('p', [4, 2, 3]), g.input('q', [4, 3, 5]));
  assert.deepEqual(g.shape(m), [4, 2, 5]);
  assert.throws(() => g.matmul(g.input('r', [2, 2, 3]), g.input('s', [4, 3, 5])), /batch dims/);
});

test('reductions drop or keep the axis; negative axes count from the end', () => {
  const g = new Graph();
  const x = g.input('x', [2, 3, 4]);
  assert.deepEqual(g.shape(g.sum(x, 1)), [2, 4]);
  assert.deepEqual(g.shape(g.sum(x, -1, true)), [2, 3, 1]);
  assert.equal(g.nodes.at(-1).attrs.axis, 2);
});

test('NumPy broadcasting rules', () => {
  assert.deepEqual(broadcastShapes([3, 1], [4]), [3, 4]);
  assert.deepEqual(broadcastShapes([], [2, 2]), [2, 2]);
  assert.equal(broadcastShapes([3], [4]), null);
});

test('a graph survives JSON and re-checks its shapes', () => {
  const g = new Graph();
  const x = g.input('x', [2, 3]);
  g.softmax(g.transpose(x), 0);
  const back = Graph.fromJSON(JSON.parse(JSON.stringify(g.toJSON())));
  assert.deepEqual(back.toJSON(), g.toJSON());
  const bad = g.toJSON();
  bad.nodes[1].shape = [9, 9];
  assert.throws(() => Graph.fromJSON(bad), /stored shape/);
});
