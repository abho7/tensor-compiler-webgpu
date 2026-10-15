import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Graph } from '../src/ir/graph.js';
import { evaluate } from '../src/interp.js';
import { grad } from '../src/autodiff.js';
import { buildSqrtCases } from '../src/testing/cases-m3.js';
import { compareAll } from '../src/testing/golden.js';
import { randomProgram, corpusProgram } from '../src/testing/random-graph.js';
import { referenceWithBounds } from '../src/gpu/bounds.js';
import { kernelSource, KERNEL_KINDS } from '../src/gpu/codegen.js';
import { lower } from '../src/gpu/lower.js';
import { emulate } from '../src/gpu/emulate.js';
import { Rng } from '../src/testing/rng.js';
import { size } from '../src/ir/shapes.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const cases = read('../golden/cases-m3.json').cases;
const torch = read('../golden/torch-m3.json');
const m3 = read('../golden/measurements.json').m3;

test('sqrt by hand: value and gradient', () => {
  const g = new Graph();
  const x = g.input('x', [3]);
  const y = g.sqrt(x);
  assert.deepEqual(Array.from(evaluate(g, { x: [4, 9, 0.25] })[y]), [2, 3, 0.5]);
  const G = grad(g, [y], [x]);
  const v = evaluate(G.graph, { x: [4, 9, 0.25], d0: [1, 1, 1] });
  assert.deepEqual(Array.from(v[G.grads[0]]), [0.25, 1 / 6, 1]);
});

test('golden/cases-m3.json is what the builder produces, and PyTorch answered every case in float64', () => {
  assert.deepEqual(JSON.parse(JSON.stringify(buildSqrtCases())), cases);
  assert.equal(torch.dtype, 'float64');
  assert.deepEqual(torch.cases.map((c) => c.name), cases.map((c) => c.name));
  assert.ok(cases.every((c) => c.graph.nodes.some((n) => n.op === 'sqrt')), 'every case contains sqrt');
});

test('the interpreter and autodiff match PyTorch on every sqrt case under metric v2', () => {
  const r = compareAll(cases, torch);
  assert.deepEqual(r.filter((x) => !x.okV2).map((x) => x.name), []);
});

test("sqrt's float32 bound is amendment 13's: 4.5 ULP own error, Hölder-bounded inheritance", () => {
  const EPS = 2 ** -23;
  const g = new Graph();
  const y = g.sqrt(g.input('x', [2]));
  const r = referenceWithBounds(g, { x: [2, 0] });
  assert.ok(Math.abs(r.bounds[y][0] - 4.5 * EPS * Math.SQRT2) <= 1e-12 * Math.SQRT2 * EPS);
  assert.equal(r.bounds[y][1], 0, 'sqrt(0) of an exact input');
  const h = new Graph();
  const z = h.sqrt(h.sub(h.input('a', [1]), h.input('b', [1])));
  assert.equal(referenceWithBounds(h, { a: [1], b: [2] }).bounds[z][0], Infinity, 'a negative operand is unbounded');
});

test('the sqrt kernel is guarded; its bounds-bug variant changes only the sqrt kernel', () => {
  assert.ok(KERNEL_KINDS.includes('sqrt'));
  assert.match(kernelSource('sqrt'), /if \(i >= P\[0\]\) \{ return; \}[\s\S]*OUT\[i\] = sqrt\(A\[i\]\);/);
  assert.match(kernelSource('sqrt', { sabotage: 'sqrt-bounds-off-by-one' }), /if \(i > P\[0\]\) \{ return; \}/);
  for (const k of KERNEL_KINDS.filter((k) => k !== 'sqrt')) {
    assert.equal(kernelSource(k, { sabotage: 'sqrt-bounds-off-by-one' }), kernelSource(k), `${k} changed`);
  }
});

test('lowered programs containing sqrt emulate to the interpreter, every output written once', () => {
  let seen = 0;
  for (let i = 1; i <= 150; i++) {
    const p = randomProgram(i, { withSqrt: true, f32: true });
    if (!p.graph.nodes.some((n) => n.op === 'sqrt')) continue;
    seen++;
    const G = grad(p.graph, p.outputs, p.graph.inputs().map((n) => n.id));
    const rng = new Rng(i);
    for (const graph of [p.graph, G.graph]) {
      const inputs = {};
      for (const n of graph.inputs()) inputs[n.attrs.name] = rng.array(size(n.shape), -2, 2);
      const ref = evaluate(graph, inputs);
      const { values, reports } = emulate(graph, lower(graph), inputs);
      for (const r of reports) {
        assert.equal(r.oob, 0);
        assert.ok(r.writes.every((w) => w === 1));
        ref[r.node].forEach((v, j) => assert.ok(Math.abs(values[r.node][j] - v) <= 1e-12 * Math.max(1, Math.abs(v)) || (Number.isNaN(v) && Number.isNaN(values[r.node][j]))));
      }
    }
  }
  assert.ok(seen > 20, `only ${seen} programs contained sqrt`);
});

test('every finite-difference and golden miss of the wrong sqrt rule has a specific explanation', () => {
  const fd = m3.sqrt.finiteDifferences;
  assert.equal(fd.honestAgree, fd.containing, 'an honest sqrt program disagreed with finite differences');
  for (const m of [...fd.misses, ...m3.sqrt.golden.misses]) {
    assert.notEqual(m.reason, 'unexplained', JSON.stringify(m));
    assert.notEqual(m.detail, 'other', JSON.stringify(m));
  }
  assert.equal(fd.caught + fd.misses.length, fd.containing);
});

test('the new generator options change nothing when off', () => {
  for (let i = 1; i <= 200; i++) {
    const base = JSON.stringify(corpusProgram(i).graph.toJSON());
    assert.equal(JSON.stringify(corpusProgram(i, { withSqrt: false, squareAware: false }).graph.toJSON()), base);
  }
});
