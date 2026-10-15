import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { kernelSource, KERNEL_KINDS, ELEMENTWISE_OPS, grid1d, WG } from '../src/gpu/codegen.js';
import { lower, isSoftmaxShift } from '../src/gpu/lower.js';
import { emulate, emulateLaunch } from '../src/gpu/emulate.js';
import { corpusProgram, randomProgram } from '../src/testing/random-graph.js';
import { grad } from '../src/autodiff.js';
import { evaluate } from '../src/interp.js';
import { Graph } from '../src/ir/graph.js';
import { Rng } from '../src/testing/rng.js';
import { size } from '../src/ir/shapes.js';

const snapshot = JSON.parse(readFileSync(new URL('../golden/wgsl.json', import.meta.url)));
const snapshotM3 = JSON.parse(readFileSync(new URL('../golden/wgsl-m3.json', import.meta.url)));

test("milestone 2's kernels are byte-identical to its snapshot", () => {
  // golden/wgsl.json is milestone 2's record and is never rewritten.
  for (const kind of Object.keys(snapshot.kernels)) assert.equal(kernelSource(kind), snapshot.kernels[kind], `${kind} changed`);
});

test('every kernel since milestone 2 matches its own snapshot', () => {
  // New kernels are recorded in golden/wgsl-m3.json by node tools/snapshot-wgsl.js.
  const later = KERNEL_KINDS.filter((k) => !(k in snapshot.kernels));
  for (const kind of later) assert.equal(kernelSource(kind), snapshotM3.kernels[kind], `${kind} changed`);
  assert.deepEqual(Object.keys(snapshotM3.kernels).sort(), later.sort());
});

test('1-D kernels bounds-check the thread index before any access', () => {
  for (const kind of [...ELEMENTWISE_OPS, 'gather', 'sum', 'max']) {
    const src = kernelSource(kind);
    const guard = src.search(/if \(i >= [^)]+\) \{ return; \}/);
    assert.ok(guard > 0, `${kind} has no bounds check`);
    const firstAccess = src.search(/\b(A|B|OUT)\[/);
    assert.ok(guard < firstAccess, `${kind} touches memory before its bounds check`);
  }
  assert.match(kernelSource('matmul-naive'), /if \(row >= M \|\| col >= N\) \{ return; \}/);
});

test('the tiled matmul never returns early, and guards every load and the store', () => {
  const src = kernelSource('matmul-tiled');
  assert.ok(!/return;/.test(src), 'an early return would skip a workgroupBarrier');
  assert.equal((src.match(/workgroupBarrier\(\)/g) ?? []).length, 2);
  assert.match(src, /if \(row < M && ak < K\) \{ av = A\[/);
  assert.match(src, /if \(bk < K && col < N\) \{ bv = B\[/);
  assert.match(src, /if \(row < M && col < N\) \{ OUT\[/);
});

test('each sabotage changes its kernel, and only in the intended place', () => {
  const differs = (kind, sabotage, pattern) => {
    const good = kernelSource(kind), bad = kernelSource(kind, { sabotage });
    assert.notEqual(good, bad, `${sabotage} did not change ${kind}`);
    assert.match(bad, pattern);
  };
  differs('add', 'elementwise-no-bounds-check', /^(?![\s\S]*if \(i >= P\[0\]\))/);
  differs('sum', 'sum-off-by-one', /k < n - 1u/);
  differs('matmul-tiled', 'tiled-off-by-one', /ak <= K/);
  differs('matmul-naive', 'matmul-reorder', /let k = K - 1u - t;/);
  differs('matmul-tiled', 'matmul-reorder', /let k = 15u - t;/);
});

test('the 1-D grid covers every element, also past 65535 workgroups', () => {
  for (const n of [1, 63, 64, 65, 1000, 100003, WG * 65535 + 1]) {
    const [x, y] = grid1d(n);
    assert.ok(x <= 65535 && y <= 65535);
    assert.ok(x * y * WG >= n, `n=${n}: ${x}x${y} groups too few`);
    assert.ok((x * (y - 1)) * WG < n, `n=${n}: a whole row of groups is idle`);
  }
});

test('without its bounds check an elementwise kernel writes past the end', () => {
  const g = new Graph();
  const y = g.neg(g.input('x', [65]));
  const [launch] = lower(g).launches;
  const x = Array.from({ length: 65 }, (_, i) => i);
  assert.equal(emulateLaunch(launch, 65, () => x).oob, 0);
  assert.ok(emulateLaunch(launch, 65, () => x, { guard: false }).oob > 0);
  assert.equal(y, launch.node);
});

test('lowered random programs, forward and gradient, emulate to the interpreter: each output written once', () => {
  for (let i = 1; i <= 120; i++) {
    const p = corpusProgram(i, { f32: true });
    const names = p.graph.inputs().map((n) => n.attrs.name);
    const G = grad(p.graph, p.outputs, names.map((n) => p.graph.inputId(n)));
    const rng = new Rng(i);
    for (const graph of [p.graph, G.graph]) {
      const inputs = {};
      for (const n of graph.inputs()) inputs[n.attrs.name] = rng.array(size(n.shape), -2, 2);
      const ref = evaluate(graph, inputs);
      for (const matmul of ['naive', 'tiled']) {
        const { values, reports } = emulate(graph, lower(graph, { matmul }), inputs);
        for (const r of reports) {
          assert.equal(r.oob, 0, `program ${i} node ${r.node}: writes out of bounds`);
          assert.ok(r.writes.every((w) => w === 1), `program ${i} node ${r.node} (${r.kind}): an element not written exactly once`);
          const want = ref[r.node], got = values[r.node];
          for (let j = 0; j < want.length; j++) {
            assert.ok(Math.abs(got[j] - want[j]) <= 1e-12 * Math.max(1, Math.abs(want[j])), `program ${i} node ${r.node} (${r.kind}) [${j}]: ${got[j]} vs ${want[j]}`);
          }
        }
      }
    }
  }
});

test('the four lowering sabotages compute wrong values where their op is exercised', () => {
  const check = (build, inputs, sabotage) => {
    const g = new Graph();
    const out = build(g);
    const ref = evaluate(g, inputs)[out];
    const got = emulate(g, lower(g, { sabotage }), inputs).values[out];
    assert.ok(Array.from(got).some((v, j) => v !== ref[j]), `${sabotage} computed the right answer`);
  };
  check((g) => g.broadcast(g.input('x', [2, 1]), [2, 3]), { x: [1, 2] }, 'broadcast-wrong-stride');
  check((g) => g.transpose(g.input('x', [2, 3])), { x: [1, 2, 3, 4, 5, 6] }, 'transpose-swapped-stride');
  check((g) => g.reshape(g.input('x', [2, 3]), [3, 2]), { x: [1, 2, 3, 4, 5, 6] }, 'reshape-stride-mismatch');
  // Dropping softmax's shift leaves the shifted node holding x itself.
  check((g) => { g.softmax(g.input('x', [3]), 0); return g.nodes.find((n) => isSoftmaxShift(g, n)).id; }, { x: [1, 2, 3] }, 'softmax-no-max');
});

test('generator float32 mode: off changes nothing, on keeps exp arguments in [-40, 4]', () => {
  // Off: the same programs milestone 1.5 measured (its record test checks the
  // figures; this checks the programs directly).
  for (let i = 1; i <= 200; i++) assert.deepEqual(corpusProgram(i).graph.toJSON(), corpusProgram(i, { f32: false }).graph.toJSON());
  // On: the guard fires rarely; seed 9604 is the first general program it changes.
  assert.notDeepEqual(randomProgram(9604).graph.toJSON(), randomProgram(9604, { f32: true }).graph.toJSON());
  for (const p of [randomProgram(9604, { f32: true }), ...Array.from({ length: 300 }, (_, k) => corpusProgram(k + 1, { f32: true }))]) {
    p.graph.nodes.forEach((n) => {
      // Softmax's own exps are exempt (amendment 12, clarified): their argument
      // is x - stopGradient(max(x)) <= 0 by construction.
      if (n.op !== 'exp' || isSoftmaxShift(p.graph, p.graph.nodes[n.inputs[0]])) return;
      const [lo, hi] = p.bounds[n.inputs[0]];
      assert.ok(lo >= -40 && hi <= 4, `seed ${p.seed}: exp of [${lo}, ${hi}]`);
    });
  }
  const unguarded = randomProgram(9604);
  assert.ok(unguarded.graph.nodes.some((n) => n.op === 'exp' && unguarded.bounds[n.inputs[0]][0] < -40),
    'seed 9604 was expected to have an exp argument below -40 without the guard');
});
