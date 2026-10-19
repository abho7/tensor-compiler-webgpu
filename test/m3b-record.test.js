import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { hashSources, SOURCE_FILES, M2_SOURCE_FILES } from '../src/gpu/sources.js';
import { M3B_SOURCE_FILES } from '../src/gpu/sources-m3b.js';
import { buildStep3b } from '../src/m3b/step.js';

const root = new URL('../', import.meta.url);
const text = (p) => readFileSync(new URL(p, root), 'utf8');
const json = (p) => JSON.parse(text(p));
const m3b = json('golden/measurements.json').m3b;

test('the M3b GPU recordings came from the sources checked in', () => {
  // CI cannot run the GPU. Any change to one of these files means running
  // web/m3b.html and web/train-transformer.html again in a browser.
  const hash = hashSources(M3B_SOURCE_FILES.map(text));
  for (const f of ['golden/gpu-m3b.json', 'golden/gpu-m3b-demo.json']) assert.equal(json(f).sourceHash, hash, `${f} is stale`);
});

test("milestone 3b edits none of milestone 2's or 3a's hashed sources (DESIGN.md amendment 21)", () => {
  const added = M3B_SOURCE_FILES.filter((f) => f.includes('m3b') || f.endsWith('train-transformer.html'));
  for (const f of added) {
    assert.ok(!SOURCE_FILES.includes(f) && !M2_SOURCE_FILES.includes(f), `${f} is in an earlier milestone's hashed list`);
  }
});

test('the M3b recordings name their adapter and power mode', () => {
  for (const f of ['golden/gpu-m3b.json', 'golden/gpu-m3b-demo.json']) {
    const r = json(f);
    assert.ok(r.adapter.vendor && r.adapter.architecture, `${f}: no adapter`);
    assert.ok(r.recorded.power.activeMode && typeof r.recorded.power.onAC === 'boolean', `${f}: no power mode`);
  }
});

test("revision R3's residual-stream node ids are what measure-m3b.js says they are", () => {
  const g = buildStep3b(16, 4).graph;
  const { embedding, afterAttention0, afterMlp0 } = m3b.vacuity.streamIds;
  const node = (id) => g.nodes[id];
  // The embedding: reshape(add(reshape(matmul(x, tok)), broadcast(pos))).
  assert.equal(node(embedding).op, 'reshape');
  const sum = node(node(embedding).inputs[0]);
  assert.equal(sum.op, 'add');
  assert.equal(node(node(sum.inputs[0]).inputs[0]).op, 'matmul');
  // Each residual: add(previous stream, block output).
  assert.equal(node(afterAttention0).op, 'add');
  assert.equal(node(afterAttention0).inputs[0], embedding);
  assert.equal(node(afterMlp0).op, 'add');
  assert.equal(node(afterMlp0).inputs[0], afterAttention0);
  assert.equal(m3b.vacuity.firstUnbounded.op, 'div');
});

test('the CPU-side checks recorded as measured: PyTorch at init and in training, both properties', () => {
  const c = m3b.cpu;
  assert.ok(c.init.every((x) => x.pass), 'init');
  assert.ok(c.training.every((x) => x.pass), 'training');
  assert.ok(c.properties.causality.every((x) => x.pass && x.laterChanged), 'causality, and not vacuously');
  assert.ok(c.properties.batchOrder.every((x) => x.pass), 'batch order');
  assert.ok(c.largestScores.every((s) => s.maxScore < 1e3), 'amendment 21 assumed scores well under 1e3');
});

test('every injected model bug is caught; causality catches the mask bug; the GPU checks catch none', () => {
  const bugs = m3b.gpu.sabotage.bugs;
  assert.equal(bugs.length, 7);
  for (const b of bugs) {
    assert.ok(b.caughtFirstBy, `${b.name} was caught by nothing`);
    assert.equal(b.checks.gpuWholeStep.caught + b.checks.gpuNodeLocal.caught, 0, `${b.name}: predicted invisible to GPU-vs-interpreter`);
  }
  assert.ok(bugs.find((b) => b.name === 'mask-off-by-one').checks.causality.caught > 0);
});
