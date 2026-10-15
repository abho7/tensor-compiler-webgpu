import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { SOURCE_FILES, hashSources } from '../src/gpu/sources.js';

const root = new URL('../', import.meta.url);
const text = (p) => readFileSync(new URL(p, root), 'utf8');
const json = (p) => JSON.parse(text(p));
const gpu = json('golden/measurements.json').m3.training.gpu;

test('the M3 GPU recordings came from the sources checked in', () => {
  // CI cannot run the GPU; it checks that the recordings were made by exactly
  // these files. Any change to one of them means re-running web/m3.html,
  // web/train.html and web/gpu.html in a browser.
  const hash = hashSources(SOURCE_FILES.map(text));
  for (const f of ['golden/gpu-m3.json', 'golden/gpu-m3-demo.json', 'golden/gpu-m2-rerun.json']) {
    assert.equal(json(f).sourceHash, hash, `${f} is stale`);
  }
});

test('the M3 recordings name their adapter and power mode', () => {
  for (const f of ['golden/gpu-m3.json', 'golden/gpu-m3-demo.json', 'golden/gpu-m2-rerun.json']) {
    const r = json(f);
    assert.ok(r.adapter.vendor && r.adapter.architecture, `${f}: no adapter`);
    assert.ok(r.recorded.power.activeMode && typeof r.recorded.power.onAC === 'boolean', `${f}: no power mode`);
  }
});

test('the extended compiler reproduces milestone 2 exactly (DESIGN.md amendment 19)', () => {
  const m2 = json('golden/gpu-m2.json'), again = json('golden/gpu-m2-rerun.json');
  for (const k of ['direct', 'fuzz', 'sabotage', 'options']) assert.deepEqual(again[k], m2[k], `${k} differs`);
  assert.equal(again.adapter.vendor, m2.adapter.vendor);
  assert.equal(again.adapter.architecture, m2.adapter.architecture);
});

test('the recorded GPU checks: direct sqrt tests and teacher-forced steps all within the float32 budget', () => {
  assert.equal(gpu.direct.pass, gpu.direct.cases);
  assert.equal(gpu.teacherForced.passed, gpu.teacherForced.steps);
  assert.ok(gpu.teacherForced.worst <= 1);
  assert.equal(gpu.consistency.identical, true, 'the trainer path differs from the checked path');
});

test("the statistical comparison's float64 side recomputes from code CI can run", () => {
  // Recorded by Chrome, recomputed by Node: both V8, but not necessarily
  // bit-identical; held to 1e-12.
  assert.ok(gpu.statistical.cpuRecomputeMaxDiff <= 1e-12, `${gpu.statistical.cpuRecomputeMaxDiff}`);
  assert.equal(gpu.statistical.runs.length, 2 * gpu.statistical.seeds);
  assert.ok(gpu.statistical.seeds >= 20);
});

test('every injected bug is caught by at least one check; graph-level bugs by none of the GPU checks', () => {
  for (const b of gpu.sabotage.bugs) {
    assert.ok(b.caughtFirstBy, `${b.name} was caught by nothing`);
    if (b.level === 'graph' || b.level === 'rule') assert.equal(b.checks.gpuTeacherForced.caught, 0, `${b.name}: the prediction was that GPU-vs-CPU cannot see it`);
  }
});
