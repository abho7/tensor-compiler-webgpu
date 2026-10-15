import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { parseOptdigits } from '../src/train/data.js';
import { buildTrainSpec, DEMO } from '../src/train/golden-spec.js';
import { buildStep, SGD, ADAM, PARAMS } from '../src/train/step.js';
import { OPS } from '../src/ir/shapes.js';

const root = new URL('../', import.meta.url);
const raw = (p) => readFileSync(new URL(p, root));
const json = (p) => JSON.parse(raw(p));
const m3 = json('golden/measurements.json').m3;

test('the optdigits files are the ones DESIGN.md amendment 15 recorded, byte for byte', () => {
  for (const f of Object.values(DEMO.files)) {
    assert.equal(createHash('sha256').update(raw(f.path)).digest('hex'), f.sha256, f.path);
  }
  const design = raw('DESIGN.md').toString();
  for (const f of Object.values(DEMO.files)) assert.ok(design.includes(f.sha256), `${f.sha256} not in DESIGN.md`);
  assert.equal(parseOptdigits(raw(DEMO.files.train.path).toString()).y.length, DEMO.trainRows);
  assert.equal(parseOptdigits(raw(DEMO.files.test.path).toString()).y.length, 1797);
});

test('golden/train-spec-m3.json is what the code builds, and PyTorch answered every run in float64', () => {
  assert.deepEqual(JSON.parse(JSON.stringify(buildTrainSpec())), json('golden/train-spec-m3.json'));
  const torch = json('golden/torch-train-m3.json');
  assert.equal(torch.dtype, 'float64');
  assert.deepEqual(torch.configs.map((c) => c.name), json('golden/train-spec-m3.json').configs.map((c) => c.name));
  assert.equal(torch.demo.losses.length, DEMO.steps);
});

test('a training step is one IR graph: only IR ops, sqrt only where Adam needs it', () => {
  const dims = { inputs: 8, hidden: 16, classes: 4 };
  const sgd = buildStep(dims, 32, SGD), adam = buildStep(dims, 32, ADAM);
  for (const s of [sgd, adam]) {
    assert.ok(s.graph.nodes.every((n) => OPS.includes(n.op)), 'a non-IR op');
    for (const p of PARAMS) assert.ok(s.outputs[p] !== undefined, `no update for ${p}`);
  }
  assert.ok(!sgd.graph.nodes.some((n) => n.op === 'sqrt'));
  assert.ok(adam.graph.nodes.some((n) => n.op === 'sqrt'));
});

test('the float64 interpreter matches PyTorch on every golden run, every step, within 1e-8', () => {
  const t = m3.training.cpuVsTorch;
  assert.equal(t.tolerance, 1e-8);
  assert.equal(t.runs.length, 8);
  for (const r of t.runs) assert.ok(r.pass, `${r.name}: worst ${r.worst} at step ${r.worstStep}`);
});

test('every optimizer bug is caught by the PyTorch goldens in every run it applies to', () => {
  for (const s of m3.training.cpuVsTorch.sabotage) {
    assert.equal(s.caught, s.trials, `${s.name}: ${s.caught} of ${s.trials}`);
    assert.ok(s.trials > 0);
  }
});
