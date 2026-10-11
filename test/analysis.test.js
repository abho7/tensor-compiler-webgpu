import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const json = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'));
const m = json('../golden/measurements.json');
const record = json('../golden/gpu-m2.json');
const sab = (name) => record.sabotage.find((s) => s.name === name).fuzz;

test('the replay sees the same programs the GPU harness ran', () => {
  // analyze-m2.js copies the harness's run construction; if the copy drifted,
  // the containing-program counts would differ from the recording.
  const f = m.m2_followup;
  for (const [key, name] of [['tiledMatmul', 'tiled-off-by-one'], ['broadcast', 'broadcast-wrong-stride']]) {
    for (const src of ['general', 'targeted']) {
      assert.equal(f[key].replay.byModel.zero[src].containing, sab(name)[src].containing, `${key} ${src}`);
    }
  }
});

test('with out-of-bounds reads returning zero, the replay reproduces the GPU counts exactly', () => {
  const f = m.m2_followup;
  for (const [key, name] of [['tiledMatmul', 'tiled-off-by-one'], ['broadcast', 'broadcast-wrong-stride']]) {
    for (const src of ['general', 'targeted']) {
      assert.equal(f[key].replay.byModel.zero[src].caught, sab(name)[src].caught, `${key} ${src}`);
    }
    assert.equal(f[key].replay.control.failing, 0, `${key}: the honest emulation failed a run`);
  }
});

test('the out-of-bounds probe recorded zeros on this adapter, with adapter and power mode', () => {
  const oob = json('../golden/gpu-oob.json');
  assert.ok(oob.adapter.vendor && oob.recorded.power.activeMode);
  for (const r of oob.results) {
    assert.ok(r.reads.filter((x) => x.inBounds).every((x) => x.kind === 'correct'), `${r.case}: an in-bounds read was wrong`);
    assert.ok(r.reads.filter((x) => !x.inBounds).every((x) => x.kind === 'zero'), `${r.case}: not every out-of-bounds read was zero`);
  }
});

test('the tile-edge bug is caught exactly in the programs with a batched matmul', () => {
  const t = m.m2_followup.tiledMatmul;
  assert.deepEqual(t.batchedVsCaught, { caughtWithoutProperty: [], propertyButNotCaught: [] });
  assert.equal(t.shapes.anyDimNotMultipleOfTile, t.shapes.programs, 'every program has a dimension off the tile grid');
});

test('every uncaught broadcast program is explained', () => {
  const b = m.m2_followup.broadcast;
  assert.deepEqual(b.stretchingVsCaught.caughtWithoutProperty, []);
  assert.deepEqual(b.uncaughtStretching.other, []);
});
