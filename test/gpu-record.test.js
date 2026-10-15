import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { M2_COMMIT, M2_SOURCE_FILES, hashSources } from '../src/gpu/sources.js';

const root = new URL('../', import.meta.url);
const text = (p) => readFileSync(new URL(p, root), 'utf8');
const json = (p) => JSON.parse(text(p));
const sha = (s) => createHash('sha256').update(s.replace(/\r\n/g, '\n')).digest('hex');

test('the milestone 1 and 1.5 records and golden cases are byte-identical to milestone 1.5 (commit 6127e48)', () => {
  // Hashes taken from commit 6127e48. Line endings are normalised: a Windows
  // checkout may carry CRLF where git stores LF.
  assert.equal(sha(text('golden/cases.json')), 'ee9b4049004a714d78ed57ea3e3166526f8054ff87295c56370ff0330b94e0d8');
  assert.equal(sha(text('golden/torch.json')), 'b77ad819e85b38feba4c35c02587c675043e821b4362c10a7f136b9e0c404efb');
  const m = json('golden/measurements.json');
  assert.equal(sha(JSON.stringify(m.m1)), 'aea39c0b61d418f5fb6c59ac429a67d8088e10870a2b17c1e007ccca079fd43f');
  assert.equal(sha(JSON.stringify(m.m1_5)), '3a03d951dc9a106e9826c2c472b51bddc7ff37b5645e911000edebe10400fbaf');
});

test('the recorded M2 GPU results came from the sources at the M2 commit', () => {
  // Pinned to the commit the recording was made at (DESIGN.md amendment 19):
  // milestone 3a edits some of these files, so they are read from git history.
  // CI checks out the full history for this.
  const record = json('golden/gpu-m2.json');
  const atCommit = (f) => execFileSync('git', ['show', `${M2_COMMIT}:${f}`], { cwd: new URL('../', import.meta.url), encoding: 'utf8', maxBuffer: 64e6 });
  assert.equal(record.sourceHash, hashSources(M2_SOURCE_FILES.map(atCommit)));
});

test('the GPU record names its adapter and the power mode it ran under', () => {
  for (const file of ['golden/gpu-m2.json', 'golden/gpu-m2-run1-direct.json']) {
    const r = json(file);
    assert.ok(r.adapter.vendor && r.adapter.architecture, `${file}: no adapter`);
    assert.ok(r.recorded.power.activeMode && typeof r.recorded.power.onAC === 'boolean', `${file}: no power mode`);
  }
});

test('the first GPU run, kept as a record, is the tanh NaN finding (DESIGN.md R2)', () => {
  const run1 = json('golden/gpu-m2-run1-direct.json');
  const failing = run1.direct.filter((d) => d.status !== 'pass');
  assert.deepEqual(failing.map((d) => d.name), ['tanh large [1000]']);
  assert.equal(failing[0].first.gpu, null, 'NaN is serialised as null');
  assert.equal(failing[0].first.ref, -1);
});
