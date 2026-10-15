import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { measureAll } from '../src/testing/measure-all.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));

test('golden/measurements.json is what the code measures today', { timeout: 600_000 }, () => {
  // Nothing in milestone 1 is timed, so every figure the report shows is
  // deterministic. A stale file, or a hand-edited number, fails here.
  const fresh = measureAll({ cases: read('../golden/cases.json').cases, golden: read('../golden/torch.json'),
  gpu: { record: read('../golden/gpu-m2.json'), run1: read('../golden/gpu-m2-run1-direct.json'), oob: read('../golden/gpu-oob.json') },
  m3: { sqrtCases: read('../golden/cases-m3.json').cases, sqrtGolden: read('../golden/torch-m3.json') } });
  assert.deepEqual(JSON.parse(JSON.stringify(fresh)), read('../golden/measurements.json'));
});
