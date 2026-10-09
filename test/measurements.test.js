import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { measureM1 } from '../src/testing/measure.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));

test('golden/measurements.json is what the code measures today', { timeout: 600_000 }, () => {
  // Nothing in milestone 1 is timed, so every figure the report shows is
  // deterministic. A stale file, or a hand-edited number, fails here.
  const fresh = measureM1({ cases: read('../golden/cases.json').cases, golden: read('../golden/torch.json') });
  assert.deepEqual(JSON.parse(JSON.stringify(fresh)), read('../golden/measurements.json'));
});
