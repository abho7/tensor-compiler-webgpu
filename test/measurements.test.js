import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { measureAll } from '../src/testing/measure-all.js';
import { m3bInputs } from '../src/m3b/inputs.js';
import { measureM3b } from '../src/m3b/measure-m3b.js';
import { m4Inputs } from '../src/fusion/inputs-m4.js';
import { measureM4 } from '../src/fusion/measure-m4.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));

test('golden/measurements.json is what the code measures today', { timeout: 1_200_000 }, async () => {
  // Nothing in milestone 1 is timed, so every figure the report shows is
  // deterministic. A stale file, or a hand-edited number, fails here.
  const fresh = { ...measureAll({ cases: read('../golden/cases.json').cases, golden: read('../golden/torch.json'),
  gpu: { record: read('../golden/gpu-m2.json'), run1: read('../golden/gpu-m2-run1-direct.json'), oob: read('../golden/gpu-oob.json') },
  m3: {
    sqrtCases: read('../golden/cases-m3.json').cases, sqrtGolden: read('../golden/torch-m3.json'),
    trainSpec: read('../golden/train-spec-m3.json'), torchTrain: read('../golden/torch-train-m3.json'),
    gpuRecord: read('../golden/gpu-m3.json'), gpuDemo: read('../golden/gpu-m3-demo.json'),
    optdigits: {
      optdigitsTrain: readFileSync(new URL('../data/optdigits/optdigits.tra', import.meta.url), 'utf8'),
      optdigitsTest: readFileSync(new URL('../data/optdigits/optdigits.tes', import.meta.url), 'utf8'),
    },
  } }), ...(await measureM3b(m3bInputs(new URL('../', import.meta.url)))),
    ...measureM4(m4Inputs(new URL('../', import.meta.url))) };
  assert.deepEqual(JSON.parse(JSON.stringify(fresh)), read('../golden/measurements.json'));
});
