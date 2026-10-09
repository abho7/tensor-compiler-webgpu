// Writes golden/cases.json: the graphs, inputs and cotangents the PyTorch
// oracle (oracle/torch_golden.py) evaluates. Deterministic; CI rebuilds the
// cases and fails if the committed file differs.
//
//   node tools/export-cases.js

import { writeFileSync } from 'node:fs';
import { buildCases } from '../src/testing/cases.js';

const cases = buildCases();
writeFileSync(new URL('../golden/cases.json', import.meta.url), JSON.stringify({ cases }) + '\n');
console.log(`wrote ${cases.length} cases`);
