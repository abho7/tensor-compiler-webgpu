// Writes golden/cases-m3.json, milestone 3a's sqrt cases for the PyTorch
// oracle:  node tools/export-cases-m3.js
// then:    .venv/Scripts/python oracle/torch_golden.py golden/cases-m3.json golden/torch-m3.json

import { writeFileSync } from 'node:fs';
import { buildSqrtCases } from '../src/testing/cases-m3.js';

const cases = buildSqrtCases();
writeFileSync(new URL('../golden/cases-m3.json', import.meta.url), JSON.stringify({ cases }) + '\n');
console.log(`wrote ${cases.length} cases`);
