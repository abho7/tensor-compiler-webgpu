// Writes golden/train-spec-m3.json, the runs oracle/torch_train.py reproduces:
//   node tools/export-train-m3.js
//   .venv/Scripts/python oracle/torch_train.py golden/train-spec-m3.json golden/torch-train-m3.json

import { writeFileSync } from 'node:fs';
import { buildTrainSpec } from '../src/train/golden-spec.js';

const spec = buildTrainSpec();
writeFileSync(new URL('../golden/train-spec-m3.json', import.meta.url), JSON.stringify(spec) + '\n');
console.log(`wrote ${spec.configs.length} golden configs and the demo (${spec.demo.steps} steps)`);
