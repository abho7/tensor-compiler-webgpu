// Writes golden/spec-m3b.json, the runs oracle/torch_transformer.py reproduces.
// A test checks the committed file is exactly what this produces.
//
//   node tools/export-spec-m3b.js

import { readFileSync, writeFileSync } from 'node:fs';
import { buildSpec3b } from '../src/m3b/spec.js';

const text = readFileSync(new URL('../data/alice/alice.txt', import.meta.url), 'utf8');
writeFileSync(new URL('../golden/spec-m3b.json', import.meta.url), JSON.stringify(buildSpec3b(text)) + '\n');
console.log('wrote golden/spec-m3b.json');
