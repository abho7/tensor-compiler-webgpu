// Recomputes golden/measurements.json, the report's only source of numbers.
//
//   node tools/measure.js
//
// Deterministic: test/measurements.test.js reruns this and fails if the
// committed file is stale.

import { readFileSync, writeFileSync } from 'node:fs';
import { measureM1 } from '../src/testing/measure.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const result = measureM1({ cases: read('../golden/cases.json').cases, golden: read('../golden/torch.json') });
writeFileSync(new URL('../golden/measurements.json', import.meta.url), JSON.stringify(result, null, 1) + '\n');
const m = result.m1;
console.log(`fuzz ${m.fuzz.agree}/${m.fuzz.programs} agree, worst ${m.fuzz.worst}; central mismatches ${m.fuzzCentral.mismatch}`);
console.log(`second order ${m.secondOrder.agree}/${m.secondOrder.programs}; torch ${m.torch.pass}/${m.torch.cases}`);
for (const s of m.sabotage.rules) console.log(`  ${s.rule.padEnd(12)} after ${s.caughtAfter} (${s.how}), ${s.caught}/${s.containing}`);
