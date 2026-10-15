// Recomputes golden/measurements.json, the report's only source of numbers.
//
//   node tools/measure.js
//
// Deterministic: test/measurements.test.js reruns this and fails if the
// committed file is stale.

import { readFileSync, writeFileSync } from 'node:fs';
import { measureAll } from '../src/testing/measure-all.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const result = measureAll({ cases: read('../golden/cases.json').cases, golden: read('../golden/torch.json'),
  gpu: { record: read('../golden/gpu-m2.json'), run1: read('../golden/gpu-m2-run1-direct.json'), oob: read('../golden/gpu-oob.json') },
  m3: { sqrtCases: read('../golden/cases-m3.json').cases, sqrtGolden: read('../golden/torch-m3.json') } });
writeFileSync(new URL('../golden/measurements.json', import.meta.url), JSON.stringify(result, null, 1) + '\n');
const { m1, m1_5: m } = result;
console.log(`M1: fuzz ${m1.fuzz.agree}/${m1.fuzz.programs}; torch ${m1.torch.pass}/${m1.torch.cases}`);
console.log(`M1.5: skips ${m.skips.before.rejected}/${m.skips.before.programs} -> ${m.skips.after.rejected}/${m.skips.after.programs}; fuzz ${m.fuzz.agree} agree, worst ${m.fuzz.worst}`);
console.log(`  metric: v1 ${m.metric.summary.all.v1Pass}, v2 ${m.metric.summary.all.v2Pass} of ${m.metric.summary.all.cases}`);
const pc = (x) => `${x.caught}/${x.containing} (${(100 * x.caught / x.containing).toFixed(1)}%)`;
for (const r of m.detection.rules) console.log(`  ${r.rule.padEnd(12)} before ${pc(r.before)}  general ${pc(r.after.general)}  targeted ${pc(r.after.targeted)}  corpus ${pc(r.after.corpus)}`);
const g = result.m2;
console.log(`M2: direct ${g.direct.pass}/${g.direct.cases}; fuzz ${g.fuzz.programs} programs, ${g.fuzz.skipped} skipped, ${g.fuzz.passed}/${g.fuzz.runs} runs pass`);
for (const s of g.sabotage) console.log(`  ${s.name.padEnd(28)} direct ${s.direct.caught}/${s.direct.cases}  fuzz general ${s.fuzz.general.caught}/${s.fuzz.general.containing}  targeted ${s.fuzz.targeted.caught}/${s.fuzz.targeted.containing}  first: ${s.caughtFirstBy}`);
