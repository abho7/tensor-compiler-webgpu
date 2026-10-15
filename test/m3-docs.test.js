import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const text = (p) => readFileSync(new URL(p, import.meta.url), 'utf8');
const m3 = JSON.parse(text('../golden/measurements.json')).m3;
const readme = text('../README.md');
const design = text('../DESIGN.md');
// Prose wraps lines, so a run of whitespace in a pattern matches any.
const loose = (s) => new RegExp(s.replace(/ /g, '\\s+'));

test('the generator limitation figures in README.md and DESIGN.md match golden/measurements.json', () => {
  const L = m3.intervalLimitation;
  for (const doc of [readme, design]) {
    assert.match(doc, loose(`${L.asRun.logNodes} \`log\` nodes in ${L.asRun.programsWithLog}`));
    assert.match(doc, loose(`${L.squareAware.logNodes} in ${L.squareAware.programsWithLog}`));
    assert.match(doc, loose(`${L.asRun.logOfSquarePlusC} times against ${L.squareAware.logOfSquarePlusC}`));
    assert.match(doc, loose(L.programs.toLocaleString('en-US')));
  }
  assert.match(design, loose(`about ${Math.round(100 * (1 - L.asRun.logNodes / L.squareAware.logNodes))}% fewer \`log\` nodes`));
  assert.match(design, loose(`${Math.round(100 * (1 - L.asRun.programsWithLog / L.squareAware.programsWithLog))}% fewer programs`));
  assert.ok(100 * L.asRun.logOfSquarePlusC / L.squareAware.logOfSquarePlusC < 8, 'DESIGN.md says under 8%');
});

test("DESIGN.md's sqrt findings (F2) match golden/measurements.json", () => {
  const S = m3.sqrt;
  const z = S.golden.zeroGradients;
  assert.match(design, loose(`Of the ${S.golden.cases} PyTorch golden cases for sqrt, ${z.cases} have`));
  assert.match(design, loose(`${z.unusedInputs} such inputs feed no output`));
  assert.match(design, loose(`${z.reachingInputs.length} reach an output`));
  assert.deepEqual(z.casesWithNoNonzeroGradient, ['sqrt-random-50']);
  assert.match(design, loose(`changes ${S.golden.flagged} of the ${S.golden.cases} cases`));
  const fd = S.finiteDifferences;
  assert.match(design, loose(`${fd.caught} of the ${fd.containing} finite-difference programs`));
});
