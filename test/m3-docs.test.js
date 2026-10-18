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

test('the div side of the generator limitation (DESIGN.md F6) matches golden/measurements.json', () => {
  const L = m3.intervalLimitation;
  for (const doc of [readme, design]) {
    assert.match(doc, loose(`${L.asRun.divNodes} \`div\` nodes in ${L.asRun.programsWithDiv}`));
    assert.match(doc, loose(`${L.squareAware.divNodes} in ${L.squareAware.programsWithDiv}`));
    assert.match(doc, loose(`denominator ${L.asRun.divBySquarePlusC} times against ${L.squareAware.divBySquarePlusC}`));
  }
  assert.match(design, loose(`about ${Math.round(100 * (1 - L.asRun.divNodes / L.squareAware.divNodes))}% fewer \`div\` nodes`));
  assert.match(design, loose(`${Math.round(100 * (1 - L.asRun.programsWithDiv / L.squareAware.programsWithDiv))}% fewer programs containing \`div\``));
  assert.ok(100 * L.asRun.divBySquarePlusC / L.squareAware.divBySquarePlusC < 5, 'DESIGN.md says under 5%');
  // F6 says the counts are the same with milestone 2's float32 guards on.
  assert.deepEqual(L.asRunFloat32, L.asRun);
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

test("the README's milestone 3a figures, and DESIGN.md F4's, match golden/measurements.json", () => {
  const g = m3.training.gpu, c = m3.training.cpuVsTorch;
  assert.match(readme, loose(`all ${c.runs.length} golden runs`));
  assert.ok(c.runs.every((r) => r.pass));
  assert.match(readme, loose(`${g.teacherForced.passed} of ${g.teacherForced.steps} steps`));
  assert.match(readme, loose(`${g.statistical.seeds} seeds per optimizer`));
  const eq = Object.values(g.statistical.equivalence).flatMap((x) => Object.values(x));
  assert.ok(eq.every((x) => x.pass)); assert.match(readme, loose(`all ${eq.length} comparisons`));
  assert.match(readme, loose(`all ${g.sabotage.bugs.length} caught`));
  assert.match(readme, loose(`${(100 * g.demo.testAccuracy).toFixed(2)}% test accuracy`));
  assert.equal(g.demo.testAccuracy, g.demo.torchTestAccuracy, 'the README says the same as PyTorch');
  // F4: which Adam runs became unjudgeable, and at which step.
  const adam = g.freeRunning.runs.filter((r) => r.name.startsWith('adam'));
  const stuck = adam.filter((r) => r.steps.some((s) => s.status !== 'pass')).map((r) => r.steps.findIndex((s) => s.status !== 'pass') + 1).sort((a, b) => a - b);
  assert.equal(stuck.length, 3, 'README: three of four Adam runs');
  assert.match(design, loose(`at steps ${stuck.slice(0, -1).join(', ')} and ${stuck.at(-1)}`));
});
