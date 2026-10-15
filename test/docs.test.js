import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TOLERANCE } from '../src/testing/gradcheck.js';
import { GOLDEN_TOLERANCE } from '../src/testing/golden.js';

const text = (p) => readFileSync(new URL(p, import.meta.url), 'utf8');
const { m1 } = JSON.parse(text('../golden/measurements.json'));
const readme = text('../README.md');
const design = text('../DESIGN.md');

test('the budgets in README.md and DESIGN.md are the ones the code enforces', () => {
  assert.equal(TOLERANCE, 1e-6);
  assert.equal(GOLDEN_TOLERANCE, 1e-12);
  for (const doc of [readme, design]) {
    assert.match(doc, /\| 1e-6\b|\*\*1e-6\*\*/);
    assert.match(doc, /\| 1e-12\b|\*\*1e-12\*\*/);
  }
});

test("the README's findings match golden/measurements.json", () => {
  const rules = m1.sabotage.rules;
  const caught = rules.filter((r) => r.caughtAfter !== null).length;
  assert.match(readme, new RegExp(`All ${caught} sabotaged gradient rules were caught`));
  assert.equal(caught, rules.length);

  const reshape = rules.find((r) => r.rule === 'reshape').caughtAfter;
  const transpose = rules.find((r) => r.rule === 'transpose').caughtAfter;
  assert.ok(reshape > 100, 'README says the reshape sabotage took over a hundred programs');
  assert.ok(transpose >= 90 && transpose < 100, 'README says the transpose sabotage took nearly a hundred');

  assert.deepEqual(m1.fuzzCentral.failures.map((f) => f.seed), [496]);
  assert.match(readme, /program 496/);
  assert.deepEqual(m1.torch.failures.map((f) => f.name), ['random-9']);
  assert.match(readme, /Program 9 computes `x0 \/ x0`/);

  assert.equal(m1.torch.kernelSabotage.length, 4);
  assert.match(readme, /Four interpreter kernels/);
  assert.equal(m1.torch.kernelSabotage.find((k) => k.op === 'transpose').randomFailing, 0);
  assert.match(readme, /failed none of the random golden cases/);
});

test("the README's milestone 1.5 figures match golden/measurements.json", () => {
  const h = JSON.parse(text('../golden/measurements.json')).m1_5;
  const pc = (x) => `${(100 * x.caught / x.containing).toFixed(1)}%`;
  const s = h.metric.summary.all;
  assert.match(readme, new RegExp(`v1 passes ${s.v1Pass} of ${s.cases}, v2 passes ${s.v2Pass} of ${s.cases}`));
  const a = h.metric.injection.allGradients;
  assert.match(readme, new RegExp(`v2 flags\\s+${a.flagged} of the ${a.affected} cases`));
  assert.deepEqual(a.missed.map((m) => m.name), ['random-9']);
  assert.deepEqual([...new Set(h.metric.injection.perRule.flatMap((r) => r.missed.map((m) => m.name)))], ['random-9']);
  const before = h.skips.before, after = h.skips.after;
  assert.match(readme, new RegExp(`${before.rejected} of ${before.programs.toLocaleString('en-US')} \\(${(100 * before.rejected / before.programs).toFixed(1)}%\\) before, ${after.rejected} of ${after.programs.toLocaleString('en-US')} after`));
  const r = Object.fromEntries(h.detection.rules.map((x) => [x.rule, x]));
  const cells = ['matmul', 'transpose', 'reshape'];
  assert.match(readme, new RegExp(cells.map((k) => `${k} ${pc(r[k].before)} to ${pc(r[k].after.corpus)}`).join(',\\s+').replace('reshape', '\\s*reshape')));
  assert.match(readme, new RegExp(`reach ${cells.map((k) => pc(r[k].after.targeted)).join(', ').replace(/, ([^,]+)$/, ' and $1')}`));
  assert.match(readme, new RegExp(`alone\\s+${cells.map((k) => pc(r[k].after.general)).join(', ').replace(/, ([^,]+)$/, ' and $1')}`));
  const met = cells.filter((k) => r[k].after.corpus.caught / r[k].after.corpus.containing > h.detection.target);
  assert.deepEqual(met, ['transpose'], 'the README says the target is met for transpose only');
});

test("the README's milestone 2 figures match golden/measurements.json", () => {
  const g = JSON.parse(text('../golden/measurements.json')).m2;
  const bugs = g.sabotage.filter((s) => s.bug);
  const ew = g.sabotage.find((s) => s.name === 'elementwise-no-bounds-check');
  assert.match(readme, new RegExp(`${g.direct.pass} of ${g.direct.cases} pass`));
  assert.match(readme, new RegExp(`${g.fuzz.passed.toLocaleString('en-US')} of ${g.fuzz.runs.toLocaleString('en-US')} runs pass`));
  assert.match(readme, new RegExp(`${g.fuzz.programs.toLocaleString('en-US')} generator-v2 programs`));
  assert.equal(g.fuzz.skipped, 0, 'the README says none skipped');
  assert.match(readme, new RegExp(`the direct tests catch all ${bugs.length}`));
  assert.ok(g.targets.directCatchesEveryBug && bugs.every((s) => s.caughtFirstBy.startsWith('direct')));
  assert.equal(g.targets.reorderFlagged, false);
  assert.deepEqual(g.targets.bugsMissedByFuzzer, ['elementwise-no-bounds-check']);
  const affected = ew.fuzz.general.containing + ew.fuzz.targeted.containing;
  assert.match(readme, new RegExp(`0 of the\\s+${affected.toLocaleString('en-US')} programs`));
  assert.equal(g.firstRun.pass, g.firstRun.cases - 1);
  assert.equal(g.tile, 16);
});

test("the README's milestone 2 follow-up figures match golden/measurements.json", () => {
  const f = JSON.parse(text('../golden/measurements.json')).m2_followup;
  const s = f.tiledMatmul.shapes, b = f.broadcast.shapes;
  assert.match(readme, new RegExp(`all ${s.programs} have an inner dimension off the ${f.tile}-wide grid; ${s.anyDimLargerThanTile} have a\\s+dimension over ${f.tile}`));
  assert.equal(s.kNotMultipleOfTile, s.programs);
  assert.match(readme, new RegExp(`caught in exactly the ${s.batched} batched programs`));
  assert.match(readme, new RegExp(`${b.targeted.leadingOnly} of ${b.targeted.programs} targeted programs only add leading axes,\\s+against ${b.general.leadingOnly} of ${b.general.programs} general ones`));
  assert.deepEqual(f.broadcast.uncaughtStretching.other, []);
  assert.ok(f.oobProbe.cases.every((c) => c.returnedZero === c.outOfBounds), 'the README says out-of-bounds reads return zero');
  assert.match(readme, /CI has no GPU and cannot re-run the kernels/);
  assert.match(readme, /float32 guard changed none of the corpus programs/);
});

test('the report page types in no measured number', () => {
  // Every figure comes from measurements.json. The page's own numerals are
  // limited to fixed context: milestone numbers, 10⁴, 20 draws, logits near 800.
  const html = text('../index.html');
  const body = html.slice(html.indexOf('<main>'), html.indexOf('<script'));
  const script = html.slice(html.indexOf('<script'));
  // Chart geometry is layout, not measurement: fenced by layout-start/layout-end markers and skipped.
  const unfenced = script.replace(/layout-start[\s\S]*?layout-end/g, '');
  const prose = (body + unfenced.replace(/\$\{[^}]*\}/g, '')).replace(/<[^>]+>/g, ' ');
  const numerals = prose.match(/\b\d+(\.\d+)?(e-?\d+)?\b/g) ?? [];
  // 1e-3 and 1e4 are sci()'s display thresholds for switching to exponent form.
  // 1.5 is the hardening milestone's name.
  const allowed = new Set(['1', '2', '3', '4', '10', '20', '800', '100', '0', '1e-3', '1e4', '1.5']);
  const stray = numerals.filter((d) => !allowed.has(d));
  assert.deepEqual(stray, [], `numbers typed into index.html: ${stray.join(', ')}`);
});
