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

test('the report page types in no measured number', () => {
  // Every figure comes from measurements.json. The page's own numerals are
  // limited to fixed context: milestone numbers, 10⁴, 20 draws, logits near 800.
  const html = text('../index.html');
  const body = html.slice(html.indexOf('<main>'), html.indexOf('<script'));
  const script = html.slice(html.indexOf('<script'));
  const prose = (body + script.replace(/\$\{[^}]*\}/g, '')).replace(/<[^>]+>/g, ' ');
  const numerals = prose.match(/\b\d+(\.\d+)?(e-?\d+)?\b/g) ?? [];
  // 1e-3 and 1e4 are sci()'s display thresholds for switching to exponent form.
  const allowed = new Set(['1', '2', '3', '4', '10', '20', '800', '100', '0', '1e-3', '1e4']);
  const stray = numerals.filter((d) => !allowed.has(d));
  assert.deepEqual(stray, [], `numbers typed into index.html: ${stray.join(', ')}`);
});
