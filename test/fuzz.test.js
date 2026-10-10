import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { randomProgram } from '../src/testing/random-graph-v1.js';
import { checkGradients, secondOrder } from '../src/testing/gradcheck.js';
import { RULE_SABOTAGE, sabotagedRules } from '../src/testing/sabotage.js';
import { RULES } from '../src/autodiff.js';

const measured = JSON.parse(readFileSync(new URL('../golden/measurements.json', import.meta.url))).m1;

test('autodiff agrees with finite differences on 300 random programs', () => {
  const tally = { agree: 0, rejected: 0, mismatch: 0, error: 0 };
  for (let seed = 1; seed <= 300; seed++) {
    const r = checkGradients(randomProgram(seed), { seed });
    tally[r.status]++;
    assert.ok(r.status === 'agree' || r.status === 'rejected', `seed ${seed}: ${r.status} ${r.detail}`);
  }
  assert.ok(tally.agree >= 270, `only ${tally.agree} of 300 programs were checkable`);
});

test('second-order gradients agree with finite differences', () => {
  for (let seed = 1; seed <= 60; seed++) {
    const r = checkGradients(secondOrder(randomProgram(seed)), { seed });
    assert.ok(r.status === 'agree' || r.status === 'rejected', `seed ${seed}: ${r.status} ${r.detail}`);
  }
});

test('the generator uses every op it claims to', () => {
  const seen = new Set();
  for (let seed = 1; seed <= 300; seed++) for (const n of randomProgram(seed).graph.nodes) seen.add(n.op);
  for (const op of ['add', 'sub', 'mul', 'div', 'neg', 'exp', 'log', 'relu', 'tanh', 'matmul', 'sum', 'max',
    'stopGradient', 'broadcast', 'reshape', 'transpose', 'const', 'input']) {
    assert.ok(seen.has(op), `no program in 300 uses ${op}`);
  }
});

test('every gradient rule has a sabotaged version', () => {
  for (const op of Object.keys(RULES)) {
    if (['step', 'max'].includes(op)) continue; // step has no gradient; max refuses one
    assert.ok(RULE_SABOTAGE[op], `no sabotage for ${op}`);
  }
});

for (const s of measured.sabotage.rules) {
  test(`sabotaged ${s.rule} (${s.bug}) is caught at the recorded program`, () => {
    // The recorded program catches it, and the same program passes with the
    // honest rules, so what was caught is the sabotage and nothing else.
    assert.ok(s.caughtAfter, `${s.rule} was never caught`);
    const program = randomProgram(s.caughtAfter);
    const bad = checkGradients(program, { seed: s.caughtAfter, rules: sabotagedRules(s.rule) });
    assert.equal(bad.status, s.how);
    const honest = checkGradients(program, { seed: s.caughtAfter });
    assert.equal(honest.status, 'agree');
  });
}

test('the old central-difference checker raises the false alarm the amendment records', () => {
  // Seed 496 divides by a sum near zero. Plain central differences break the
  // budget on truncation alone; Richardson extrapolation does not, and the
  // gradient itself is right (DESIGN.md, amendment 1).
  const program = randomProgram(496);
  assert.equal(checkGradients(program, { seed: 496, method: 'central' }).status, 'mismatch');
  assert.equal(checkGradients(program, { seed: 496 }).status, 'agree');
});
