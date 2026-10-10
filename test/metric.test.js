import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { compareAll, tensorScore, tensorError, GOLDEN_TOLERANCE } from '../src/testing/golden.js';
import { RULES } from '../src/autodiff.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const { cases } = read('../golden/cases.json');
const golden = read('../golden/torch.json');
const honest = compareAll(cases, golden);

test('v2 by hand: an absolute floor near zero, relative for large values', () => {
  assert.equal(tensorScore([1, 2], [1, 2]), 0);
  assert.ok(tensorScore([1e-13, 0], [0, 0]) <= 1, 'residue near an exact zero should pass');
  assert.ok(tensorScore([1e-11, 0], [0, 0]) > 1, 'an absolute error past the floor should fail');
  assert.ok(tensorScore([1e6 + 1e-7], [1e6]) <= 1);
  assert.ok(tensorScore([1e6 + 1e-5], [1e6]) > 1, '1e-11 relative at 1e6 should fail');
});

test('random-9 fails v1 and passes v2; nothing else changes verdict', () => {
  const r9 = honest.find((r) => r.name === 'random-9');
  assert.ok(!r9.ok, 'random-9 was expected to keep failing v1');
  assert.ok(r9.okV2, 'random-9 should pass v2');
  assert.deepEqual(honest.filter((r) => !r.ok).map((r) => r.name), ['random-9']);
  assert.deepEqual(honest.filter((r) => !r.okV2).map((r) => r.name), []);
});

test('v2 is not too loose: every gradient wrong by 1 + 1e-4 is flagged', () => {
  // In every case whose true gradient is not zero. random-9's is exactly
  // zero, and scaling its rounding residue by 1.0001 is not an error.
  const scaled = compareAll(cases, golden, { scaleGrads: 1 + 1e-4 });
  scaled.forEach((r, i) => {
    if (honest[i].absolute.gradScale < 1e-8) return;
    assert.ok(!r.okV2, `${r.name}: a 1e-4 gradient error passed v2 (score ${r.scores.grad})`);
  });
});

test('v2 flags one rule wrong by 1 + 1e-4, wherever it changes a real gradient', () => {
  for (const op of ['mul', 'matmul', 'sum', 'broadcast', 'transpose', 'tanh']) {
    const rules = { ...RULES, [op]: (G, n, gy) => RULES[op](G, n, gy).map((c) => (c === null ? null : G.mul(c, G.const(1 + 1e-4)))) };
    const r = compareAll(cases, golden, { rules });
    let affected = 0;
    r.forEach((x, i) => {
      const change = Math.max(0, ...Object.keys(x.grads).flatMap((k) => Array.from(x.grads[k], (v, j) => Math.abs(v - honest[i].grads[k][j]))));
      if (change < 1e-10) return;
      affected++;
      assert.ok(!x.okV2, `${op} x (1 + 1e-4) passed v2 on ${x.name} (change ${change})`);
    });
    assert.ok(affected > 0, `no case exercises ${op}`);
  }
});

test('v1 is unchanged by the rewrite: same per-tensor errors as recorded in milestone 1', () => {
  const m1 = read('../golden/measurements.json').m1.torch;
  const r9 = honest.find((r) => r.name === 'random-9');
  assert.deepEqual(r9.errors, m1.failures[0].errors);
  assert.equal(tensorError([0], [0]), 0);
  assert.equal(GOLDEN_TOLERANCE, 1e-12);
});
