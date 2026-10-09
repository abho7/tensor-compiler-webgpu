import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildCases } from '../src/testing/cases.js';
import { compareAll, GOLDEN_TOLERANCE } from '../src/testing/golden.js';
import { KERNEL_SABOTAGE, sabotagedKernels } from '../src/testing/sabotage.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const { cases } = read('../golden/cases.json');
const golden = read('../golden/torch.json');

test('golden/cases.json is exactly what the case builder produces today', () => {
  // If the generator or a hand-built case changes, the PyTorch goldens are
  // stale until oracle/torch_golden.py is rerun; this is where that shows.
  assert.deepEqual(JSON.parse(JSON.stringify(buildCases())), cases);
});

test('golden/torch.json answers every case, in float64', () => {
  assert.equal(golden.dtype, 'float64');
  assert.deepEqual(golden.cases.map((c) => c.name), cases.map((c) => c.name));
});

test('the interpreter and autodiff match PyTorch within 1e-12, except one recorded failure', () => {
  const results = compareAll(cases, golden);
  const failing = results.filter((r) => !r.ok);
  // random-9 computes x0 / x0, whose gradient is exactly zero. Both sides
  // return different rounding residue near 1e-16, which the per-tensor metric
  // (normalised by PyTorch's largest value) scores as an error of 2. It is
  // recorded as a failure of the stated budget (DESIGN.md, amendment 3). This
  // pins it to exactly that: any other failure, or a larger one, fails here.
  assert.deepEqual(failing.map((r) => r.name), ['random-9']);
  const [r9] = failing;
  assert.ok(r9.errors.output <= GOLDEN_TOLERANCE, 'random-9 outputs should still match');
  assert.ok(r9.absolute.gradDiff < 1e-15 && r9.absolute.gradScale < 1e-15,
    `random-9 is no longer rounding residue: ${JSON.stringify(r9.absolute)}`);
});

test('softmax and log-softmax match torch.softmax and torch.log_softmax themselves', () => {
  const composites = compareAll(cases, golden).filter((r) => r.kind === 'composite');
  assert.equal(composites.length, 9);
  for (const r of composites) {
    assert.ok(r.errors.nativeOutput <= GOLDEN_TOLERANCE, `${r.name}: ${r.errors.nativeOutput}`);
    assert.ok(r.errors.nativeGrad <= GOLDEN_TOLERANCE, `${r.name}: ${r.errors.nativeGrad}`);
  }
});

for (const [op, { bug }] of Object.entries(KERNEL_SABOTAGE)) {
  test(`the golden comparison catches a broken ${op} kernel (${bug})`, () => {
    // Counted only among cases that pass with the honest kernels, so the one
    // recorded failure (random-9) cannot make a sabotage look caught.
    const honest = compareAll(cases, golden);
    const failing = compareAll(cases, golden, { kernels: sabotagedKernels(op) }).filter((r, i) => honest[i].ok && !r.ok);
    assert.ok(failing.length >= 1, `no case failed with ${op} broken`);
  });
}
