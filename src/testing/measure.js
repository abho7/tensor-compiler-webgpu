// Milestone 1's measurements: everything the report shows, computed from
// scratch. Deterministic (every program and every input comes from a seed, and
// nothing is timed), so CI recomputes it and fails if the committed
// golden/measurements.json differs from what the code produces now.

import { randomProgram, GENERATED } from './random-graph.js';
import { checkGradients, secondOrder, TOLERANCE } from './gradcheck.js';
import { RULE_SABOTAGE, sabotagedRules, KERNEL_SABOTAGE, sabotagedKernels } from './sabotage.js';
import { compareAll, GOLDEN_TOLERANCE } from './golden.js';

export const FUZZ_PROGRAMS = 3000;
export const SECOND_ORDER_PROGRAMS = 500;
export const SABOTAGE_LIMIT = 3000;
export const SABOTAGE_RATE_PROGRAMS = 1000;

function campaign(count, check) {
  const tally = { agree: 0, mismatch: 0, error: 0, rejected: 0 };
  let worst = 0, worstSeed = null, rejectedDraws = 0;
  const failures = [];
  const opsChecked = {};
  for (let seed = 1; seed <= count; seed++) {
    const program = randomProgram(seed);
    const r = check(program, seed);
    tally[r.status]++;
    rejectedDraws += r.rejections;
    if (r.status === 'agree') {
      if (r.worst > worst) { worst = r.worst; worstSeed = seed; }
      for (const op of new Set(program.graph.nodes.map((n) => n.op))) opsChecked[op] = (opsChecked[op] ?? 0) + 1;
    } else if (r.status !== 'rejected') {
      failures.push({ seed, status: r.status, worst: r.worst, detail: r.detail });
    }
  }
  return { programs: count, ...tally, rejectedDraws, worst, worstSeed, failures, opsChecked };
}

/** The ops a sabotaged rule can be reached through in a program. */
const REACHED_BY = { stopGradient: 'max' };

function sabotage() {
  return Object.entries(RULE_SABOTAGE).map(([rule, { bug }]) => {
    const rules = sabotagedRules(rule);
    let first = null;
    let containing = 0, caught = 0;
    // Run until the first catch, and at least the first SABOTAGE_RATE_PROGRAMS
    // programs, which give the catch rate among programs using the op.
    for (let seed = 1; seed <= SABOTAGE_LIMIT && (!first || seed <= SABOTAGE_RATE_PROGRAMS); seed++) {
      const program = randomProgram(seed);
      const r = checkGradients(program, { seed, rules });
      const hit = r.status === 'mismatch' || r.status === 'error';
      if (hit && !first) first = { programs: seed, how: r.status, detail: r.detail };
      if (seed <= SABOTAGE_RATE_PROGRAMS && program.graph.nodes.some((n) => n.op === (REACHED_BY[rule] ?? rule))) {
        containing++;
        if (hit) caught++;
      }
    }
    return { rule, bug, caughtAfter: first?.programs ?? null, how: first?.how ?? 'not caught', detail: first?.detail ?? null,
      containing, caught };
  });
}

export function measureM1({ cases, golden }) {
  const fuzz = campaign(FUZZ_PROGRAMS, (p, seed) => checkGradients(p, { seed }));
  const central = campaign(FUZZ_PROGRAMS, (p, seed) => checkGradients(p, { seed, method: 'central' }));
  const second = campaign(SECOND_ORDER_PROGRAMS, (p, seed) => checkGradients(secondOrder(p), { seed }));

  const results = compareAll(cases, golden);
  const worstOf = (k) => Math.max(0, ...results.map((r) => r.errors?.[k] ?? 0));
  const kinds = {};
  for (const r of results) {
    kinds[r.kind] ??= { cases: 0, pass: 0 };
    kinds[r.kind].cases++;
    if (r.ok) kinds[r.kind].pass++;
  }
  const kernelSabotage = Object.entries(KERNEL_SABOTAGE).map(([op, { bug }]) => {
    // Only cases that pass honestly count: random-9 fails either way.
    const r = compareAll(cases, golden, { kernels: sabotagedKernels(op) });
    const newlyFailing = r.filter((x, i) => results[i].ok && !x.ok);
    return { op, bug, casesFailing: newlyFailing.length, cases: r.length,
      randomFailing: newlyFailing.filter((x) => x.kind === 'random').length,
      names: newlyFailing.filter((x) => x.kind !== 'random').map((x) => x.name) };
  });

  return {
    m1: {
      producedBy: 'node tools/measure.js',
      budgets: { finiteDifference: TOLERANCE, torch: GOLDEN_TOLERANCE },
      generatedOps: Object.keys(GENERATED),
      fuzz: { method: 'richardson', ...fuzz },
      fuzzCentral: { method: 'central', ...central },
      secondOrder: second,
      sabotage: { limit: SABOTAGE_LIMIT, ratePrograms: SABOTAGE_RATE_PROGRAMS, rules: sabotage() },
      torch: {
        torchVersion: golden.torch,
        dtype: golden.dtype,
        cases: results.length,
        pass: results.filter((r) => r.ok).length,
        kinds,
        worst: { output: worstOf('output'), grad: worstOf('grad'), nativeOutput: worstOf('nativeOutput'), nativeGrad: worstOf('nativeGrad') },
        // The same, over passing cases only: with the failure included, the
        // gradient figure is the failure's.
        worstPassing: {
          output: Math.max(0, ...results.filter((r) => r.ok).map((r) => r.errors.output)),
          grad: Math.max(0, ...results.filter((r) => r.ok).map((r) => r.errors.grad)),
        },
        failures: results.filter((r) => !r.ok).map((r) => ({ name: r.name, errors: r.errors, absolute: r.absolute ?? null, detail: r.detail ?? null })),
        kernelSabotage,
      },
    },
  };
}
