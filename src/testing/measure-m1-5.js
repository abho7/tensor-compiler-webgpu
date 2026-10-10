// Milestone 1.5's measurements: metric v2, the v2 generator's skip rate and
// agreement, and per-bug detection rates before (v1 generator) and after
// (v2). Deterministic, like milestone 1's, and recomputed by CI.

import { corpusProgram, TARGETS, MAX_SIZE } from './random-graph.js';
import { checkGradients, secondOrder } from './gradcheck.js';
import { RULE_SABOTAGE, sabotagedRules } from './sabotage.js';
import { compareAll, RTOL, ATOL_SCALE, GOLDEN_TOLERANCE } from './golden.js';
import { RULES } from '../autodiff.js';

export const CORPUS_PROGRAMS = 2000;
export const SECOND_ORDER_PROGRAMS = 300;
export const RATE_PROGRAMS_V1 = 1000;
export const INJECTED = 1 + 1e-4;

/** The ops a sabotaged rule can be reached through in a program. */
const REACHED_BY = { stopGradient: 'max' };
const contains = (program, rule) => program.graph.nodes.some((n) => n.op === (REACHED_BY[rule] ?? rule));
const caught = (r) => r.status === 'mismatch' || r.status === 'error';
const source = (i) => (i % 2 === 1 ? 'general' : 'targeted');

function corpus(count) {
  return Array.from({ length: count }, (_, k) => [k + 1, corpusProgram(k + 1)]);
}

function fuzz(programs, check) {
  const tally = { agree: 0, mismatch: 0, error: 0, rejected: 0 };
  let worst = 0, worstIndex = null, rejectedDraws = 0;
  const failures = [];
  const opsChecked = {};
  for (const [i, program] of programs) {
    const r = check(program, i);
    tally[r.status]++;
    rejectedDraws += r.rejections;
    if (r.status === 'agree') {
      if (r.worst > worst) { worst = r.worst; worstIndex = i; }
      for (const op of new Set(program.graph.nodes.map((n) => n.op))) opsChecked[op] = (opsChecked[op] ?? 0) + 1;
    } else if (r.status !== 'rejected') {
      failures.push({ index: i, status: r.status, worst: r.worst, detail: r.detail });
    }
  }
  return { programs: programs.length, ...tally, rejectedDraws, worst, worstIndex, failures, opsChecked };
}

/**
 * Per sabotaged rule: the fraction of programs containing the op that catch
 * it. "Before" is milestone 1's figure over the v1 generator's first
 * RATE_PROGRAMS_V1 programs (same definition, taken from its record); "after"
 * is the v2 corpus, split by source.
 */
function detection(programs, m1) {
  return Object.entries(RULE_SABOTAGE).map(([rule, { bug }]) => {
    const rules = sabotagedRules(rule);
    const old = m1.sabotage.rules.find((r) => r.rule === rule);
    const before = { containing: old.containing, caught: old.caught };
    const after = { general: { containing: 0, caught: 0 }, targeted: { containing: 0, caught: 0 } };
    let firstCatch = null;
    for (const [i, p] of programs) {
      // A rule is only applied where its op appears, so a program without the
      // op computes exactly what the honest rules do and cannot catch it.
      if (!contains(p, rule)) continue;
      const hit = caught(checkGradients(p, { seed: i, rules }));
      if (hit && firstCatch === null) firstCatch = i;
      after[source(i)].containing++;
      if (hit) after[source(i)].caught++;
    }
    after.corpus = {
      containing: after.general.containing + after.targeted.containing,
      caught: after.general.caught + after.targeted.caught,
    };
    return { rule, bug, targeted: TARGETS.includes(rule), before, after, firstCatch };
  });
}

/** Every case under both metrics, and the (1 + 1e-4) injections v2 must flag. */
function metric(cases, golden) {
  const honest = compareAll(cases, golden);
  const worst = (r, k) => Math.max(0, ...Object.values(r[k] ?? {}));
  const table = honest.map((r) => ({
    name: r.name, set: r.set, kind: r.kind,
    v1: worst(r, 'errors'), v1Pass: r.ok, v2: worst(r, 'scores'), v2Pass: r.okV2,
  }));
  const summary = (set) => {
    const rows = table.filter((t) => set === 'all' || t.set === set);
    return { cases: rows.length, v1Pass: rows.filter((t) => t.v1Pass).length, v2Pass: rows.filter((t) => t.v2Pass).length,
      worstV2: Math.max(0, ...rows.map((t) => t.v2)) };
  };

  const changed = (a, b) => Object.keys(a).some((k) => a[k].some((v, j) => v !== b[k][j]));
  // Array.from: flatMap does not flatten typed arrays, and Math.max of one is NaN.
  const maxChange = (a, b) => Math.max(0, ...Object.keys(a).flatMap((k) => Array.from(a[k], (v, j) => Math.abs(v - b[k][j]))));

  // (a) every gradient of every case multiplied by 1 + 1e-4.
  const scaled = compareAll(cases, golden, { scaleGrads: INJECTED });
  const allGrads = { affected: 0, flagged: 0, missed: [] };
  scaled.forEach((r, i) => {
    if (!honest[i].okV2 || !changed(r.grads, honest[i].grads)) return;
    allGrads.affected++;
    if (!r.okV2) allGrads.flagged++;
    else allGrads.missed.push({ name: r.name, change: maxChange(r.grads, honest[i].grads), gradScale: honest[i].absolute.gradScale });
  });

  // (b) one rule at a time, its contribution multiplied by 1 + 1e-4.
  const perRule = Object.keys(RULES).filter((op) => !['step', 'stopGradient', 'max'].includes(op)).map((op) => {
    const rules = { ...RULES, [op]: (G, n, gy) => RULES[op](G, n, gy).map((c) => (c === null ? null : G.mul(c, G.const(INJECTED)))) };
    const r = compareAll(cases, golden, { rules });
    const out = { op, affected: 0, flagged: 0, missed: [] };
    r.forEach((x, i) => {
      if (!honest[i].okV2 || !x.grads || !changed(x.grads, honest[i].grads)) return;
      out.affected++;
      if (!x.okV2) out.flagged++;
      else out.missed.push({ name: x.name, change: maxChange(x.grads, honest[i].grads), gradScale: honest[i].absolute.gradScale });
    });
    return out;
  });

  return {
    formula: 'abs(ours - torch) <= atol + rtol * abs(torch), atol = atolScale * max(1, max abs(torch over the tensor))',
    rtol: RTOL, atolScale: ATOL_SCALE, v1Tolerance: GOLDEN_TOLERANCE,
    summary: { m1: summary('m1'), 'm1.5': summary('m1.5'), all: summary('all') },
    cases: table,
    injection: { factor: INJECTED, allGradients: allGrads, perRule },
  };
}

export function measureM15({ cases, golden, m1 }) {
  const programs = corpus(CORPUS_PROGRAMS);
  const f = fuzz(programs, (p, i) => checkGradients(p, { seed: i }));
  const second = fuzz(programs.slice(0, SECOND_ORDER_PROGRAMS), (p, i) => checkGradients(secondOrder(p), { seed: i }));
  return {
    m1_5: {
      producedBy: 'node tools/measure.js',
      generator: { maxSize: MAX_SIZE, targets: TARGETS, corpus: 'odd indices general, even indices targeted, cycling through the targets' },
      skips: {
        before: { programs: m1.fuzz.programs, rejected: m1.fuzz.rejected },
        after: { programs: f.programs, rejected: f.rejected },
      },
      fuzz: f,
      secondOrder: second,
      detection: { target: 0.5, ratePrograms: RATE_PROGRAMS_V1, rules: detection(programs, m1) },
      metric: metric(cases, golden),
    },
  };
}
