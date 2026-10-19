// Milestone 3b's checks that need no GPU (DESIGN.md amendments 23, 26, 27):
// the interpreter against PyTorch at initialisation and per training step,
// the causality and batch-order properties, and each model bug against all of
// them. Deterministic, so CI recomputes it.

import { Rng } from '../testing/rng.js';
import { CONFIG, MODEL_BUGS, initParams, paramSpec, count } from './model.js';
import { trainCpu3b, forwardCpu, gradientsCpu } from './cpu.js';
import { projectionVector, CAUSALITY_POSITIONS } from './config.js';

export const TAU1 = 1e-10;
export const TAU2 = 1e-8;

/** Metric v2 at tolerance tau: the largest ratio of a difference to its allowance (<= 1 passes). */
export function v2Score(ours, ref, tau) {
  let scale = 0;
  for (const v of ref) scale = Math.max(scale, Math.abs(v));
  const atol = tau * Math.max(1, scale);
  let worst = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(ours[i] - ref[i]);
    worst = Math.max(worst, Number.isNaN(d) ? Infinity : d / (atol + tau * Math.abs(ref[i])));
  }
  return worst;
}

/**
 * The projected check of amendment 23 (ii): |<r, ours> - <r, torch>| against
 * sum_i |r_i| (atol + rtol |p_i|), implied by the elementwise check. PyTorch's
 * parameters are stored only after step N, so |p_i| and the tensor's scale are
 * taken from this side's; where the check passes the two differ by at most
 * 1e-8 relative, which moves the allowance by no more than that.
 */
function projectionScore(ours, r, refProj, tau) {
  let scale = 0;
  for (const v of ours) scale = Math.max(scale, Math.abs(v));
  const atol = tau * Math.max(1, scale);
  let p = 0, allow = 0;
  for (let i = 0; i < r.length; i++) { p += r[i] * ours[i]; allow += Math.abs(r[i]) * (atol + tau * Math.abs(ours[i])); }
  const d = Math.abs(p - refProj);
  return Number.isNaN(d) ? Infinity : d / allow;
}

/** Reads golden/torch-m3b.f64 through the JSON's index. */
export function torchBlob(meta, bytes) {
  const all = new Float64Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 8);
  return (key) => {
    const at = meta.blob.index[key];
    if (!at) throw new Error(`torch-m3b has no ${key}`);
    return all.subarray(at[0], at[0] + at[1]);
  };
}

/** The sequences each init case and golden step uses. */
export function caseData(spec, textIds) {
  const syn = spec.synthetic.train;
  const T1 = CONFIG.T + 1;
  const cases = spec.golden.map((g) => ({ name: g.name, V: g.V, seed: g.seed, seqs: g.schedule[0].map((i) => syn[i]) }));
  const ti = spec.text.init;
  cases.push({ name: 'text-seed1', V: spec.text.V, seed: ti.seed, seqs: ti.offsets.map((o) => textIds.slice(o, o + T1)) });
  return cases;
}

// ---- (i) at initialisation ------------------------------------------------------

export function initCheck(spec, torch, blob, textIds, { bug } = {}) {
  return caseData(spec, textIds).map((c, k) => {
    const params = initParams(c.V, c.seed);
    const ours = gradientsCpu(c.V, params, c.seqs, { bug });
    const scores = { loss: v2Score([ours.loss], [torch.init[k].loss], TAU1), logits: v2Score(ours.logits, blob(`init/${c.name}/logits`), TAU1) };
    for (const [n] of paramSpec(c.V)) scores[`grad ${n}`] = v2Score(ours.grads[n], blob(`init/${c.name}/grad/${n}`), TAU1);
    const [worstOf, worst] = Object.entries(scores).reduce((a, b) => (b[1] > a[1] ? b : a));
    return { name: c.name, worst, worstOf, pass: worst <= 1, logitsScore: scores.logits, lossScore: scores.loss };
  });
}

// ---- (ii) training, N steps ----------------------------------------------------------

export function trainingCheck(spec, torch, blob, { bug, stopAtFirstFailure = false } = {}) {
  const syn = spec.synthetic.train;
  const names = paramSpec(16).map(([n]) => n);
  const shapes = Object.fromEntries(paramSpec(16));
  const r = Object.fromEntries(names.map((n) => [n, Array.from({ length: spec.projections.k }, (_, j) => projectionVector(n, count(shapes[n]), j))]));
  return spec.golden.map((g, k) => {
    const ref = torch.training[k];
    const steps = [];
    let firstFailure = null;
    const run = trainCpu3b({ V: g.V, B: g.batch, init: initParams(g.V, g.seed), steps: g.steps,
      batchAt: (t) => g.schedule[t].map((i) => syn[i]), bug,
      onStep: (t, loss, params) => {
        const ls = v2Score([loss], [ref.losses[t]], TAU2);
        let proj = 0;
        for (const n of names) for (let j = 0; j < spec.projections.k; j++) proj = Math.max(proj, projectionScore(params[n], r[n][j], ref.projections[t][n][j], TAU2));
        steps.push({ loss: ls, projection: proj });
        if (Math.max(ls, proj) > 1 && firstFailure === null) firstFailure = t;
        return stopAtFirstFailure && firstFailure !== null;
      } });
    let finalWorst = null, finalWorstOf = null;
    if (steps.length === g.steps) {
      finalWorst = 0;
      for (const n of names) {
        const s = v2Score(run.final[n], blob(`train/${g.name}/final/${n}`), TAU2);
        if (s > finalWorst) { finalWorst = s; finalWorstOf = n; }
      }
      if (finalWorst > 1 && firstFailure === null) firstFailure = g.steps - 1;
    }
    return {
      name: g.name, steps: steps.length, worstLoss: Math.max(...steps.map((s) => s.loss)), worstProjection: Math.max(...steps.map((s) => s.projection)),
      finalWorst, finalWorstOf, firstFailure, pass: firstFailure === null,
      maxLossDiff: Math.max(...run.losses.map((l, t) => Math.abs(l - ref.losses[t]))),
    };
  });
}

// ---- properties -------------------------------------------------------------------------

const bitwiseEqual = (a, b, from, to) => { for (let i = from; i < to; i++) if (!Object.is(a[i], b[i])) return false; return true; };

/**
 * Causality (amendment 26): replace the token at t in sequence 0 and require
 * bitwise-identical logits at every earlier position of every sequence, and
 * some change at t or later. `logitsOf(seqs)` gives [B·T·V] logits.
 */
export async function causalityTrials(V, seqs, logitsOf, { seed = 1 } = {}) {
  const { T } = CONFIG;
  const base = await logitsOf(seqs);
  const rng = new Rng((seed * 31337 + 1) >>> 0);
  const out = [];
  for (const t of CAUSALITY_POSITIONS) {
    const changed = seqs.map((s) => [...s]);
    changed[0][t] = (changed[0][t] + 1 + rng.int(0, V - 2)) % V;
    const after = await logitsOf(changed);
    const before = bitwiseEqual(base, after, 0, t * V);
    let others = true;
    for (let b = 1; b < seqs.length; b++) others &&= bitwiseEqual(base, after, b * T * V, (b + 1) * T * V);
    const later = !bitwiseEqual(base, after, t * V, T * V);
    out.push({ t, earlierIdentical: before, otherSequencesIdentical: others, laterChanged: later, pass: before && others && later });
  }
  return out;
}

/** Batch order (amendment 26): reversing the batch reverses the logits bitwise; loss and gradients within tau1. */
export function batchOrderTrial(V, params, seqs, { bug, logitsOnly = false } = {}) {
  const { T } = CONFIG;
  const rev = [...seqs].reverse();
  const a = logitsOnly ? forwardCpu(V, params, seqs, { bug }) : gradientsCpu(V, params, seqs, { bug });
  const b = logitsOnly ? forwardCpu(V, params, rev, { bug }) : gradientsCpu(V, params, rev, { bug });
  const B = seqs.length, row = T * V;
  let logitsPermuted = true;
  for (let i = 0; i < B; i++) logitsPermuted &&= bitwiseEqual(a.logits.subarray(i * row, (i + 1) * row), b.logits.subarray((B - 1 - i) * row, (B - i) * row), 0, row);
  let gradScore = logitsOnly ? 0 : v2Score([b.loss], [a.loss], TAU1);
  if (!logitsOnly) for (const n of Object.keys(a.grads)) gradScore = Math.max(gradScore, v2Score(b.grads[n], a.grads[n], TAU1));
  return { logitsPermuted, gradScore, pass: logitsPermuted && gradScore <= 1 };
}

// ---- all of it, and the sabotage ---------------------------------------------------------

export async function propertiesCpu(spec, { bug } = {}) {
  const syn = spec.synthetic.train;
  const causality = [], batchOrder = [];
  for (const g of spec.golden) {
    const params = initParams(g.V, g.seed);
    const seqs = g.schedule[0].map((i) => syn[i]);
    for (const tr of await causalityTrials(g.V, seqs, (s) => forwardCpu(g.V, params, s, { bug }).logits, { seed: g.seed })) causality.push({ config: g.name, ...tr });
    batchOrder.push({ config: g.name, ...batchOrderTrial(g.V, params, seqs, { bug }) });
  }
  return { causality, batchOrder };
}

/** The largest attention score's size (amendment 21 assumed well under 1e3): each golden config at initialisation. */
export function largestScores(spec) {
  const syn = spec.synthetic.train;
  return spec.golden.map((g) => ({ config: g.name, maxScore: forwardCpu(g.V, initParams(g.V, g.seed), g.schedule[0].map((i) => syn[i])).maxScore }));
}

export async function verifyCpu3b(spec, torch, bytes, textIds) {
  const blob = torchBlob(torch, bytes);
  const init = initCheck(spec, torch, blob, textIds);
  const training = trainingCheck(spec, torch, blob);
  const properties = await propertiesCpu(spec);
  const scores = largestScores(spec);
  const sabotage = [];
  for (const bug of Object.keys(MODEL_BUGS)) {
    const p = await propertiesCpu(spec, { bug });
    const i = initCheck(spec, torch, blob, textIds, { bug });
    const t = trainingCheck(spec, torch, blob, { bug, stopAtFirstFailure: true });
    sabotage.push({
      name: bug, bug: MODEL_BUGS[bug],
      checks: {
        causality: { trials: p.causality.length, caught: p.causality.filter((c) => !c.earlierIdentical || !c.otherSequencesIdentical).length },
        batchOrder: { trials: p.batchOrder.length, caught: p.batchOrder.filter((c) => !c.pass).length },
        torchInit: { trials: i.length, caught: i.filter((c) => !c.pass).length },
        torchTraining: { trials: t.length, caught: t.filter((c) => !c.pass).length, firstFailures: t.map((c) => c.firstFailure) },
      },
    });
  }
  return { tau1: TAU1, tau2: TAU2, init, training, properties, largestScores: scores, sabotage };
}
