// Why a wrong sqrt gradient (g / y instead of g / 2y) goes uncaught in some
// programs: each miss is classified at the exact point its check used.
//
//   sqrt-on-constants      no sqrt depends on any input, so autodiff never
//                          applies sqrt's rule
//   zero-cotangent         the gradient arriving at every input-dependent sqrt
//                          is exactly zero (e.g. a softmax over a size-1 axis,
//                          whose output is constant)
//   argument-constant      sqrt's argument does not vary with the inputs
//                          (e.g. x - x), so what sqrt's rule sends back is
//                          multiplied by an exact zero
//   below-tolerance        the rule's error reaches an input, but by less than
//                          the check's tolerance at that point
//   unexplained            none of the above

import { grad } from '../autodiff.js';
import { evaluate } from '../interp.js';
import { RULES } from '../autodiff.js';
import { sabotagedRules } from './sabotage.js';
import { randomProgram } from './random-graph.js';
import { checkGradients, TOLERANCE } from './gradcheck.js';

const maxAbs = (a) => Math.max(0, ...Array.from(a, Math.abs));

function gradientsAt(graph, outputs, wrt, inputs, cotangents, rules) {
  const G = grad(graph, outputs, wrt, { rules });
  const feed = { ...inputs };
  G.cotangents.forEach((n, i) => { feed[n] = cotangents[i]; });
  const v = evaluate(G.graph, feed);
  return G.grads.map((id) => v[id]);
}

function ancestors(g, ids) {
  const set = new Set(ids);
  for (let i = g.nodes.length - 1; i >= 0; i--) if (set.has(i)) g.nodes[i].inputs.forEach((j) => set.add(j));
  return set;
}
function descendants(g, ids) {
  const set = new Set(ids);
  for (const n of g.nodes) if (n.inputs.some((i) => set.has(i))) set.add(n.id);
  return set;
}
/** A softmax over a size-1 axis: its max reduces an axis of extent 1, and its output is constant. */
const size1Softmax = (g, set) => [...set].some((i) => g.nodes[i].op === 'max' && g.shape(g.nodes[i].inputs[0])[g.nodes[i].attrs.axis] === 1);
/** x - x: exactly zero whatever x is. */
const selfSubtraction = (g, set) => [...set].some((i) => g.nodes[i].op === 'sub' && g.nodes[i].inputs[0] === g.nodes[i].inputs[1]);
/** y / y: exactly one, so its true gradient is zero and only rounding residue flows back. */
const selfDivision = (g, set) => [...set].some((i) => g.nodes[i].op === 'div' && g.nodes[i].inputs[0] === g.nodes[i].inputs[1]);

function explain(g, set) {
  if (size1Softmax(g, set)) return 'a softmax over a size-1 axis';
  if (selfSubtraction(g, set)) return 'x - x';
  if (selfDivision(g, set)) return 'y / y';
  return 'other';
}

/**
 * Classify one miss. `tolerance(honest, sabotaged)` returns the largest ratio
 * of the change in an input gradient to what the check allows.
 */
export function classifySqrtMiss(graph, outputs, inputs, cotangents, tolerance) {
  const inputIds = graph.inputs().map((n) => n.id);
  const dep = new Array(graph.nodes.length).fill(false);
  for (const id of inputIds) dep[id] = true;
  for (const n of graph.nodes) if (n.inputs.some((i) => dep[i])) dep[n.id] = true;
  const sqrts = graph.nodes.filter((n) => n.op === 'sqrt' && dep[n.id]);
  if (sqrts.length === 0) return { reason: 'sqrt-on-constants' };

  const cot = sqrts.map((n) => maxAbs(gradientsAt(graph, outputs, [n.id], inputs, cotangents, RULES)[0]));
  if (cot.every((c) => c === 0)) {
    return { reason: 'zero-cotangent', detail: explain(graph, descendants(graph, sqrts.map((n) => n.id))) };
  }
  // Sensitivity of each live sqrt's argument to the inputs, with unit weights.
  const live = sqrts.filter((n, k) => cot[k] !== 0);
  const argConstant = live.every((n) => {
    const arg = n.inputs[0];
    const ones = [new Float64Array(graph.shape(arg).reduce((p, d) => p * d, 1)).fill(1)];
    return gradientsAt(graph, [arg], inputIds, inputs, ones, RULES).every((g) => maxAbs(g) === 0);
  });
  if (argConstant) return { reason: 'argument-constant', detail: explain(graph, ancestors(graph, live.map((n) => n.inputs[0]))) };

  const honest = gradientsAt(graph, outputs, inputIds, inputs, cotangents, RULES);
  const bad = gradientsAt(graph, outputs, inputIds, inputs, cotangents, sabotagedRules('sqrt'));
  const ratio = tolerance(honest, bad);
  return ratio <= 1
    ? { reason: 'below-tolerance', ratio, detail: explain(graph, descendants(graph, live.map((n) => n.id))) }
    : { reason: 'unexplained', ratio };
}

/** The finite-difference check's allowance, abs(g - fd) <= 1e-6 * max(1, abs(g)), applied to the change. */
function fdRatio(honest, bad) {
  let worst = 0;
  honest.forEach((h, i) => h.forEach((v, j) => {
    worst = Math.max(worst, Math.abs(bad[i][j] - v) / (TOLERANCE * Math.max(1, Math.abs(v))));
  }));
  return worst;
}

/**
 * The finite-difference sabotage campaign for sqrt: programs 1..n from the
 * withSqrt generator; for each containing sqrt, whether the wrong rule is
 * caught, and for each miss, why.
 */
export function sqrtFdCampaign(n = 400) {
  const rules = sabotagedRules('sqrt');
  const out = { programs: n, containing: 0, honestAgree: 0, honestOther: [], caught: 0, misses: [] };
  for (let i = 1; i <= n; i++) {
    const p = randomProgram(i, { withSqrt: true });
    if (!p.graph.nodes.some((x) => x.op === 'sqrt')) continue;
    out.containing++;
    const h = checkGradients(p, { seed: i });
    if (h.status === 'agree') out.honestAgree++; else out.honestOther.push({ seed: i, status: h.status, detail: h.detail });
    const b = checkGradients(p, { seed: i, rules });
    if (b.status === 'mismatch' || b.status === 'error') { out.caught++; continue; }
    const why = b.point
      ? classifySqrtMiss(p.graph, p.outputs, b.point.inputs, b.point.w, fdRatio)
      : { reason: 'unexplained', detail: `check status ${b.status}` };
    out.misses.push({ seed: i, ...why });
  }
  return out;
}

/** Metric v2's allowance for a gradient change, as compareAll applies it per tensor. */
function v2Ratio(honest, bad) {
  let worst = 0;
  honest.forEach((h, i) => {
    let scale = 0;
    for (const v of h) scale = Math.max(scale, Math.abs(v));
    const atol = 1e-12 * Math.max(1, scale);
    h.forEach((v, j) => { worst = Math.max(worst, Math.abs(bad[i][j] - v) / (atol + 1e-12 * Math.abs(v))); });
  });
  return worst;
}

/**
 * The PyTorch golden cases for sqrt: how many flag the wrong rule, why the
 * rest do not, and which inputs have a gradient that is exactly zero (about
 * which a gradient check says nothing).
 */
export function sqrtGoldenAnalysis(cases, golden, { compareAll, Graph }) {
  const honest = compareAll(cases, golden);
  const bad = compareAll(cases, golden, { rules: sabotagedRules('sqrt') });
  const flagged = honest.filter((h, i) => h.okV2 && !bad[i].okV2).length;
  const misses = [];
  const zeroGradients = { cases: 0, inputs: 0, unusedInputs: 0, reachingInputs: [], casesWithNoNonzeroGradient: [] };
  cases.forEach((c, k) => {
    const g = Graph.fromJSON(c.graph);
    if (honest[k].okV2 && bad[k].okV2) {
      misses.push({ name: c.name, ...classifySqrtMiss(g, c.outputs, c.inputs, c.cotangents, v2Ratio) });
    }
    const ref = golden.cases[k].grads;
    const zero = Object.keys(ref).filter((n) => ref[n].every((v) => v === 0));
    if (zero.length === 0) return;
    zeroGradients.cases++;
    const live = ancestors(g, c.outputs);
    for (const n of zero) {
      zeroGradients.inputs++;
      if (live.has(g.inputId(n))) zeroGradients.reachingInputs.push(`${c.name}:${n}`); else zeroGradients.unusedInputs++;
    }
    if (zero.length === Object.keys(ref).length) zeroGradients.casesWithNoNonzeroGradient.push(c.name);
  });
  return { cases: cases.length, honestPass: honest.filter((h) => h.okV2).length, flagged, misses, zeroGradients };
}
