// Compares the interpreter and autodiff with the PyTorch float64 golden files,
// under two metrics, both recorded.
//
// v1 (milestone 1, DESIGN.md "Error budget"): per tensor,
//   max abs(ours - torch) / max abs(torch) <= 1e-12.
// It fails on an exact zero computed as rounding residue (random-9: x0 / x0),
// because it divides by the residue. It is kept, and still reported.
//
// v2 (decided in milestone 1.5, before any GPU measurement; DESIGN.md
// "Metric v2"): per element,
//   abs(ours - torch) <= atol + rtol * abs(torch),
//   atol = ATOL_SCALE * max(1, max abs(torch over the tensor)),  rtol = RTOL.
// A tensor's score is the largest ratio of the left side to the right; it
// passes at <= 1.

import { Graph } from '../ir/graph.js';
import { evaluate, KERNELS } from '../interp.js';
import { grad, RULES } from '../autodiff.js';

export const GOLDEN_TOLERANCE = 1e-12;
export const RTOL = 1e-12;
export const ATOL_SCALE = 1e-12;

/** v1: per-tensor relative error; zero only if both are exactly zero. */
export function tensorError(ours, ref) {
  if (ours.length !== ref.length) return Infinity;
  let diff = 0, scale = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(ours[i] - ref[i]);
    diff = Number.isNaN(d) ? Infinity : Math.max(diff, d);
    scale = Math.max(scale, Math.abs(ref[i]));
  }
  return diff === 0 ? 0 : diff / Math.max(scale, 1e-300);
}

/** v2: the largest abs(ours - ref) / (atol + rtol * abs(ref)) over the tensor; passes at <= 1. */
export function tensorScore(ours, ref) {
  if (ours.length !== ref.length) return Infinity;
  let scale = 0;
  for (const v of ref) scale = Math.max(scale, Math.abs(v));
  const atol = ATOL_SCALE * Math.max(1, scale);
  let worst = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(ours[i] - ref[i]);
    const r = Number.isNaN(d) ? Infinity : d / (atol + RTOL * Math.abs(ref[i]));
    if (r > worst) worst = r;
  }
  return worst;
}

/**
 * One case under both metrics:
 *   errors:   v1 errors { output, grad, nativeOutput?, nativeGrad? }
 *   scores:   v2 scores, same keys
 *   absolute: { gradDiff, gradScale } for telling residue from error
 *   grads:    our gradients, by input name
 * `scaleGrads` multiplies every gradient before comparing: the injected
 * (1 + 1e-4) error the v2 metric must still flag. Throws if the graph fails.
 */
export function compareCase(c, ref, { kernels = KERNELS, rules = RULES, scaleGrads = 1 } = {}) {
  const graph = Graph.fromJSON(c.graph);
  const inputs = graph.inputs();
  const G = grad(graph, c.outputs, inputs.map((n) => n.id), { rules });
  const feed = { ...c.inputs };
  G.cotangents.forEach((name, i) => { feed[name] = c.cotangents[i]; });
  const values = evaluate(G.graph, feed, { kernels });

  const outs = c.outputs.map((id) => values[id]);
  const grads = inputs.map((n, i) => values[G.grads[i]].map((v) => v * scaleGrads));
  const pairs = {
    output: outs.map((o, i) => [o, ref.outputs[i]]),
    grad: inputs.map((n, i) => [grads[i], ref.grads[n.attrs.name]]),
  };
  if (ref.native) {
    pairs.nativeOutput = [[outs[0], ref.native.output]];
    pairs.nativeGrad = [[grads[0], ref.native.grad]];
  }
  const errors = {}, scores = {};
  for (const [k, list] of Object.entries(pairs)) {
    errors[k] = Math.max(0, ...list.map(([a, b]) => tensorError(a, b)));
    scores[k] = Math.max(0, ...list.map(([a, b]) => tensorScore(a, b)));
  }

  // Absolute figures for the gradients, so a failure can be told apart from
  // two different roundings of an exact zero.
  let gradDiff = 0, gradScale = 0;
  pairs.grad.forEach(([ours, theirs]) => {
    for (let j = 0; j < theirs.length; j++) {
      gradDiff = Math.max(gradDiff, Math.abs(ours[j] - theirs[j]));
      gradScale = Math.max(gradScale, Math.abs(theirs[j]));
    }
  });
  const byName = Object.fromEntries(inputs.map((n, i) => [n.attrs.name, grads[i]]));
  return { errors, scores, absolute: { gradDiff, gradScale }, grads: byName };
}

/** Every case: [{ name, kind, set, errors, scores, absolute, ok, okV2 }]. */
export function compareAll(cases, golden, options) {
  const refs = new Map(golden.cases.map((r) => [r.name, r]));
  return cases.map((c) => {
    const base = { name: c.name, kind: c.kind, set: c.set ?? 'm1' };
    const ref = refs.get(c.name);
    if (!ref) return { ...base, errors: null, ok: false, okV2: false, detail: 'missing from torch.json' };
    let r;
    try {
      r = compareCase(c, ref, options);
    } catch (error) {
      return { ...base, errors: null, ok: false, okV2: false, detail: error.message };
    }
    return {
      ...base, errors: r.errors, scores: r.scores, absolute: r.absolute, grads: r.grads,
      ok: Object.values(r.errors).every((e) => e <= GOLDEN_TOLERANCE),
      okV2: Object.values(r.scores).every((s) => s <= 1),
    };
  });
}
