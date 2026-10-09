// Compares the interpreter and autodiff with the PyTorch float64 golden files.
// Budget (DESIGN.md): per tensor, max abs(ours - torch) / max abs(torch) <= 1e-12.

import { Graph } from '../ir/graph.js';
import { evaluate, KERNELS } from '../interp.js';
import { grad, RULES } from '../autodiff.js';

export const GOLDEN_TOLERANCE = 1e-12;

/** Per-tensor relative error; zero only if both are exactly zero. */
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

/**
 * Errors for one case: { errors: { output, grad, nativeOutput?, nativeGrad? },
 * absolute: { gradDiff, gradScale } }, each error the worst per-tensor error of
 * its kind. Throws if the graph fails to run.
 */
export function compareCase(c, ref, { kernels = KERNELS, rules = RULES } = {}) {
  let absolute;
  const graph = Graph.fromJSON(c.graph);
  const inputs = graph.inputs();
  const G = grad(graph, c.outputs, inputs.map((n) => n.id), { rules });
  const feed = { ...c.inputs };
  G.cotangents.forEach((name, i) => { feed[name] = c.cotangents[i]; });
  const values = evaluate(G.graph, feed, { kernels });

  const outs = c.outputs.map((id) => values[id]);
  const result = {
    output: Math.max(...outs.map((o, i) => tensorError(o, ref.outputs[i]))),
    grad: Math.max(0, ...inputs.map((n, i) => tensorError(values[G.grads[i]], ref.grads[n.attrs.name]))),
  };
  // Absolute figures for the gradients, so a failure can be told apart from
  // two different roundings of an exact zero.
  let gradDiff = 0, gradScale = 0;
  inputs.forEach((n, i) => {
    const ours = values[G.grads[i]], theirs = ref.grads[n.attrs.name];
    for (let j = 0; j < theirs.length; j++) {
      gradDiff = Math.max(gradDiff, Math.abs(ours[j] - theirs[j]));
      gradScale = Math.max(gradScale, Math.abs(theirs[j]));
    }
  });
  absolute = { gradDiff, gradScale };
  if (ref.native) {
    result.nativeOutput = tensorError(outs[0], ref.native.output);
    result.nativeGrad = tensorError(values[G.grads[0]], ref.native.grad);
  }
  return { errors: result, absolute };
}

/** Every case: [{ name, kind, errors, ok }]. */
export function compareAll(cases, golden, options) {
  const refs = new Map(golden.cases.map((r) => [r.name, r]));
  return cases.map((c) => {
    const ref = refs.get(c.name);
    if (!ref) return { name: c.name, kind: c.kind, errors: null, ok: false, detail: 'missing from torch.json' };
    let errors, absolute;
    try {
      ({ errors, absolute } = compareCase(c, ref, options));
    } catch (error) {
      return { name: c.name, kind: c.kind, errors: null, ok: false, detail: error.message };
    }
    const ok = Object.values(errors).every((e) => e <= GOLDEN_TOLERANCE);
    return { name: c.name, kind: c.kind, errors, absolute, ok };
  });
}
