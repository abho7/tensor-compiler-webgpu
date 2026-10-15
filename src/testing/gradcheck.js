// Checks autodiff's gradients against central finite differences.
//
// For a program with outputs y_i and a fixed random cotangent w_i per output,
// the scalar L = Σ_i Σ w_i ⊙ y_i has gradient exactly what grad() computes
// when fed w_i as cotangents. Central differences of L, one input element at a
// time, give an independent estimate. The budget is in DESIGN.md:
// abs(g - fd) / max(1, abs(g), abs(fd)) <= 1e-6 per element.
//
// The estimate is Richardson-extrapolated central differences,
// (4·D(h/2) - D(h)) / 3, which cancels the h² truncation term. Plain central
// differences (`method: 'central'`) were the first version, and they raised
// false alarms: near a pole their truncation error alone broke the budget
// while autodiff was right (seed 496: -24679.7021 against an exact
// -24679.6668). The budget was not widened; the instrument was fixed, and the
// old method is kept so the measurement records how often it misfired.
//
// Finite differences are only meaningful where L is smooth, so a point is
// rejected, and new inputs drawn, if
// - any value is non-finite or larger than LIMIT in magnitude (exp chains);
// - a log argument or a division's denominator is within DELTA of zero; or
// - any perturbed evaluation lands on the other side of a kink: a relu or
//   step input changes sign. This is checked exactly, by comparing the sign
//   pattern of every relu/step input at each perturbed point with the base.
// Rejections are counted and reported, never silently dropped.

import { evaluate, KERNELS } from '../interp.js';
import { grad, RULES } from '../autodiff.js';
import { size } from '../ir/shapes.js';
import { Rng } from './rng.js';

export const TOLERANCE = 1e-6;
const DELTA = 1e-3;
const LIMIT = 1e4;

function pointIsUsable(graph, inputs, kernels) {
  let ok = true;
  const signs = [];
  const values = evaluate(graph, inputs, {
    kernels,
    onNode(node, args, value) {
      for (const v of value) if (!Number.isFinite(v) || Math.abs(v) > LIMIT) ok = false;
      if (node.op === 'log') for (const v of args[0]) if (v < DELTA) ok = false;
      if (node.op === 'div') for (const v of args[1]) if (Math.abs(v) < DELTA) ok = false;
      if (node.op === 'relu' || node.op === 'step') for (const v of args[0]) signs.push(v > 0 ? 1 : 0);
    },
  });
  return { ok, signs: signs.join(''), values };
}

/**
 * Check one program. `program` is { graph, outputs, wrt? } where `wrt` names
 * the inputs to differentiate with respect to (default: all of them); every
 * other input is drawn at random and held fixed.
 *
 * Returns { status, worst, rejections, elements, detail }, status being
 * 'agree', 'mismatch', 'rejected' (no usable point found) or 'error' (the
 * transform or interpreter threw).
 */
export function checkGradients(program, { seed = 1, rules = RULES, kernels = KERNELS, tolerance = TOLERANCE, tries = 20, method = 'richardson' } = {}) {
  const { graph, outputs } = program;
  const allInputs = graph.inputs();
  const wrtNames = program.wrt ?? allInputs.map((n) => n.attrs.name);
  const wrtIds = wrtNames.map((name) => graph.inputId(name));

  let G;
  try {
    G = grad(graph, outputs, wrtIds, { rules, prefix: `d${graph.nodes.length}_` });
  } catch (error) {
    return { status: 'error', worst: Infinity, rejections: 0, elements: 0, detail: `grad: ${error.message}` };
  }

  const rng = new Rng((seed * 2654435761) >>> 0);
  let rejections = 0;
  for (let attempt = 0; attempt < tries; attempt++) {
    const inputs = {};
    for (const n of allInputs) inputs[n.attrs.name] = rng.array(size(n.shape), -2, 2);
    const w = outputs.map((id) => rng.array(size(graph.nodes[id].shape), -1, 1));

    let base;
    try {
      base = pointIsUsable(graph, inputs, kernels);
    } catch (error) {
      return { status: 'error', worst: Infinity, rejections, elements: 0, detail: `forward: ${error.message}` };
    }
    if (!base.ok) { rejections++; continue; }

    // L from an already-evaluated point: each perturbed point is evaluated once,
    // for its usability and its loss together.
    const loss = (values) => {
      let total = 0;
      outputs.forEach((id, i) => { for (let j = 0; j < w[i].length; j++) total += w[i][j] * values[id][j]; });
      return total;
    };

    // Finite differences first, so a kink crossing rejects the point before
    // anything is compared.
    const fd = [];
    let crossed = false;
    for (const name of wrtNames) {
      const x = inputs[name];
      const column = [];
      for (let j = 0; j < x.length && !crossed; j++) {
        const h = 1e-5 * Math.max(1, Math.abs(x[j]));
        const central = (step) => {
          const plus = { ...inputs, [name]: x.map((v, k) => (k === j ? v + step : v)) };
          const minus = { ...inputs, [name]: x.map((v, k) => (k === j ? v - step : v)) };
          const p = pointIsUsable(graph, plus, kernels);
          const m = pointIsUsable(graph, minus, kernels);
          if (!p.ok || !m.ok || p.signs !== base.signs || m.signs !== base.signs) return null;
          return (loss(p.values) - loss(m.values)) / (2 * step);
        };
        const d1 = central(h);
        const d2 = method === 'richardson' ? central(h / 2) : 0;
        if (d1 === null || d2 === null) { crossed = true; break; }
        column.push(method === 'richardson' ? (4 * d2 - d1) / 3 : d1);
      }
      fd.push(column);
    }
    if (crossed) { rejections++; continue; }

    let analytic;
    try {
      const feed = { ...inputs };
      G.cotangents.forEach((name, i) => { feed[name] = w[i]; });
      const values = evaluate(G.graph, feed, { kernels });
      analytic = G.grads.map((id) => values[id]);
    } catch (error) {
      return { status: 'error', worst: Infinity, rejections, elements: 0, detail: `gradient graph: ${error.message}` };
    }

    let worst = 0;
    let detail = '';
    let elements = 0;
    wrtNames.forEach((name, i) => {
      for (let j = 0; j < fd[i].length; j++) {
        elements++;
        const a = analytic[i][j];
        const f = fd[i][j];
        const err = Number.isFinite(a) ? Math.abs(a - f) / Math.max(1, Math.abs(a), Math.abs(f)) : Infinity;
        if (!(err <= worst)) { worst = err; detail = `${name}[${j}]: autodiff ${a}, finite difference ${f}`; }
      }
    });
    // point: the inputs and output weights this verdict was reached at, for
    // explaining a result afterwards (milestone 3a); no record stores it.
    return { status: worst <= tolerance ? 'agree' : 'mismatch', worst, rejections, elements, detail, point: { inputs, w } };
  }
  return { status: 'rejected', worst: 0, rejections, elements: 0, detail: `no usable point in ${tries} draws` };
}

/**
 * The second-order program of `program`: its gradient graph, whose outputs are
 * the gradients, differentiated with respect to the original inputs while the
 * cotangent inputs are drawn at random and held fixed.
 */
export function secondOrder(program, { rules = RULES } = {}) {
  const { graph, outputs } = program;
  const names = graph.inputs().map((n) => n.attrs.name);
  const G = grad(graph, outputs, names.map((n) => graph.inputId(n)), { rules });
  return { graph: G.graph, outputs: G.grads, wrt: names };
}
