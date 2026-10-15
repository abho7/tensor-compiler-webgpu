// Training on the float64 reference interpreter: the step graph run once per
// batch, its outputs fed back as the next step's parameters and state.

import { evaluate } from '../interp.js';
import { buildStep, buildEval, initialState, PARAMS } from './step.js';
import { batchSchedule, gather, initMlp } from './data.js';

/**
 * Train for `steps` steps. Returns { losses, params } where losses[t] is the
 * loss computed at step t (from the parameters before that step's update) and
 * params[t] the parameters after it (only when `keepParams`). `init` and
 * `schedule` default to the seeded ones, so the same seed gives the same run.
 */
export function trainCpu({ dims, data, opt, seed, steps, batch, sabotage, keepParams = false, init, schedule }) {
  const step = buildStep(dims, batch, opt, { sabotage });
  let params = Object.fromEntries(Object.entries(init ?? initMlp(dims, seed)).map(([k, v]) => [k, Float64Array.from(v)]));
  let state = initialState(opt, dims);
  const order = schedule ?? batchSchedule(data.y.length, batch, steps, seed);
  const losses = [], kept = [];
  for (let t = 0; t < steps; t++) {
    const { x, y } = gather(data, order[t], dims.inputs, dims.classes);
    const values = evaluate(step.graph, { ...params, ...state, x, y, [step.cotangent]: [1] });
    losses.push(values[step.loss][0]);
    params = Object.fromEntries(PARAMS.map((p) => [p, values[step.outputs[p]]]));
    state = Object.fromEntries(Object.keys(state).map((k) => [k, values[step.outputs[k]]]));
    if (keepParams) kept.push(params);
  }
  return { losses, params: keepParams ? kept : undefined, final: params };
}

/** Mean cross-entropy and accuracy of `params` on `data`, in float64. */
export function evaluateMlp(dims, params, data) {
  const rows = data.y.length;
  const { graph, logp } = buildEval(dims, rows);
  const lp = evaluate(graph, { ...params, x: data.x })[logp];
  let nll = 0, correct = 0;
  for (let r = 0; r < rows; r++) {
    nll -= lp[r * dims.classes + data.y[r]];
    let best = 0;
    for (let c = 1; c < dims.classes; c++) if (lp[r * dims.classes + c] > lp[r * dims.classes + best]) best = c;
    if (best === data.y[r]) correct++;
  }
  return { loss: nll / rows, accuracy: correct / rows };
}
