// Milestone 3b on the float64 reference interpreter: training, evaluation,
// gradients at a point and greedy generation. Shared by CI (verify-cpu.js)
// and the browser harness, so both run the same code.

import { evaluate } from '../interp.js';
import { grad } from '../autodiff.js';
import { CONFIG, buildForward, paramSpec, count } from './model.js';
import { buildStep3b, paramShapes } from './step.js';
import { adamInitialState } from './adam.js';
import { encodeBatch } from './data.js';

const toF64 = (p) => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, Float64Array.from(v)]));

/**
 * Train for `steps` steps; `batchAt(t)` returns that step's T + 1 token
 * sequences. Returns { losses, params (after each step, with keepParams),
 * states (with keepState), final, finalState }.
 */
export function trainCpu3b({ V, B, init, steps, batchAt, bug, keepParams = false, keepState = false, onStep }) {
  const step = buildStep3b(V, B, { bug });
  let params = toF64(init);
  let state = toF64(adamInitialState(step.names, paramShapes(V)));
  const losses = [], kept = [], keptState = [];
  for (let t = 0; t < steps; t++) {
    const { x, y } = encodeBatch(batchAt(t), V);
    const v = evaluate(step.graph, { ...params, ...state, x, y, [step.cotangent]: [1] });
    losses.push(v[step.loss][0]);
    params = Object.fromEntries(step.names.map((n) => [n, v[step.outputs[n]]]));
    state = Object.fromEntries(step.stateNames.map((n) => [n, v[step.outputs[n]]]));
    if (keepParams) kept.push(params);
    if (keepState) keptState.push(state);
    // onStep may return true to stop early (the sabotage runs stop at their first detection).
    if (onStep?.(t, losses[t], params)) break;
  }
  return { losses, params: keepParams ? kept : undefined, states: keepState ? keptState : undefined, final: params, finalState: state };
}

const forwards = new Map();
function forwardFor(V, B, bug) {
  const key = `${V}/${B}/${bug ?? ''}`;
  if (!forwards.has(key)) forwards.set(key, buildForward(V, B, { bug }));
  return forwards.get(key);
}

/** Logits [B·T·V], loss, and the largest attention score's size, for a batch of sequences. */
export function forwardCpu(V, params, seqs, { bug } = {}) {
  const f = forwardFor(V, seqs.length, bug);
  const { x, y } = encodeBatch(seqs, V);
  const v = evaluate(f.graph, { ...params, x, y });
  let maxScore = 0;
  for (const id of f.scores) for (const s of v[id]) maxScore = Math.max(maxScore, Math.abs(s));
  return { logits: v[f.logits], loss: v[f.loss][0], maxScore };
}

/** Loss, logits and every parameter's gradient at a point. */
export function gradientsCpu(V, params, seqs, { bug } = {}) {
  const f = buildForward(V, seqs.length, { bug });
  const names = paramSpec(V).map(([n]) => n);
  const G = grad(f.graph, [f.loss], names.map((n) => f.ids[n]));
  const { x, y } = encodeBatch(seqs, V);
  const v = evaluate(G.graph, { ...params, x, y, [G.cotangents[0]]: [1] });
  return { loss: v[f.loss][0], logits: v[f.logits], grads: Object.fromEntries(names.map((n, k) => [n, v[G.grads[k]]])) };
}

/** Mean loss over sequences, evaluated `chunk` sequences at a time. */
export function evalLossCpu(V, params, seqs, { chunk = 8 } = {}) {
  let total = 0;
  for (let i = 0; i < seqs.length; i += chunk) {
    const part = seqs.slice(i, i + chunk);
    total += forwardCpu(V, params, part).loss * part.length;
  }
  return total / seqs.length;
}

/**
 * Greedy generation in one context window: from `prompt` (token ids), K tokens.
 * `logitsAt(window)` returns the [T·V] logits for one sequence; positions
 * after the last real token hold token 0 and, by causality, cannot matter.
 * Returns { tokens, gaps } with the top-two logit gap at each step.
 */
export async function generateGreedy(V, prompt, K, logitsAt) {
  const { T } = CONFIG;
  const tokens = [...prompt];
  const steps = [];
  for (let k = 0; k < K; k++) {
    const pos = tokens.length - 1;
    const seq = [...tokens, ...new Array(T + 1 - tokens.length).fill(0)];
    const logits = await logitsAt(seq);
    const row = logits.slice(pos * V, (pos + 1) * V);
    let best = 0;
    for (let c = 1; c < V; c++) if (row[c] > row[best]) best = c;
    steps.push({ pos, token: best, row: Array.from(row) });
    tokens.push(best);
  }
  return { tokens: tokens.slice(prompt.length), steps };
}

export const interpLogitsAt = (V, params) => (seq) => forwardCpu(V, params, [seq]).logits;

export { count };
