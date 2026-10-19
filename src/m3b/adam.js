// Milestone 3a's Adam update (DESIGN.md amendment 14), generalised to any
// parameter set (amendment 25). The construction is step.js's, node for node:
// test/m3b-model.test.js builds milestone 3a's MLP step both ways and checks
// that the graphs are identical. step.js itself is not edited, because its
// text is part of milestone 3a's recorded source hash.

/** Adam's state, in step.js's order: each parameter's m and v, then the two beta powers. */
export function adamStateSpec(names, shapes) {
  return [...names.flatMap((p) => [[`${p}.m`, shapes[p]], [`${p}.v`, shapes[p]]]), ['beta1Pow', []], ['beta2Pow', []]];
}

/** Initial state: moments 0, beta powers 1. */
export function adamInitialState(names, shapes) {
  return Object.fromEntries(adamStateSpec(names, shapes).map(([name, shape]) => [
    name, new Array(shape.reduce((a, b) => a * b, 1)).fill(name.endsWith('Pow') ? 1 : 0)]));
}

/**
 * Append Adam's update to graph B. `ids` and `grads` map parameter names to
 * node ids; `state` maps state names to input ids. Returns the outputs: each
 * parameter's and state tensor's new value. `sabotage` takes milestone 3a's
 * optimizer bug names that apply to Adam.
 */
export function adamUpdate(B, names, ids, grads, state, opt, { sabotage } = {}) {
  const c = (v) => B.const(v);
  const outputs = {};
  const decayed = (p) => (sabotage === 'weight-decay-sign'
    ? B.sub(grads[p], B.mul(c(opt.weightDecay), ids[p]))
    : B.add(grads[p], B.mul(c(opt.weightDecay), ids[p])));
  const b1 = B.mul(state.beta1Pow, c(opt.beta1));
  const b2 = B.mul(state.beta2Pow, c(opt.beta2));
  outputs.beta1Pow = b1;
  outputs.beta2Pow = b2;
  const bc1 = B.sub(c(1), b1);
  const bc2 = B.sub(c(1), sabotage === 'adam-bias-correction' ? b1 : b2);
  const stepSize = B.div(c(opt.lr), bc1);
  const sqrtBc2 = B.sqrt(bc2);
  for (const p of names) {
    const gp = decayed(p);
    const m = B.add(B.mul(c(opt.beta1), state[`${p}.m`]), B.mul(c(1 - opt.beta1), gp));
    const v = B.add(B.mul(c(opt.beta2), state[`${p}.v`]), B.mul(c(1 - opt.beta2), B.mul(gp, gp)));
    const denom = B.add(B.div(B.sqrt(v), sqrtBc2), c(opt.eps));
    outputs[p] = B.sub(ids[p], B.mul(stepSize, B.div(m, denom)));
    outputs[`${p}.m`] = m;
    outputs[`${p}.v`] = v;
  }
  return outputs;
}
