// The float32 budget (DESIGN.md amendment 8): the float64 reference for a
// program, and beside every value a first-order bound E on the error a
// float32 GPU evaluation may make, propagated node by node from the WGSL
// specification's per-op accuracy. A GPU value passes when
// abs(gpu - ref) <= S · E.
//
// Inputs and constants are rounded to float32 first (amendment 9), so both
// sides start from identical values.

import { KERNELS } from '../interp.js';
import { size } from '../ir/shapes.js';

export const S = 2;
export const U = 2 ** -24;
export const EPS = 2 ** -23;
export const TAU = 2 ** -126;
export const F32_MAX = 3.4028234663852886e38;
export const gamma = (n) => (n * U) / (1 - n * U);

const fround = (v) => Math.fround(v);
const ftz = (r) => (r !== 0 && Math.abs(r) < TAU ? TAU : 0);
const abs = (a) => a.map(Math.abs);

/** Elementwise bound rules: (r, operands, operand bounds) -> bound. */
const ELEMENTWISE = {
  add: (r, [a, b], [ea, eb]) => ea + eb + U * Math.abs(r),
  sub: (r, [a, b], [ea, eb]) => ea + eb + U * Math.abs(r),
  mul: (r, [a, b], [ea, eb]) => Math.abs(b) * ea + Math.abs(a) * eb + ea * eb + U * Math.abs(r),
  div: (r, [a, b], [ea, eb]) => {
    const m = Math.abs(b);
    if (eb >= m || m < 2 ** -126 || m > 2 ** 126) return Infinity;
    return (ea + Math.abs(r) * eb) / (m - eb) + 2.5 * EPS * Math.abs(r);
  },
  neg: (r, a, [ex]) => ex,
  relu: (r, a, [ex]) => ex,
  stopGradient: (r, a, [ex]) => ex,
  exp: (r, [x], [ex]) => {
    const grow = Math.exp(ex);
    return Math.abs(r) * (grow - 1) + (3 + 2 * (Math.abs(x) + ex)) * EPS * Math.abs(r) * grow;
  },
  log: (r, [x], [ex]) => {
    if (!(x > 0) || ex >= x) return Infinity;
    const inherited = -Math.log1p(-ex / x);
    return inherited + Math.max(2 ** -21, 3 * EPS * (Math.abs(r) + inherited));
  },
  tanh: (r, [x], [ex]) => ex + Math.max(1e-5, (9.5 + 4 * (Math.abs(x) + ex)) * EPS),
  // Amendment 13: inherited from 1/inverseSqrt(x), 2 + 2.5 ULP; sqrt is
  // 1/2-Hölder, so the inherited part stays finite even when e_x reaches x.
  sqrt: (r, [x], [ex]) => {
    if (x < 0) return Infinity;
    const inherited = x > 0 ? Math.min(Math.sqrt(ex), ex / Math.sqrt(x)) : Math.sqrt(ex);
    return inherited + 4.5 * EPS * Math.abs(r);
  },
  step: (r, [x], [ex]) => (ex > 0 && Math.abs(x) <= ex ? 1 : 0),
};

/** Bounds for reductions and matmul, built from the interpreter's own kernels on bound arrays. */
const STRUCTURED = {
  sum(node, [x], [ex], g) {
    const n = g.nodes[node.inputs[0]].shape[node.attrs.axis];
    const inherited = KERNELS.sum(node, [ex], g);
    const mass = KERNELS.sum(node, [abs(x)], g);
    return inherited.map((e, i) => e + gamma(n) * mass[i]);
  },
  max(node, [x], [ex], g) {
    return KERNELS.max(node, [ex], g);
  },
  matmul(node, [a, b], [ea, eb], g) {
    const K = g.nodes[node.inputs[0]].shape.at(-1);
    const mm = (p, q) => KERNELS.matmul(node, [p, q], g);
    const aa = abs(a), bb = abs(b);
    const t1 = mm(aa, eb), t2 = mm(ea, bb), t3 = mm(ea, eb), mass = mm(aa, bb);
    return t1.map((v, i) => v + t2[i] + t3[i] + gamma(K) * mass[i]);
  },
  // Data movement: the bound moves with the value.
  broadcast: (node, xs, [ex], g) => KERNELS.broadcast(node, [ex], g),
  reshape: (node, xs, [ex], g) => KERNELS.reshape(node, [ex], g),
  transpose: (node, xs, [ex], g) => KERNELS.transpose(node, [ex], g),
};

/**
 * Evaluate `graph` in float64 from float32-rounded inputs and constants, with
 * a bound per element. Returns { values, bounds, inputs }: arrays per node,
 * and the rounded inputs actually used (what the GPU must be given).
 *
 * `carried` (milestone 3a, amendment 17) maps input names to { value, bound }:
 * an input that is a previous step's output, taken at its exact float64 value
 * with that step's bound, rather than rounded to float32 with bound zero. Its
 * entry in `inputs` is then the float64 value; the GPU is given its own.
 */
export function referenceWithBounds(graph, rawInputs, { carried = {} } = {}) {
  const values = new Array(graph.nodes.length);
  const bounds = new Array(graph.nodes.length);
  const inputs = {};
  for (const node of graph.nodes) {
    let value, bound;
    if (node.op === 'input' && carried[node.attrs.name]) {
      const c = carried[node.attrs.name];
      if (c.value.length !== size(node.shape)) throw new Error(`carried input '${node.attrs.name}' has ${c.value.length} values, needs ${size(node.shape)}`);
      value = Float64Array.from(c.value);
      inputs[node.attrs.name] = value;
      bound = Float64Array.from(c.bound);
    } else if (node.op === 'input') {
      const raw = rawInputs[node.attrs.name];
      if (raw === undefined) throw new Error(`no value for input '${node.attrs.name}'`);
      if (raw.length !== size(node.shape)) throw new Error(`input '${node.attrs.name}' has ${raw.length} values, needs ${size(node.shape)}`);
      value = Float64Array.from(raw, fround);
      inputs[node.attrs.name] = value;
      bound = value.map(ftz);
    } else if (node.op === 'const') {
      value = KERNELS.const(node).map(fround);
      bound = value.map(ftz);
    } else {
      const args = node.inputs.map((id) => values[id]);
      const errs = node.inputs.map((id) => bounds[id]);
      value = KERNELS[node.op](node, args, graph);
      if (STRUCTURED[node.op]) {
        bound = STRUCTURED[node.op](node, args, errs, graph);
      } else {
        const rule = ELEMENTWISE[node.op];
        if (!rule) throw new Error(`no bound rule for '${node.op}'`);
        bound = new Float64Array(value.length);
        for (let i = 0; i < value.length; i++) {
          bound[i] = rule(value[i], args.map((a) => a[i]), errs.map((e) => e[i]));
        }
      }
      // Plus ftz on every node, as amendment 8 states it: a subnormal result
      // may have been flushed to zero. (For pure copies this is slightly more
      // than the specification needs, by at most 2^-126.)
      for (let i = 0; i < value.length; i++) bound[i] += ftz(value[i]);
    }
    values[node.id] = value;
    bounds[node.id] = bound;
  }
  return { values, bounds, inputs };
}

/** Elements that cannot be judged: an unbounded error, or a reference outside float32's range. */
export function unjudgeable(values, bounds) {
  let count = 0;
  for (let n = 0; n < values.length; n++) {
    for (let i = 0; i < values[n].length; i++) {
      const r = values[n][i];
      if (!Number.isFinite(r) || Math.abs(r) > F32_MAX || !Number.isFinite(bounds[n][i])) count++;
    }
  }
  return count;
}

/**
 * Judge GPU values against the reference, node by node. Returns
 * { pass, worst, first } where worst is the largest abs(gpu - ref) / (S·E)
 * (pass when <= 1) and first describes the first failing node in graph order.
 */
export function judge(graph, gpu, values, bounds, { only } = {}) {
  let worst = 0, first = null;
  for (const node of graph.nodes) {
    if (only && !only.has(node.id)) continue;
    const g = gpu[node.id], r = values[node.id], e = bounds[node.id];
    for (let i = 0; i < r.length; i++) {
      const d = Math.abs(g[i] - r[i]);
      const allowed = S * e[i];
      const ratio = d === 0 ? 0 : (Number.isNaN(d) ? Infinity : d / allowed);
      if (ratio > worst) worst = ratio;
      if (ratio > 1 && first === null) {
        first = { node: node.id, op: node.op, index: i, gpu: g[i], ref: r[i], bound: e[i], ratio };
      }
    }
  }
  return { pass: first === null, worst, first };
}
