// Milestone 3b's model (DESIGN.md amendments 20 and 21): a decoder-only,
// pre-LayerNorm character transformer built only from IR ops that milestones 1
// to 3a verified. No op is added.
//
// Tokens arrive one-hot, as [B·T, V], so the embedding lookup is a matmul.
// Activations are kept as [B·T, d] between blocks; attention reshapes them to
// [B, H, T, d_h] and back.

import { Graph } from '../ir/graph.js';
import { Rng } from '../testing/rng.js';

export const CONFIG = { T: 64, d: 64, H: 4, dh: 16, L: 2, ff: 256, eps: 1e-5, mask: -1e4 };

/** Amendment 27's model bugs, each injected alone by the builder. */
export const MODEL_BUGS = {
  'mask-off-by-one': 'causal mask lets row i see position i + 1',
  'scale-missing': 'attention scores not scaled by 1/sqrt(d_h)',
  'softmax-wrong-axis': 'attention softmax over the query axis',
  'ln-divisor': 'LayerNorm variance divided by d - 1',
  'ln-eps': 'LayerNorm eps 1e-3 instead of 1e-5',
  'merge-no-transpose': 'heads merged by a reshape without the transpose',
  'position-off-by-one': 'position p gets embedding row (p + 1) mod T',
};

/**
 * Parameters in their fixed order: [name, shape, init], where init is
 * ['uniform', bound], ['ones'] or ['zeros'].
 */
export function paramSpec(V, { T, d, L, ff } = CONFIG) {
  const linear = (fanIn) => ['uniform', 1 / Math.sqrt(fanIn)];
  const spec = [['tok', [V, d], ['uniform', 1]], ['pos', [T, d], ['uniform', 1]]];
  for (let l = 0; l < L; l++) {
    const p = `b${l}.`;
    spec.push(
      [`${p}ln1.g`, [d], ['ones']], [`${p}ln1.b`, [d], ['zeros']],
      [`${p}wq`, [d, d], linear(d)], [`${p}wk`, [d, d], linear(d)], [`${p}wv`, [d, d], linear(d)],
      [`${p}wo`, [d, d], linear(d)], [`${p}bo`, [d], linear(d)],
      [`${p}ln2.g`, [d], ['ones']], [`${p}ln2.b`, [d], ['zeros']],
      [`${p}w1`, [d, ff], linear(d)], [`${p}b1`, [ff], linear(d)],
      [`${p}w2`, [ff, d], linear(ff)], [`${p}b2`, [d], linear(ff)],
    );
  }
  spec.push(['lnf.g', [d], ['ones']], ['lnf.b', [d], ['zeros']], ['head', [d, V], linear(d)], ['headb', [V], linear(d)]);
  return spec;
}

export const count = (shape) => shape.reduce((a, b) => a * b, 1);
export const paramCount = (V) => paramSpec(V).reduce((n, [, s]) => n + count(s), 0);

/**
 * Initial parameters for `seed`, drawn in paramSpec order, each tensor in
 * row-major order (oracle/torch_transformer.py replays this with a port of the RNG).
 */
export function initParams(V, seed) {
  const rng = new Rng((seed * 7001 + 13) >>> 0);
  const out = {};
  for (const [name, shape, [kind, bound]] of paramSpec(V)) {
    const n = count(shape);
    out[name] = kind === 'uniform' ? rng.array(n, -bound, bound) : new Array(n).fill(kind === 'ones' ? 1 : 0);
  }
  return out;
}

/** The causal mask: 0 where row i may see column j, `mask` elsewhere. */
export function causalMask(T, bug) {
  const reach = bug === 'mask-off-by-one' ? 1 : 0;
  const m = [];
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) m.push(j <= i + reach ? 0 : CONFIG.mask);
  return m;
}

function layerNorm(g, x, gain, bias, bug) {
  const { d, eps } = CONFIG;
  const mean = g.div(g.sum(x, 1, true), g.const(d));
  const xc = g.sub(x, mean);
  const variance = g.div(g.sum(g.mul(xc, xc), 1, true), g.const(bug === 'ln-divisor' ? d - 1 : d));
  const y = g.div(xc, g.sqrt(g.add(variance, g.const(bug === 'ln-eps' ? 1e-3 : eps))));
  return g.add(g.mul(y, gain), bias);
}

function gelu(g, x) {
  const cube = g.mul(x, g.mul(x, x));
  const inner = g.mul(g.const(Math.sqrt(2 / Math.PI)), g.add(x, g.mul(g.const(0.044715), cube)));
  return g.mul(g.mul(g.const(0.5), x), g.add(g.const(1), g.tanh(inner)));
}

function attention(g, a, p, B, bug, trace) {
  const { T, d, H, dh } = CONFIG;
  const heads = (w) => g.transpose(g.reshape(g.matmul(a, p[w]), [B, T, H, dh]), [0, 2, 1, 3]);
  const q = heads('wq'), k = heads('wk'), v = heads('wv');
  let s = g.matmul(q, g.transpose(k, [0, 1, 3, 2]));
  if (bug !== 'scale-missing') s = g.mul(s, g.const(1 / Math.sqrt(dh)));
  trace?.push(s);
  s = g.add(s, g.const(causalMask(T, bug), [T, T]));
  const probs = g.softmax(s, bug === 'softmax-wrong-axis' ? 2 : 3);
  const o = g.matmul(probs, v);
  const merged = bug === 'merge-no-transpose' ? g.reshape(o, [B * T, d]) : g.reshape(g.transpose(o, [0, 2, 1, 3]), [B * T, d]);
  return g.add(g.matmul(merged, p.wo), p.bo);
}

/** A cyclic shift by one row, for the position bug: (S·P)[p] = P[(p + 1) mod T]. */
function shiftMatrix(T) {
  const m = new Array(T * T).fill(0);
  for (let p = 0; p < T; p++) m[p * T + ((p + 1) % T)] = 1;
  return m;
}

/**
 * The model on graph `g`: `ids` maps every parameter name and 'x' (one-hot
 * tokens, [B·T, V]) to node ids. Returns the logits' id, [B·T, V].
 */
export function logitsOf(g, ids, V, B, { bug, trace } = {}) {
  const { T, d, L } = CONFIG;
  let pos = ids.pos;
  if (bug === 'position-off-by-one') pos = g.matmul(g.const(shiftMatrix(T), [T, T]), pos);
  let h = g.reshape(g.add(g.reshape(g.matmul(ids.x, ids.tok), [B, T, d]), pos), [B * T, d]);
  for (let l = 0; l < L; l++) {
    const p = Object.fromEntries(['ln1.g', 'ln1.b', 'wq', 'wk', 'wv', 'wo', 'bo', 'ln2.g', 'ln2.b', 'w1', 'b1', 'w2', 'b2'].map((k) => [k, ids[`b${l}.${k}`]]));
    h = g.add(h, attention(g, layerNorm(g, h, p['ln1.g'], p['ln1.b'], bug), p, B, bug, trace));
    const m = layerNorm(g, h, p['ln2.g'], p['ln2.b'], bug);
    h = g.add(h, g.add(g.matmul(gelu(g, g.add(g.matmul(m, p.w1), p.b1)), p.w2), p.b2));
  }
  return g.add(g.matmul(layerNorm(g, h, ids['lnf.g'], ids['lnf.b'], bug), ids.head), ids.headb);
}

/** Mean next-token negative log-likelihood over all B·T positions; `y` is one-hot [B·T, V]. */
export function lossOf(g, logits, y, rows) {
  const logp = g.logSoftmax(logits, 1);
  return g.div(g.neg(g.sum(g.sum(g.mul(y, logp), 1), 0)), g.const(rows));
}

/** A forward graph: inputs are the parameters, x and y. Returns { graph, ids, logits, loss }. */
export function buildForward(V, B, { bug } = {}) {
  const { T } = CONFIG;
  const g = new Graph();
  const ids = {};
  for (const [name, shape] of paramSpec(V)) ids[name] = g.input(name, shape);
  ids.x = g.input('x', [B * T, V]);
  ids.y = g.input('y', [B * T, V]);
  // The scaled attention scores, before the mask (amendment 21 records their largest size).
  const scores = [];
  const logits = logitsOf(g, ids, V, B, { bug, trace: scores });
  const loss = lossOf(g, logits, ids.y, B * T);
  return { graph: g, ids, logits, loss, scores };
}
