// Random well-typed programs for the gradient fuzzer, version 2.
//
// A program is { seed, focus, graph, outputs, inputs, bounds }: bounds holds
// each node's interval. The same seed (and focus) always produces the same
// program.
//
// What changed from v1 (random-graph-v1.js, frozen as the M1 record):
//
// - Valid by construction. 3.8% of v1's programs had no point where finite
//   differences mean anything: almost all a `log` whose argument could go
//   negative, or a division whose denominator could reach zero. v2 carries an
//   interval for every value, sound for inputs drawn in [-2, 2], and only emits
//   log, div and exp where the interval proves them safe, wrapping the operand
//   (x*x + c, exp of something bounded, tanh) where it does not. Ops whose
//   result could exceed BOUND in magnitude are not emitted.
// - Larger, non-square shapes: dimensions up to 6, size-1 dimensions on
//   purpose, ranks up to 4, tensors up to MAX_SIZE elements.
// - Targeted programs (`focus`): built around one op, so matmul, transpose,
//   reshape, broadcast and sum appear often and in the shapes where their
//   bugs are visible. v1's matmul operands were often reshaped to [1, n],
//   where a reshape and a transpose agree; its transposes were mostly rank 2,
//   where every permutation is its own inverse.
//
// As in v1, max and stopGradient only appear inside softmax and log-softmax,
// where their combined gradient is mathematically zero.

import { Graph } from '../ir/graph.js';
import { broadcastShapes, size } from '../ir/shapes.js';
import { Rng } from './rng.js';

export const MAX_SIZE = 96;
export const BOUND = 1e3;
const INPUT_RANGE = [-2, 2];

/** Ops the general generator chooses between, with relative weights. */
export const GENERATED = {
  add: 3, sub: 3, mul: 3, div: 3, neg: 1, exp: 2, log: 2, relu: 3, tanh: 2,
  matmul: 4, sum: 3, broadcast: 2, reshape: 3, transpose: 3, softmax: 2, logSoftmax: 2,
};

/** Ops with a targeted generator. */
export const TARGETS = ['matmul', 'transpose', 'reshape', 'broadcast', 'sum'];

// ---- intervals -------------------------------------------------------------

const mulIv = ([a, b], [c, d]) => {
  const p = [a * c, a * d, b * c, b * d];
  return [Math.min(...p), Math.max(...p)];
};
const magnitude = ([lo, hi]) => Math.max(Math.abs(lo), Math.abs(hi));

/** The interval of a node's value, given its inputs' intervals. Sound, not tight. */
function interval(g, id, iv) {
  const n = g.nodes[id];
  const [a, b] = n.inputs.map((i) => iv.get(i));
  switch (n.op) {
    case 'input': return [...INPUT_RANGE];
    case 'const': {
      const v = n.attrs.value;
      return typeof v === 'number' ? [v, v] : [Math.min(...v), Math.max(...v)];
    }
    case 'add': return [a[0] + b[0], a[1] + b[1]];
    case 'sub': return [a[0] - b[1], a[1] - b[0]];
    case 'mul': return mulIv(a, b);
    case 'div':
      if (b[0] <= 0 && b[1] >= 0) return [-Infinity, Infinity];
      return mulIv(a, [1 / b[1], 1 / b[0]]);
    case 'neg': return [-a[1], -a[0]];
    case 'exp': return [Math.exp(a[0]), Math.exp(a[1])];
    case 'log': return a[0] > 0 ? [Math.log(a[0]), Math.log(a[1])] : [-Infinity, Infinity];
    case 'relu': return [Math.max(0, a[0]), Math.max(0, a[1])];
    case 'step': return [0, 1];
    case 'tanh': return [Math.tanh(a[0]), Math.tanh(a[1])];
    case 'matmul': {
      const k = g.shape(n.inputs[0]).at(-1);
      // A sum of k products, each within the product interval.
      const [lo, hi] = mulIv(a, b);
      return [k * lo, k * hi];
    }
    case 'sum': {
      const len = g.shape(n.inputs[0])[n.attrs.axis];
      return [len * a[0], len * a[1]];
    }
    case 'max': case 'stopGradient': case 'broadcast': case 'reshape': case 'transpose':
      return [...a];
    default: throw new Error(`no interval rule for ${n.op}`);
  }
}

// ---- the generator ---------------------------------------------------------

function round(v) { return Math.round(v * 1024) / 1024; }

function factorizations(n, rank) {
  if (rank === 1) return [[n]];
  const out = [];
  for (let d = 1; d <= n; d++) {
    if (n % d) continue;
    for (const rest of factorizations(n / d, rank - 1)) out.push([d, ...rest]);
  }
  return out;
}

const isInvolution = (perm) => perm.every((p, i) => perm[p] === i);

class Builder {
  constructor(rng) {
    this.rng = rng;
    this.g = new Graph();
    this.iv = new Map();
    this.pool = [];
    this.leaves = 0;
    // Set while a targeted program is built: no size-1 dimensions, matmul's
    // right operand always an input, and the shapes and permutations that hide
    // bugs avoided. The general generator keeps all of those on purpose.
    this.strict = false;
  }

  /** Record a node's interval; returns the id, or undefined if it is too large. */
  track(id) {
    if (!this.iv.has(id)) {
      // Builder helpers may have emitted intermediate nodes (broadcasts).
      for (let i = 0; i <= id; i++) if (!this.iv.has(i)) this.iv.set(i, interval(this.g, i, this.iv));
    }
    return id;
  }

  ok(id) { return magnitude(this.iv.get(this.track(id))) <= BOUND; }

  dim() {
    if (this.strict) return this.rng.int(2, 5);
    return this.rng.chance(0.2) ? 1 : this.rng.int(2, 6);
  }

  shape(minRank = 0, maxRank = 4) {
    for (;;) {
      const rank = this.rng.int(minRank, maxRank);
      const s = Array.from({ length: rank }, () => this.dim());
      if (size(s) <= MAX_SIZE) return s;
    }
  }

  input(shape) { return this.push(this.g.input(`x${this.leaves++}`, shape)); }

  leaf(shape) {
    // Inputs are what gradients are taken with respect to; constants are not.
    if (this.rng.chance(0.7)) return this.input(shape);
    return this.push(this.g.const(this.rng.array(size(shape), -2, 2).map(round), shape));
  }

  push(id) { this.track(id); this.pool.push(id); return id; }

  /** A value, favouring recent ones so programs are deep rather than wide. */
  choose(filter = () => true) {
    const c = this.pool.filter((id) => filter(this.g.shape(id), id));
    if (c.length === 0) return undefined;
    const r = this.rng.next();
    return c[c.length - 1 - Math.floor(r * r * c.length)];
  }

  /** The first operand of the op being applied: the forced one if given, else chosen. */
  first(filter = () => true) {
    if (this.operand === undefined) return this.choose(filter);
    return filter(this.g.shape(this.operand), this.operand) ? this.operand : undefined;
  }

  scalarConst(lo, hi) { return this.track(this.g.const(round(this.rng.uniform(lo, hi)))); }

  /** A value with the same shape as x whose interval is bounded away from zero. */
  safeDenominator(x) {
    const [lo, hi] = this.iv.get(x);
    if (lo >= 0.1 || hi <= -0.1) return x;
    const r = this.rng.next();
    if (r < 0.4) return this.track(this.g.add(this.g.mul(x, x), this.scalarConst(0.2, 1)));
    if (r < 0.6) return this.track(this.g.neg(this.g.add(this.g.mul(x, x), this.scalarConst(0.2, 1))));
    return this.track(this.g.exp(this.boundedForExp(x)));
  }

  /** x, or tanh(x) when exp(x) could exceed the bound. */
  boundedForExp(x) {
    return this.iv.get(x)[1] <= 4 ? x : this.track(this.g.tanh(x));
  }

  /** A value with x's shape whose interval is strictly positive. */
  safeLogArgument(x) {
    if (this.iv.get(x)[0] >= 0.1) return x;
    return this.rng.chance(0.6)
      ? this.track(this.g.add(this.g.mul(x, x), this.scalarConst(0.2, 1)))
      : this.track(this.g.exp(this.boundedForExp(x)));
  }

  /** A partner for a binary op with x: an existing compatible value, or a new leaf. */
  partner(x) {
    const xs = this.g.shape(x);
    const b = this.rng.chance(0.5)
      ? this.choose((s) => { const o = broadcastShapes(xs, s); return o && size(o) <= MAX_SIZE; })
      : undefined;
    if (b !== undefined) return b;
    const drop = this.rng.chance(0.3) ? this.rng.int(0, xs.length) : 0;
    return this.leaf(xs.slice(drop).map((d) => (this.rng.chance(0.3) ? 1 : d)));
  }

  /**
   * Apply `op` to values from the pool (its first operand is `operand` when
   * given). Returns the new id, or undefined if it does not fit.
   */
  apply(op, operand) {
    const { g, rng } = this;
    this.operand = operand;
    switch (op) {
      case 'add': case 'sub': case 'mul': {
        const a = this.first();
        return g[op](a, this.partner(a));
      }
      case 'div': {
        const a = this.first();
        return g.div(a, this.safeDenominator(this.partner(a)));
      }
      case 'log': return g.log(this.safeLogArgument(this.first()));
      case 'exp': return g.exp(this.boundedForExp(this.first()));
      case 'neg': case 'relu': case 'tanh': return g[op](this.first());
      case 'matmul': return this.matmul(this.first());
      case 'sum': {
        const x = this.first((s) => s.length >= 1);
        return x === undefined ? undefined : g.sum(x, rng.int(0, g.shape(x).length - 1), rng.chance(0.4));
      }
      case 'softmax': case 'logSoftmax': {
        const x = this.first((s) => s.length >= 1);
        return x === undefined ? undefined : g[op](x, rng.int(0, g.shape(x).length - 1));
      }
      case 'broadcast': return this.broadcast(this.first());
      case 'reshape': return this.reshape(this.first());
      case 'transpose': {
        const x = this.first((s) => s.length >= 2);
        return x === undefined ? undefined : this.transpose(x);
      }
      default: throw new Error(`generator has no case for '${op}'`);
    }
  }

  /** a @ b for a new or existing b, reshaping a to rank 2 if needed. */
  matmul(a) {
    const { g, rng } = this;
    let as = g.shape(a);
    if (as.length < 2) {
      const n = size(as);
      // Prefer a factorization with both sides > 1: at [1, n] a transpose and a
      // reshape agree, which hides exactly the bugs worth finding.
      const options = factorizations(n, 2);
      const square = options.filter(([m, k]) => m > 1 && k > 1);
      a = this.track(g.reshape(a, rng.pick(square.length && rng.chance(0.8) ? square : options)));
      as = g.shape(a);
    }
    const batch = as.slice(0, -2);
    const k = as.at(-1);
    const n = this.dim();
    if (size([...batch, as.at(-2), n]) > MAX_SIZE || size([...batch, k, n]) > MAX_SIZE) return undefined;
    const fits = (s) => s.length === as.length && s.at(-2) === k && batch.every((d, i) => d === s[i]) &&
      size([...batch, as.at(-2), s.at(-1)]) <= MAX_SIZE;
    const b = this.strict
      ? this.input([...batch, k, n])
      : (rng.chance(0.3) ? this.choose(fits) : undefined) ?? this.leaf([...batch, k, n]);
    return g.matmul(a, b);
  }

  broadcast(x) {
    const { g, rng } = this;
    const xs = g.shape(x);
    const lead = rng.chance(0.5) ? Array.from({ length: rng.int(1, 2) }, () => rng.int(2, 4)) : [];
    const to = [...lead, ...xs.map((d) => (d === 1 && rng.chance(0.7) ? rng.int(2, 5) : d))];
    if (to.length > 4 || size(to) > MAX_SIZE || (to.length === xs.length && to.every((d, i) => d === xs[i]))) return undefined;
    return g.broadcast(x, to);
  }

  reshape(x) {
    const { g, rng } = this;
    const n = size(g.shape(x));
    const all = factorizations(n, rng.int(1, 4)).filter((s) => s.length <= 4);
    // Favour shapes whose last two dims both exceed 1, where a layout mix-up shows.
    const wide = all.filter((s) => s.length >= 2 && s.at(-1) > 1 && s.at(-2) > 1);
    const shape = rng.pick(wide.length && rng.chance(this.strict ? 0.9 : 0.7) ? wide : all);
    const id = g.reshape(x, shape);
    return id === x ? undefined : id;
  }

  transpose(x) {
    const { g, rng } = this;
    const rank = g.shape(x).length;
    let perm;
    for (let t = 0; t < 10; t++) {
      perm = rng.shuffle([...Array(rank).keys()]);
      // Favour permutations that are not their own inverse; at rank 2 there are none.
      if (!perm.every((p, i) => p === i) && (!isInvolution(perm) || rng.chance(this.strict ? 0.1 : 0.3))) break;
    }
    const id = g.transpose(x, perm);
    return id === x ? undefined : id;
  }

  /** A non-uniform weighting of x: x * w for a new input w of x's shape. */
  weigh(x) {
    return this.g.mul(x, this.input(this.g.shape(x)));
  }
}

// ---- targeted programs -------------------------------------------------------

/**
 * Programs built around one op. Each places the op where its gradient is
 * observable: on a path from an input to the output, followed by an
 * elementwise weighting so its cotangent is not uniform.
 */
const TARGETED = {
  matmul(b) {
    const rank = b.rng.chance(0.6) ? 2 : 3;
    const batch = rank === 3 ? [b.rng.int(2, 3)] : [];
    const m = b.rng.int(2, 5), k = b.rng.int(2, 5);
    let x = b.input([...batch, m, k]);
    for (let i = b.rng.int(1, 2); i > 0; i--) {
      if (b.rng.chance(0.5)) x = b.push(b.g.tanh(x));
      const y = b.matmul(x);
      if (y === undefined || !b.ok(y)) break;
      x = b.push(y);
    }
    return x;
  },
  transpose(b) {
    const x0 = b.input(b.shape(3, 4));
    let x = b.push(b.transpose(x0) ?? x0);
    x = b.push(b.weigh(b.g.tanh(x)));
    if (b.rng.chance(0.5)) x = b.push(b.transpose(x) ?? x);
    return x;
  },
  reshape(b) {
    // Retry until a reshape is actually emitted: a factorization equal to the
    // current shape emits nothing, and the program would not contain the op.
    for (let t = 0; t < 10; t++) {
      const x0 = b.input(b.shape(2, 4));
      let x = b.push(b.g.tanh(x0));
      const r = b.reshape(x);
      if (r === undefined) continue;
      x = b.push(r);
      x = b.push(b.weigh(x));
      if (b.rng.chance(0.4)) { const r2 = b.reshape(x); if (r2 !== undefined) x = b.push(r2); }
      return x;
    }
    throw new Error('no reshape in 10 tries');
  },
  broadcast(b) {
    // Likewise: a shape with no size-1 dim and no added leading axes broadcasts to itself.
    for (let t = 0; t < 10; t++) {
      const x0 = b.input(b.shape(1, 3).map((d) => (b.rng.chance(0.5) ? 1 : d)));
      const y = b.broadcast(b.push(b.g.tanh(x0)));
      if (y === undefined) continue;
      return b.push(b.weigh(b.push(y)));
    }
    throw new Error('no broadcast in 10 tries');
  },
  sum(b) {
    const x0 = b.input(b.shape(2, 4));
    const x = b.push(b.g.tanh(x0));
    const axis = b.rng.int(0, b.g.shape(x).length - 1);
    return b.push(b.weigh(b.push(b.g.sum(x, axis, b.rng.chance(0.4)))));
  },
};

/**
 * A random program. `focus` (one of TARGETS) builds it around that op;
 * otherwise it is a general program over every op.
 */
export function randomProgram(seed, { focus } = {}) {
  const rng = new Rng(focus ? (seed * 31 + TARGETS.indexOf(focus) + 1) >>> 0 : seed);
  const b = new Builder(rng);
  let last;

  if (focus) {
    b.strict = true;
    last = TARGETED[focus](b);
    b.strict = false;
    // A few general ops after it, applied to its result, so the op is not
    // always in the same context and stays on the path to an output.
    for (let i = rng.int(0, 2); i > 0; i--) {
      const id = tryOp(b, rng.pick(['add', 'mul', 'tanh', 'exp', 'log', 'div', 'relu', 'sum', 'softmax']), last);
      if (id !== undefined) last = id;
    }
  } else {
    for (let i = rng.int(1, 3); i > 0; i--) b.input(b.shape());
    const ops = Object.entries(GENERATED).flatMap(([op, w]) => Array(w).fill(op));
    const steps = rng.int(2, 10);
    let made = 0;
    for (let attempt = 0; made < steps && attempt < steps * 20; attempt++) {
      if (tryOp(b, rng.pick(ops)) !== undefined) made++;
    }
  }

  // Every computed value nothing else consumes is an output, so every op is on
  // a path to one. (v1 output only the last value, which left earlier ops as
  // dead code more often than not, and a bug in dead code cannot be caught.)
  const consumed = new Set(b.g.nodes.flatMap((n) => n.inputs));
  let outputs = b.g.nodes.filter((n) => n.op !== 'input' && n.op !== 'const' && !consumed.has(n.id)).map((n) => n.id);
  if (outputs.length === 0) outputs = [b.pool.at(-1)];
  return {
    seed, focus: focus ?? null, graph: b.g, outputs, inputs: b.g.inputs().map((n) => n.attrs.name),
    // Each node's interval, for testing that the analysis is sound.
    bounds: b.g.nodes.map((n) => b.iv.get(n.id) ?? interval(b.g, n.id, b.iv)),
  };
}

/** Apply an op, keeping it only if its result is provably within BOUND. */
function tryOp(b, op, operand) {
  const before = b.g.nodes.length;
  const id = b.apply(op, operand);
  if (id === undefined) return undefined;
  if (!b.ok(id)) {
    // Unwind every node this attempt emitted, so no unbounded value is left behind.
    b.g.nodes.length = before;
    for (const k of [...b.iv.keys()]) if (k >= before) b.iv.delete(k);
    b.pool = b.pool.filter((p) => p < before);
    return undefined;
  }
  return b.push(id);
}

/**
 * The fuzzing corpus for program index i: alternately a general program and a
 * targeted one, cycling through TARGETS.
 */
export function corpusProgram(i) {
  if (i % 2 === 1) return randomProgram(i);
  return randomProgram(i, { focus: TARGETS[(i / 2) % TARGETS.length] });
}
