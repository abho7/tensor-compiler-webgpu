// Random well-typed programs over every op, for the gradient fuzzer.
//
// A program is { seed, graph, outputs, inputs }: a graph, the ids of its
// outputs, and the names of its inputs. The same seed always produces the same
// program, so a failure is reproduced by its seed alone.
//
// What it will not generate, and why:
// - a bare `max` or `stopGradient`. Those two only appear inside softmax and
//   log-softmax, the one place their combined gradient is mathematically zero.
//   Anywhere else stopGradient deliberately differs from the true derivative,
//   and a finite-difference check would report that as a bug.
// - tensors over MAX_SIZE elements, so finite differences stay cheap.

import { Graph } from '../ir/graph.js';
import { broadcastShapes, size } from '../ir/shapes.js';
import { Rng } from './rng.js';

const MAX_SIZE = 48;

/** Every op the generator chooses between, with its relative weight. */
export const GENERATED = {
  add: 3, sub: 3, mul: 3, div: 3, neg: 1, exp: 2, log: 2, relu: 3, tanh: 2,
  matmul: 4, sum: 3, broadcast: 2, reshape: 2, transpose: 3, softmax: 2, logSoftmax: 2,
};

function randomShape(rng, maxRank = 3) {
  for (;;) {
    const rank = rng.int(0, maxRank);
    const shape = Array.from({ length: rank }, () => rng.int(1, 4));
    if (size(shape) <= MAX_SIZE) return shape;
  }
}

/** A shape that broadcasts against `shape`: some dims set to 1, maybe some leading ones dropped. */
function compatibleShape(rng, shape) {
  const drop = rng.chance(0.3) ? rng.int(0, shape.length) : 0;
  return shape.slice(drop).map((d) => (rng.chance(0.35) ? 1 : d));
}

function factorizations(n, rank) {
  if (rank === 1) return [[n]];
  const out = [];
  for (let d = 1; d <= n; d++) {
    if (n % d) continue;
    for (const rest of factorizations(n / d, rank - 1)) out.push([d, ...rest]);
  }
  return out;
}

export function randomProgram(seed) {
  const rng = new Rng(seed);
  const g = new Graph();
  const pool = [];
  let leaves = 0;

  const leaf = (shape) => {
    // Inputs are what gradients are taken with respect to; constants are not.
    const id = rng.chance(0.65)
      ? g.input(`x${leaves++}`, shape)
      : g.const(rng.array(size(shape), -2, 2).map(round), shape);
    pool.push(id);
    return id;
  };

  // Recent values are favoured, which makes programs deep rather than wide.
  const choose = (filter = () => true) => {
    const candidates = pool.filter((id) => filter(g.shape(id)));
    if (candidates.length === 0) return undefined;
    const r = rng.next();
    return candidates[candidates.length - 1 - Math.floor(r * r * candidates.length)];
  };

  const nInputs = rng.int(1, 3);
  for (let i = 0; i < nInputs; i++) {
    const id = g.input(`x${leaves++}`, randomShape(rng));
    pool.push(id);
  }

  const ops = Object.entries(GENERATED).flatMap(([op, w]) => Array(w).fill(op));
  const steps = rng.int(2, 10);
  let made = 0;
  for (let attempt = 0; made < steps && attempt < steps * 20; attempt++) {
    const id = apply(rng.pick(ops));
    if (id !== undefined) { pool.push(id); made++; }
  }

  function partner(a) {
    const as = g.shape(a);
    const b = rng.chance(0.5)
      ? choose((s) => { const o = broadcastShapes(as, s); return o && size(o) <= MAX_SIZE; })
      : undefined;
    return b ?? leaf(compatibleShape(rng, as));
  }

  function apply(op) {
    switch (op) {
      case 'add': case 'sub': case 'mul': {
        const a = choose();
        return g[op](a, partner(a));
      }
      case 'div': {
        const a = choose();
        let b = partner(a);
        // Mostly a denominator kept away from zero; sometimes a raw one, whose
        // near-zero draws the checker rejects and counts.
        const r = rng.next();
        if (r < 0.3) b = g.exp(b);
        else if (r < 0.6) b = g.add(g.mul(b, b), g.const(round(rng.uniform(0.2, 1))));
        return g.div(a, b);
      }
      case 'log': {
        let x = choose();
        if (rng.chance(0.7)) x = g.add(g.mul(x, x), g.const(round(rng.uniform(0.2, 1))));
        return g.log(x);
      }
      case 'neg': case 'exp': case 'relu': case 'tanh':
        return g[op](choose());
      case 'matmul': {
        let a = choose();
        let as = g.shape(a);
        if (as.length < 2) {
          const n = size(as);
          a = g.reshape(a, rng.chance(0.5) ? [1, n] : [n, 1]);
          as = g.shape(a);
        }
        const batch = as.slice(0, -2);
        const k = as.at(-1);
        const fits = (s) => s.length === as.length && s.at(-2) === k &&
          batch.every((d, i) => d === s[i]) && size([...batch, as.at(-2), s.at(-1)]) <= MAX_SIZE;
        let b = rng.chance(0.4) ? choose(fits) : undefined;
        if (b === undefined) {
          const n = rng.int(1, 4);
          if (size([...batch, as.at(-2), n]) > MAX_SIZE || size([...batch, k, n]) > MAX_SIZE) return undefined;
          b = leaf([...batch, k, n]);
        }
        return g.matmul(a, b);
      }
      case 'sum': {
        const x = choose((s) => s.length >= 1);
        if (x === undefined) return undefined;
        return g.sum(x, rng.int(0, g.shape(x).length - 1), rng.chance(0.4));
      }
      case 'softmax': case 'logSoftmax': {
        const x = choose((s) => s.length >= 1);
        if (x === undefined) return undefined;
        return g[op](x, rng.int(0, g.shape(x).length - 1));
      }
      case 'broadcast': {
        const x = choose();
        const xs = g.shape(x);
        const lead = rng.chance(0.5) ? Array.from({ length: rng.int(1, 2) }, () => rng.int(2, 3)) : [];
        const to = [...lead, ...xs.map((d) => (d === 1 && rng.chance(0.7) ? rng.int(2, 3) : d))];
        if (to.length > 4 || size(to) > MAX_SIZE || to.length === xs.length && to.every((d, i) => d === xs[i])) return undefined;
        return g.broadcast(x, to);
      }
      case 'reshape': {
        const x = choose();
        const n = size(g.shape(x));
        const shape = rng.pick(factorizations(n, rng.int(1, 3)));
        const id = g.reshape(x, shape);
        return id === x ? undefined : id;
      }
      case 'transpose': {
        const x = choose((s) => s.length >= 2);
        if (x === undefined) return undefined;
        const perm = rng.shuffle([...g.shape(x).keys()]);
        const id = g.transpose(x, perm);
        return id === x ? undefined : id;
      }
      default:
        throw new Error(`generator has no case for '${op}'`);
    }
  }

  const last = pool.at(-1);
  const outputs = [last];
  if (rng.chance(0.4)) {
    const other = choose((s) => true);
    if (other !== undefined && other !== last) outputs.push(other);
  }
  return { seed, graph: g, outputs, inputs: g.inputs().map((n) => n.attrs.name) };
}

/** Constants are rounded so they print exactly and survive JSON unchanged. */
function round(v) { return Math.round(v * 1024) / 1024; }
