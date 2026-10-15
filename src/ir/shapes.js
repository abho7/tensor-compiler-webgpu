// Shape rules for every op. A node's shape is computed once, when it is
// created, so a graph that exists is a graph whose shapes are consistent.

export class ShapeError extends Error {
  constructor(message) { super(message); this.name = 'ShapeError'; }
}

export const size = (shape) => shape.reduce((n, d) => n * d, 1);
export const sameShape = (a, b) => a.length === b.length && a.every((d, i) => d === b[i]);
export const fmt = (shape) => `[${shape.join(',')}]`;

function checkShape(shape, what) {
  if (!Array.isArray(shape) || !shape.every((d) => Number.isInteger(d) && d >= 1)) {
    throw new ShapeError(`${what}: shape must be a list of positive integers, got ${JSON.stringify(shape)}`);
  }
}

/** Resolve a possibly negative axis against a rank. */
export function normAxis(axis, rank, op) {
  const a = axis < 0 ? axis + rank : axis;
  if (!Number.isInteger(a) || a < 0 || a >= rank) {
    throw new ShapeError(`${op}: axis ${axis} out of range for rank ${rank}`);
  }
  return a;
}

/** The shape `a` and `b` broadcast to under NumPy's rules, or null if they do not. */
export function broadcastShapes(a, b) {
  const rank = Math.max(a.length, b.length);
  const out = [];
  for (let i = 0; i < rank; i++) {
    const da = a[a.length - rank + i] ?? 1;
    const db = b[b.length - rank + i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) return null;
    out.push(Math.max(da, db));
  }
  return out;
}

/** True if `from` broadcasts to exactly `to`. */
export function broadcastsTo(from, to) {
  if (from.length > to.length) return false;
  const lead = to.length - from.length;
  return from.every((d, i) => d === 1 || d === to[lead + i]);
}

const UNARY = new Set(['neg', 'exp', 'log', 'relu', 'tanh', 'step', 'stopGradient', 'sqrt']);
const BINARY = new Set(['add', 'sub', 'mul', 'div']);
const REDUCE = new Set(['sum', 'max']);

export const OPS = ['input', 'const', ...BINARY, ...UNARY, 'matmul', ...REDUCE,
  'broadcast', 'reshape', 'transpose'];

/** The result shape of `op` applied to inputs of `shapes`, or a ShapeError. */
export function inferShape(op, shapes, attrs) {
  const arity = { input: 0, const: 0, matmul: 2 };
  const want = arity[op] ?? (BINARY.has(op) ? 2 : 1);
  if (!OPS.includes(op)) throw new ShapeError(`unknown op '${op}'`);
  if (shapes.length !== want) throw new ShapeError(`${op}: takes ${want} inputs, got ${shapes.length}`);

  if (op === 'input' || op === 'const') {
    checkShape(attrs.shape, op);
    if (op === 'const') {
      const v = attrs.value;
      const ok = typeof v === 'number' || (Array.isArray(v) && v.length === size(attrs.shape));
      if (!ok) throw new ShapeError(`const: value must be a number or ${size(attrs.shape)} numbers`);
    }
    return [...attrs.shape];
  }
  if (BINARY.has(op)) {
    if (!sameShape(shapes[0], shapes[1])) {
      throw new ShapeError(`${op}: shapes ${fmt(shapes[0])} and ${fmt(shapes[1])} differ; broadcast explicitly`);
    }
    return [...shapes[0]];
  }
  if (UNARY.has(op)) return [...shapes[0]];
  if (op === 'matmul') {
    const [a, b] = shapes;
    if (a.length < 2 || b.length < 2) throw new ShapeError(`matmul: needs rank >= 2, got ${fmt(a)} and ${fmt(b)}`);
    if (a.length !== b.length || !sameShape(a.slice(0, -2), b.slice(0, -2))) {
      throw new ShapeError(`matmul: batch dims of ${fmt(a)} and ${fmt(b)} differ`);
    }
    if (a.at(-1) !== b.at(-2)) throw new ShapeError(`matmul: inner dims of ${fmt(a)} and ${fmt(b)} differ`);
    return [...a.slice(0, -1), b.at(-1)];
  }
  if (REDUCE.has(op)) {
    const x = shapes[0];
    const axis = normAxis(attrs.axis, x.length, op);
    if (attrs.axis !== axis) throw new ShapeError(`${op}: axis must be stored normalised`);
    return attrs.keepDims ? x.map((d, i) => (i === axis ? 1 : d)) : x.filter((_, i) => i !== axis);
  }
  if (op === 'broadcast') {
    checkShape(attrs.shape, op);
    if (!broadcastsTo(shapes[0], attrs.shape)) {
      throw new ShapeError(`broadcast: ${fmt(shapes[0])} does not broadcast to ${fmt(attrs.shape)}`);
    }
    return [...attrs.shape];
  }
  if (op === 'reshape') {
    checkShape(attrs.shape, op);
    if (size(attrs.shape) !== size(shapes[0])) {
      throw new ShapeError(`reshape: ${fmt(shapes[0])} has ${size(shapes[0])} elements, ${fmt(attrs.shape)} has ${size(attrs.shape)}`);
    }
    return [...attrs.shape];
  }
  if (op === 'transpose') {
    const x = shapes[0];
    const p = attrs.perm;
    const isPerm = Array.isArray(p) && p.length === x.length &&
      [...p].sort((u, v) => u - v).every((v, i) => v === i);
    if (!isPerm) throw new ShapeError(`transpose: ${JSON.stringify(p)} is not a permutation of rank ${x.length}`);
    return p.map((i) => x[i]);
  }
  throw new ShapeError(`no shape rule for '${op}'`);
}
