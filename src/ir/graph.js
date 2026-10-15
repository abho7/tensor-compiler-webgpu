// The tensor IR: a list of nodes in SSA form.
//
// A node names its inputs by id, and ids are positions in the list, so the
// list is always in topological order and a node can only use values defined
// before it. Every node's shape is inferred when it is created.
//
// Two layers of construction:
//   emit(op, inputs, attrs)  the raw op, with the IR's strict rules (binary ops
//                            need identical shapes)
//   add(), mul(), ...        the builder: NumPy-style broadcasting, which it
//                            makes explicit by inserting `broadcast` nodes

import { inferShape, broadcastShapes, normAxis, sameShape, size, fmt, ShapeError } from './shapes.js';

export class Graph {
  constructor() {
    this.nodes = [];
  }

  /** Append a node and return its id. */
  emit(op, inputs = [], attrs = {}) {
    for (const id of inputs) {
      if (!Number.isInteger(id) || id < 0 || id >= this.nodes.length) {
        throw new ShapeError(`${op}: input ${id} is not a value of this graph`);
      }
    }
    const shape = inferShape(op, inputs.map((id) => this.nodes[id].shape), attrs);
    const id = this.nodes.length;
    this.nodes.push({ id, op, inputs: [...inputs], attrs: structuredClone(attrs), shape });
    return id;
  }

  shape(id) { return this.nodes[id].shape; }

  /** The id of the input named `name`, or undefined. */
  inputId(name) {
    return this.nodes.find((n) => n.op === 'input' && n.attrs.name === name)?.id;
  }

  inputs() { return this.nodes.filter((n) => n.op === 'input'); }

  clone() {
    const g = new Graph();
    g.nodes = structuredClone(this.nodes);
    return g;
  }

  toJSON() {
    return { nodes: this.nodes.map(({ op, inputs, attrs, shape }) => ({ op, inputs, attrs, shape })) };
  }

  /** Rebuild a graph from toJSON() output, re-checking every shape. */
  static fromJSON(json) {
    const g = new Graph();
    for (const n of json.nodes) {
      const id = g.emit(n.op, n.inputs, n.attrs);
      if (!sameShape(g.shape(id), n.shape)) {
        throw new ShapeError(`node ${id}: stored shape ${fmt(n.shape)} but inferred ${fmt(g.shape(id))}`);
      }
    }
    return g;
  }

  // ---- builder -----------------------------------------------------------

  input(name, shape) {
    if (this.inputId(name) !== undefined) throw new Error(`input '${name}' already exists`);
    return this.emit('input', [], { name, shape });
  }

  const(value, shape = []) { return this.emit('const', [], { shape, value }); }

  /** `x` broadcast to `shape`; `x` itself if it already has that shape. */
  broadcast(x, shape) {
    return sameShape(this.shape(x), shape) ? x : this.emit('broadcast', [x], { shape: [...shape] });
  }

  // Node ids are numbers, so a bare number here is always an id, never a
  // literal: scalars go through const(). (Accepting literals once turned every
  // add(x, y) into an add of two constants equal to the ids.)
  #binary(op, a, b) {
    const shape = broadcastShapes(this.shape(a), this.shape(b));
    if (!shape) throw new ShapeError(`${op}: ${fmt(this.shape(a))} and ${fmt(this.shape(b))} do not broadcast`);
    return this.emit(op, [this.broadcast(a, shape), this.broadcast(b, shape)]);
  }

  add(a, b) { return this.#binary('add', a, b); }
  sub(a, b) { return this.#binary('sub', a, b); }
  mul(a, b) { return this.#binary('mul', a, b); }
  div(a, b) { return this.#binary('div', a, b); }

  neg(x) { return this.emit('neg', [x]); }
  exp(x) { return this.emit('exp', [x]); }
  log(x) { return this.emit('log', [x]); }
  relu(x) { return this.emit('relu', [x]); }
  tanh(x) { return this.emit('tanh', [x]); }
  sqrt(x) { return this.emit('sqrt', [x]); }
  step(x) { return this.emit('step', [x]); }
  stopGradient(x) { return this.emit('stopGradient', [x]); }

  matmul(a, b) { return this.emit('matmul', [a, b]); }

  sum(x, axis, keepDims = false) {
    return this.emit('sum', [x], { axis: normAxis(axis, this.shape(x).length, 'sum'), keepDims });
  }

  max(x, axis, keepDims = false) {
    return this.emit('max', [x], { axis: normAxis(axis, this.shape(x).length, 'max'), keepDims });
  }

  reshape(x, shape) {
    return sameShape(this.shape(x), shape) ? x : this.emit('reshape', [x], { shape: [...shape] });
  }

  /** Permute axes; with no `perm`, reverse them. */
  transpose(x, perm) {
    const rank = this.shape(x).length;
    perm ??= [...Array(rank).keys()].reverse();
    return perm.every((p, i) => p === i) ? x : this.emit('transpose', [x], { perm: [...perm] });
  }

  /** Swap the last two axes. */
  transposeLast2(x) {
    const rank = this.shape(x).length;
    const perm = [...Array(rank).keys()];
    [perm[rank - 2], perm[rank - 1]] = [perm[rank - 1], perm[rank - 2]];
    return this.transpose(x, perm);
  }

  // ---- composites, built from the ops above ------------------------------

  /** x minus its (gradient-free) maximum along `axis`. Softmax is invariant to the shift. */
  #shifted(x, axis) {
    const m = this.stopGradient(this.max(x, axis, true));
    return this.sub(x, m);
  }

  softmax(x, axis = -1) {
    const z = this.exp(this.#shifted(x, axis));
    return this.div(z, this.sum(z, axis, true));
  }

  logSoftmax(x, axis = -1) {
    const z = this.#shifted(x, axis);
    return this.sub(z, this.log(this.sum(this.exp(z), axis, true)));
  }
}

export { size };
