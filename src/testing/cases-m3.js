// Milestone 3a's PyTorch golden cases for the new op, sqrt (DESIGN.md
// amendment 13): generated programs containing it, and hand-built graphs
// shaped like Adam's update. In their own file so the milestone 1 goldens stay
// byte-identical. Deterministic; CI rebuilds them and compares.

import { Graph } from '../ir/graph.js';
import { size } from '../ir/shapes.js';
import { evaluate } from '../interp.js';
import { randomProgram } from './random-graph.js';
import { Rng } from './rng.js';

export const SQRT_RANDOM_CASES = 30;

function usable(graph, inputs) {
  let ok = true;
  evaluate(graph, inputs, {
    onNode(node, args, value) {
      for (const v of value) if (!Number.isFinite(v) || Math.abs(v) > 1e4) ok = false;
      if (node.op === 'log' || node.op === 'sqrt') for (const v of args[0]) if (v < 1e-3) ok = false;
      if (node.op === 'div') for (const v of args[1]) if (Math.abs(v) < 1e-3) ok = false;
      if (node.op === 'relu' || node.op === 'step') for (const v of args[0]) if (Math.abs(v) < 1e-3) ok = false;
    },
  });
  return ok;
}

function draw(rng, graph, outputs, ranges = {}) {
  const inputs = {};
  for (const n of graph.inputs()) {
    const [lo, hi] = ranges[n.attrs.name] ?? [-2, 2];
    inputs[n.attrs.name] = rng.array(size(n.shape), lo, hi);
  }
  const cotangents = outputs.map((id) => rng.array(size(graph.nodes[id].shape), -1, 1));
  return { inputs, cotangents };
}

const pack = (name, kind, graph, outputs, point) =>
  ({ name, kind, set: 'm3', graph: graph.toJSON(), outputs, ...point });

export function buildSqrtCases() {
  const cases = [];
  for (let seed = 1; cases.length < SQRT_RANDOM_CASES; seed++) {
    const p = randomProgram(seed, { withSqrt: true });
    if (!p.graph.nodes.some((n) => n.op === 'sqrt')) continue;
    const rng = new Rng(seed * 13 + 5);
    for (let t = 0; t < 20; t++) {
      const point = draw(rng, p.graph, p.outputs);
      if (usable(p.graph, point.inputs)) { cases.push(pack(`sqrt-random-${seed}`, 'random', p.graph, p.outputs, point)); break; }
    }
  }

  const rng = new Rng(20261014);
  const hand = [
    ['sqrt-plain', (g) => g.sqrt(g.input('x', [3, 5])), { x: [0.01, 9] }],
    ['sqrt-nested', (g) => g.sqrt(g.sqrt(g.input('x', [7]))), { x: [0.05, 50] }],
    ['sqrt-of-sum', (g) => g.sqrt(g.sum(g.input('x', [4, 6]), 1)), { x: [0.1, 3] }],
    // Adam's update: m / (sqrt(v) + eps), with v >= 0 and m of either sign.
    ['adam-update', (g) => g.div(g.input('m', [4, 6]), g.add(g.sqrt(g.input('v', [4, 6])), g.const(1e-8))), { m: [-1, 1], v: [1e-6, 1] }],
  ];
  for (const [name, build, ranges] of hand) {
    const g = new Graph();
    const out = build(g);
    cases.push(pack(name, 'hand', g, [out], draw(rng, g, [out], ranges)));
  }
  return cases;
}
