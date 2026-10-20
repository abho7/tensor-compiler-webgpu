// Lowering a fused graph: a fused node to its generated kernel, every other
// node to exactly the launch milestone 2's lowering gives it. Plain nodes are
// lowered one at a time, as one-op graphs through lower(), so this file does
// not touch milestone 2's lowering (amendment 29). Pure JavaScript.

import { Graph } from '../ir/graph.js';
import { lower } from '../gpu/lower.js';
import { fusedKernel } from './codegen-fused.js';

const plainCache = new Map();

/** The launch milestone 2's lowering gives `node` alone, with its ids put back. */
function plainLaunch(graph, node, matmul) {
  const shapes = node.inputs.map((id) => graph.nodes[id].shape);
  const key = JSON.stringify([node.op, shapes, node.attrs, matmul]);
  let one = plainCache.get(key);
  if (!one) {
    const g = new Graph();
    const ins = shapes.map((s, k) => g.input(`a${k}`, s));
    g.emit(node.op, ins, node.attrs);
    one = lower(g, { matmul }).launches[0];
    plainCache.set(key, one);
  }
  return { ...one, node: node.id, inputs: [...node.inputs] };
}

/**
 * Lower `graph` (which may contain fused nodes). Options: matmul 'naive' or
 * 'tiled' (for plain matmuls and epilogues alike); sabotage, one of
 * FUSED_SABOTAGE, applied to fused kernels only.
 */
export function lowerFused(graph, { matmul = 'naive', sabotage } = {}) {
  const launches = [];
  for (const node of graph.nodes) {
    if (node.op === 'input' || node.op === 'const') continue;
    if (node.op === 'fused') {
      const k = fusedKernel(node, { matmul, sabotage });
      launches.push({ node: node.id, op: 'fused', kind: `fused-${node.attrs.kind}`, pattern: node.attrs.pattern,
        inputs: [...node.inputs], params: k.params, grid: k.grid, code: k.code });
    } else {
      launches.push(plainLaunch(graph, node, matmul));
    }
  }
  return { launches };
}
