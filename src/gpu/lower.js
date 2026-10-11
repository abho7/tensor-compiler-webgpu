// Lowering: a graph to a list of kernel launches, one per op (no fusion).
//
// Each launch names its kernel, the node ids it reads and writes, its u32
// parameters and its workgroup grid. Inputs and constants are not launches;
// the runtime uploads them. Pure JavaScript: nothing here touches a GPU.

import { size } from '../ir/shapes.js';
import { strides } from '../interp.js';
import { kernelSource, grid1d, ELEMENTWISE_OPS, MAX_RANK, MM_WG, TILE } from './codegen.js';

/** Sabotages that live in the lowering (the parameters), not in kernel text. */
export const LOWERING_SABOTAGE = ['transpose-swapped-stride', 'broadcast-wrong-stride', 'reshape-stride-mismatch', 'softmax-no-max'];

/** True for the x - stopGradient(max(x)) shift a softmax or log-softmax builds. */
export function isSoftmaxShift(graph, node) {
  if (node.op !== 'sub') return false;
  let m = graph.nodes[node.inputs[1]];
  while (m.op === 'broadcast') m = graph.nodes[m.inputs[0]];
  return m.op === 'stopGradient';
}

function gatherParams(outShape, srcStrides) {
  if (outShape.length > MAX_RANK) throw new Error(`rank ${outShape.length} exceeds ${MAX_RANK}`);
  const p = new Uint32Array(2 + 2 * MAX_RANK);
  p[0] = size(outShape);
  p[1] = outShape.length;
  outShape.forEach((d, i) => { p[2 + i] = d; p[2 + MAX_RANK + i] = srcStrides[i]; });
  return p;
}

function gatherStrides(node, inShape, sabotage) {
  const out = node.shape;
  switch (node.op) {
    case 'broadcast': {
      const lead = out.length - inShape.length;
      const natural = strides(inShape);
      return out.map((d, i) => {
        if (i < lead) return 0;
        const s = inShape[i - lead];
        if (s !== 1) return natural[i - lead];
        // Correct: a stretched axis reads the same element, stride 0.
        return sabotage === 'broadcast-wrong-stride' && d !== 1 ? natural[i - lead] : 0;
      });
    }
    case 'transpose': {
      const natural = strides(inShape);
      const st = node.attrs.perm.map((p) => natural[p]);
      if (sabotage === 'transpose-swapped-stride' && st.length >= 2) {
        [st[st.length - 1], st[st.length - 2]] = [st[st.length - 2], st[st.length - 1]];
      }
      return st;
    }
    case 'reshape': {
      if (sabotage !== 'reshape-stride-mismatch') return strides(out);
      // The bug: the source's strides, right-aligned to the output's axes.
      const natural = strides(inShape);
      const lead = out.length - inShape.length;
      return out.map((_, i) => (i - lead >= 0 ? natural[i - lead] : size(inShape)));
    }
    default: throw new Error(`not a gather op: ${node.op}`);
  }
}

/**
 * Lower `graph`. Options: matmul 'naive' (default) or 'tiled'; sabotage, one
 * of the kernel or lowering sabotage names.
 */
export function lower(graph, { matmul = 'naive', sabotage } = {}) {
  const launches = [];
  for (const node of graph.nodes) {
    if (node.op === 'input' || node.op === 'const') continue;
    const inShape = node.inputs.length ? graph.nodes[node.inputs[0]].shape : [];
    let kind, params, grid;
    let inputs = [...node.inputs];
    if (sabotage === 'softmax-no-max' && isSoftmaxShift(graph, node)) {
      // The bug: the shift is never subtracted; the "shifted" value is x itself.
      kind = 'copy';
      inputs = [node.inputs[0]];
      const n = size(node.shape);
      params = Uint32Array.of(n);
      grid = grid1d(n);
    } else if (node.op === 'stopGradient' || ELEMENTWISE_OPS.includes(node.op)) {
      kind = node.op === 'stopGradient' ? 'copy' : node.op;
      const n = size(node.shape);
      params = Uint32Array.of(n);
      grid = grid1d(n);
    } else if (node.op === 'broadcast' || node.op === 'transpose' || node.op === 'reshape') {
      kind = 'gather';
      params = gatherParams(node.shape, gatherStrides(node, inShape, sabotage));
      grid = grid1d(size(node.shape));
    } else if (node.op === 'sum' || node.op === 'max') {
      kind = node.op;
      const axis = node.attrs.axis;
      const outer = size(inShape.slice(0, axis)), n = inShape[axis], inner = size(inShape.slice(axis + 1));
      params = Uint32Array.of(outer, n, inner);
      grid = grid1d(outer * inner);
    } else if (node.op === 'matmul') {
      kind = matmul === 'tiled' ? 'matmul-tiled' : 'matmul-naive';
      const as = inShape, bs = graph.nodes[node.inputs[1]].shape;
      const batch = size(as.slice(0, -2)), M = as.at(-2), K = as.at(-1), N = bs.at(-1);
      params = Uint32Array.of(batch, M, K, N);
      const t = kind === 'matmul-tiled' ? TILE : MM_WG;
      grid = [Math.ceil(N / t), Math.ceil(M / t), batch];
    } else {
      throw new Error(`no lowering for '${node.op}'`);
    }
    const kernelSabotage = LOWERING_SABOTAGE.includes(sabotage) ? undefined : sabotage;
    launches.push({
      node: node.id, op: node.op, kind, inputs, params, grid,
      code: kernelSource(kind, { sabotage: kernelSabotage }),
    });
  }
  return { launches };
}
