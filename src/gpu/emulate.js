// A JavaScript model of what each kernel's threads do, driven by the same
// launch parameters and grid the GPU gets. It runs every thread of the grid,
// including the ones past the end that the bounds checks must stop, and
// records writes: so a test can check, without a GPU, that every output
// element is written exactly once, that nothing is written out of bounds, and
// that the lowering's strides and extents give the interpreter's answer.
//
// It models the kernels' index logic as codegen.js writes it; it does not run
// WGSL. The GPU itself is the final check.

import { WG, MM_WG, TILE, MAX_RANK } from './codegen.js';
import { size } from '../ir/shapes.js';
import { KERNELS } from '../interp.js';

const UNARY = {
  neg: (x) => -x, exp: Math.exp, log: Math.log, relu: (x) => (x > 0 ? x : 0),
  tanh: Math.tanh, step: (x) => (x > 0 ? 1 : 0), copy: (x) => x,
};
const BINARY = { add: (a, b) => a + b, sub: (a, b) => a - b, mul: (a, b) => a * b, div: (a, b) => a / b };

function threads1d(grid, body) {
  const [gx, gy] = grid;
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx * WG; x++) body(x + y * gx * WG);
}

/**
 * Run one launch. `read(id)` gives an input's values; returns
 * { out, writes, oob }: the output, how many times each element was written,
 * and the number of writes past the end.
 */
export function emulateLaunch(launch, outLength, read, { guard = true } = {}) {
  const out = new Float64Array(outLength);
  const writes = new Uint32Array(outLength);
  let oob = 0;
  const write = (i, v) => {
    if (i < 0 || i >= outLength) { oob++; return; }
    out[i] = v; writes[i]++;
  };
  const [A, B] = launch.inputs.map(read);
  const P = launch.params;
  const k = launch.kind;
  if (UNARY[k] || BINARY[k]) {
    threads1d(launch.grid, (i) => {
      if (guard && i >= P[0]) return;
      write(i, UNARY[k] ? UNARY[k](A[i]) : BINARY[k](A[i], B[i]));
    });
  } else if (k === 'gather') {
    threads1d(launch.grid, (i) => {
      if (i >= P[0]) return;
      let rem = i, src = 0;
      for (let d = P[1] - 1; d >= 0; d--) { src += (rem % P[2 + d]) * P[2 + MAX_RANK + d]; rem = Math.floor(rem / P[2 + d]); }
      write(i, A[src]);
    });
  } else if (k === 'sum' || k === 'max') {
    const [outer, n, inner] = P;
    threads1d(launch.grid, (i) => {
      if (i >= outer * inner) return;
      const base = Math.floor(i / inner) * n * inner + (i % inner);
      let acc = k === 'sum' ? 0 : A[base];
      for (let t = k === 'sum' ? 0 : 1; t < n; t++) acc = k === 'sum' ? acc + A[base + t * inner] : Math.max(acc, A[base + t * inner]);
      write(i, acc);
    });
  } else if (k === 'matmul-naive' || k === 'matmul-tiled') {
    const [, M, K, N] = P;
    const t = k === 'matmul-tiled' ? TILE : MM_WG;
    const [gx, gy, gz] = launch.grid;
    for (let b = 0; b < gz; b++) for (let row = 0; row < gy * t; row++) for (let col = 0; col < gx * t; col++) {
      if (row >= M || col >= N) continue;
      let acc = 0;
      if (k === 'matmul-naive') {
        for (let p = 0; p < K; p++) acc += A[b * M * K + row * K + p] * B[b * K * N + p * N + col];
      } else {
        // Tile by tile, loads past the edge reading zero, as the kernel does.
        for (let t0 = 0; t0 * TILE < K; t0++) {
          for (let p = 0; p < TILE; p++) {
            const kk = t0 * TILE + p;
            const av = kk < K ? A[b * M * K + row * K + kk] : 0;
            const bv = kk < K ? B[b * K * N + kk * N + col] : 0;
            acc += av * bv;
          }
        }
      }
      write(b * M * N + row * N + col, acc);
    }
  } else {
    throw new Error(`no emulation for '${k}'`);
  }
  return { out, writes, oob };
}

/** Emulate a whole lowered graph; inputs and constants come from the interpreter. */
export function emulate(graph, lowered, inputs, options) {
  const values = new Array(graph.nodes.length);
  const reports = [];
  for (const node of graph.nodes) {
    if (node.op === 'input') values[node.id] = Float64Array.from(inputs[node.attrs.name]);
    if (node.op === 'const') values[node.id] = KERNELS.const(node);
  }
  for (const launch of lowered.launches) {
    const r = emulateLaunch(launch, size(graph.nodes[launch.node].shape), (id) => values[id], options);
    values[launch.node] = r.out;
    reports.push({ node: launch.node, kind: launch.kind, writes: r.writes, oob: r.oob });
  }
  return { values, reports };
}
