// WGSL for fused nodes (DESIGN.md amendment 30), generated as text in plain
// JavaScript so it can be tested without a GPU.
//
// A kernel is generated from the fused node's body. Every body value is a
// `let` computed once per index it is needed at; scalar constants are inlined
// as their exact float32 bits; sizes and strides come from the parameter block
// P at run time, as in the unfused kernels, so one kernel text serves every
// shape of the same structure.
//
//   elementwise  one thread per element (64 per workgroup, as the unfused kernels)
//   epilogue     milestone 2's naive or tiled matmul, then the body on the sum
//   rowwise      one 64-thread workgroup per row; each reduction is a strided
//                per-thread partial and a tree through workgroup memory
//
// `sabotage` names one of amendment 31's kernel bugs; the default is correct.

import { size } from '../ir/shapes.js';
import { strides } from '../interp.js';
import { WG, MM_WG, TILE, MAX_RANK, grid1d } from '../gpu/codegen.js';
import { bodyOf } from './fuse.js';

export const FUSED_SABOTAGE = ['epilogue-before-bias', 'layernorm-tree-short', 'softmax-no-max', 'fused-bounds-off-by-one'];
export const ROW_WG = 64;
const NEG_INF_BITS = 0xff800000;

const f32bits = (v) => { const b = new Uint32Array(new Float32Array([v]).buffer)[0]; return b; };
export const literal = (v) => `bitcast<f32>(0x${f32bits(v).toString(16).padStart(8, '0')}u)`;

const UNARY = {
  neg: (x) => `-${x}`,
  exp: (x) => `exp(${x})`,
  log: (x) => `log(${x})`,
  relu: (x) => `max(${x}, 0.0)`,
  tanh: (x) => `tanh(clamp(${x}, -20.0, 20.0))`, // as the unfused kernel (DESIGN.md revision R2)
  sqrt: (x) => `sqrt(${x})`,
  step: (x) => `select(0.0, 1.0, ${x} > 0.0)`,
  stopGradient: (x) => x,
};
const BINARY = { add: '+', sub: '-', mul: '*', div: '/' };

const bindings = (k) => [
  ...Array.from({ length: k }, (_, i) => `@group(0) @binding(${i}) var<storage, read> A${i}: array<f32>;`),
  `@group(0) @binding(${k}) var<storage, read_write> OUT: array<f32>;`,
  `@group(0) @binding(${k + 1}) var<storage, read> P: array<u32>;`,
].join('\n');

const GIDX = `fn gidx(i: u32, base: u32) -> u32 {
  let rank = P[base];
  var rem = i;
  var src = 0u;
  for (var k = 0u; k < rank; k = k + 1u) {
    let d = rank - 1u - k;
    let extent = P[base + 1u + d];
    src = src + (rem % extent) * P[base + ${1 + MAX_RANK}u + d];
    rem = rem / extent;
  }
  return src;
}`;

/** Source strides that make a gather over `node`'s output read its input (as lower.js computes them). */
function gatherStrides(node, inShape) {
  const out = node.shape;
  if (node.op === 'broadcast') {
    const lead = out.length - inShape.length;
    const natural = strides(inShape);
    return out.map((_, i) => (i < lead || inShape[i - lead] === 1 ? 0 : natural[i - lead]));
  }
  if (node.op === 'transpose') {
    const natural = strides(inShape);
    return node.attrs.perm.map((p) => natural[p]);
  }
  throw new Error(`not a strided move: ${node.op}`);
}

/**
 * The expression builder shared by the three kinds. `index(ctx)` gives the WGSL
 * flat index for a context; a context is an index variable, and every value is
 * memoised per context so it is computed once there.
 */
class Emitter {
  constructor(b, params, opts) {
    this.b = b;              // body graph
    this.params = params;    // u32 parameter list (appended to for gathers)
    this.opts = opts;
    this.memo = new Map();
    this.counter = 0;
    this.usesGidx = false;
  }

  gatherBase(node) {
    const inShape = this.b.nodes[node.inputs[0]].shape;
    if (node.shape.length > MAX_RANK) throw new Error(`rank ${node.shape.length} exceeds ${MAX_RANK}`);
    const base = this.params.length;
    const st = gatherStrides(node, inShape);
    const block = new Array(1 + 2 * MAX_RANK).fill(0);
    block[0] = node.shape.length;
    node.shape.forEach((d, i) => { block[1 + i] = d; block[1 + MAX_RANK + i] = st[i]; });
    this.params.push(...block);
    this.usesGidx = true;
    return base;
  }

  /** The WGSL value of body node `id` in context `ctx`, appending `let`s to `lines`. */
  value(id, ctx, lines) {
    const key = `${ctx.name}|${id}`;
    if (this.memo.has(key)) return this.memo.get(key);
    const special = this.opts.special?.(id, ctx, lines);
    if (special !== undefined) { this.memo.set(key, special); return special; }
    const node = this.b.nodes[id];
    let v;
    if (node.op === 'input') {
      v = `A${Number(node.attrs.name.slice(1))}[${ctx.index}]`;
    } else if (node.op === 'const') {
      v = literal(typeof node.attrs.value === 'number' ? node.attrs.value : node.attrs.value[0]);
      this.memo.set(key, v);
      return v;
    } else if (node.op === 'reshape') {
      v = this.value(node.inputs[0], ctx, lines);
      this.memo.set(key, v);
      return v;
    } else if (node.op === 'broadcast' || node.op === 'transpose') {
      const src = this.b.nodes[node.inputs[0]];
      if (src.op === 'const') {
        v = this.value(src.id, ctx, lines);
        this.memo.set(key, v);
        return v;
      }
      const base = this.gatherBase(node);
      const j = `j${this.counter++}`;
      lines.push(`  let ${j} = gidx(${ctx.index}, ${base}u);`);
      // The source is read, or (recompute) re-evaluated, at the gathered index.
      v = this.value(src.id, { name: j, index: j }, lines);
      this.memo.set(key, v);
      return v;
    } else if (node.op in BINARY) {
      const [a, c] = node.inputs.map((i) => this.value(i, ctx, lines));
      v = `${a} ${BINARY[node.op]} ${c}`;
    } else if (node.op in UNARY) {
      v = UNARY[node.op](this.value(node.inputs[0], ctx, lines));
    } else {
      throw new Error(`fused codegen: no rule for '${node.op}'`);
    }
    const name = `v${this.counter++}`;
    lines.push(`  let ${name}: f32 = ${v};`);
    this.memo.set(key, name);
    return name;
  }
}

const operandCount = (b) => b.nodes.filter((n) => n.op === 'input').length;

// ---- elementwise --------------------------------------------------------------

function elementwise(node, { sabotage }) {
  const b = bodyOf(node);
  const params = [size(node.shape)];
  const em = new Emitter(b, params, {});
  const lines = [];
  const out = em.value(node.attrs.output, { name: 'i', index: 'i' }, lines);
  const guard = sabotage === 'fused-bounds-off-by-one' ? 'i > P[0]' : 'i >= P[0]';
  const code = `${bindings(operandCount(b))}
${em.usesGidx ? `\n${GIDX}\n` : ''}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * ${WG}u;
  if (${guard}) { return; }
${lines.join('\n')}
  OUT[i] = ${out};
}
`;
  return { code, params: Uint32Array.from(params), grid: grid1d(size(node.shape)) };
}

// ---- matmul epilogue ------------------------------------------------------------

function epilogue(node, { matmul, sabotage }) {
  const b = bodyOf(node);
  const root = b.nodes.find((n) => n.op === 'matmul');
  const [ia, ib] = root.inputs.map((i) => Number(b.nodes[i].attrs.name.slice(1)));
  const as = b.nodes[root.inputs[0]].shape, bs = b.nodes[root.inputs[1]].shape;
  const batch = size(as.slice(0, -2)), M = as.at(-2), K = as.at(-1), N = bs.at(-1);
  const params = [batch, M, K, N];
  // Sabotage 1: the activation applied to the bare sum, the bias added after it.
  let biasAdd = null;
  if (sabotage === 'epilogue-before-bias') {
    const add = b.nodes.find((n) => n.op === 'add' && n.inputs.includes(root.id) && !n.inputs.every((i) => i === root.id));
    if (add && add.id !== node.attrs.output) biasAdd = add;
  }
  const em = new Emitter(b, params, {
    special: (id) => (id === root.id || id === biasAdd?.id ? 'acc' : undefined),
  });
  const lines = [];
  const ctx = { name: 'i', index: 'i' };
  let out = em.value(node.attrs.output, ctx, lines);
  if (biasAdd) {
    const other = biasAdd.inputs.find((i) => i !== root.id);
    out = `${out} + ${em.value(other, ctx, lines)}`;
  }
  const store = `let i = b * M * N + row * N + col;
${lines.join('\n')}
  OUT[i] = ${out};`;
  let code;
  if (matmul === 'tiled') {
    code = `${bindings(operandCount(b))}
${em.usesGidx ? `\n${GIDX}\n` : ''}
var<workgroup> As: array<array<f32, ${TILE}>, ${TILE}>;
var<workgroup> Bs: array<array<f32, ${TILE}>, ${TILE}>;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let M = P[1];
  let K = P[2];
  let N = P[3];
  let b = wg.z;
  let row = wg.y * ${TILE}u + l.y;
  let col = wg.x * ${TILE}u + l.x;
  let aBase = b * M * K;
  let bBase = b * K * N;
  var acc: f32 = 0.0;
  let tiles = (K + ${TILE - 1}u) / ${TILE}u;
  for (var t0 = 0u; t0 < tiles; t0 = t0 + 1u) {
    let ak = t0 * ${TILE}u + l.x;
    let bk = t0 * ${TILE}u + l.y;
    var av: f32 = 0.0;
    if (row < M && ak < K) { av = A${ia}[aBase + row * K + ak]; }
    var bv: f32 = 0.0;
    if (bk < K && col < N) { bv = A${ib}[bBase + bk * N + col]; }
    As[l.y][l.x] = av;
    Bs[l.y][l.x] = bv;
    workgroupBarrier();
    for (var k = 0u; k < ${TILE}u; k = k + 1u) {
      acc = acc + As[l.y][k] * Bs[k][l.x];
    }
    workgroupBarrier();
  }
  if (row < M && col < N) {
  ${store}
  }
}
`;
  } else {
    code = `${bindings(operandCount(b))}
${em.usesGidx ? `\n${GIDX}\n` : ''}
@compute @workgroup_size(${MM_WG}, ${MM_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let M = P[1];
  let K = P[2];
  let N = P[3];
  let col = gid.x;
  let row = gid.y;
  let b = gid.z;
  if (row >= M || col >= N) { return; }
  let aBase = b * M * K + row * K;
  let bBase = b * K * N + col;
  var acc: f32 = 0.0;
  for (var k = 0u; k < K; k = k + 1u) {
    acc = acc + A${ia}[aBase + k] * A${ib}[bBase + k * N];
  }
  ${store}
}
`;
  }
  const t = matmul === 'tiled' ? TILE : MM_WG;
  return { code, params: Uint32Array.from(params), grid: [Math.ceil(N / t), Math.ceil(M / t), batch] };
}

// ---- rowwise ----------------------------------------------------------------------

const isReduce = (n) => n.op === 'sum' || n.op === 'max';

function rowwise(node, { sabotage }) {
  const b = bodyOf(node);
  const reductions = b.nodes.filter(isReduce);
  const S = b.nodes[reductions[0].inputs[0]].shape;
  const C = S.at(-1), R = size(S) / C;
  const params = [R, C, NEG_INF_BITS];
  const pattern = node.attrs.pattern;
  const rowTier = (id) => size(b.nodes[id].shape) === R && size(b.nodes[id].shape) !== R * C;
  const reduced = new Map();
  // Row-tier lets go before the loop that first needs them; element-tier lets inside it.
  let pre = null;
  const em = new Emitter(b, params, {
    special(id, ctx, lines) {
      const n = b.nodes[id];
      if (reduced.has(id)) return reduced.get(id);
      // A row-tier value read in the element context: evaluate it once per row, outside the loop.
      if (ctx.name === 'e' && rowTier(id)) return em.value(id, ROW, pre);
      // A row value broadcast back over the row.
      if (n.op === 'broadcast' && ctx.name === 'e' && rowTier(n.inputs[0]) && b.nodes[n.inputs[0]].op !== 'input'
        && b.nodes[n.inputs[0]].op !== 'const') return em.value(n.inputs[0], ROW, pre);
      return undefined;
    },
  });
  const ROW = { name: 'r', index: 'r' };
  const ELEM = { name: 'e', index: 'e' };
  const body = [];
  const resetElem = () => { for (const k of [...em.memo.keys()]) if (k.startsWith('e|') || k.startsWith('j')) em.memo.delete(k); };
  const shortTree = sabotage === 'layernorm-tree-short' && pattern === 'layernorm';
  reductions.forEach((rd, k) => {
    pre = [];
    const inner = [];
    resetElem();
    const v = em.value(rd.inputs[0], ELEM, inner);
    const isMax = rd.op === 'max';
    const combine = (a, c) => (isMax ? `max(${a}, ${c})` : `${a} + ${c}`);
    body.push(...pre);
    body.push(`  var acc${k}: f32 = ${isMax ? 'negInf' : '0.0'};
  for (var c = lid; c < C; c = c + ${ROW_WG}u) {
    let e = r * C + c;
${inner.map((l) => `  ${l}`).join('\n')}
    acc${k} = ${combine(`acc${k}`, v)};
  }
  red[lid] = acc${k};
  workgroupBarrier();
  for (var s = ${ROW_WG / 2}u; s > ${shortTree ? '1u' : '0u'}; s = s >> 1u) {
    if (lid < s) { red[lid] = ${combine('red[lid]', 'red[lid + s]')}; }
    workgroupBarrier();
  }
  let red${k}: f32 = ${sabotage === 'softmax-no-max' && isMax && (pattern === 'softmax' || pattern === 'log-softmax') ? '0.0' : 'red[0]'};
  workgroupBarrier();`);
    reduced.set(rd.id, `red${k}`);
  });
  const out = node.attrs.output;
  pre = [];
  resetElem();
  if (rowTier(out)) {
    const v = em.value(out, ROW, pre);
    body.push(...pre, `  if (lid == 0u) { OUT[r] = ${v}; }`);
  } else {
    const inner = [];
    const v = em.value(out, ELEM, inner);
    body.push(...pre, `  for (var c = lid; c < C; c = c + ${ROW_WG}u) {
    let e = r * C + c;
${inner.map((l) => `  ${l}`).join('\n')}
    OUT[e] = ${v};
  }`);
  }
  const code = `${bindings(operandCount(b))}
${em.usesGidx ? `\n${GIDX}\n` : ''}
var<workgroup> red: array<f32, ${ROW_WG}>;

@compute @workgroup_size(${ROW_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let r = wg.x + wg.y * nwg.x;
  if (r >= P[0]) { return; }
  let C = P[1];
  let negInf = bitcast<f32>(P[2]);
  let lid = l.x;
${body.join('\n')}
}
`;
  const x = Math.min(R, 65535);
  return { code, params: Uint32Array.from(params), grid: [x, Math.ceil(R / x), 1] };
}

/** The kernel for a fused node: { code, params, grid }. */
export function fusedKernel(node, { matmul = 'naive', sabotage } = {}) {
  if (node.attrs.kind === 'elementwise') return elementwise(node, { sabotage });
  if (node.attrs.kind === 'epilogue') return epilogue(node, { matmul, sabotage });
  if (node.attrs.kind === 'rowwise') return rowwise(node, { sabotage });
  throw new Error(`no fused kernel kind '${node.attrs.kind}'`);
}
