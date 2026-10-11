// WGSL for every kernel, generated as text in plain JavaScript so it can be
// tested without a GPU.
//
// Kernels are generic over shape: sizes and strides arrive at run time in a
// small u32 parameter buffer P, so one compiled pipeline serves every shape.
// Every kernel bounds-checks the thread index against the element count it was
// given, because the grid is rounded up to whole workgroups.
//
// `sabotage` names one deliberately wrong variant (DESIGN.md, milestone 2's
// sabotage list); the default is the correct kernel.

export const WG = 64;          // threads per workgroup, 1-D kernels
export const MM_WG = 8;        // naive matmul: 8 x 8 threads, one output each
export const TILE = 16;        // tiled matmul: 16 x 16 tile, one output per thread
export const MAX_RANK = 6;     // strided gather kernels
export const MAX_GROUPS = 65535;

const UNARY = {
  neg: (x) => `-${x}`,
  exp: (x) => `exp(${x})`,
  log: (x) => `log(${x})`,
  relu: (x) => `max(${x}, 0.0)`,
  // Clamped: the specification lets tanh inherit sinh/cosh's overflow, and
  // this project's adapter returns NaN from |x| >= 89 (DESIGN.md revision R2).
  // tanh(20) = 1 - 8.5e-18, exactly 1 in float32, so the clamp loses nothing.
  tanh: (x) => `tanh(clamp(${x}, -20.0, 20.0))`,
  step: (x) => `select(0.0, 1.0, ${x} > 0.0)`,
  copy: (x) => x,
};
const BINARY = { add: '+', sub: '-', mul: '*', div: '/' };

export const ELEMENTWISE_OPS = [...Object.keys(UNARY), ...Object.keys(BINARY)];

const bindings = (inputs) => [
  ...inputs.map((name, i) => `@group(0) @binding(${i}) var<storage, read> ${name}: array<f32>;`),
  `@group(0) @binding(${inputs.length}) var<storage, read_write> OUT: array<f32>;`,
  `@group(0) @binding(${inputs.length + 1}) var<storage, read> P: array<u32>;`,
].join('\n');

/** The flat thread index of a 1-D kernel dispatched over up to 65535 x Y workgroups. */
const FLAT = `let i = gid.x + gid.y * nwg.x * ${WG}u;`;
const ENTRY_1D = `@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {`;

function elementwise(op, sabotage) {
  // The sabotaged variant still reads P (a phony assignment), or the 'auto'
  // pipeline layout would drop the binding and every launch would fail to bind.
  const guard = sabotage === 'elementwise-no-bounds-check' ? '  _ = P[0];\n' : '  if (i >= P[0]) { return; }\n';
  const isBinary = op in BINARY;
  const expr = isBinary ? `A[i] ${BINARY[op]} B[i]` : UNARY[op]('A[i]');
  return `${bindings(isBinary ? ['A', 'B'] : ['A'])}

${ENTRY_1D}
  ${FLAT}
${guard}  OUT[i] = ${expr};
}
`;
}

/**
 * out[i] = A[src(i)], where src decomposes i over the output shape and
 * recomposes it with per-axis source strides. Broadcast (stride 0 on expanded
 * axes), transpose (permuted strides) and reshape (the output's own strides)
 * are all this kernel with different parameters.
 * P = [n, rank, shape[0..MAX_RANK), stride[0..MAX_RANK)].
 */
function gather() {
  return `${bindings(['A'])}

${ENTRY_1D}
  ${FLAT}
  if (i >= P[0]) { return; }
  let rank = P[1];
  var rem = i;
  var src = 0u;
  for (var k = 0u; k < rank; k = k + 1u) {
    let d = rank - 1u - k;
    let extent = P[2u + d];
    src = src + (rem % extent) * P[${2 + MAX_RANK}u + d];
    rem = rem / extent;
  }
  OUT[i] = A[src];
}
`;
}

/** Reduce along one axis. P = [outer, n, inner]; one thread per output element. */
function reduce(op, sabotage) {
  const extent = sabotage === 'sum-off-by-one' && op === 'sum' ? 'n - 1u' : 'n';
  const init = op === 'sum' ? '0.0' : 'A[base]';
  const start = op === 'sum' ? '0u' : '1u';
  const combine = op === 'sum' ? 'acc + v' : 'max(acc, v)';
  return `${bindings(['A'])}

${ENTRY_1D}
  ${FLAT}
  let outer = P[0];
  let n = P[1];
  let inner = P[2];
  if (i >= outer * inner) { return; }
  let o = i / inner;
  let j = i % inner;
  let base = o * n * inner + j;
  var acc: f32 = ${init};
  for (var k = ${start}; k < ${extent}; k = k + 1u) {
    let v = A[base + k * inner];
    acc = ${combine};
  }
  OUT[i] = acc;
}
`;
}

/** One thread per output element. P = [batch, M, K, N]. */
function matmulNaive(sabotage) {
  const loop = sabotage === 'matmul-reorder'
    ? `for (var t = 0u; t < K; t = t + 1u) {
    let k = K - 1u - t;`
    : `for (var k = 0u; k < K; k = k + 1u) {`;
  return `${bindings(['A', 'B'])}

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
  ${loop}
    acc = acc + A[aBase + k] * B[bBase + k * N];
  }
  OUT[b * M * N + row * N + col] = acc;
}
`;
}

/**
 * A TILE x TILE output tile per workgroup, with A and B staged through
 * workgroup memory one TILE-wide slice of K at a time. Loads past the edge of
 * A or B read zero, so M, N and K need not be multiples of TILE. No thread
 * returns early: every thread must reach every barrier.
 */
function matmulTiled(sabotage) {
  const lt = sabotage === 'tiled-off-by-one' ? '<=' : '<';
  const inner = sabotage === 'matmul-reorder'
    ? `for (var t = 0u; t < ${TILE}u; t = t + 1u) {
      let k = ${TILE - 1}u - t;`
    : `for (var k = 0u; k < ${TILE}u; k = k + 1u) {`;
  return `${bindings(['A', 'B'])}

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
    if (row < M && ak ${lt} K) { av = A[aBase + row * K + ak]; }
    var bv: f32 = 0.0;
    if (bk ${lt} K && col < N) { bv = B[bBase + bk * N + col]; }
    As[l.y][l.x] = av;
    Bs[l.y][l.x] = bv;
    workgroupBarrier();
    ${inner}
      acc = acc + As[l.y][k] * Bs[k][l.x];
    }
    workgroupBarrier();
  }
  if (row < M && col < N) { OUT[b * M * N + row * N + col] = acc; }
}
`;
}

/**
 * The WGSL source for a kernel `kind`: an elementwise op name, 'gather',
 * 'sum', 'max', 'matmul-naive' or 'matmul-tiled'.
 */
export function kernelSource(kind, { sabotage } = {}) {
  if (ELEMENTWISE_OPS.includes(kind)) return elementwise(kind, sabotage);
  if (kind === 'gather') return gather();
  if (kind === 'sum' || kind === 'max') return reduce(kind, sabotage);
  if (kind === 'matmul-naive') return matmulNaive(sabotage);
  if (kind === 'matmul-tiled') return matmulTiled(sabotage);
  throw new Error(`no kernel for '${kind}'`);
}

export const KERNEL_KINDS = [...ELEMENTWISE_OPS, 'gather', 'sum', 'max', 'matmul-naive', 'matmul-tiled'];

/** Workgroup counts for a 1-D kernel over `n` threads: up to 65535 in x, the rest in y. */
export function grid1d(n) {
  const groups = Math.max(1, Math.ceil(n / WG));
  const x = Math.min(groups, MAX_GROUPS);
  return [x, Math.ceil(groups / x), 1];
}
