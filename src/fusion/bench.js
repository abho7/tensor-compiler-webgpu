// The timing harness (DESIGN.md amendment 32; browser only).
//
// One timed repetition is one command buffer with one compute pass,
// timestamped at its beginning and end, holding a program's dispatches k
// times over; the time per program run is the pass time over k. Wall-clock
// time from submit to onSubmittedWorkDone is recorded beside it. Every GPU
// object a repetition uses is made beforehand, and the device's create*
// methods and queue.writeBuffer are wrapped with counters so that this can be
// checked rather than assumed.

const align4 = (n) => Math.ceil(n / 4) * 4;
export const WARMUP = 5;
export const REPS = 32;
export const TARGET_NS = 2e6;
export const MAX_K = 4096;

export const median = (a) => { const s = [...a].sort((x, y) => x - y); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

export class Bench {
  static async create() {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter');
    const timestamps = adapter.features.has('timestamp-query');
    // shader-f16 for the hand-written kernels that store f16 weights (amendment 37).
    const features = ['timestamp-query', 'shader-f16'].filter((f) => adapter.features.has(f));
    const device = await adapter.requestDevice({ requiredFeatures: features });
    const bench = new Bench(adapter, device, timestamps);
    device.lost.then((info) => { bench.lost = info; });
    return bench;
  }

  constructor(adapter, device, timestamps) {
    this.adapter = adapter;
    this.device = device;
    this.timestamps = timestamps;
    this.pipelines = new Map();
    this.lost = null;
    this.counts = {};
    // The counters (amendment 32): every object-creating call, and every write.
    const count = (target, name) => {
      const original = target[name].bind(target);
      target[name] = (...args) => { this.counts[name] = (this.counts[name] ?? 0) + 1; return original(...args); };
    };
    for (const name of ['createBuffer', 'createBindGroup', 'createComputePipeline', 'createShaderModule', 'createQuerySet',
      'createCommandEncoder', 'createBindGroupLayout', 'createPipelineLayout']) count(device, name);
    count(device.queue, 'writeBuffer');
    if (timestamps) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    this.stamps = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  info() {
    const i = this.adapter.info ?? {};
    return { vendor: i.vendor ?? '', architecture: i.architecture ?? '', device: i.device ?? '', description: i.description ?? '',
      timestampQuery: this.timestamps, shaderF16: this.device.features.has('shader-f16'), userAgent: navigator.userAgent };
  }

  pipeline(code) {
    let p = this.pipelines.get(code);
    if (!p) {
      p = this.device.createComputePipeline({ layout: 'auto', compute: { module: this.device.createShaderModule({ code }), entryPoint: 'main' } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  /** Make every buffer, pipeline and bind group a program needs. */
  prepare(program) {
    const device = this.device;
    const buffers = program.buffers.map((b) => {
      const usage = b.uniform ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      const buffer = device.createBuffer({ size: Math.max(16, align4(b.bytes)), usage });
      if (b.data) device.queue.writeBuffer(buffer, 0, b.data.buffer, b.data.byteOffset, b.data.byteLength);
      return buffer;
    });
    const dispatches = program.dispatches.map((d) => {
      const pipeline = this.pipeline(d.code);
      const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: d.bindings.map((i, binding) => ({ binding, resource: { buffer: buffers[i] } })) });
      return { pipeline, bind, grid: d.grid };
    });
    return { program, buffers, dispatches };
  }

  /** Reset a prepared program's data buffers to their initial contents (outside any timed region). */
  reset(prepared) {
    prepared.program.buffers.forEach((b, i) => {
      if (b.data && !b.uniform) this.device.queue.writeBuffer(prepared.buffers[i], 0, b.data.buffer, b.data.byteOffset, b.data.byteLength);
    });
  }

  /** Run once and read back the output buffers (for the correctness gate; not timed). */
  async outputs(prepared) {
    const device = this.device;
    device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const d of prepared.dispatches) { pass.setPipeline(d.pipeline); pass.setBindGroup(0, d.bind); pass.dispatchWorkgroups(...d.grid); }
    pass.end();
    const ids = prepared.program.outputs;
    const sizes = ids.map((i) => prepared.program.buffers[i].bytes);
    const staging = device.createBuffer({ size: Math.max(16, align4(sizes.reduce((a, b) => a + b, 0))), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    let off = 0;
    const offs = ids.map((i, k) => { const o = off; encoder.copyBufferToBuffer(prepared.buffers[i], 0, staging, o, align4(sizes[k])); off += align4(sizes[k]); return o; });
    device.queue.submit([encoder.finish()]);
    const err = await device.popErrorScope();
    if (err) { staging.destroy(); throw new Error(`WebGPU validation error: ${err.message}`); }
    await staging.mapAsync(GPUMapMode.READ);
    const all = staging.getMappedRange();
    const out = ids.map((i, k) => Float32Array.from(new Float32Array(all, offs[k], sizes[k] / 4)));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /**
   * One timed repetition: k runs of the program in one compute pass.
   * `leak` (the counter's self-test only) makes a bind group inside the timed region.
   */
  async rep(prepared, k, { leak = false } = {}) {
    const device = this.device;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(this.timestamps
      ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    for (let r = 0; r < k; r++) {
      for (const d of prepared.dispatches) {
        pass.setPipeline(d.pipeline);
        pass.setBindGroup(0, leak ? device.createBindGroup({ layout: d.pipeline.getBindGroupLayout(0), entries: this.entriesOf(prepared, d) }) : d.bind);
        pass.dispatchWorkgroups(...d.grid);
      }
    }
    pass.end();
    if (this.timestamps) {
      encoder.resolveQuerySet(this.querySet, 0, 2, this.resolve, 0);
      encoder.copyBufferToBuffer(this.resolve, 0, this.stamps, 0, 16);
    }
    const t0 = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const wallMs = performance.now() - t0;
    let gpuNs = null;
    if (this.timestamps) {
      await this.stamps.mapAsync(GPUMapMode.READ);
      const t = new BigInt64Array(this.stamps.getMappedRange().slice(0));
      this.stamps.unmap();
      gpuNs = Number(t[1] - t[0]);
    }
    return { gpuNs, wallMs };
  }

  entriesOf(prepared, d) {
    const i = prepared.dispatches.indexOf(d);
    return prepared.program.dispatches[i].bindings.map((b, binding) => ({ binding, resource: { buffer: prepared.buffers[b] } }));
  }

  snapshot() { return { ...this.counts }; }
  static created(before, after) {
    const out = {};
    for (const k of new Set([...Object.keys(before), ...Object.keys(after)])) {
      const d = (after[k] ?? 0) - (before[k] ?? 0);
      if (d) out[k] = d;
    }
    return out;
  }

  /**
   * Time the variants of one workload per amendment 32: k from the first
   * variant's first run, 5 warm-ups each, then 32 repetitions each in ABBA
   * order. Returns per-variant repetitions and the objects created while timing.
   */
  async time(prepared, { leak = false } = {}) {
    const names = Object.keys(prepared);
    // k from the first variant's second single run (the first also pays for first use).
    await this.rep(prepared[names[0]], 1);
    const first = await this.rep(prepared[names[0]], 1);
    const one = first.gpuNs ?? first.wallMs * 1e6;
    const k = Math.max(1, Math.min(MAX_K, Math.ceil(TARGET_NS / Math.max(one, 1))));
    for (const n of names) for (let i = 0; i < WARMUP; i++) await this.rep(prepared[n], k);
    const reps = Object.fromEntries(names.map((n) => [n, []]));
    const before = this.snapshot();
    for (let b = 0; b < REPS; b++) {
      const order = b % 2 === 0 ? names : [...names].reverse();
      for (const n of order) {
        const r = await this.rep(prepared[n], k, { leak });
        reps[n].push({ gpuNs: r.gpuNs === null ? null : r.gpuNs / k, wallMs: r.wallMs / k });
      }
    }
    const created = Bench.created(before, this.snapshot());
    if (this.lost) throw new Error(`device lost: ${this.lost.message}`);
    return { k, reps, created };
  }

  destroy(prepared) { for (const b of prepared.buffers) b.destroy(); }
}

/** Created objects allowed inside the timed region: one command encoder per repetition, nothing else. */
export function creationViolations(created, repetitions) {
  const v = {};
  for (const [k, n] of Object.entries(created)) {
    if (k === 'createCommandEncoder' && n === repetitions) continue;
    v[k] = n;
  }
  return v;
}

/** Summary of one variant's repetitions: median, min and max per program run, in microseconds. */
export function summarize(reps) {
  const g = reps.map((r) => r.gpuNs).filter((x) => x !== null).map((x) => x / 1e3);
  const w = reps.map((r) => r.wallMs * 1e3);
  const s = (a) => (a.length ? { median: median(a), min: Math.min(...a), max: Math.max(...a) } : null);
  return { gpuUs: s(g), wallUs: s(w) };
}
