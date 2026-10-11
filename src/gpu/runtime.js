// Runs a lowered graph on WebGPU (browser only). One buffer per node, one
// command buffer per program, every node's value read back so every node can be
// judged.
//
// With `canary`, every buffer is padded past its end with a known bit pattern
// that is checked after the run: an out-of-bounds write that WebGPU's
// robustness rules would otherwise drop or clamp silently becomes visible.

import { size } from '../ir/shapes.js';
import { WG } from './codegen.js';

export const CANARY_BITS = 0xdeadbeef;

const align4 = (n) => Math.ceil(n / 4) * 4;

export class GpuRunner {
  static async create() {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter');
    const device = await adapter.requestDevice();
    const runner = new GpuRunner(adapter, device);
    device.lost.then((info) => { runner.lost = info; });
    return runner;
  }

  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    this.pipelines = new Map();
    this.lost = null;
  }

  info() {
    const i = this.adapter.info ?? {};
    return {
      vendor: i.vendor ?? '', architecture: i.architecture ?? '', device: i.device ?? '', description: i.description ?? '',
      features: [...this.adapter.features].sort(),
      limits: { maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize, maxComputeWorkgroupsPerDimension: this.adapter.limits.maxComputeWorkgroupsPerDimension },
      userAgent: navigator.userAgent,
    };
  }

  pipeline(code) {
    let p = this.pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  /**
   * Run `lowered` (from lower()) for `graph`. `inputs` maps input names to
   * float32-representable numbers. Returns { values, canaryIntact }: a
   * Float32Array per node and, with `canary`, whether each node's padding
   * survived. Throws on a WebGPU validation or memory error.
   */
  async run(graph, lowered, inputs, { canary = false } = {}) {
    if (this.lost) throw new Error(`device lost: ${this.lost.message}`);
    const device = this.device;
    const n = graph.nodes.map((node) => size(node.shape));
    // Padding past the last element: a whole extra workgroup's worth, so any
    // write a missing bounds check lets through lands in it.
    const capacity = n.map((k) => (canary ? Math.ceil(k / WG) * WG + WG : k));
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const buffers = capacity.map((c) => device.createBuffer({ size: align4(Math.max(1, c) * 4), usage }));
    const temporary = [...buffers];

    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');

    for (const node of graph.nodes) {
      const data = new Float32Array(capacity[node.id]);
      if (canary) new Uint32Array(data.buffer).fill(CANARY_BITS);
      if (node.op === 'input') data.set(inputs[node.attrs.name]);
      else if (node.op === 'const') {
        const v = node.attrs.value;
        if (typeof v === 'number') data.fill(v, 0, n[node.id]); else data.set(v);
      } else if (!canary) continue;
      device.queue.writeBuffer(buffers[node.id], 0, data);
    }

    const encoder = device.createCommandEncoder();
    for (const launch of lowered.launches) {
      const pipeline = this.pipeline(launch.code);
      const params = device.createBuffer({ size: align4(Math.max(1, launch.params.length) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, launch.params);
      temporary.push(params);
      const entries = [
        ...launch.inputs.map((id, binding) => ({ binding, resource: { buffer: buffers[id] } })),
        { binding: launch.inputs.length, resource: { buffer: buffers[launch.node] } },
        { binding: launch.inputs.length + 1, resource: { buffer: params } },
      ];
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(...launch.grid);
      pass.end();
    }

    const offsets = [];
    let total = 0;
    for (const b of buffers) { offsets.push(total); total += b.size; }
    const staging = device.createBuffer({ size: Math.max(4, total), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    temporary.push(staging);
    buffers.forEach((b, i) => encoder.copyBufferToBuffer(b, 0, staging, offsets[i], b.size));
    device.queue.submit([encoder.finish()]);

    const oom = await device.popErrorScope();
    const invalid = await device.popErrorScope();
    try {
      if (invalid) throw new Error(`WebGPU validation error: ${invalid.message}`);
      if (oom) throw new Error(`WebGPU out of memory: ${oom.message}`);
      await staging.mapAsync(GPUMapMode.READ);
      const all = staging.getMappedRange();
      const values = [], canaryIntact = [];
      graph.nodes.forEach((node, i) => {
        const view = new Float32Array(all, offsets[i], capacity[i]);
        values.push(Float32Array.from(view.subarray(0, n[i])));
        if (canary) canaryIntact.push(new Uint32Array(all, offsets[i] + n[i] * 4, capacity[i] - n[i]).every((w) => w === CANARY_BITS));
      });
      staging.unmap();
      return { values, canaryIntact };
    } finally {
      for (const b of temporary) b.destroy();
    }
  }
}
