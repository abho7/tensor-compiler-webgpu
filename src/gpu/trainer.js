// Training end to end on the GPU (browser only). The step graph is lowered
// once; every buffer, parameter block and bind group is created once. Each
// step writes the batch, runs every kernel, then copies each parameter's and
// state tensor's new value back into its input buffer on the GPU, so they
// never leave it. Only the loss is read back, for display.

import { size } from '../ir/shapes.js';
import { lower } from './lower.js';
import { PARAMS } from '../train/step.js';

const align4 = (n) => Math.ceil(n / 4) * 4;

export class GpuTrainer {
  /**
   * `runner` is a GpuRunner (for its device and pipeline cache); `step` comes
   * from buildStep(); `params` and `state` are initial values by name.
   */
  constructor(runner, step, params, state, { matmul = 'naive', sabotage } = {}) {
    this.runner = runner;
    this.device = runner.device;
    this.step = step;
    const graph = step.graph;
    const lowered = lower(graph, { matmul, sabotage });
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.buffers = graph.nodes.map((n) => this.device.createBuffer({ size: align4(Math.max(1, size(n.shape)) * 4), usage }));
    const write = (id, values) => this.device.queue.writeBuffer(this.buffers[id], 0, Float32Array.from(values));
    for (const n of graph.nodes) {
      if (n.op === 'const') write(n.id, typeof n.attrs.value === 'number' ? new Array(size(n.shape)).fill(n.attrs.value) : n.attrs.value);
    }
    write(graph.inputId(step.cotangent), [1]);
    for (const [name, v] of Object.entries({ ...params, ...state })) write(step.inputs[name], v);

    // Launches, each with its own parameter block and bind group, made once.
    this.passes = lowered.launches.map((l) => {
      const pipeline = runner.pipeline(l.code);
      const p = this.device.createBuffer({ size: align4(Math.max(1, l.params.length) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(p, 0, l.params);
      const entries = [
        ...l.inputs.map((id, binding) => ({ binding, resource: { buffer: this.buffers[id] } })),
        { binding: l.inputs.length, resource: { buffer: this.buffers[l.node] } },
        { binding: l.inputs.length + 1, resource: { buffer: p } },
      ];
      return { pipeline, bind: this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }), grid: l.grid, extra: p };
    });
    // What feeds back: each parameter and state output into its input.
    this.feedback = Object.keys({ ...params, ...state }).map((name) => [step.outputs[name], step.inputs[name], this.buffers[step.inputs[name]].size]);
    this.staging = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  /** One step on batch (x, y). Resolves to the loss computed from the parameters before the update. */
  async train(x, y) {
    const { device, step, buffers } = this;
    device.queue.writeBuffer(buffers[step.inputs.x], 0, Float32Array.from(x));
    device.queue.writeBuffer(buffers[step.inputs.y], 0, Float32Array.from(y));
    device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder();
    for (const p of this.passes) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(p.pipeline);
      pass.setBindGroup(0, p.bind);
      pass.dispatchWorkgroups(...p.grid);
      pass.end();
    }
    for (const [from, to, bytes] of this.feedback) encoder.copyBufferToBuffer(buffers[from], 0, buffers[to], 0, bytes);
    encoder.copyBufferToBuffer(buffers[step.loss], 0, this.staging, 0, 4);
    device.queue.submit([encoder.finish()]);
    const err = await device.popErrorScope();
    if (err) throw new Error(`WebGPU validation error: ${err.message}`);
    await this.staging.mapAsync(GPUMapMode.READ);
    const loss = new Float32Array(this.staging.getMappedRange().slice(0))[0];
    this.staging.unmap();
    return loss;
  }

  /** Read the current parameters (and state, by name) back from the GPU. */
  async read(names = PARAMS) {
    const { device, step, buffers } = this;
    const sizes = names.map((n) => buffers[step.inputs[n]].size);
    const total = sizes.reduce((a, b) => a + b, 0);
    const staging = device.createBuffer({ size: total, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    let off = 0;
    names.forEach((n, k) => { encoder.copyBufferToBuffer(buffers[step.inputs[n]], 0, staging, off, sizes[k]); off += sizes[k]; });
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = staging.getMappedRange();
    const out = {};
    off = 0;
    names.forEach((n, k) => {
      const len = size(step.graph.nodes[step.inputs[n]].shape);
      out[n] = Array.from(new Float32Array(all, off, len));
      off += sizes[k];
    });
    staging.unmap();
    staging.destroy();
    return out;
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
    for (const p of this.passes) p.extra.destroy();
    this.staging.destroy();
  }
}
