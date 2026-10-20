// A training step with fused nodes, for the GPU (browser only for the trainer).
//
// fusedStep() fuses milestone 3b's step graph, keeping the loss and every new
// parameter and state value, and renumbers the step's ids into the fused graph.
// FusedTrainer is milestone 3a's GpuTrainer (src/gpu/trainer.js) with the
// lowering passed in, so it runs a fused graph: every buffer, parameter block
// and bind group is made once; each step writes the batch, runs every launch,
// and copies the new parameters and state back into their input buffers on
// the GPU.

import { size } from '../ir/shapes.js';
import { fuse } from './fuse.js';
import { lowerFused } from './lower-fused.js';

const align4 = (n) => Math.ceil(n / 4) * 4;

/** Fuse a step from buildStep3b; ids are renumbered into the fused graph. */
export function fusedStep(step, { sabotage, passSabotage } = {}) {
  const keep = [step.loss, step.logits, ...Object.values(step.outputs)];
  const f = fuse(step.graph, { keep, sabotage: passSabotage });
  const at = (id) => f.index.get(id);
  return {
    ...step, graph: f.graph, fusion: f, original: step.graph,
    loss: at(step.loss), logits: at(step.logits),
    inputs: Object.fromEntries(Object.entries(step.inputs).map(([k, v]) => [k, at(v)])),
    outputs: Object.fromEntries(Object.entries(step.outputs).map(([k, v]) => [k, at(v)])),
    lowered: lowerFused(f.graph, { sabotage }),
  };
}

export class FusedTrainer {
  constructor(runner, step, params, state) {
    this.device = runner.device;
    this.step = step;
    const graph = step.graph;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.buffers = graph.nodes.map((n) => this.device.createBuffer({ size: align4(Math.max(1, size(n.shape)) * 4), usage }));
    const write = (id, values) => this.device.queue.writeBuffer(this.buffers[id], 0, Float32Array.from(values));
    for (const n of graph.nodes) {
      if (n.op === 'const') write(n.id, typeof n.attrs.value === 'number' ? new Array(size(n.shape)).fill(n.attrs.value) : n.attrs.value);
    }
    write(graph.inputId(step.cotangent), [1]);
    for (const [name, v] of Object.entries({ ...params, ...state })) write(step.inputs[name], v);
    this.passes = step.lowered.launches.map((l) => {
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
    this.feedback = Object.keys({ ...params, ...state }).map((name) => [step.outputs[name], step.inputs[name], this.buffers[step.inputs[name]].size]);
    this.staging = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

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

  async read(names) {
    const { device, step, buffers } = this;
    const sizes = names.map((n) => buffers[step.inputs[n]].size);
    const staging = device.createBuffer({ size: sizes.reduce((a, b) => a + b, 0), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    let off = 0;
    names.forEach((n, k) => { encoder.copyBufferToBuffer(buffers[step.inputs[n]], 0, staging, off, sizes[k]); off += sizes[k]; });
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = staging.getMappedRange();
    const out = {};
    off = 0;
    names.forEach((n, k) => {
      out[n] = Array.from(new Float32Array(all, off, size(step.graph.nodes[step.inputs[n]].shape)));
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
