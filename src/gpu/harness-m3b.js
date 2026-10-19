// Milestone 3b's GPU checks, run in the browser by web/m3b.html (DESIGN.md
// amendments 22, 23, 26 and 27). Nothing is timed.
//
//   1. teacher-forced: the four golden runs, all N steps: the whole-step
//      check as registered (amendment 22) and the node-local check (R3)
//   2. causality and batch order on the GPU (bitwise, GPU against itself)
//   3. the trainer path against the checked path, bit for bit
//   4. statistical: 24 seeds x 300 steps trained on the GPU; validation loss by the interpreter
//   5. sabotage: each model bug against the teacher-forced check

import { Graph } from '../ir/graph.js';
import { size } from '../ir/shapes.js';
import { referenceWithBounds, S, F32_MAX } from './bounds.js';
import { lower } from './lower.js';
import { GpuTrainer } from './trainer.js';
import { initParams, buildForward, MODEL_BUGS } from '../m3b/model.js';
import { buildStep3b, paramShapes } from '../m3b/step.js';
import { adamInitialState } from '../m3b/adam.js';
import { encodeBatch } from '../m3b/data.js';
import { trainCpu3b, evalLossCpu } from '../m3b/cpu.js';
import { causalityTrials } from '../m3b/verify-cpu.js';

const fround = (p) => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, Float32Array.from(v)]));

/**
 * Amendment 22's judgement: every judgeable element is judged; unjudgeable
 * ones (unbounded error, or a reference outside float32's range) are counted.
 */
export function judge3b(graph, gpu, values, bounds) {
  let worst = 0, first = null, unjudgeable = 0, elements = 0;
  for (const node of graph.nodes) {
    const g = gpu[node.id], r = values[node.id], e = bounds[node.id];
    for (let i = 0; i < r.length; i++) {
      elements++;
      if (!Number.isFinite(r[i]) || Math.abs(r[i]) > F32_MAX || !Number.isFinite(e[i])) { unjudgeable++; continue; }
      const d = Math.abs(g[i] - r[i]);
      const ratio = d === 0 ? 0 : (Number.isNaN(d) ? Infinity : d / (S * e[i]));
      if (ratio > worst) worst = ratio;
      if (ratio > 1 && first === null) first = { node: node.id, op: node.op, index: i, gpu: g[i], ref: r[i], bound: e[i], ratio };
    }
  }
  const unjudged = unjudgeable > 0.01 * elements;
  return { pass: first === null && !unjudged, unjudged, worst, first, unjudgeable, elements };
}

/** The bound relative to the value over the new parameters: how informative the budget was. */
function boundRatios(step, ref) {
  const ratios = [];
  for (const n of step.names) {
    const id = step.outputs[n];
    ref.values[id].forEach((v, i) => { if (v !== 0) ratios.push(S * ref.bounds[id][i] / Math.abs(v)); });
  }
  ratios.sort((a, b) => a - b);
  return { median: ratios[ratios.length >> 1] ?? null, p99: ratios[Math.floor(ratios.length * 0.99)] ?? null };
}

// ---- 1. teacher-forced -----------------------------------------------------------

/**
 * Revision R3's node-local reference: for every computed node, the float64
 * result of that op alone on the float32-rounded reference values of its
 * operands, with amendment 8's bound for that op (operand errors zero), built
 * by running bounds.js on a one-op graph.
 */
export function localReference(graph, values) {
  const expected = new Array(graph.nodes.length), bound = new Array(graph.nodes.length);
  for (const node of graph.nodes) {
    if (node.op === 'input' || node.op === 'const') continue;
    const g = new Graph();
    const ins = node.inputs.map((id, k) => g.input(`a${k}`, graph.nodes[id].shape));
    const out = g.emit(node.op, ins, node.attrs);
    const r = referenceWithBounds(g, Object.fromEntries(node.inputs.map((id, k) => [`a${k}`, values[id]])));
    expected[node.id] = r.values[out];
    bound[node.id] = r.bounds[out];
  }
  return { expected, bound };
}

const align4 = (n) => Math.ceil(n / 4) * 4;

/**
 * Every launch of `lowered` reads float32-rounded reference operands and writes
 * its own output buffer, so no launch sees another's error. Returns the GPU
 * output per launched node.
 */
export async function runLocal(runner, graph, lowered, values) {
  const device = runner.device;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const temporary = [];
  const make = (n) => { const b = device.createBuffer({ size: align4(Math.max(1, n) * 4), usage }); temporary.push(b); return b; };
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  const ref = graph.nodes.map((node) => {
    const b = make(size(node.shape));
    device.queue.writeBuffer(b, 0, Float32Array.from(values[node.id]));
    return b;
  });
  const outs = new Map();
  const encoder = device.createCommandEncoder();
  for (const l of lowered.launches) {
    const out = make(size(graph.nodes[l.node].shape));
    outs.set(l.node, out);
    const pipeline = runner.pipeline(l.code);
    const p = device.createBuffer({ size: align4(Math.max(1, l.params.length) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    temporary.push(p);
    device.queue.writeBuffer(p, 0, l.params);
    const entries = [
      ...l.inputs.map((id, binding) => ({ binding, resource: { buffer: ref[id] } })),
      { binding: l.inputs.length, resource: { buffer: out } },
      { binding: l.inputs.length + 1, resource: { buffer: p } },
    ];
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
    pass.dispatchWorkgroups(...l.grid);
    pass.end();
  }
  let total = 0;
  const offsets = new Map();
  for (const [id, b] of outs) { offsets.set(id, total); total += b.size; }
  const staging = device.createBuffer({ size: Math.max(4, total), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  temporary.push(staging);
  for (const [id, b] of outs) encoder.copyBufferToBuffer(b, 0, staging, offsets.get(id), b.size);
  device.queue.submit([encoder.finish()]);
  const oom = await device.popErrorScope();
  const invalid = await device.popErrorScope();
  try {
    if (invalid) throw new Error(`WebGPU validation error: ${invalid.message}`);
    if (oom) throw new Error(`WebGPU out of memory: ${oom.message}`);
    await staging.mapAsync(GPUMapMode.READ);
    const all = staging.getMappedRange();
    const result = new Array(graph.nodes.length);
    for (const [id] of outs) result[id] = Float32Array.from(new Float32Array(all, offsets.get(id), size(graph.nodes[id].shape)));
    staging.unmap();
    return result;
  } finally {
    for (const b of temporary) b.destroy();
  }
}

/** Judge GPU outputs against expectations; nodes without one (inputs, constants) are skipped. */
function judgeNodes(graph, gpu, expected, bound) {
  let worst = 0, first = null, unjudgeable = 0, elements = 0;
  for (const node of graph.nodes) {
    if (!expected[node.id]) continue;
    const g = gpu[node.id], r = expected[node.id], e = bound[node.id];
    for (let i = 0; i < r.length; i++) {
      elements++;
      if (!Number.isFinite(r[i]) || Math.abs(r[i]) > F32_MAX || !Number.isFinite(e[i])) { unjudgeable++; continue; }
      const d = Math.abs(g[i] - r[i]);
      const ratio = d === 0 ? 0 : (Number.isNaN(d) ? Infinity : d / (S * e[i]));
      if (ratio > worst) worst = ratio;
      if (ratio > 1 && first === null) first = { node: node.id, op: node.op, index: i, gpu: g[i], ref: r[i], bound: e[i], ratio };
    }
  }
  const unjudged = unjudgeable > 0.01 * elements;
  return { pass: first === null && !unjudged, unjudged, worst, first, unjudgeable, elements };
}

/** For information only (R3): how far the GPU's loss and new parameters are from the interpreter's. */
function observed(step, gpu, ref) {
  let paramMaxRel = 0;
  for (const n of step.names) {
    const a = gpu[step.outputs[n]], b = ref.values[step.outputs[n]];
    let scale = 0, d = 0;
    for (let i = 0; i < b.length; i++) { scale = Math.max(scale, Math.abs(b[i])); d = Math.max(d, Math.abs(a[i] - b[i])); }
    if (scale > 0) paramMaxRel = Math.max(paramMaxRel, d / scale);
  }
  return { lossDiff: Math.abs(gpu[step.loss][0] - ref.values[step.loss][0]), paramMaxRel };
}

const brief = (j) => ({ pass: j.pass, unjudged: j.unjudged, worst: j.worst, unjudgeable: j.unjudgeable, elements: j.elements,
  first: j.first ? { op: j.first.op, node: j.first.node, ratio: j.first.ratio } : null });

export async function runTeacherForced3b(runner, spec, { bug, kernelSabotage, configs = spec.golden, steps, progress } = {}) {
  const syn = spec.synthetic.train;
  const runs = [];
  for (const g of configs) {
    const n = steps ?? g.steps;
    const step = buildStep3b(g.V, g.batch, { bug });
    // kernelSabotage: milestone 2's kernel and lowering bugs, only to test the checks themselves.
    const lowered = lower(step.graph, { sabotage: kernelSabotage });
    const batchAt = (t) => g.schedule[t].map((i) => syn[i]);
    const cpu = trainCpu3b({ V: g.V, B: g.batch, init: initParams(g.V, g.seed), steps: n, batchAt, bug, keepParams: true, keepState: true });
    const initState = adamInitialState(step.names, paramShapes(g.V));
    const out = [];
    for (let t = 0; t < n; t++) {
      const params = t === 0 ? initParams(g.V, g.seed) : cpu.params[t - 1];
      const state = t === 0 ? initState : cpu.states[t - 1];
      const { x, y } = encodeBatch(batchAt(t), g.V);
      const ref = referenceWithBounds(step.graph, { ...params, ...state, x, y, [step.cotangent]: [1] });
      const entry = { t };
      try {
        const gpu = await runner.run(step.graph, lowered, ref.inputs);
        entry.whole = brief(judge3b(step.graph, gpu.values, ref.values, ref.bounds));
        entry.observed = observed(step, gpu.values, ref);
        const local = localReference(step.graph, ref.values);
        const gpuLocal = await runLocal(runner, step.graph, lowered, ref.values);
        entry.local = brief(judgeNodes(step.graph, gpuLocal, local.expected, local.bound));
      } catch (error) {
        entry.error = String(error.message ?? error);
      }
      out.push(entry);
      progress?.(`teacher-forced${bug || kernelSabotage ? ` [${bug ?? kernelSabotage}]` : ''} ${g.name} step ${t + 1}/${n}`);
    }
    const summary = (k) => ({
      passed: out.filter((s) => s[k]?.pass).length, unjudged: out.filter((s) => s[k]?.unjudged).length,
      judgedFailures: out.filter((s) => s[k]?.first).length, worst: Math.max(0, ...out.map((s) => s[k]?.worst ?? 0)),
      unjudgeable: out.reduce((a, s) => a + (s[k]?.unjudgeable ?? 0), 0), elements: out.reduce((a, s) => a + (s[k]?.elements ?? 0), 0),
      firstJudgedFailure: out.findIndex((s) => s[k]?.first),
    });
    runs.push({ name: g.name, steps: out.length, errors: out.filter((s) => s.error).length, whole: summary('whole'), local: summary('local'),
      observed: { maxLossDiff: Math.max(0, ...out.map((s) => s.observed?.lossDiff ?? 0)), maxParamRel: Math.max(0, ...out.map((s) => s.observed?.paramMaxRel ?? 0)) },
      detail: out });
  }
  return runs;
}

// ---- 2. properties on the GPU ------------------------------------------------------

/** Logits on the GPU for a batch of sequences, from float32 parameters. */
export function gpuLogits(runner, V, params) {
  const cache = new Map();
  return async (seqs) => {
    const B = seqs.length;
    if (!cache.has(B)) { const f = buildForward(V, B); cache.set(B, { f, lowered: lower(f.graph) }); }
    const { f, lowered } = cache.get(B);
    const { x, y } = encodeBatch(seqs, V);
    const out = await runner.run(f.graph, lowered, { ...params, x, y });
    return out.values[f.logits];
  };
}

export async function runPropertiesGpu(runner, spec, { progress } = {}) {
  const syn = spec.synthetic.train;
  const causality = [], batchOrder = [];
  for (const g of spec.golden) {
    const params = fround(initParams(g.V, g.seed));
    const logits = gpuLogits(runner, g.V, params);
    const seqs = g.schedule[0].map((i) => syn[i]);
    for (const tr of await causalityTrials(g.V, seqs, logits, { seed: g.seed })) causality.push({ config: g.name, ...tr });
    const a = await logits(seqs), b = await logits([...seqs].reverse());
    const row = a.length / seqs.length;
    let permuted = true;
    for (let i = 0; i < seqs.length; i++) for (let j = 0; j < row; j++) permuted &&= Object.is(a[i * row + j], b[(seqs.length - 1 - i) * row + j]);
    batchOrder.push({ config: g.name, logitsPermuted: permuted });
    progress?.(`properties ${g.name}`);
  }
  return { causality, batchOrder };
}

// ---- 3. trainer path against the checked path ----------------------------------------

export async function runConsistency3b(runner, spec, steps = 3) {
  const g = spec.golden[0];
  const syn = spec.synthetic.train;
  const step = buildStep3b(g.V, g.batch);
  const names = [...step.names, ...step.stateNames];
  const init = fround(initParams(g.V, g.seed)), state = adamInitialState(step.names, paramShapes(g.V));
  const trainer = new GpuTrainer(runner, step, init, state);
  const lowered = lower(step.graph);
  let inputs = { ...init, ...state };
  let identical = true;
  for (let t = 0; t < steps; t++) {
    const { x, y } = encodeBatch(g.schedule[t].map((i) => syn[i]), g.V);
    const a = await trainer.train(x, y);
    const b = await runner.run(step.graph, lowered, { ...inputs, x, y, [step.cotangent]: [1] });
    if (a !== b.values[step.loss][0]) identical = false;
    inputs = Object.fromEntries(names.map((n) => [n, b.values[step.outputs[n]]]));
  }
  const fromTrainer = await trainer.read(names);
  for (const n of names) if (fromTrainer[n].some((v, i) => v !== inputs[n][i])) identical = false;
  trainer.destroy();
  return { config: g.name, steps, identical };
}

// ---- 4. statistical --------------------------------------------------------------

export async function runStatistical3b(runner, spec, { progress } = {}) {
  const syn = spec.synthetic;
  const st = spec.statistical;
  const V = syn.vocab;
  const step = buildStep3b(V, st.batch);
  const out = [];
  for (const run of st.runs) {
    const trainer = new GpuTrainer(runner, step, fround(initParams(V, run.seed)), adamInitialState(step.names, paramShapes(V)));
    const losses = [];
    for (let t = 0; t < st.steps; t++) {
      const { x, y } = encodeBatch(run.schedule[t].map((i) => syn.train[i]), V);
      losses.push(await trainer.train(x, y));
    }
    const final = await trainer.read(step.names);
    trainer.destroy();
    const valLoss = evalLossCpu(V, final, syn.val);
    out.push({ seed: run.seed, valLoss, finalTrainLoss: losses.at(-1), ...(run.seed === 1 ? { losses } : {}) });
    progress?.(`statistical seed ${run.seed}/${st.runs.length}`);
  }
  return out;
}

// ---- 5. sabotage against the GPU checks ---------------------------------------------------

/** A detection needs a judged element over its bound (R3: an unjudged step detects nothing). */
export async function runSabotage3b(runner, spec, { progress } = {}) {
  const out = [];
  for (const bug of Object.keys(MODEL_BUGS)) {
    const [r] = await runTeacherForced3b(runner, spec, { bug, configs: [spec.golden[0]], steps: 2, progress });
    out.push({ name: bug, trials: r.steps, caughtWhole: r.whole.judgedFailures, caughtLocal: r.local.judgedFailures, errors: r.errors,
      detail: r.detail.map((s) => ({ t: s.t, whole: s.whole, local: s.local, error: s.error ?? null })) });
  }
  return out;
}

/**
 * The teacher-forced checks tested by breaking the kernels: milestone 2's
 * kernel and lowering bugs, each on golden run 1's first step. The node-local
 * check should catch every one that changes a value beyond its op's budget.
 */
export const SELF_TEST_SABOTAGE = ['sum-off-by-one', 'transpose-swapped-stride', 'broadcast-wrong-stride', 'softmax-no-max', 'matmul-reorder'];

export async function runSelfTest3b(runner, spec, { progress } = {}) {
  const out = [];
  for (const kernelSabotage of SELF_TEST_SABOTAGE) {
    const [r] = await runTeacherForced3b(runner, spec, { kernelSabotage, configs: [spec.golden[0]], steps: 1, progress });
    const s = r.detail[0];
    out.push({ name: kernelSabotage, whole: s.whole ?? null, local: s.local ?? null, error: s.error ?? null });
  }
  return out;
}
