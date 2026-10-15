// Milestone 3a's GPU checks, run in the browser by web/m3.html (DESIGN.md
// amendments 13, 17 and 18). Every comparison is per element under the float32
// budget (amendment 8). Nothing is timed.
//
//   1. direct sqrt kernel tests (canary-padded buffers)
//   2. teacher-forced per step: the 8 golden runs, every step
//   3. free-running, K = 5 steps, bounds carried across steps
//   4. statistical: 24 seeds x {SGD, Adam} x 200 steps, trained on the GPU
//   5. the trainer path against the checked path, bit for bit
//   6. sabotage: each of the five bugs against the GPU checks

import { size } from '../ir/shapes.js';
import { referenceWithBounds, judge, unjudgeable, S } from './bounds.js';
import { lower } from './lower.js';
import { GpuTrainer } from './trainer.js';
import { sqrtDirectCases } from './direct-cases-m3.js';
import { buildStep, initialState, PARAMS, SGD, ADAM, OPTIMIZER_SABOTAGE } from '../train/step.js';
import { trainCpu, evaluateMlp } from '../train/cpu.js';
import { syntheticSpirals, initMlp, batchSchedule, gather } from '../train/data.js';
import { sabotagedRules } from '../testing/sabotage.js';

export const FREE_RUNNING_K = 5;
export const STAT_SEEDS = 24;
export const STAT_STEPS = 200;
const DIMS = { inputs: 8, hidden: 16, classes: 4 };
const BATCH = 32;

/** The five bugs (amendment 18): where each is injected. */
export const BUGS = [
  { name: 'adam-bias-correction', level: 'graph', applies: ['adam'] },
  { name: 'momentum-off-by-one', level: 'graph', applies: ['sgd'] },
  { name: 'weight-decay-sign', level: 'graph', applies: ['sgd', 'adam'] },
  { name: 'sqrt-gradient', level: 'rule', applies: ['sgd', 'adam'] },
  { name: 'sqrt-bounds-off-by-one', level: 'kernel', applies: ['adam'] },
];

function stepFor(opt, bug) {
  if (bug?.level === 'graph') return buildStep(DIMS, BATCH, opt, { sabotage: bug.name });
  if (bug?.level === 'rule') return buildStep(DIMS, BATCH, opt, { rules: sabotagedRules('sqrt') });
  return buildStep(DIMS, BATCH, opt);
}

async function runChecked(runner, graph, inputs, { sabotage, canary = false, carried } = {}) {
  const ref = referenceWithBounds(graph, inputs, { carried });
  if (unjudgeable(ref.values, ref.bounds) > 0) return { status: 'skip', ref };
  const gpuInputs = { ...ref.inputs };
  if (carried) for (const k of Object.keys(carried)) gpuInputs[k] = inputs[k];
  let gpu;
  try {
    gpu = await runner.run(graph, lower(graph, { sabotage }), gpuInputs, { canary });
  } catch (error) {
    return { status: 'error', error: String(error.message ?? error), ref };
  }
  const j = judge(graph, gpu.values, ref.values, ref.bounds);
  const canaryOk = !canary || gpu.canaryIntact.every(Boolean);
  return { status: j.pass && canaryOk ? 'pass' : 'fail', worst: j.worst, first: j.first, canaryOk, ref, gpu };
}

// ---- 1. direct sqrt tests ---------------------------------------------------

export async function runSqrtDirect(runner, { sabotage, progress } = {}) {
  const out = [];
  for (const c of sqrtDirectCases()) {
    const r = await runChecked(runner, c.graph, c.inputs, { sabotage, canary: true });
    out.push({ name: c.name, kernel: c.kernel, status: r.status, worst: r.worst ?? null, first: r.first ?? null, canaryOk: r.canaryOk ?? null, error: r.error ?? null });
  }
  progress?.(`sqrt direct${sabotage ? ` [${sabotage}]` : ''} done`);
  return out;
}

// ---- 2. teacher-forced --------------------------------------------------------

/**
 * Every step of each golden run: both sides get the same float32-rounded
 * parameters, state and batch (the float64 trajectory's, before that step) and
 * run one step, judged on every node.
 */
export async function runTeacherForced(runner, spec, { bug, progress } = {}) {
  const runs = [];
  for (const cfg of spec.configs) {
    if (bug && !bug.applies.includes(cfg.optimizer.kind)) continue;
    const step = stepFor(cfg.optimizer, bug);
    const cpu = trainCpu({ dims: spec.dims, data: spec.synthetic.train, opt: cfg.optimizer, seed: cfg.seed, steps: cfg.steps, batch: cfg.batch,
      keepParams: true, keepState: true, init: cfg.init, schedule: cfg.schedule,
      sabotage: bug?.level === 'graph' ? bug.name : undefined, rules: bug?.level === 'rule' ? sabotagedRules('sqrt') : undefined });
    const steps = [];
    for (let t = 0; t < cfg.steps; t++) {
      const params = t === 0 ? cfg.init : cpu.params[t - 1];
      const state = t === 0 ? initialState(cfg.optimizer, spec.dims) : cpu.states[t - 1];
      const { x, y } = gather(spec.synthetic.train, cfg.schedule[t], spec.dims.inputs, spec.dims.classes);
      const r = await runChecked(runner, step.graph, { ...params, ...state, x, y, [step.cotangent]: [1] },
        { sabotage: bug?.level === 'kernel' ? bug.name : undefined });
      steps.push({ status: r.status, worst: r.worst ?? null, first: r.first ? { op: r.first.op, node: r.first.node } : null, error: r.error ?? null });
    }
    const failing = steps.filter((s) => s.status !== 'pass' && s.status !== 'skip');
    runs.push({ name: cfg.name, steps: steps.length, passed: steps.filter((s) => s.status === 'pass').length,
      skipped: steps.filter((s) => s.status === 'skip').length, failed: failing.length,
      firstFailure: failing.length ? steps.indexOf(failing[0]) : null, worst: Math.max(0, ...steps.map((s) => s.worst ?? 0)),
      failures: failing.slice(0, 5) });
    progress?.(`teacher-forced${bug ? ` [${bug.name}]` : ''} ${cfg.name}`);
  }
  return runs;
}

// ---- 3. free-running, K steps ---------------------------------------------------

export async function runFreeRunning(runner, spec, { progress } = {}) {
  const runs = [];
  for (const cfg of spec.configs) {
    const step = buildStep(spec.dims, cfg.batch, cfg.optimizer);
    const names = [...PARAMS, ...Object.keys(initialState(cfg.optimizer, spec.dims))];
    let inputs = { ...cfg.init, ...initialState(cfg.optimizer, spec.dims) };
    let carried;
    const steps = [];
    for (let t = 0; t < FREE_RUNNING_K; t++) {
      const { x, y } = gather(spec.synthetic.train, cfg.schedule[t], spec.dims.inputs, spec.dims.classes);
      const r = await runChecked(runner, step.graph, { ...inputs, x, y, [step.cotangent]: [1] }, { carried });
      // How loose the carried bound has become: S*E / |value| over the new parameters.
      const ratios = [];
      for (const p of PARAMS) {
        const id = step.outputs[p];
        r.ref.values[id].forEach((v, i) => { if (v !== 0) ratios.push(S * r.ref.bounds[id][i] / Math.abs(v)); });
      }
      ratios.sort((a, b) => a - b);
      steps.push({ status: r.status, worst: r.worst ?? null, boundToValueMedian: ratios[ratios.length >> 1] ?? null,
        boundToValueMax: ratios.length ? ratios[ratios.length - 1] : null, first: r.first ? { op: r.first.op, node: r.first.node } : null });
      if (r.status !== 'pass') break;
      // Each side continues from its own outputs; the reference carries its bounds.
      carried = Object.fromEntries(names.map((n) => [n, { value: r.ref.values[step.outputs[n]], bound: r.ref.bounds[step.outputs[n]] }]));
      inputs = Object.fromEntries(names.map((n) => [n, Array.from(r.gpu.values[step.outputs[n]])]));
    }
    runs.push({ name: cfg.name, steps });
    progress?.(`free-running ${cfg.name}`);
  }
  return runs;
}

// ---- 4. statistical, 5. trainer vs checked path ------------------------------------

export async function runStatistical(runner, { progress } = {}) {
  const data = syntheticSpirals();
  const out = [];
  for (const opt of [SGD, ADAM]) {
    for (let seed = 1; seed <= STAT_SEEDS; seed++) {
      const init = initMlp(DIMS, seed);
      const schedule = batchSchedule(data.train.y.length, BATCH, STAT_STEPS, seed);
      const step = buildStep(DIMS, BATCH, opt);
      const trainer = new GpuTrainer(runner, step, init, initialState(opt, DIMS));
      const gpuLosses = [];
      for (let t = 0; t < STAT_STEPS; t++) {
        const { x, y } = gather(data.train, schedule[t], DIMS.inputs, DIMS.classes);
        gpuLosses.push(await trainer.train(x, y));
      }
      const gpuFinal = await trainer.read();
      trainer.destroy();
      const cpu = trainCpu({ dims: DIMS, data: data.train, opt, seed, steps: STAT_STEPS, batch: BATCH, init, schedule });
      const e = (p) => ({ trainLoss: evaluateMlp(DIMS, p, data.train).loss, testAccuracy: evaluateMlp(DIMS, p, data.test).accuracy });
      out.push({ optimizer: opt.kind, seed, gpu: e(gpuFinal), cpu: e(cpu.final),
        ...(seed === 1 ? { gpuLosses, cpuLosses: cpu.losses } : {}) });
      progress?.(`statistical ${opt.kind} seed ${seed}`);
    }
  }
  return out;
}

/** The trainer's path must compute what the checked path computes: first steps, bit for bit. */
export async function runConsistency(runner, spec, steps = 3) {
  const cfg = spec.configs.find((c) => c.optimizer.kind === 'adam');
  const step = buildStep(spec.dims, cfg.batch, cfg.optimizer);
  const names = [...PARAMS, ...Object.keys(initialState(cfg.optimizer, spec.dims))];
  const trainer = new GpuTrainer(runner, step, cfg.init, initialState(cfg.optimizer, spec.dims));
  let inputs = { ...cfg.init, ...initialState(cfg.optimizer, spec.dims) };
  let identical = true;
  const lowered = lower(step.graph);
  for (let t = 0; t < steps; t++) {
    const { x, y } = gather(spec.synthetic.train, cfg.schedule[t], spec.dims.inputs, spec.dims.classes);
    const a = await trainer.train(x, y);
    const b = await runner.run(step.graph, lowered, { ...inputs, x, y, [step.cotangent]: [1] });
    if (a !== b.values[step.loss][0]) identical = false;
    inputs = Object.fromEntries(names.map((n) => [n, Array.from(b.values[step.outputs[n]])]));
  }
  const fromTrainer = await trainer.read(names);
  for (const n of names) if (fromTrainer[n].some((v, i) => v !== inputs[n][i])) identical = false;
  trainer.destroy();
  return { config: cfg.name, steps, identical };
}

// ---- 6. sabotage ---------------------------------------------------------------

export async function runSabotage(runner, spec, honestDirect, { progress } = {}) {
  const out = [];
  for (const bug of BUGS) {
    const tf = await runTeacherForced(runner, spec, { bug, progress });
    let direct = null;
    if (bug.level === 'kernel') {
      const d = await runSqrtDirect(runner, { sabotage: bug.name, progress });
      const caught = d.filter((x, k) => x.status !== 'pass' && honestDirect[k].status === 'pass');
      direct = { cases: d.length, caught: caught.length, first: caught[0]?.name ?? null,
        how: caught[0] ? (caught[0].canaryOk === false && !caught[0].first ? 'canary' : 'values') : null };
    }
    out.push({ name: bug.name, level: bug.level, teacherForced: { runs: tf.length, caught: tf.filter((r) => r.failed > 0).length,
      detail: tf.map((r) => ({ name: r.name, failed: r.failed, firstFailure: r.firstFailure })) }, direct });
  }
  return out;
}

export { OPTIMIZER_SABOTAGE };
