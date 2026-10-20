// Milestone 4's GPU correctness checks, run in the browser by web/m4.html
// (DESIGN.md amendments 31 and 36). Nothing here is timed.
//
//   1. direct adversarial tests of every fused kernel kind (canary-padded)
//   2. the differential fuzzer: unfused and fused graphs on the GPU, both
//      against the float64 interpreter, milestone 2's corpus
//   3. the fused training step, node-local, every step of the four golden runs
//   4. the fused trainer against the checked path, bit for bit
//   5. statistical: 24 seeds trained with the fused step
//   6. sabotage: amendment 31's five bugs against the GPU checks

import { Graph } from '../ir/graph.js';
import { size } from '../ir/shapes.js';
import { grad } from '../autodiff.js';
import { Rng } from '../testing/rng.js';
import { corpusProgram } from '../testing/random-graph.js';
import { referenceWithBounds, judge, unjudgeable, S, F32_MAX } from './bounds.js';
import { lower } from './lower.js';
import { runLocal } from './harness-m3b.js';
import { fuse, bodyOf } from '../fusion/fuse.js';
import { lowerFused } from '../fusion/lower-fused.js';
import { FUSED_SABOTAGE } from '../fusion/codegen-fused.js';
import { fusedDirectCases } from '../fusion/direct-cases-m4.js';
import { fusedStep, FusedTrainer } from '../fusion/trainer-fused.js';
import { initParams, buildForward } from '../m3b/model.js';
import { buildStep3b, paramShapes } from '../m3b/step.js';
import { adamInitialState } from '../m3b/adam.js';
import { encodeBatch } from '../m3b/data.js';
import { trainCpu3b, evalLossCpu } from '../m3b/cpu.js';

export const FUZZ_PROGRAMS_M4 = 2000;
export const SABOTAGE_FUZZ_PROGRAMS = 500;
/** Amendment 31's bugs, in its order: four in kernels, one in the pass. */
export const SABOTAGES_M4 = [
  { name: 'epilogue-before-bias', where: 'kernel', kinds: ['epilogue'] },
  { name: 'layernorm-tree-short', where: 'kernel', patterns: ['layernorm'] },
  { name: 'softmax-no-max', where: 'kernel', patterns: ['softmax', 'log-softmax'] },
  { name: 'fused-bounds-off-by-one', where: 'kernel', kinds: ['elementwise'] },
  { name: 'fusion-absorbs-shared', where: 'pass' },
];
if (SABOTAGES_M4.filter((s) => s.where === 'kernel').some((s) => !FUSED_SABOTAGE.includes(s.name))) throw new Error('sabotage lists disagree');

const fround = (p) => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, Float32Array.from(v)]));

/** Judge a fused graph's GPU values at every surviving node against the original's reference. */
export function judgeFused(fused, gpu, ref) {
  let worst = 0, first = null;
  for (const node of fused.graph.nodes) {
    const o = fused.origin[node.id];
    const g = gpu[node.id], r = ref.values[o], e = ref.bounds[o];
    if (g.length !== r.length) return { pass: false, worst: Infinity, first: { node: node.id, origin: o, op: node.op, reason: 'length' } };
    for (let i = 0; i < r.length; i++) {
      const d = Math.abs(g[i] - r[i]);
      const ratio = d === 0 ? 0 : (Number.isNaN(d) ? Infinity : d / (S * e[i]));
      if (ratio > worst) worst = ratio;
      if (ratio > 1 && first === null) first = { node: node.id, origin: o, op: node.op, pattern: node.attrs.pattern ?? null, index: i, gpu: g[i], ref: r[i], bound: e[i], ratio };
    }
  }
  return { pass: first === null, worst, first };
}

/** Whether a fused graph's GPU values equal the unfused graph's bit for bit at every surviving node. */
function bitwiseSame(fused, gpuFused, gpuUnfused) {
  for (const node of fused.graph.nodes) {
    const a = gpuFused[node.id], b = gpuUnfused[fused.origin[node.id]];
    for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  }
  return true;
}

const fusedKinds = (f) => [...new Set(f.graph.nodes.filter((n) => n.op === 'fused').map((n) => n.attrs.kind))];
const fusedPatterns = (f) => [...new Set(f.graph.nodes.filter((n) => n.op === 'fused').map((n) => n.attrs.pattern))];

/**
 * One graph: the fused version (and, with `unfused`, the unfused one too) on
 * the GPU against the float64 interpreter. Returns a status per variant.
 */
async function runPair(runner, graph, rawInputs, { matmul = 'naive', sabotage, canary = false, keep = [], unfused = true, ref } = {}) {
  ref ??= referenceWithBounds(graph, rawInputs);
  if (unjudgeable(ref.values, ref.bounds) > 0) return { status: 'skip' };
  const pass = SABOTAGES_M4.find((s) => s.name === sabotage)?.where === 'pass';
  let fused;
  try {
    fused = fuse(graph, { keep, sabotage: pass ? sabotage : undefined });
  } catch (error) {
    return { status: 'fuse-error', error: String(error.message ?? error) };
  }
  const out = { status: 'ok', kinds: fusedKinds(fused), patterns: fusedPatterns(fused), fusedNodes: fused.graph.nodes.filter((n) => n.op === 'fused').length };
  try {
    const gf = await runner.run(fused.graph, lowerFused(fused.graph, { matmul, sabotage: pass ? undefined : sabotage }), ref.inputs, { canary });
    const j = judgeFused(fused, gf.values, ref);
    const canaryOk = !canary || gf.canaryIntact.every(Boolean);
    out.fused = { pass: j.pass && canaryOk, worst: j.worst, first: j.first, canaryOk };
    if (unfused) {
      const gu = await runner.run(graph, lower(graph, { matmul }), ref.inputs, { canary });
      const ju = judge(graph, gu.values, ref.values, ref.bounds);
      out.unfused = { pass: ju.pass && (!canary || gu.canaryIntact.every(Boolean)), worst: ju.worst, first: ju.first };
      out.bitwiseSame = bitwiseSame(fused, gf.values, gu.values);
    }
  } catch (error) {
    out.status = 'error';
    out.error = String(error.message ?? error);
  }
  return out;
}

// ---- 1. direct tests ------------------------------------------------------------

export async function runDirectM4(runner, { sabotage, progress } = {}) {
  const out = [];
  const cases = fusedDirectCases();
  for (const [k, c] of cases.entries()) {
    const r = await runPair(runner, c.graph, c.inputs, { matmul: c.matmul ?? 'naive', sabotage, canary: true, unfused: false });
    const intended = r.kinds?.length === 1 && r.kinds[0] === c.kind && r.patterns[0] === c.pattern;
    out.push({ name: c.name, kind: c.kind, pattern: c.pattern, matmul: c.matmul ?? null, intended, status: r.status,
      pass: r.status === 'ok' && intended && r.fused.pass, valuesPass: r.status === 'ok' ? r.fused.pass : null, worst: r.fused?.worst ?? null, first: r.fused?.first ?? null,
      canaryOk: r.fused?.canaryOk ?? null, error: r.error ?? null });
    progress?.(`direct${sabotage ? ` [${sabotage}]` : ''} ${k + 1}/${cases.length}`);
  }
  return out;
}

// ---- 2. fuzzing ---------------------------------------------------------------

/** Milestone 2's runs for corpus program i (harness.js programRuns, repeated here so that file stays as recorded). */
export function programRunsM4(i) {
  const p = corpusProgram(i, { f32: true });
  const rng = new Rng((i * 7919 + 17) >>> 0);
  const inputs = {};
  for (const n of p.graph.inputs()) inputs[n.attrs.name] = rng.array(size(n.shape), -2, 2);
  const G = grad(p.graph, p.outputs, p.graph.inputs().map((n) => n.id));
  const gradInputs = { ...inputs };
  G.cotangents.forEach((name, k) => { gradInputs[name] = rng.array(size(p.graph.nodes[p.outputs[k]].shape), -1, 1); });
  const hasMatmul = p.graph.nodes.some((n) => n.op === 'matmul');
  const runs = [];
  for (const matmul of hasMatmul ? ['naive', 'tiled'] : ['naive']) {
    runs.push({ label: 'forward', graph: p.graph, inputs, outputs: p.outputs, matmul });
    runs.push({ label: 'gradient', graph: G.graph, inputs: gradInputs, outputs: [...p.outputs, ...G.grads], matmul });
  }
  return runs;
}

const source = (i) => (i % 2 === 1 ? 'general' : 'targeted');

export async function runFuzzM4(runner, { programs = FUZZ_PROGRAMS_M4, progress } = {}) {
  const result = { programs, skipped: 0, runs: 0, fusedRuns: 0, unfusedPassed: 0, fusedPassed: 0, errors: 0, bitwiseSame: 0,
    worstFused: 0, worstUnfused: 0, failures: [], byKind: {}, byPattern: {} };
  for (let i = 1; i <= programs; i++) {
    const runs = programRunsM4(i);
    const outcomes = [];
    for (const run of runs) outcomes.push({ run, r: await runPair(runner, run.graph, run.inputs, { matmul: run.matmul, keep: run.outputs }) });
    if (outcomes.some((o) => o.r.status === 'skip')) { result.skipped++; continue; }
    for (const { run, r } of outcomes) {
      result.runs++;
      if (r.status !== 'ok') { result.errors++; result.failures.push({ index: i, source: source(i), run: run.label, matmul: run.matmul, status: r.status, error: r.error }); continue; }
      if (r.fusedNodes > 0) result.fusedRuns++;
      for (const k of r.kinds) result.byKind[k] = (result.byKind[k] ?? 0) + 1;
      for (const p of r.patterns) result.byPattern[p] = (result.byPattern[p] ?? 0) + 1;
      if (r.unfused.pass) result.unfusedPassed++;
      if (r.fused.pass) result.fusedPassed++;
      if (r.bitwiseSame) result.bitwiseSame++;
      result.worstFused = Math.max(result.worstFused, r.fused.worst);
      result.worstUnfused = Math.max(result.worstUnfused, r.unfused.worst);
      if (!r.fused.pass || !r.unfused.pass) {
        result.failures.push({ index: i, source: source(i), run: run.label, matmul: run.matmul, fused: { pass: r.fused.pass, worst: r.fused.worst, first: r.fused.first },
          unfused: { pass: r.unfused.pass, worst: r.unfused.worst, first: r.unfused.first } });
      }
    }
    if (i % 25 === 0) progress?.(`fuzz ${i}/${programs}`);
  }
  return result;
}

// ---- 3. the fused training step, node-local -----------------------------------------

/**
 * Node-local expectations for a fused graph: every launched node judged on the
 * float32-rounded reference values of its own operands. A fused node is
 * expected to give its body's float64 result, within amendment 8's bound
 * composed over its body; a plain node within its op's own bound.
 */
export function localReferenceFused(graph, values) {
  const expected = new Array(graph.nodes.length), bound = new Array(graph.nodes.length);
  for (const node of graph.nodes) {
    if (node.op === 'input' || node.op === 'const') continue;
    let g, out;
    if (node.op === 'fused') {
      g = bodyOf(node);
      out = node.attrs.output;
    } else {
      g = new Graph();
      const ins = node.inputs.map((id, k) => g.input(`a${k}`, graph.nodes[id].shape));
      out = g.emit(node.op, ins, node.attrs);
    }
    const r = referenceWithBounds(g, Object.fromEntries(node.inputs.map((id, k) => [`a${k}`, values[id]])));
    expected[node.id] = r.values[out];
    bound[node.id] = r.bounds[out];
  }
  return { expected, bound };
}

function judgeLocal(graph, gpu, expected, bound) {
  let worst = 0, first = null, unjudgeableCount = 0, elements = 0;
  const byKind = {};
  for (const node of graph.nodes) {
    if (!expected[node.id]) continue;
    const g = gpu[node.id], r = expected[node.id], e = bound[node.id];
    const kind = node.op === 'fused' ? node.attrs.pattern : 'plain';
    const k = (byKind[kind] ??= { nodes: 0, worst: 0 });
    k.nodes++;
    for (let i = 0; i < r.length; i++) {
      elements++;
      if (!Number.isFinite(r[i]) || Math.abs(r[i]) > F32_MAX || !Number.isFinite(e[i])) { unjudgeableCount++; continue; }
      const d = Math.abs(g[i] - r[i]);
      const ratio = d === 0 ? 0 : (Number.isNaN(d) ? Infinity : d / (S * e[i]));
      if (ratio > worst) worst = ratio;
      if (ratio > k.worst) k.worst = ratio;
      if (ratio > 1 && first === null) first = { node: node.id, op: node.op, pattern: node.attrs.pattern ?? null, index: i, gpu: g[i], ref: r[i], bound: e[i], ratio };
    }
  }
  const unjudged = unjudgeableCount > 0.01 * elements;
  return { pass: first === null && !unjudged, unjudged, worst, first, unjudgeable: unjudgeableCount, elements, byKind };
}

export async function runTeacherForcedM4(runner, spec, { sabotage, configs = spec.golden, steps, progress } = {}) {
  const syn = spec.synthetic.train;
  const pass = SABOTAGES_M4.find((s) => s.name === sabotage)?.where === 'pass';
  const runs = [];
  for (const g of configs) {
    const n = steps ?? g.steps;
    const step = buildStep3b(g.V, g.batch);
    let fs;
    try {
      fs = fusedStep(step, pass ? { passSabotage: sabotage } : { sabotage });
    } catch (error) {
      runs.push({ name: g.name, steps: 0, errors: 0, passed: 0, unjudged: 0, judgedFailures: 0, worst: 0, fuseError: String(error.message ?? error), detail: [] });
      continue;
    }
    const batchAt = (t) => g.schedule[t].map((i) => syn[i]);
    const cpu = trainCpu3b({ V: g.V, B: g.batch, init: initParams(g.V, g.seed), steps: n, batchAt, keepParams: true, keepState: true });
    const initState = adamInitialState(step.names, paramShapes(g.V));
    const out = [];
    for (let t = 0; t < n; t++) {
      const params = t === 0 ? initParams(g.V, g.seed) : cpu.params[t - 1];
      const state = t === 0 ? initState : cpu.states[t - 1];
      const { x, y } = encodeBatch(batchAt(t), g.V);
      const ref = referenceWithBounds(step.graph, { ...params, ...state, x, y, [step.cotangent]: [1] });
      const values = fs.graph.nodes.map((node) => ref.values[fs.fusion.origin[node.id]]);
      const entry = { t };
      try {
        const local = localReferenceFused(fs.graph, values);
        const gpuLocal = await runLocal(runner, fs.graph, fs.lowered, values);
        const j = judgeLocal(fs.graph, gpuLocal, local.expected, local.bound);
        entry.local = { pass: j.pass, unjudged: j.unjudged, worst: j.worst, unjudgeable: j.unjudgeable, elements: j.elements, byKind: j.byKind,
          first: j.first ? { op: j.first.op, pattern: j.first.pattern, node: j.first.node, ratio: j.first.ratio } : null };
      } catch (error) {
        entry.error = String(error.message ?? error);
      }
      out.push(entry);
      progress?.(`fused teacher-forced${sabotage ? ` [${sabotage}]` : ''} ${g.name} step ${t + 1}/${n}`);
    }
    runs.push({ name: g.name, steps: out.length, errors: out.filter((s) => s.error).length,
      passed: out.filter((s) => s.local?.pass).length, unjudged: out.filter((s) => s.local?.unjudged).length,
      judgedFailures: out.filter((s) => s.local?.first).length, worst: Math.max(0, ...out.map((s) => s.local?.worst ?? 0)),
      launches: fs.lowered.launches.length, unfusedLaunches: lower(step.graph).launches.length, detail: out });
  }
  return runs;
}

/**
 * Revision R5: the fused forward pass (logits and loss from parameters), node-local,
 * on the four golden configs at initialisation with their first batch. Unlike the
 * training step it contains the full LayerNorm, softmax and GELU-epilogue kernels.
 */
export async function runForwardLocalM4(runner, spec, { sabotage, configs = spec.golden, progress } = {}) {
  const syn = spec.synthetic.train;
  const pass = SABOTAGES_M4.find((s) => s.name === sabotage)?.where === 'pass';
  const out = [];
  for (const g of configs) {
    const f = buildForward(g.V, g.batch);
    let fused;
    try {
      fused = fuse(f.graph, { keep: [f.logits, f.loss], sabotage: pass ? sabotage : undefined });
    } catch (error) {
      out.push({ name: g.name, fuseError: String(error.message ?? error) });
      continue;
    }
    const lowered = lowerFused(fused.graph, { sabotage: pass ? undefined : sabotage });
    const { x, y } = encodeBatch(g.schedule[0].map((i) => syn[i]), g.V);
    const ref = referenceWithBounds(f.graph, { ...initParams(g.V, g.seed), x, y });
    const values = fused.graph.nodes.map((node) => ref.values[fused.origin[node.id]]);
    const entry = { name: g.name, patterns: fusedPatterns(fused), launches: lowered.launches.length, unfusedLaunches: lower(f.graph).launches.length };
    try {
      const local = localReferenceFused(fused.graph, values);
      const gpu = await runLocal(runner, fused.graph, lowered, values);
      const j = judgeLocal(fused.graph, gpu, local.expected, local.bound);
      Object.assign(entry, { pass: j.pass, unjudged: j.unjudged, worst: j.worst, unjudgeable: j.unjudgeable, elements: j.elements, byKind: j.byKind,
        first: j.first ? { op: j.first.op, pattern: j.first.pattern, node: j.first.node, ratio: j.first.ratio } : null });
    } catch (error) {
      entry.error = String(error.message ?? error);
    }
    out.push(entry);
    progress?.(`fused forward node-local${sabotage ? ` [${sabotage}]` : ''} ${g.name}`);
  }
  return out;
}

// ---- 4. trainer path against the checked path -------------------------------------------

export async function runConsistencyM4(runner, spec, steps = 3) {
  const g = spec.golden[0];
  const syn = spec.synthetic.train;
  const fs = fusedStep(buildStep3b(g.V, g.batch));
  const names = [...fs.names, ...fs.stateNames];
  const init = fround(initParams(g.V, g.seed)), state = adamInitialState(fs.names, paramShapes(g.V));
  const trainer = new FusedTrainer(runner, fs, init, state);
  let inputs = { ...init, ...state };
  let identical = true;
  for (let t = 0; t < steps; t++) {
    const { x, y } = encodeBatch(g.schedule[t].map((i) => syn[i]), g.V);
    const a = await trainer.train(x, y);
    const b = await runner.run(fs.graph, fs.lowered, { ...inputs, x, y, [fs.cotangent]: [1] });
    if (a !== b.values[fs.loss][0]) identical = false;
    inputs = Object.fromEntries(names.map((n) => [n, b.values[fs.outputs[n]]]));
  }
  const fromTrainer = await trainer.read(names);
  for (const n of names) if (fromTrainer[n].some((v, i) => v !== inputs[n][i])) identical = false;
  trainer.destroy();
  return { config: g.name, steps, identical };
}

// ---- 5. statistical -------------------------------------------------------------------

export async function runStatisticalM4(runner, spec, { progress } = {}) {
  const syn = spec.synthetic;
  const st = spec.statistical;
  const V = syn.vocab;
  const fs = fusedStep(buildStep3b(V, st.batch));
  const out = [];
  for (const run of st.runs) {
    const trainer = new FusedTrainer(runner, fs, fround(initParams(V, run.seed)), adamInitialState(fs.names, paramShapes(V)));
    const losses = [];
    for (let t = 0; t < st.steps; t++) {
      const { x, y } = encodeBatch(run.schedule[t].map((i) => syn.train[i]), V);
      losses.push(await trainer.train(x, y));
    }
    const final = await trainer.read(fs.names);
    trainer.destroy();
    out.push({ seed: run.seed, valLoss: evalLossCpu(V, final, syn.val), finalTrainLoss: losses.at(-1), losses });
    progress?.(`statistical (fused) seed ${run.seed}/${st.runs.length}`);
  }
  return out;
}

// ---- 6. sabotage ------------------------------------------------------------------------

/** Does sabotage `s` have a target in this fused graph? */
function targets(s, kinds, patterns) {
  if (s.where === 'pass') return true;
  if (s.kinds) return s.kinds.some((k) => kinds.includes(k));
  return s.patterns.some((p) => patterns.includes(p));
}

export async function runSabotageM4(runner, spec, honestDirect, { programs = SABOTAGE_FUZZ_PROGRAMS, progress } = {}) {
  const out = [];
  for (const s of SABOTAGES_M4) {
    const direct = await runDirectM4(runner, { sabotage: s.name, progress });
    // Caught: a case that passed honestly now fails on its values, canaries or a GPU error.
    // A sabotaged pass that throws is counted apart, as a loud failure of the pass itself.
    const caught = direct.filter((d, k) => honestDirect[k].pass && d.status !== 'fuse-error' && d.valuesPass !== true);
    const threw = direct.filter((d) => d.status === 'fuse-error').length;
    const fuzz = { general: { containing: 0, caught: 0 }, targeted: { containing: 0, caught: 0 }, skipped: 0, fuseErrors: 0, firstCatch: null };
    for (let i = 1; i <= programs; i++) {
      let containing = false, isCaught = false, skipped = false, threwHere = false;
      for (const run of programRunsM4(i)) {
        const honest = fuse(run.graph, { keep: run.outputs });
        if (!targets(s, fusedKinds(honest), fusedPatterns(honest))) continue;
        const r = await runPair(runner, run.graph, run.inputs, { matmul: run.matmul, keep: run.outputs, sabotage: s.name, unfused: false });
        if (r.status === 'skip') { skipped = true; break; }
        containing = true;
        if (r.status === 'fuse-error') threwHere = true;
        else if (r.status !== 'ok' || !r.fused.pass) isCaught = true;
      }
      if (skipped) { fuzz.skipped++; continue; }
      if (!containing) continue;
      fuzz[source(i)].containing++;
      if (threwHere) fuzz.fuseErrors++;
      if (isCaught) { fuzz[source(i)].caught++; fuzz.firstCatch ??= i; }
      if (i % 50 === 0) progress?.(`sabotage ${s.name}: fuzz ${i}/${programs}`);
    }
    const [tf] = await runTeacherForcedM4(runner, spec, { sabotage: s.name, configs: [spec.golden[0]], steps: 2, progress });
    const fwd = await runForwardLocalM4(runner, spec, { sabotage: s.name, progress });
    out.push({ name: s.name, where: s.where,
      direct: { cases: direct.length, caught: caught.length, fuseErrors: threw, caughtCases: caught.map((d) => d.name),
        how: caught.map((d) => (d.status !== 'ok' ? d.status : d.canaryOk === false && !d.first ? 'canary' : 'values')) },
      fuzz,
      teacherForced: { trials: tf.steps, caught: tf.judgedFailures + tf.errors, fuseError: tf.fuseError ?? null,
        detail: tf.detail.map((d) => ({ t: d.t, local: d.local ? { pass: d.local.pass, worst: d.local.worst, first: d.local.first } : null, error: d.error ?? null })) },
      forward: { trials: fwd.length, caught: fwd.filter((r) => r.first || r.error).length, fuseErrors: fwd.filter((r) => r.fuseError).length,
        detail: fwd.map((r) => ({ name: r.name, pass: r.pass ?? null, worst: r.worst ?? null, first: r.first ?? null, error: r.error ?? null, fuseError: r.fuseError ?? null })) } });
  }
  return out;
}
