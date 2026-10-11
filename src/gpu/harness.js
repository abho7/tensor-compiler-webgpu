// Milestone 2's GPU checks, run in the browser by web/gpu.html:
//   1. direct per-op adversarial tests (the correctness claim per kernel)
//   2. differential fuzzing of forward and gradient graphs (supporting evidence)
//   3. sabotage: each injected kernel bug against both
// Every comparison is per element under the float32 budget (DESIGN.md
// amendment 8). Nothing is timed.

import { grad } from '../autodiff.js';
import { size } from '../ir/shapes.js';
import { Rng } from '../testing/rng.js';
import { corpusProgram, TARGETS } from '../testing/random-graph.js';
import { referenceWithBounds, judge, unjudgeable } from './bounds.js';
import { lower } from './lower.js';
import { directCases } from './direct-cases.js';
import { ELEMENTWISE_OPS } from './codegen.js';

export const FUZZ_PROGRAMS = 2000;

/** The injected bugs, in the order the brief lists them. `kinds`: the kernels a bug can affect. */
export const SABOTAGES = [
  { name: 'tiled-off-by-one', label: 'tiled matmul: off-by-one at the tile edge', kinds: ['matmul-tiled'], bug: true },
  { name: 'transpose-swapped-stride', label: 'transpose: the last two source strides swapped', kinds: ['gather'], ops: ['transpose'], bug: true },
  { name: 'elementwise-no-bounds-check', label: 'elementwise: no bounds check in the last workgroup', kinds: ELEMENTWISE_OPS, bug: true },
  { name: 'broadcast-wrong-stride', label: 'broadcast: a stretched axis reads with its natural stride, not 0', kinds: ['gather'], ops: ['broadcast'], bug: true },
  { name: 'sum-off-by-one', label: 'sum(axis): reduction extent one short', kinds: ['sum'], bug: true },
  { name: 'reshape-stride-mismatch', label: "reshape: the source's strides used for the output's axes", kinds: ['gather'], ops: ['reshape'], bug: true },
  { name: 'softmax-no-max', label: 'softmax: the maximum never subtracted', kinds: ['max'], ops: ['max'], bug: true },
  { name: 'matmul-reorder', label: 'matmul: accumulation in reverse order (mathematically correct)', kinds: ['matmul-naive', 'matmul-tiled'], bug: false },
];

const source = (i) => (i % 2 === 1 ? 'general' : 'targeted');

async function runChecked(runner, graph, rawInputs, outputs, { matmul, sabotage, canary = false } = {}) {
  const ref = referenceWithBounds(graph, rawInputs);
  if (unjudgeable(ref.values, ref.bounds) > 0) return { status: 'skip' };
  const lowered = lower(graph, { matmul, sabotage });
  let gpu;
  try {
    gpu = await runner.run(graph, lowered, ref.inputs, { canary });
  } catch (error) {
    return { status: 'error', error: String(error.message ?? error) };
  }
  const j = judge(graph, gpu.values, ref.values, ref.bounds);
  // The verdict is per node (amendment 8). Whether the outputs alone would have
  // shown it is recorded too: a node-level miscompile can leave outputs right.
  const outputsOnly = judge(graph, gpu.values, ref.values, ref.bounds, { only: new Set(outputs) });
  const canaryOk = !canary || gpu.canaryIntact.every(Boolean);
  const pass = j.pass && canaryOk;
  return { status: pass ? 'pass' : 'fail', outputsPass: outputsOnly.pass, worst: j.worst, first: j.first, canaryOk, kinds: [...new Set(lowered.launches.map((l) => l.kind))] };
}

// ---- 1. direct tests --------------------------------------------------------

export async function runDirect(runner, { sabotage, progress } = {}) {
  const cases = directCases();
  const out = [];
  for (const [k, c] of cases.entries()) {
    const r = await runChecked(runner, c.graph, c.inputs, c.outputs, { matmul: c.matmul, sabotage, canary: true });
    out.push({ name: c.name, op: c.op, kernel: c.kernel, status: r.status, worst: r.worst ?? null, first: r.first ?? null, canaryOk: r.canaryOk ?? null, error: r.error ?? null });
    progress?.(`direct${sabotage ? ` [${sabotage}]` : ''} ${k + 1}/${cases.length}`);
  }
  return out;
}

// ---- 2. fuzzing -------------------------------------------------------------

/** The runs for corpus program i: forward and gradient graphs, under each matmul variant it needs. */
function programRuns(i) {
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
  return { program: p, runs, ops: new Set([...p.graph.nodes, ...G.graph.nodes].map((n) => n.op)) };
}

export async function runFuzz(runner, { programs = FUZZ_PROGRAMS, progress } = {}) {
  const result = { programs, skipped: 0, runs: 0, passed: 0, failed: 0, errors: 0, worst: 0, failures: [], skips: [],
    bySource: { general: { programs: 0, skipped: 0 }, targeted: { programs: 0, skipped: 0 } }, opsChecked: {} };
  for (let i = 1; i <= programs; i++) {
    const { runs, ops } = programRuns(i);
    result.bySource[source(i)].programs++;
    const outcomes = [];
    for (const run of runs) outcomes.push({ run, r: await runChecked(runner, run.graph, run.inputs, run.outputs, { matmul: run.matmul }) });
    if (outcomes.some((o) => o.r.status === 'skip')) {
      // A program is skipped whole if any of its runs cannot be judged (amendment 8).
      result.skipped++;
      result.bySource[source(i)].skipped++;
      result.skips.push(i);
      continue;
    }
    for (const op of ops) result.opsChecked[op] = (result.opsChecked[op] ?? 0) + 1;
    for (const { run, r } of outcomes) {
      result.runs++;
      if (r.status === 'pass') result.passed++;
      else if (r.status === 'error') result.errors++;
      else result.failed++;
      if (r.worst > result.worst) result.worst = r.worst;
      if (r.status !== 'pass') {
        result.failures.push({ index: i, source: source(i), run: run.label, matmul: run.matmul, status: r.status, worst: r.worst ?? null, first: r.first ?? null, error: r.error ?? null });
      }
    }
    if (i % 25 === 0) progress?.(`fuzz ${i}/${programs}`);
  }
  return result;
}

// ---- 3. sabotage ------------------------------------------------------------

function affects(sab, ops, kinds) {
  if (sab.ops) return sab.ops.some((op) => ops.has(op));
  return sab.kinds.some((k) => kinds.has(k));
}

export async function runSabotage(runner, honestDirect, { programs = FUZZ_PROGRAMS, progress } = {}) {
  const out = [];
  for (const sab of SABOTAGES) {
    const direct = await runDirect(runner, { sabotage: sab.name, progress });
    // Caught by a direct case: it fails now and passed with the honest kernels.
    const caughtBy = direct.filter((d, k) => d.status !== 'pass' && honestDirect[k].status === 'pass');
    const fuzz = { general: { containing: 0, caught: 0, caughtByOutputs: 0 }, targeted: { containing: 0, caught: 0, caughtByOutputs: 0 }, firstCatch: null, firstCatchSource: null, skipped: 0 };
    for (let i = 1; i <= programs; i++) {
      const { runs, ops } = programRuns(i);
      const kinds = new Set();
      for (const run of runs) for (const l of lower(run.graph, { matmul: run.matmul }).launches) kinds.add(l.kind);
      if (!affects(sab, ops, kinds)) continue;
      // The tiled-matmul bug can only show in the tiled runs.
      const relevant = sab.name === 'tiled-off-by-one' ? runs.filter((r) => r.matmul === 'tiled') : runs;
      let caught = false, caughtByOutputs = false, skipped = false;
      for (const run of relevant) {
        const r = await runChecked(runner, run.graph, run.inputs, run.outputs, { matmul: run.matmul, sabotage: sab.name });
        if (r.status === 'skip') { skipped = true; break; }
        if (r.status !== 'pass') caught = true;
        if (r.status === 'error' || r.outputsPass === false) caughtByOutputs = true;
      }
      if (skipped) { fuzz.skipped++; continue; }
      fuzz[source(i)].containing++;
      if (caughtByOutputs) fuzz[source(i)].caughtByOutputs++;
      if (caught) {
        fuzz[source(i)].caught++;
        if (fuzz.firstCatch === null) { fuzz.firstCatch = i; fuzz.firstCatchSource = source(i); }
      }
      if (i % 50 === 0) progress?.(`sabotage ${sab.name}: fuzz ${i}/${programs}`);
    }
    out.push({
      name: sab.name, label: sab.label, bug: sab.bug,
      direct: { cases: direct.length, caught: caughtBy.length, first: caughtBy[0]?.name ?? null,
        firstHow: caughtBy[0] ? (caughtBy[0].status === 'error' ? 'error' : caughtBy[0].canaryOk === false && caughtBy[0].first === null ? 'canary' : 'values') : null,
        caughtCases: caughtBy.map((d) => d.name) },
      fuzz,
    });
  }
  return out;
}

export { TARGETS };
