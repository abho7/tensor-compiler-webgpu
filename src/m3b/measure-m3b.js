// Milestone 3b's figures for golden/measurements.json: everything that needs
// no GPU is computed from scratch (the PyTorch comparisons, the properties on
// the interpreter, the sabotage against them, and revision R3's measurement of
// the vacuous bound); the GPU recordings are summarised, never recomputed.

import { referenceWithBounds, F32_MAX } from '../gpu/bounds.js';
import { tost } from '../train/equivalence.js';
import { CONFIG, MODEL_BUGS, initParams, buildForward, paramCount } from './model.js';
import { buildStep3b, paramShapes } from './step.js';
import { adamInitialState } from './adam.js';
import { encodeBatch, encodeText, periodicCopyFloor } from './data.js';
import { verifyCpu3b } from './verify-cpu.js';

export const STAT_MARGIN = 0.02;

/**
 * Revision R3's figures: on golden run 1's first step, how much of the step
 * the composed bound can judge, where it first becomes unbounded, and how the
 * largest bound grows along block 0's residual stream; and how many forward
 * logits have a finite bound at initialisation.
 */
export function vacuity(spec, textIds) {
  const g = spec.golden[0];
  const step = buildStep3b(g.V, g.batch);
  const { x, y } = encodeBatch(g.schedule[0].map((i) => spec.synthetic.train[i]), g.V);
  const ref = referenceWithBounds(step.graph, { ...initParams(g.V, g.seed), ...adamInitialState(step.names, paramShapes(g.V)), x, y, [step.cotangent]: [1] });
  let elements = 0, unjudgeable = 0, firstUnbounded = null;
  for (const n of step.graph.nodes) {
    const v = ref.values[n.id], e = ref.bounds[n.id];
    for (let i = 0; i < v.length; i++) {
      elements++;
      if (!Number.isFinite(v[i]) || Math.abs(v[i]) > F32_MAX || !Number.isFinite(e[i])) {
        unjudgeable++;
        if (firstUnbounded === null) firstUnbounded = { node: n.id, op: n.op };
      }
    }
  }
  const maxBound = (id) => Math.max(...ref.bounds[id]);
  // The residual stream after the embedding, after block 0's attention, and
  // after block 0's MLP (test/m3b-record.test.js checks these ids structurally).
  const stream = { embedding: 38, afterAttention0: 91, afterMlp0: 136 };
  const logits = [[g.V, spec.synthetic.train[0]], [spec.text.V, textIds.slice(spec.text.init.offsets[0], spec.text.init.offsets[0] + CONFIG.T + 1)]].map(([V, seq]) => {
    const f = buildForward(V, 1);
    const enc = encodeBatch([seq], V);
    const r = referenceWithBounds(f.graph, { ...initParams(V, 1), ...enc });
    return { V, logits: r.bounds[f.logits].length, finite: Array.from(r.bounds[f.logits]).filter(Number.isFinite).length };
  });
  return {
    run: g.name, step: 0, nodes: step.graph.nodes.length, elements, unjudgeable, firstUnbounded,
    streamIds: stream, streamMaxBound: Object.fromEntries(Object.entries(stream).map(([k, id]) => [k, maxBound(id)])), logits,
  };
}

/** The pre-registered predictions of amendment 27, for checking against what happened. */
const PREDICTED = {
  causality: ['mask-off-by-one', 'softmax-wrong-axis', 'merge-no-transpose'],
  batchOrder: [],
  torchInit: Object.keys(MODEL_BUGS),
  gpuTeacherForced: [],
};
export const CHECK_ORDER_3B = ['causality', 'batchOrder', 'torchInit', 'torchTraining', 'gpuWholeStep', 'gpuNodeLocal'];

function summarizeGpu(cpu, record, demo, torch) {
  const tf = record.teacherForced;
  const total = (k, f) => tf.reduce((a, r) => a + r[k][f], 0);
  const stepsTotal = tf.reduce((a, r) => a + r.steps, 0);
  const teacherForced = {
    runs: tf.length, steps: stepsTotal, errors: tf.reduce((a, r) => a + r.errors, 0),
    whole: { passed: total('whole', 'passed'), unjudged: total('whole', 'unjudged'), judgedFailures: total('whole', 'judgedFailures'),
      worstJudged: Math.max(...tf.map((r) => r.whole.worst)), unjudgeable: total('whole', 'unjudgeable'), elements: total('whole', 'elements') },
    local: { passed: total('local', 'passed'), unjudged: total('local', 'unjudged'), judgedFailures: total('local', 'judgedFailures'),
      worst: Math.max(...tf.map((r) => r.local.worst)), unjudgeable: total('local', 'unjudgeable'), elements: total('local', 'elements') },
    observed: { maxLossDiff: Math.max(...tf.map((r) => r.observed.maxLossDiff)), maxParamRel: Math.max(...tf.map((r) => r.observed.maxParamRel)) },
    perRun: tf.map((r) => ({ name: r.name, steps: r.steps, whole: r.whole, local: r.local, observed: r.observed })),
  };

  const t64 = Object.fromEntries(torch.statistical.float64.map((r) => [r.seed, r]));
  const t32 = Object.fromEntries(torch.statistical.float32.map((r) => [r.seed, r]));
  const gpuRuns = record.statistical;
  const statistical = {
    seeds: gpuRuns.length, margin: STAT_MARGIN, alpha: 0.05,
    gpuVsTorch64: tost(gpuRuns.map((r) => r.valLoss - t64[r.seed].valLoss), STAT_MARGIN),
    torch32VsTorch64: tost(torch.statistical.float32.map((r) => r.valLoss - t64[r.seed].valLoss), STAT_MARGIN),
    meanValLoss: { gpu: mean(gpuRuns.map((r) => r.valLoss)), torch64: mean(torch.statistical.float64.map((r) => r.valLoss)), torch32: mean(torch.statistical.float32.map((r) => r.valLoss)) },
    floor: periodicCopyFloor(),
    seed1: { gpu: gpuRuns.find((r) => r.seed === 1)?.losses ?? null, torch64: t64[1]?.losses ?? null, torch32: t32[1]?.losses ?? null },
  };

  const gpuSab = Object.fromEntries(record.sabotage.map((s) => [s.name, s]));
  const bugs = cpu.sabotage.map((s) => {
    const g = gpuSab[s.name];
    const checks = { ...s.checks, gpuWholeStep: { trials: g.trials, caught: g.caughtWhole }, gpuNodeLocal: { trials: g.trials, caught: g.caughtLocal } };
    const caughtFirstBy = CHECK_ORDER_3B.find((k) => checks[k].caught > 0) ?? null;
    const asPredicted = {
      causality: (checks.causality.caught > 0) === PREDICTED.causality.includes(s.name),
      batchOrder: (checks.batchOrder.caught > 0) === PREDICTED.batchOrder.includes(s.name),
      torchInit: (checks.torchInit.caught > 0) === PREDICTED.torchInit.includes(s.name),
      gpuTeacherForced: (checks.gpuWholeStep.caught + checks.gpuNodeLocal.caught > 0) === PREDICTED.gpuTeacherForced.includes(s.name),
    };
    return { name: s.name, bug: s.bug, checks, caughtFirstBy, asPredicted };
  });

  const props = record.properties;
  return {
    sourceHash: record.sourceHash, recorded: record.recorded, adapter: { vendor: record.adapter.vendor, architecture: record.adapter.architecture, userAgent: record.adapter.userAgent },
    selfTest: record.selfTest.map((s) => ({ name: s.name, wholeJudgedFailure: !!s.whole?.first, localCaught: !!s.local?.first, localFirstOp: s.local?.first?.op ?? null, localWorst: s.local?.worst ?? null, error: s.error })),
    teacherForced,
    properties: {
      causality: { trials: props.causality.length, passed: props.causality.filter((c) => c.pass).length },
      batchOrder: { trials: props.batchOrder.length, passed: props.batchOrder.filter((c) => c.logitsPermuted).length },
    },
    consistency: record.consistency,
    statistical,
    sabotage: { checkOrder: CHECK_ORDER_3B, predicted: PREDICTED, bugs },
    demo: demo && {
      sourceHash: demo.sourceHash, recorded: demo.recorded, adapter: { vendor: demo.adapter.vendor, architecture: demo.adapter.architecture },
      steps: demo.steps, losses: demo.losses, torchLosses64: torch.demo.float64.losses, torchLosses32: torch.demo.float32.losses, valLoss: demo.valLoss, torchValLoss64: torch.demo.float64.valLoss, torchValLoss32: torch.demo.float32.valLoss,
      maxLossDifferenceFromTorch64: demo.maxLossDifferenceFromTorch64,
      torch32VsTorch64MaxLossDiff: Math.max(...torch.demo.float64.losses.map((l, t) => Math.abs(l - torch.demo.float32.losses[t]))),
      greedy: { k: demo.greedy.k, agree: demo.greedy.agree, firstDisagreement: demo.greedy.firstDisagreement, gap: demo.greedy.gap },
      generated: demo.generated,
      causalityGpu: { trials: demo.causalityGpu.length, passed: demo.causalityGpu.filter((c) => c.pass).length },
      causalityCpu: { trials: demo.causalityCpu.length, passed: demo.causalityCpu.filter((c) => c.pass).length },
      maxScore: demo.maxScore,
    },
  };
}

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

export async function measureM3b({ spec, torch, torchBytes, aliceText, gpuRecord, gpuDemo }) {
  const textIds = encodeText(aliceText).ids;
  const cpu = await verifyCpu3b(spec, torch, torchBytes, textIds);
  return {
    m3b: {
      producedBy: 'node tools/measure.js (src/m3b/measure-m3b.js)',
      model: { ...CONFIG, params: { synthetic: paramCount(spec.synthetic.vocab), text: paramCount(spec.text.V) }, vocab: { synthetic: spec.synthetic.vocab, text: spec.text.V } },
      torch: torch.torch,
      cpu,
      vacuity: vacuity(spec, textIds),
      gpu: gpuRecord ? summarizeGpu(cpu, gpuRecord, gpuDemo, torch) : null,
    },
  };
}
