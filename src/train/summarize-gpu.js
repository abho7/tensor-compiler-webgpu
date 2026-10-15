// Summarises milestone 3a's recorded GPU run (golden/gpu-m3.json) and demo run
// (golden/gpu-m3-demo.json) for golden/measurements.json, and assembles the
// sabotage table across every check (DESIGN.md amendment 18). The float64 side
// of the statistical runs is recomputed here, so CI checks the recorded CPU
// numbers against code it can run.

import { trainCpu, evaluateMlp } from './cpu.js';
import { syntheticSpirals, initMlp, batchSchedule } from './data.js';
import { SGD, ADAM, PARAMS } from './step.js';
import { trainScore } from './verify-cpu.js';
import { equivalence, MARGINS, ALPHA } from './equivalence.js';
import { sabotagedRules } from '../testing/sabotage.js';

const DIMS = { inputs: 8, hidden: 16, classes: 4 };

/** The checks in the order they run; "caught first" is the earliest that caught a bug. */
export const CHECK_ORDER = ['finiteDifferences', 'opGoldens', 'directKernel', 'trainingGoldens', 'gpuTeacherForced'];

/** The wrong sqrt rule against the PyTorch training goldens (predicted invisible: sqrt is never differentiated in training). */
function sqrtRuleAgainstTrainingGoldens(spec, torch) {
  const trials = spec.configs.map((cfg, k) => {
    const r = trainCpu({ dims: spec.dims, data: spec.synthetic.train, opt: cfg.optimizer, seed: cfg.seed, steps: cfg.steps, batch: cfg.batch,
      keepParams: true, init: cfg.init, schedule: cfg.schedule, rules: sabotagedRules('sqrt') });
    const ref = torch.configs[k];
    const first = r.losses.findIndex((l, t) => Math.max(trainScore([l], [ref.losses[t]]), ...PARAMS.map((p) => trainScore(r.params[t][p], ref.params[t][p]))) > 1);
    return { name: cfg.name, caught: first >= 0 };
  });
  return { trials: trials.length, caught: trials.filter((t) => t.caught).length };
}

function recomputeCpuStatistical(record) {
  const data = syntheticSpirals();
  let maxDiff = 0;
  for (const r of record.statistical) {
    const opt = r.optimizer === 'sgd' ? SGD : ADAM;
    const steps = 200, batch = 32;
    const cpu = trainCpu({ dims: DIMS, data: data.train, opt, seed: r.seed, steps, batch,
      init: initMlp(DIMS, r.seed), schedule: batchSchedule(data.train.y.length, batch, steps, r.seed) });
    const loss = evaluateMlp(DIMS, cpu.final, data.train).loss, acc = evaluateMlp(DIMS, cpu.final, data.test).accuracy;
    maxDiff = Math.max(maxDiff, Math.abs(loss - r.cpu.trainLoss), Math.abs(acc - r.cpu.testAccuracy));
  }
  return maxDiff;
}

export function summarizeM3Gpu({ record, demo, spec, torch, sqrt, cpuVsTorch }) {
  const tf = record.teacherForced;
  const stat = record.statistical;
  const rate = (caught, trials) => ({ caught, trials });
  const gpuSab = Object.fromEntries(record.sabotage.map((s) => [s.name, s]));
  const trainSab = Object.fromEntries(cpuVsTorch.sabotage.map((s) => [s.name, s]));
  const sqrtTrain = sqrtRuleAgainstTrainingGoldens(spec, torch);

  const table = ['adam-bias-correction', 'momentum-off-by-one', 'weight-decay-sign', 'sqrt-gradient', 'sqrt-bounds-off-by-one'].map((name) => {
    const g = gpuSab[name];
    const checks = {
      finiteDifferences: name === 'sqrt-gradient' ? rate(sqrt.finiteDifferences.caught, sqrt.finiteDifferences.containing) : null,
      opGoldens: name === 'sqrt-gradient' ? rate(sqrt.golden.flagged, sqrt.golden.cases) : null,
      directKernel: g.direct ? rate(g.direct.caught, g.direct.cases) : null,
      trainingGoldens: trainSab[name] ? rate(trainSab[name].caught, trainSab[name].trials)
        : name === 'sqrt-gradient' ? rate(sqrtTrain.caught, sqrtTrain.trials) : null,
      gpuTeacherForced: rate(g.teacherForced.caught, g.teacherForced.runs),
    };
    const first = CHECK_ORDER.find((c) => checks[c] && checks[c].caught > 0) ?? null;
    return { name, level: g.level, checks, caughtFirstBy: first };
  });

  return {
    recorded: record.recorded, adapter: record.adapter, sourceHash: record.sourceHash,
    direct: { cases: record.direct.length, pass: record.direct.filter((d) => d.status === 'pass').length,
      worst: Math.max(0, ...record.direct.filter((d) => d.status === 'pass').map((d) => d.worst)),
      failures: record.direct.filter((d) => d.status !== 'pass') },
    teacherForced: {
      runs: tf.length, steps: tf.reduce((a, r) => a + r.steps, 0), passed: tf.reduce((a, r) => a + r.passed, 0),
      skipped: tf.reduce((a, r) => a + r.skipped, 0), failed: tf.reduce((a, r) => a + r.failed, 0),
      worst: Math.max(...tf.map((r) => r.worst)), perRun: tf.map(({ name, passed, steps, worst, failures }) => ({ name, passed, steps, worst, failures })),
    },
    freeRunning: { k: record.freeRunning[0]?.steps.length ?? 0, runs: record.freeRunning },
    consistency: record.consistency,
    statistical: {
      seeds: stat.length / 2, steps: 200, margins: MARGINS, alpha: ALPHA,
      equivalence: equivalence(stat),
      runs: stat.map(({ optimizer, seed, gpu, cpu }) => ({ optimizer, seed, gpu, cpu })),
      curves: stat.filter((r) => r.gpuLosses).map(({ optimizer, seed, gpuLosses, cpuLosses }) => ({ optimizer, seed, gpuLosses, cpuLosses })),
      cpuRecomputeMaxDiff: recomputeCpuStatistical(record),
    },
    sabotage: { checkOrder: CHECK_ORDER, bugs: table },
    demo: demo ? {
      recorded: demo.recorded, adapter: demo.adapter, dims: spec.demo.dims, steps: demo.steps, losses: demo.losses,
      testAccuracy: demo.testAccuracy, testLoss: demo.testLoss, maxLossDifferenceFromTorch: demo.maxLossDifferenceFromTorch,
      torchTestAccuracy: torch.demo.testAccuracy, torchLosses: torch.demo.losses,
    } : null,
  };
}
