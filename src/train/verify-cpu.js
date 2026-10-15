// The float64 interpreter's training against PyTorch float64 (DESIGN.md
// amendment 16), and the optimizer sabotages against the same goldens
// (amendment 18). No GPU; deterministic, so CI recomputes it.

import { trainCpu, evaluateMlp } from './cpu.js';
import { PARAMS, OPTIMIZER_SABOTAGE } from './step.js';
import { parseOptdigits } from './data.js';

export const TRAIN_TOLERANCE = 1e-8;

/** Metric v2 with atolScale = rtol = 1e-8: the largest ratio of a difference to its allowance. */
export function trainScore(ours, ref) {
  let scale = 0;
  for (const v of ref) scale = Math.max(scale, Math.abs(v));
  const atol = TRAIN_TOLERANCE * Math.max(1, scale);
  let worst = 0;
  ref.forEach((r, i) => {
    const d = Math.abs(ours[i] - r);
    worst = Math.max(worst, Number.isNaN(d) ? Infinity : d / (atol + TRAIN_TOLERANCE * Math.abs(r)));
  });
  return worst;
}

function runConfig(spec, cfg, sabotage) {
  return trainCpu({ dims: spec.dims, data: spec.synthetic.train, opt: cfg.optimizer, seed: cfg.seed, steps: cfg.steps,
    batch: cfg.batch, keepParams: true, init: cfg.init, schedule: cfg.schedule, sabotage });
}

/** Per step, the worst ratio over the loss and every parameter tensor. */
function stepScores(run, ref) {
  return run.losses.map((loss, t) => Math.max(trainScore([loss], [ref.losses[t]]),
    ...PARAMS.map((p) => trainScore(run.params[t][p], ref.params[t][p]))));
}

/** Which golden runs each optimizer bug applies to. */
export const APPLIES = { 'adam-bias-correction': ['adam'], 'momentum-off-by-one': ['sgd'], 'weight-decay-sign': ['sgd', 'adam'] };

export function verifyCpuTraining(spec, torch, { optdigitsTrain, optdigitsTest } = {}) {
  const runs = spec.configs.map((cfg, k) => {
    const r = runConfig(spec, cfg);
    const scores = stepScores(r, torch.configs[k]);
    const worst = Math.max(...scores);
    const maxLossDiff = Math.max(...r.losses.map((l, t) => Math.abs(l - torch.configs[k].losses[t])));
    return { name: cfg.name, optimizer: cfg.optimizer.kind, steps: cfg.steps, worst, worstStep: scores.indexOf(worst), maxLossDiff, pass: worst <= 1 };
  });

  const sabotage = Object.entries(OPTIMIZER_SABOTAGE).map(([name, bug]) => {
    const trials = [];
    spec.configs.forEach((cfg, k) => {
      if (!APPLIES[name].includes(cfg.optimizer.kind)) return;
      const scores = stepScores(runConfig(spec, cfg, name), torch.configs[k]);
      const first = scores.findIndex((s) => s > 1);
      trials.push({ name: cfg.name, caught: first >= 0, firstStep: first >= 0 ? first : null });
    });
    return { name, bug, trials: trials.length, caught: trials.filter((t) => t.caught).length, detail: trials };
  });

  let demo = null;
  if (optdigitsTrain) {
    // Information only: amendment 16 pre-registers no tolerance for the demo.
    const train = parseOptdigits(optdigitsTrain), test = parseOptdigits(optdigitsTest);
    const d = spec.demo;
    const r = trainCpu({ dims: d.dims, data: train, opt: d.optimizer, seed: d.seed, steps: d.steps, batch: d.batch, init: d.init, schedule: d.schedule });
    const evalTest = evaluateMlp(d.dims, r.final, test);
    demo = {
      steps: d.steps, maxLossDiff: Math.max(...r.losses.map((l, t) => Math.abs(l - torch.demo.losses[t]))),
      cpuTestAccuracy: evalTest.accuracy, cpuTestLoss: evalTest.loss,
      torchTestAccuracy: torch.demo.testAccuracy, torchTestLoss: torch.demo.testLoss,
    };
  }
  return { tolerance: TRAIN_TOLERANCE, torch: torch.torch, runs, sabotage, demo };
}
