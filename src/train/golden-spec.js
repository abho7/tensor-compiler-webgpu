// The runs PyTorch is asked to reproduce (DESIGN.md amendments 15 and 16):
// four seeds x {SGD, Adam} x 50 steps on the synthetic set, and the demo run on
// optdigits. Initial parameters and batch indices are written out explicitly,
// so PyTorch starts from exactly the numbers this project's RNG produced.

import { syntheticSpirals, initMlp, batchSchedule } from './data.js';
import { SGD, ADAM } from './step.js';

export const GOLDEN_SEEDS = [1, 2, 3, 4];
export const GOLDEN_STEPS = 50;
export const SYNTH_DIMS = { inputs: 8, hidden: 16, classes: 4 };
export const SYNTH_BATCH = 32;

export const DEMO = {
  dims: { inputs: 64, hidden: 64, classes: 10 },
  optimizer: { kind: 'adam', lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0 },
  batch: 64, epochs: 10, seed: 1, trainRows: 3823,
  files: {
    train: { path: 'data/optdigits/optdigits.tra', sha256: 'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd' },
    test: { path: 'data/optdigits/optdigits.tes', sha256: '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8' },
  },
};
DEMO.steps = Math.floor(DEMO.trainRows / DEMO.batch) * DEMO.epochs;

export function buildTrainSpec() {
  const data = syntheticSpirals();
  const configs = [];
  for (const seed of GOLDEN_SEEDS) {
    for (const optimizer of [SGD, ADAM]) {
      configs.push({
        name: `${optimizer.kind}-seed${seed}`, seed, optimizer, steps: GOLDEN_STEPS, batch: SYNTH_BATCH,
        init: initMlp(SYNTH_DIMS, seed), schedule: batchSchedule(data.train.y.length, SYNTH_BATCH, GOLDEN_STEPS, seed),
      });
    }
  }
  const demo = { ...DEMO, init: initMlp(DEMO.dims, DEMO.seed), schedule: batchSchedule(DEMO.trainRows, DEMO.batch, DEMO.steps, DEMO.seed) };
  return { dims: SYNTH_DIMS, synthetic: data, configs, demo };
}
