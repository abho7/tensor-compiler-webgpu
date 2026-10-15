// Milestone 3a's data (DESIGN.md amendment 15): the seeded synthetic set used
// for every golden comparison, the optdigits parser for the demo, the
// initialisation, and the batch schedule. Everything is a function of a seed,
// so both sides of a comparison see identical numbers.

import { Rng } from '../testing/rng.js';

export const SYNTHETIC = { seed: 3001, features: 8, classes: 4, train: 256, test: 256 };

/**
 * Four interleaved spiral arms in two dimensions, lifted to 8 features by a
 * fixed seeded linear map plus noise. Balanced across the 4 classes.
 * Returns { train, test }, each { x: number[n*8], y: number[n] }.
 */
export function syntheticSpirals({ seed, features, classes, train, test } = SYNTHETIC) {
  const rng = new Rng(seed);
  // The lift: a fixed 2 x features map.
  const lift = rng.array(2 * features, -1, 1);
  const make = (n) => {
    const x = [], y = [];
    for (let i = 0; i < n; i++) {
      const k = i % classes;
      const t = rng.uniform(0.05, 1);
      const angle = (2 * Math.PI * k) / classes + 3.5 * t + rng.uniform(-0.15, 0.15);
      const p = [t * Math.cos(angle), t * Math.sin(angle)];
      for (let f = 0; f < features; f++) x.push(p[0] * lift[f] + p[1] * lift[features + f] + rng.uniform(-0.05, 0.05));
      y.push(k);
    }
    return { x, y };
  };
  return { train: make(train), test: make(test) };
}

/**
 * Parse UCI optdigits CSV text: 64 counts (0 to 16) and a label per row.
 * Features are divided by 16, as amendment 15 states.
 */
export function parseOptdigits(text) {
  const x = [], y = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const v = line.split(',').map(Number);
    if (v.length !== 65 || v.some((n) => !Number.isFinite(n))) throw new Error(`bad optdigits row: ${line.slice(0, 40)}`);
    for (let f = 0; f < 64; f++) x.push(v[f] / 16);
    y.push(v[64]);
  }
  return { x, y };
}

/** Parameters as PyTorch's nn.Linear initialises them: uniform in +-1/sqrt(fan_in), weights and biases. */
export function initMlp({ inputs, hidden, classes }, seed) {
  const rng = new Rng((seed * 1000 + 7) >>> 0);
  const u = (n, fanIn) => rng.array(n, -1 / Math.sqrt(fanIn), 1 / Math.sqrt(fanIn));
  return {
    W1: u(inputs * hidden, inputs), b1: u(hidden, inputs),
    W2: u(hidden * classes, hidden), b2: u(classes, hidden),
  };
}

/**
 * The batch schedule: for each epoch a seeded permutation of the training
 * indices, cut into batches (a trailing short batch is dropped, so every step
 * has the same shape). Returns an array of index arrays, `steps` long.
 */
export function batchSchedule(n, batch, steps, seed) {
  const rng = new Rng((seed * 7919 + 3) >>> 0);
  const out = [];
  while (out.length < steps) {
    const perm = rng.shuffle([...Array(n).keys()]);
    for (let s = 0; s + batch <= n && out.length < steps; s += batch) out.push(perm.slice(s, s + batch));
  }
  return out;
}

/** Gather a batch: x as [batch * features], y as a one-hot [batch * classes]. */
export function gather(data, indices, features, classes) {
  const x = [], y = new Array(indices.length * classes).fill(0);
  indices.forEach((idx, b) => {
    for (let f = 0; f < features; f++) x.push(data.x[idx * features + f]);
    y[b * classes + data.y[idx]] = 1;
  });
  return { x, y };
}
