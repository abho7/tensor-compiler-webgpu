// What PyTorch is asked to reproduce for milestone 3b (DESIGN.md amendments
// 23, 24 and 26), written to golden/spec-m3b.json by tools/export-spec-m3b.js.
// Initial parameters and projection vectors are not stored: the oracle
// regenerates them with a port of this project's RNG and must match the
// SHA-256 recorded here before it runs anything.

import { createHash } from 'node:crypto';
import { CONFIG, initParams, paramSpec, count } from './model.js';
import { ADAM_3B } from './step.js';
import { GOLDEN_3B, TEXT_INIT_3B, STAT_3B, DEMO_3B, PROJECTIONS, projectionVector } from './config.js';
import { SYNTHETIC_3B, TEXT, periodicCopy, syntheticSchedule, encodeText, textSchedule, textValOffsets } from './data.js';

/** SHA-256 of float64 little-endian bytes of `arrays`, in order. */
export function sha256F64(arrays) {
  const h = createHash('sha256');
  for (const a of arrays) h.update(Buffer.from(Float64Array.from(a).buffer));
  return h.digest('hex');
}

export const initSha = (V, seed) => {
  const p = initParams(V, seed);
  return sha256F64(paramSpec(V).map(([n]) => p[n]));
};

export const projectionSha = (V) => sha256F64(paramSpec(V).flatMap(([n, s]) =>
  Array.from({ length: PROJECTIONS }, (_, k) => projectionVector(n, count(s), k))));

export function buildSpec3b(aliceText) {
  const V = SYNTHETIC_3B.vocab;
  const text = encodeText(aliceText);
  const Vt = text.vocab.length;
  return {
    config: CONFIG, optimizer: ADAM_3B, paramOrder: paramSpec(V).map(([n, s, i]) => ({ name: n, shape: s, init: i })),
    synthetic: { ...SYNTHETIC_3B, ...periodicCopy() },
    golden: GOLDEN_3B.seeds.map((seed) => ({
      name: `golden-seed${seed}`, seed, V, batch: GOLDEN_3B.batch, steps: GOLDEN_3B.steps,
      schedule: syntheticSchedule(GOLDEN_3B.batch, GOLDEN_3B.steps, seed), initSha: initSha(V, seed),
    })),
    projections: { k: PROJECTIONS, sha256: projectionSha(V) },
    text: {
      ...TEXT, vocab: text.vocab, V: Vt, n: text.ids.length, nTrain: text.nTrain,
      init: { ...TEXT_INIT_3B, offsets: textSchedule(text.nTrain, TEXT_INIT_3B.batch, 1, TEXT_INIT_3B.seed)[0], initSha: initSha(Vt, TEXT_INIT_3B.seed) },
    },
    statistical: {
      ...STAT_3B,
      runs: Array.from({ length: STAT_3B.seeds }, (_, i) => ({
        seed: i + 1, schedule: syntheticSchedule(STAT_3B.batch, STAT_3B.steps, i + 1), initSha: initSha(V, i + 1),
      })),
    },
    demo: {
      ...DEMO_3B, V: Vt, initSha: initSha(Vt, DEMO_3B.seed),
      offsets: textSchedule(text.nTrain, DEMO_3B.batch, DEMO_3B.steps, DEMO_3B.seed),
      valOffsets: textValOffsets(text.ids.length, text.nTrain),
    },
  };
}
