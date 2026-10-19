// Milestone 3b's data (DESIGN.md amendment 24): the seeded "periodic copy"
// task for every golden comparison, and the demo text. Everything is a
// function of a seed, so both sides of a comparison see the same tokens.

import { Rng } from '../testing/rng.js';
import { batchSchedule } from '../train/data.js';
import { CONFIG } from './model.js';

export const SYNTHETIC_3B = { seed: 3101, vocab: 16, period: 16, train: 512, val: 128 };

export const TEXT = {
  path: 'data/alice/alice.txt',
  sha256: 'e3917589f48b4e9f4b731a6b7e127cf9ddd6e1b3819c78be02a2702ee73bfdb1',
  source: 'https://www.gutenberg.org/cache/epub/11/pg11.txt',
  sourceSha256: '01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e',
  trainFraction: 0.9, valWindows: 64,
};

const SEQ = CONFIG.T + 1;

/** Sequences of T + 1 tokens: a random block of `period` symbols, repeated. */
export function periodicCopy({ seed, vocab, period, train, val } = SYNTHETIC_3B) {
  const rng = new Rng(seed);
  const make = (n) => Array.from({ length: n }, () => {
    const block = Array.from({ length: period }, () => rng.int(0, vocab - 1));
    return Array.from({ length: SEQ }, (_, i) => block[i % period]);
  });
  return { train: make(train), val: make(val) };
}

/** The best possible mean loss on periodicCopy: only the first period - 1 targets are unpredictable. */
export const periodicCopyFloor = ({ vocab, period } = SYNTHETIC_3B) => ((period - 1) / CONFIG.T) * Math.log(vocab);

/** The synthetic batch schedule: milestone 3a's seeded per-epoch permutation of sequence indices. */
export const syntheticSchedule = (B, steps, seed) => batchSchedule(SYNTHETIC_3B.train, B, steps, seed);

/** Characters in code-point order, and the text as token ids. */
export function encodeText(text) {
  const chars = [...text];
  const vocab = [...new Set(chars)].sort((a, b) => a.codePointAt(0) - b.codePointAt(0));
  const index = new Map(vocab.map((c, i) => [c, i]));
  const ids = chars.map((c) => index.get(c));
  const nTrain = Math.floor(TEXT.trainFraction * ids.length);
  return { vocab, ids, nTrain };
}

/** Window start offsets for training batches: B uniform offsets per step within the training part. */
export function textSchedule(nTrain, B, steps, seed) {
  const rng = new Rng((seed * 104729 + 5) >>> 0);
  return Array.from({ length: steps }, () => Array.from({ length: B }, () => rng.int(0, nTrain - SEQ)));
}

/** The validation windows: `valWindows` offsets evenly spaced across the validation part. */
export function textValOffsets(nTotal, nTrain, k = TEXT.valWindows) {
  const span = nTotal - nTrain - SEQ;
  return Array.from({ length: k }, (_, i) => nTrain + Math.floor((i * span) / (k - 1)));
}

export const textWindows = (ids, offsets) => offsets.map((o) => ids.slice(o, o + SEQ));

/** One-hot inputs and targets for a list of T + 1 token sequences: x, y as [B·T, V] arrays. */
export function encodeBatch(seqs, V) {
  const { T } = CONFIG;
  const x = new Float64Array(seqs.length * T * V), y = new Float64Array(seqs.length * T * V);
  seqs.forEach((s, b) => {
    for (let t = 0; t < T; t++) {
      x[(b * T + t) * V + s[t]] = 1;
      y[(b * T + t) * V + s[t + 1]] = 1;
    }
  });
  return { x, y };
}
