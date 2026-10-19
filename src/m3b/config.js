// Milestone 3b's run sizes and seeds (DESIGN.md amendments 23, 24 and 26), in
// a module the browser can load (spec.js needs node:crypto).

import { Rng } from '../testing/rng.js';

export const GOLDEN_3B = { seeds: [1, 2, 3, 4], batch: 4, steps: 20 };
export const TEXT_INIT_3B = { seed: 1, batch: 4 };
export const STAT_3B = { seeds: 24, batch: 8, steps: 300 };
export const DEMO_3B = { seed: 1, batch: 16, steps: 600, prompt: 'Alice ' };
export const GREEDY_K = 58;
export const PROJECTIONS = 4;
export const CAUSALITY_POSITIONS = [1, 17, 40, 63];

/** FNV-1a over a name's characters (ASCII). */
function fnv(name) {
  let h = 0x811c9dc5;
  for (let i = 0; i < name.length; i++) { h ^= name.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h;
}

/** The k-th projection vector for a parameter: uniform in [-1, 1], one per element. */
export function projectionVector(name, n, k) {
  return new Rng((fnv(name) + 7919 * k) >>> 0).array(n, -1, 1);
}
