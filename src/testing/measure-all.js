// Every milestone's measurements, as golden/measurements.json holds them.

import { measureM1 } from './measure.js';
import { measureM15 } from './measure-m1-5.js';
import { measureM2 } from './measure-m2.js';

/** `gpu`, when given, is { record, run1 }: milestone 2's recorded GPU results. */
export function measureAll({ cases, golden, gpu }) {
  const { m1 } = measureM1({ cases, golden });
  return { m1, ...measureM15({ cases, golden, m1 }), ...(gpu ? measureM2(gpu) : {}) };
}
