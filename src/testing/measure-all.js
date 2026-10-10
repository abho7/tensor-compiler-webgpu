// Every milestone's measurements, as golden/measurements.json holds them.

import { measureM1 } from './measure.js';
import { measureM15 } from './measure-m1-5.js';

export function measureAll({ cases, golden }) {
  const { m1 } = measureM1({ cases, golden });
  return { m1, ...measureM15({ cases, golden, m1 }) };
}
