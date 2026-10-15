// Every milestone's measurements, as golden/measurements.json holds them.

import { measureM1 } from './measure.js';
import { measureM15 } from './measure-m1-5.js';
import { measureM2 } from './measure-m2.js';
import { analyzeM2 } from './analyze-m2.js';
import { measureM3 } from './measure-m3.js';

/**
 * `gpu`, when given, is { record, run1, oob }: milestone 2's recorded GPU
 * results and the out-of-bounds read probe. `m3`, when given, is
 * { sqrtCases, sqrtGolden }: milestone 3a's sqrt cases and their PyTorch answers.
 */
export function measureAll({ cases, golden, gpu, m3 }) {
  const { m1 } = measureM1({ cases, golden });
  const out = { m1, ...measureM15({ cases, golden, m1 }) };
  if (!gpu) return out;
  const { m2_followup } = analyzeM2({ programs: gpu.record.options.programs });
  m2_followup.oobProbe = {
    adapter: gpu.oob.adapter, recorded: gpu.oob.recorded,
    cases: gpu.oob.results.map((r) => ({
      case: r.case,
      outOfBounds: r.reads.filter((x) => !x.inBounds).length,
      returnedZero: r.reads.filter((x) => !x.inBounds && x.kind === 'zero').length,
      inBoundsCorrect: r.reads.filter((x) => x.inBounds && x.kind === 'correct').length,
      inBounds: r.reads.filter((x) => x.inBounds).length,
      kinds: [...new Set(r.reads.filter((x) => !x.inBounds).map((x) => x.kind))],
    })),
  };
  return { ...out, ...measureM2(gpu), m2_followup, ...(m3 ? measureM3(m3) : {}) };
}
