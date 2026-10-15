// Milestone 2's figures, summarised from what the browser harness recorded
// (golden/gpu-m2.json, and the first run kept as golden/gpu-m2-run1-direct.json).
// The GPU cannot run in CI, so these are recorded, not recomputed; CI checks
// that the recording came from the sources checked in (test/gpu-record.test.js)
// and that this summary is what the code makes of it.

import { TILE, WG } from '../gpu/codegen.js';

/** The kernels milestone 2 recorded, frozen: later milestones add kernels. */
const M2_KERNELS = ['neg', 'exp', 'log', 'relu', 'tanh', 'step', 'copy', 'add', 'sub', 'mul', 'div', 'gather', 'sum', 'max', 'matmul-naive', 'matmul-tiled'];
import { S, U, EPS, TAU } from '../gpu/bounds.js';

const rate = (x) => ({ containing: x.containing, caught: x.caught, caughtByOutputs: x.caughtByOutputs });

export function measureM2({ record, run1 }) {
  const direct = record.direct;
  const byKernel = {};
  for (const d of direct) {
    byKernel[d.kernel] ??= { cases: 0, pass: 0 };
    byKernel[d.kernel].cases++;
    if (d.status === 'pass') byKernel[d.kernel].pass++;
  }
  const f = record.fuzz;
  const sabotage = record.sabotage.map((s) => ({
    name: s.name, label: s.label, bug: s.bug,
    direct: { cases: s.direct.cases, caught: s.direct.caught, first: s.direct.first, firstHow: s.direct.firstHow },
    fuzz: {
      general: rate(s.fuzz.general), targeted: rate(s.fuzz.targeted),
      firstCatch: s.fuzz.firstCatch, firstCatchSource: s.fuzz.firstCatchSource, skipped: s.fuzz.skipped,
    },
    // Direct tests run first in the harness, so a bug they catch is caught there first.
    caughtFirstBy: s.direct.caught > 0 ? `direct: ${s.direct.first}` : s.fuzz.firstCatch !== null ? `fuzzer: program ${s.fuzz.firstCatch}` : 'nothing',
  }));
  const bugs = sabotage.filter((s) => s.bug);
  return {
    m2: {
      producedBy: 'web/gpu.html, summarised by node tools/measure.js',
      recorded: record.recorded,
      adapter: record.adapter,
      sourceHash: record.sourceHash,
      budget: { safetyFactor: S, unitRoundoff: U, ulpFactor: EPS, smallestNormal: TAU, divisionUlp: 2.5 },
      tile: TILE,
      workgroup: WG,
      kernels: M2_KERNELS,
      direct: {
        cases: direct.length,
        pass: direct.filter((d) => d.status === 'pass').length,
        worst: Math.max(0, ...direct.filter((d) => d.status === 'pass').map((d) => d.worst)),
        byKernel,
        failures: direct.filter((d) => d.status !== 'pass').map((d) => ({ name: d.name, status: d.status, first: d.first, canaryOk: d.canaryOk, error: d.error })),
      },
      firstRun: {
        recorded: run1.recorded,
        sourceHash: run1.sourceHash,
        cases: run1.direct.length,
        pass: run1.direct.filter((d) => d.status === 'pass').length,
        failures: run1.direct.filter((d) => d.status !== 'pass').map((d) => ({ name: d.name, first: d.first })),
        note: run1.note,
      },
      fuzz: {
        programs: f.programs, skipped: f.skipped, runs: f.runs, passed: f.passed, failed: f.failed, errors: f.errors,
        worst: f.worst, bySource: f.bySource, opsChecked: f.opsChecked,
        failures: f.failures.slice(0, 50), failureCount: f.failures.length,
      },
      sabotage,
      targets: {
        directCatchesEveryBug: bugs.every((s) => s.direct.caught > 0),
        bugsMissedByDirect: bugs.filter((s) => s.direct.caught === 0).map((s) => s.name),
        bugsMissedByFuzzer: bugs.filter((s) => s.fuzz.firstCatch === null).map((s) => s.name),
        reorderFlagged: sabotage.filter((s) => !s.bug).some((s) => s.direct.caught > 0 || s.fuzz.firstCatch !== null),
        unexplainedFuzzFailures: f.failures.length,
      },
    },
  };
}
