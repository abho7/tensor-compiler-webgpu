// Milestone 2 follow-up: why two injected bugs were caught less often than
// the others. Deterministic and GPU-free: it replays the harness's fuzz runs
// (same corpus, same inputs) through the kernel emulator with the bug
// injected, under a model of what an out-of-bounds read returns, and judges
// the result with the same float32 bounds. If the model reproduces the GPU's
// recorded counts exactly, the model is the explanation.

import { grad } from '../autodiff.js';
import { size } from '../ir/shapes.js';
import { Rng } from './rng.js';
import { corpusProgram } from './random-graph.js';
import { referenceWithBounds, judge, S } from '../gpu/bounds.js';
import { lower } from '../gpu/lower.js';
import { emulate } from '../gpu/emulate.js';
import { TILE } from '../gpu/codegen.js';

const source = (i) => (i % 2 === 1 ? 'general' : 'targeted');

/**
 * The runs the harness makes for corpus program i (a copy of
 * src/gpu/harness.js programRuns, which is not exported; test/analysis.test.js
 * checks the copy against the containing-program counts the GPU recorded).
 */
export function programRuns(i) {
  const p = corpusProgram(i, { f32: true });
  const rng = new Rng((i * 7919 + 17) >>> 0);
  const inputs = {};
  for (const n of p.graph.inputs()) inputs[n.attrs.name] = rng.array(size(n.shape), -2, 2);
  const G = grad(p.graph, p.outputs, p.graph.inputs().map((n) => n.id));
  const gradInputs = { ...inputs };
  G.cotangents.forEach((name, k) => { gradInputs[name] = rng.array(size(p.graph.nodes[p.outputs[k]].shape), -1, 1); });
  const hasMatmul = p.graph.nodes.some((n) => n.op === 'matmul');
  const runs = [];
  for (const matmul of hasMatmul ? ['naive', 'tiled'] : ['naive']) {
    runs.push({ label: 'forward', graph: p.graph, inputs, matmul });
    runs.push({ label: 'gradient', graph: G.graph, inputs: gradInputs, matmul });
  }
  return { program: p, runs, ops: new Set([...p.graph.nodes, ...G.graph.nodes].map((n) => n.op)) };
}

/** Would the emulated, sabotaged run fail the float32 budget? */
function emulatedCatch(run, sabotage, oobRead) {
  const ref = referenceWithBounds(run.graph, run.inputs);
  const tiledEdge = sabotage === 'tiled-off-by-one' ? '<=' : '<';
  const lowered = lower(run.graph, { matmul: run.matmul, sabotage: sabotage === 'tiled-off-by-one' ? undefined : sabotage });
  const { values } = emulate(run.graph, lowered, ref.inputs, { leaves: ref.values, oobRead, tiledEdge });
  return !judge(run.graph, values, ref.values, ref.bounds).pass;
}

const rate = () => ({ containing: 0, caught: 0 });

function replay(programs, sabotage, relevant, select, oobReads) {
  const out = Object.fromEntries(oobReads.map((m) => [m, { general: rate(), targeted: rate() }]));
  const control = { runs: 0, failing: 0 };
  const caughtUnderZero = new Set();
  for (let i = 1; i <= programs; i++) {
    const pr = programRuns(i);
    if (!relevant(pr)) continue;
    const runs = select(pr.runs);
    // Control: the honest emulation must pass every run, or the model is wrong.
    for (const run of runs) {
      control.runs++;
      const ref = referenceWithBounds(run.graph, run.inputs);
      const { values } = emulate(run.graph, lower(run.graph, { matmul: run.matmul }), ref.inputs, { leaves: ref.values, oobRead: 'nan' });
      if (!judge(run.graph, values, ref.values, ref.bounds).pass) control.failing++;
    }
    for (const m of oobReads) {
      const r = out[m][source(i)];
      r.containing++;
      if (runs.some((run) => emulatedCatch(run, sabotage, m))) {
        r.caught++;
        if (m === 'zero') caughtUnderZero.add(i);
      }
    }
  }
  return { byModel: out, control, caughtUnderZero };
}

/** Item 1: the shapes of every matmul the tiled kernel ran on in the corpus. */
function matmulShapes(programs) {
  const batchedSet = new Set();
  const counts = { programs: 0, anyDimNotMultipleOfTile: 0, anyDimLargerThanTile: 0, kNotMultipleOfTile: 0, kLargerThanTile: 0, batched: 0, unbatchedOnly: 0, maxDim: 0 };
  const bySource = { general: { programs: 0, batched: 0 }, targeted: { programs: 0, batched: 0 } };
  for (let i = 1; i <= programs; i++) {
    const pr = programRuns(i);
    const mms = pr.runs.filter((r) => r.matmul === 'tiled').flatMap((r) => r.graph.nodes.filter((n) => n.op === 'matmul').map((n) => {
      const as = r.graph.shape(n.inputs[0]), bs = r.graph.shape(n.inputs[1]);
      return { batch: size(as.slice(0, -2)), M: as.at(-2), K: as.at(-1), N: bs.at(-1) };
    }));
    if (mms.length === 0) continue;
    counts.programs++;
    bySource[source(i)].programs++;
    const dims = mms.flatMap((m) => [m.M, m.K, m.N]);
    counts.maxDim = Math.max(counts.maxDim, ...dims);
    if (dims.some((d) => d % TILE !== 0)) counts.anyDimNotMultipleOfTile++;
    if (dims.some((d) => d > TILE)) counts.anyDimLargerThanTile++;
    if (mms.some((m) => m.K % TILE !== 0)) counts.kNotMultipleOfTile++;
    if (mms.some((m) => m.K > TILE)) counts.kLargerThanTile++;
    if (mms.some((m) => m.batch > 1)) { counts.batched++; bySource[source(i)].batched++; batchedSet.add(i); } else counts.unbatchedOnly++;
  }
  return { ...counts, bySource, batchedSet };
}

/** Item 2: broadcasts that stretch an axis (the only kind the stride bug can change). */
function broadcastShapes(programs) {
  const out = { general: { programs: 0, stretching: 0, leadingOnly: 0 }, targeted: { programs: 0, stretching: 0, leadingOnly: 0 } };
  const stretchingSet = new Set();
  for (let i = 1; i <= programs; i++) {
    const pr = programRuns(i);
    if (!pr.ops.has('broadcast')) continue;
    const graphs = [...new Set(pr.runs.map((r) => r.graph))];
    const stretches = graphs.some((g) => g.nodes.some((n) => {
      if (n.op !== 'broadcast') return false;
      const from = g.shape(n.inputs[0]), lead = n.shape.length - from.length;
      return from.some((d, k) => d === 1 && n.shape[lead + k] !== 1);
    }));
    const o = out[source(i)];
    o.programs++;
    if (stretches) { o.stretching++; stretchingSet.add(i); } else o.leadingOnly++;
  }
  return { ...out, stretchingSet };
}

/**
 * Why a program that stretches an axis still was not caught: every stretched
 * source is all zeros (reading the wrong element still reads zero), or the
 * wrong values differ by less than the float32 budget allows at that node.
 */
function uncaughtBroadcastReasons(indices) {
  const reasons = { allZeroSource: [], withinBudget: [], other: [] };
  for (const i of indices) {
    let allZero = true, worst = 0;
    for (const run of programRuns(i).runs) {
      const g = run.graph;
      const ref = referenceWithBounds(g, run.inputs);
      const { values } = emulate(g, lower(g, { matmul: run.matmul, sabotage: 'broadcast-wrong-stride' }), ref.inputs, { leaves: ref.values, oobRead: 'zero' });
      for (const n of g.nodes) {
        if (n.op !== 'broadcast') continue;
        const from = g.shape(n.inputs[0]), lead = n.shape.length - from.length;
        if (!from.some((d, k) => d === 1 && n.shape[lead + k] !== 1)) continue;
        if (Array.from(ref.values[n.inputs[0]]).some((v) => v !== 0)) allZero = false;
        values[n.id].forEach((v, j) => {
          const d = Math.abs(v - ref.values[n.id][j]);
          if (d > 0) worst = Math.max(worst, d / (S * ref.bounds[n.id][j]));
        });
      }
    }
    if (allZero) reasons.allZeroSource.push(i);
    else if (worst <= 1) reasons.withinBudget.push(i);
    else reasons.other.push(i);
  }
  return reasons;
}

/** Program-by-program agreement between a shape property and being caught. */
function agreement(propertySet, caughtSet) {
  const only = (a, b) => [...a].filter((i) => !b.has(i)).sort((x, y) => x - y);
  return { caughtWithoutProperty: only(caughtSet, propertySet), propertyButNotCaught: only(propertySet, caughtSet) };
}

export function analyzeM2({ programs = 2000 } = {}) {
  const models = ['zero', 'clamp'];
  const mmShapes = matmulShapes(programs);
  const mmReplay = replay(programs, 'tiled-off-by-one', (pr) => pr.ops.has('matmul'), (runs) => runs.filter((r) => r.matmul === 'tiled'), models);
  const bcShapes = broadcastShapes(programs);
  const bcReplay = replay(programs, 'broadcast-wrong-stride', (pr) => pr.ops.has('broadcast'), (runs) => runs, models);
  const strip = ({ batchedSet, stretchingSet, ...rest }) => rest;
  const stripReplay = ({ caughtUnderZero, ...rest }) => rest;
  return {
    m2_followup: {
      producedBy: 'node tools/measure.js (src/testing/analyze-m2.js)',
      tile: TILE,
      tiledMatmul: {
        shapes: strip(mmShapes),
        replay: stripReplay(mmReplay),
        // Caught under the zero model exactly when the program has a batched matmul?
        batchedVsCaught: agreement(mmShapes.batchedSet, mmReplay.caughtUnderZero),
      },
      broadcast: {
        shapes: strip(bcShapes),
        replay: stripReplay(bcReplay),
        stretchingVsCaught: agreement(bcShapes.stretchingSet, bcReplay.caughtUnderZero),
        uncaughtStretching: uncaughtBroadcastReasons(agreement(bcShapes.stretchingSet, bcReplay.caughtUnderZero).propertyButNotCaught),
      },
    },
  };
}
