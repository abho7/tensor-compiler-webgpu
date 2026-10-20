// Milestone 4's figures for golden/measurements.json. Everything that needs no
// GPU is computed from scratch: semantic preservation over milestone 2's corpus
// and the model's graphs, the pass sabotage against it, and what the pass does
// to the model (revision R5). The GPU recordings are summarised, never
// recomputed.

import { corpusProgram } from '../testing/random-graph.js';
import { grad } from '../autodiff.js';
import { Rng } from '../testing/rng.js';
import { size } from '../ir/shapes.js';
import { fuse, checkSemantics, fusionSummary } from './fuse.js';
import { buildForward, initParams } from '../m3b/model.js';
import { buildStep3b, paramShapes } from '../m3b/step.js';
import { adamInitialState } from '../m3b/adam.js';
import { encodeBatch } from '../m3b/data.js';
import { MAX_OPERANDS } from './fuse.js';
import { STAT_MARGIN } from '../m3b/measure-m3b.js';
import { ROW_WG } from './codegen-fused.js';
import { WG, TILE, MM_WG } from '../gpu/codegen.js';
import { REPS, WARMUP, TARGET_NS } from './bench.js';
import { ROW_CAP, CEILING_BYTES } from './workloads.js';
import { FUZZ_PROGRAMS_M4, SABOTAGE_FUZZ_PROGRAMS } from '../gpu/harness-m4.js';

export const SEMANTIC_PROGRAMS = 2000;
export const PASS_SABOTAGE_PROGRAMS = 500;

/** Milestone 2's forward and gradient graphs for corpus program i, with its inputs. */
export function corpusGraphs(i) {
  const p = corpusProgram(i, { f32: true });
  const rng = new Rng((i * 7919 + 17) >>> 0);
  const inputs = {};
  for (const n of p.graph.inputs()) inputs[n.attrs.name] = rng.array(size(n.shape), -2, 2);
  const G = grad(p.graph, p.outputs, p.graph.inputs().map((n) => n.id));
  const gradInputs = { ...inputs };
  G.cotangents.forEach((name, k) => { gradInputs[name] = rng.array(size(p.graph.nodes[p.outputs[k]].shape), -1, 1); });
  return [
    { label: 'forward', graph: p.graph, inputs, keep: p.outputs },
    { label: 'gradient', graph: G.graph, inputs: gradInputs, keep: [...p.outputs, ...G.grads] },
  ];
}

/** The model's graphs: the forward pass and the training step of golden config 1, with its first batch. */
export function modelGraphs(spec) {
  const g = spec.golden[0];
  const { x, y } = encodeBatch(g.schedule[0].map((i) => spec.synthetic.train[i]), g.V);
  const params = initParams(g.V, g.seed);
  const f = buildForward(g.V, g.batch);
  const s = buildStep3b(g.V, g.batch);
  return [
    { label: 'forward', graph: f.graph, inputs: { ...params, x, y }, keep: [f.logits, f.loss] },
    { label: 'step', graph: s.graph, inputs: { ...params, ...adamInitialState(s.names, paramShapes(g.V)), x, y, [s.cotangent]: [1] },
      keep: [s.loss, s.logits, ...Object.values(s.outputs)] },
  ];
}

/** Amendment 29: bitwise semantic preservation, over the corpus and the model. */
export function semantics(spec) {
  const corpus = { programs: SEMANTIC_PROGRAMS, graphs: 0, withFused: 0, identical: 0, compared: 0, fusedNodes: 0, byPattern: {}, failures: [] };
  for (let i = 1; i <= SEMANTIC_PROGRAMS; i++) {
    for (const c of corpusGraphs(i)) {
      const f = fuse(c.graph, { keep: c.keep });
      const s = checkSemantics(c.graph, f, c.inputs);
      corpus.graphs++;
      const fused = f.graph.nodes.filter((n) => n.op === 'fused');
      if (fused.length) corpus.withFused++;
      corpus.fusedNodes += fused.length;
      for (const n of fused) corpus.byPattern[n.attrs.pattern] = (corpus.byPattern[n.attrs.pattern] ?? 0) + 1;
      corpus.compared += s.compared;
      if (s.identical) corpus.identical++;
      else corpus.failures.push({ program: i, graph: c.label, first: s.first });
    }
  }
  const model = modelGraphs(spec).map((c) => {
    const f = fuse(c.graph, { keep: c.keep });
    const s = checkSemantics(c.graph, f, c.inputs);
    return { graph: c.label, identical: s.identical, compared: s.compared, ...fusionSummary(c.graph, f) };
  });
  return { corpus, model };
}

/**
 * Sabotage 5 (amendment 31) against semantic preservation: corpus programs 1
 * to 500 and the model's graphs. A graph is caught if its fused values differ;
 * the sabotaged pass throwing is counted apart, as a loud failure of the pass.
 */
export function passSabotage(spec) {
  const out = { programs: PASS_SABOTAGE_PROGRAMS, graphs: 0, changed: 0, caught: 0, threw: 0, firstCatch: null };
  for (let i = 1; i <= PASS_SABOTAGE_PROGRAMS; i++) {
    for (const c of corpusGraphs(i)) {
      out.graphs++;
      const honest = fuse(c.graph, { keep: c.keep });
      let bad;
      try {
        bad = fuse(c.graph, { keep: c.keep, sabotage: 'fusion-absorbs-shared' });
      } catch {
        out.threw++;
        out.changed++;
        continue;
      }
      if (JSON.stringify(bad.graph.nodes) !== JSON.stringify(honest.graph.nodes)) out.changed++;
      if (!checkSemantics(c.graph, bad, c.inputs).identical) { out.caught++; out.firstCatch ??= i; }
    }
  }
  out.model = modelGraphs(spec).map((c) => {
    try {
      const bad = fuse(c.graph, { keep: c.keep, sabotage: 'fusion-absorbs-shared' });
      return { graph: c.label, threw: false, caught: !checkSemantics(c.graph, bad, c.inputs).identical };
    } catch (error) {
      return { graph: c.label, threw: true, error: String(error.message ?? error) };
    }
  });
  return out;
}

/** Revision R5: what the pass makes of the model's forward pass and training step. */
export function modelFusion(spec) {
  return modelGraphs(spec).map((c) => {
    const f = fuse(c.graph, { keep: c.keep });
    const epilogues = f.groups.filter((g) => g.kind === 'epilogue');
    const withActivation = epilogues.filter((g) => [...g.members].some((id) => ['tanh', 'relu'].includes(c.graph.nodes[id].op))).length;
    return { graph: c.label, ...fusionSummary(c.graph, f), epilogues: epilogues.length, epiloguesWithActivation: withActivation };
  });
}

export function measureM4({ spec, torch, gpuM3b, gpuRecord, gpuRun1, benchRecord, discarded = [] }) {
  const passSab = passSabotage(spec);
  return { m4: {
    // The fixed numbers of amendments 30 to 36, from the code that uses them.
    config: { maxOperands: MAX_OPERANDS, rowWorkgroup: ROW_WG, elementwiseWorkgroup: WG, tile: TILE, naiveMatmulWorkgroup: MM_WG,
      reps: REPS, warmup: WARMUP, targetMsPerRep: TARGET_NS / 1e6, rowCap: ROW_CAP, ceilingBytes: CEILING_BYTES,
      fuzzPrograms: FUZZ_PROGRAMS_M4, sabotageFuzzPrograms: SABOTAGE_FUZZ_PROGRAMS, semanticPrograms: SEMANTIC_PROGRAMS,
      passSabotagePrograms: PASS_SABOTAGE_PROGRAMS, statMargin: STAT_MARGIN_M4, m3bStatMargin: STAT_MARGIN, thresholds: THRESHOLDS },
    semantics: semantics(spec), passSabotage: passSab, fusion: modelFusion(spec),
    gpu: gpuRecord ? summarizeGpuM4(gpuRecord, { torch, gpuM3b, passSab }) : null,
    // The first recording, kept as it came out: the fuzzer found a bug in the pass (DESIGN.md F10).
    gpuRun1: gpuRun1 ? { sourceHash: gpuRun1.sourceHash, errors: gpuRun1.fuzz.errors, runs: gpuRun1.fuzz.runs,
      failures: gpuRun1.fuzz.failures.map((f) => ({ program: f.index, run: f.run, matmul: f.matmul, error: f.error ?? null })) } : null,
    bench: benchRecord ? summarizeBench(benchRecord) : null,
    discardedSessions: discarded.map((d) => ({ at: d.recorded?.at ?? null, after: d.discarded.after, start: d.discarded.start?.activeMode ?? null,
      now: d.discarded.now?.activeMode ?? null, onAC: d.discarded.now?.onAC ?? null })),
  } };
}

// ---- the GPU correctness record (golden/gpu-m4.json) ---------------------------------------

import { tost, ALPHA } from '../train/equivalence.js';

export const STAT_MARGIN_M4 = 1e-5;

/**
 * Amendment 31's predictions, per bug and check: true (catches), false (does
 * not), or null (no prediction). The forward-pass check was added by R5, after
 * the predictions, so it carries none.
 */
export const PREDICTED_M4 = {
  'epilogue-before-bias': { semantics: false, direct: true, fuzz: null, step: true, forward: null },
  'layernorm-tree-short': { semantics: false, direct: true, fuzz: false, step: true, forward: null },
  'softmax-no-max': { semantics: false, direct: true, fuzz: null, step: false, forward: null },
  'fused-bounds-off-by-one': { semantics: false, direct: true, fuzz: false, step: false, forward: null },
  'fusion-absorbs-shared': { semantics: true, direct: null, fuzz: null, step: null, forward: null },
};
const CHECK_ORDER = ['semantics', 'direct', 'fuzz', 'step', 'forward'];

export function summarizeGpuM4(record, { torch, gpuM3b, passSab }) {
  const r = record;
  const direct = { cases: r.direct.length, passed: r.direct.filter((d) => d.pass).length, worst: Math.max(...r.direct.map((d) => d.worst ?? 0)),
    byPattern: {} };
  for (const d of r.direct) {
    const p = (direct.byPattern[d.pattern] ??= { cases: 0, passed: 0 });
    p.cases++; if (d.pass) p.passed++;
  }
  const f = r.fuzz;
  const fuzz = { programs: f.programs, skipped: f.skipped, runs: f.runs, fusedRuns: f.fusedRuns, unfusedPassed: f.unfusedPassed, fusedPassed: f.fusedPassed,
    errors: f.errors, failures: f.failures.length, bitwiseSame: f.bitwiseSame, worstFused: f.worstFused, worstUnfused: f.worstUnfused, byPattern: f.byPattern };
  const tf = r.teacherForced;
  const kinds = {};
  for (const run of tf) for (const s of run.detail) for (const [k, v] of Object.entries(s.local?.byKind ?? {})) kinds[k] = Math.max(kinds[k] ?? 0, v.worst);
  const teacherForced = { runs: tf.length, steps: tf.reduce((a, x) => a + x.steps, 0), passed: tf.reduce((a, x) => a + x.passed, 0),
    unjudged: tf.reduce((a, x) => a + x.unjudged, 0), judgedFailures: tf.reduce((a, x) => a + x.judgedFailures, 0), errors: tf.reduce((a, x) => a + x.errors, 0),
    worst: Math.max(...tf.map((x) => x.worst)), worstByKind: kinds, launches: tf[0].launches, unfusedLaunches: tf[0].unfusedLaunches };
  const fl = r.forwardLocal;
  const forward = { configs: fl.length, passed: fl.filter((x) => x.pass).length, worst: Math.max(...fl.map((x) => x.worst ?? 0)),
    launches: fl[0].launches, unfusedLaunches: fl[0].unfusedLaunches, patterns: fl[0].patterns };

  // Amendment 36: fused GPU against PyTorch float64, and against milestone 3b's unfused GPU, paired by seed.
  const t64 = new Map(torch.statistical.float64.map((e) => [e.seed, e.valLoss]));
  const m3b = new Map((gpuM3b?.statistical ?? []).map((e) => [e.seed, e]));
  const st = r.statistical;
  const statistical = {
    seeds: st.length, margin: STAT_MARGIN_M4, alpha: ALPHA,
    fusedVsTorch64: tost(st.map((e) => e.valLoss - t64.get(e.seed)), STAT_MARGIN_M4),
    fusedVsUnfused: tost(st.map((e) => e.valLoss - m3b.get(e.seed).valLoss), STAT_MARGIN_M4),
    // For context: milestone 3b's unfused run under the new margin.
    unfusedVsTorch64: tost(st.map((e) => m3b.get(e.seed).valLoss - t64.get(e.seed)), STAT_MARGIN_M4),
    identicalValLoss: st.filter((e) => e.valLoss === m3b.get(e.seed).valLoss).length,
    identicalFinalTrainLoss: st.filter((e) => e.finalTrainLoss === m3b.get(e.seed).finalTrainLoss).length,
    seed1: { fused: st.find((e) => e.seed === 1).losses, unfused: m3b.get(1).losses,
      firstDifferentStep: st.find((e) => e.seed === 1).losses.findIndex((v, i) => v !== m3b.get(1).losses[i]) },
  };

  const sabotage = r.sabotage.map((s) => {
    const fz = s.fuzz;
    const checks = {
      // Semantic preservation never runs a kernel, so a kernel bug cannot change it; the pass bug is measured in pure JS.
      semantics: s.where === 'pass' ? { trials: passSab.graphs, caught: passSab.caught, threw: passSab.threw } : { trials: passSab.graphs, caught: 0, threw: 0 },
      direct: { trials: s.direct.cases, caught: s.direct.caught, threw: s.direct.fuseErrors },
      fuzz: { trials: fz.general.containing + fz.targeted.containing, caught: fz.general.caught + fz.targeted.caught, threw: fz.fuseErrors,
        general: fz.general, targeted: fz.targeted, firstCatch: fz.firstCatch },
      step: { trials: s.teacherForced.trials, caught: s.teacherForced.caught, threw: s.teacherForced.fuseError ? 1 : 0 },
      forward: { trials: s.forward.trials, caught: s.forward.caught, threw: s.forward.fuseErrors },
    };
    const caughtFirstBy = CHECK_ORDER.find((c) => checks[c].caught > 0) ?? null;
    const threwFirstIn = CHECK_ORDER.find((c) => checks[c].threw > 0) ?? null;
    const predicted = PREDICTED_M4[s.name];
    const asPredicted = Object.fromEntries(CHECK_ORDER.filter((c) => predicted[c] !== null).map((c) => [c, (checks[c].caught > 0) === predicted[c]]));
    return { name: s.name, where: s.where, checks, caughtFirstBy, threwFirstIn, predicted, asPredicted,
      caughtOrThrew: CHECK_ORDER.some((c) => checks[c].caught > 0 || checks[c].threw > 0) };
  });
  return { sourceHash: r.sourceHash, adapter: { vendor: r.adapter.vendor, architecture: r.adapter.architecture }, power: r.recorded?.power ?? null,
    direct, fuzz, teacherForced, forward, consistency: r.consistency, statistical, sabotage };
}

// ---- the timing record (golden/gpu-m4-bench.json) ----------------------------------------

/**
 * Amendment 32's rule: a variant wins or loses against another only if their
 * min-to-max ranges do not overlap. Speedup is the ratio of medians.
 */
export function verdict(base, other) {
  if (!base || !other) return null;
  const speedup = base.gpuUs.median / other.gpuUs.median;
  const v = other.gpuUs.max < base.gpuUs.min ? 'win' : other.gpuUs.min > base.gpuUs.max ? 'loss' : 'noise';
  return { speedup, verdict: v };
}

/** Amendment 35's thresholds, as registered. */
export const THRESHOLDS = { p1Fraction: 0.5, p2Ceiling: 3, p3Neutral: 0.1, p3ReluK: 512, p4Speedup: 2, p4WideC: 64, p4WideR: 64, p4NarrowC: 7,
  p4NarrowR: 65536, p5MinR: 256, p7Dominated: 0.8, p7Within: 1.25, p8Matvec: 2, p8Add: 0.2, p8Within: 2, p9Fraction: 0.5, p1MinN: 1048573, bigN: 4194301 };
const TH = THRESHOLDS;

const briefTiming = (v) => (v ? { median: v.gpuUs.median, min: v.gpuUs.min, max: v.gpuUs.max, wallMedian: v.wallUs.median, launches: v.launches, bytes: v.bytes, gbs: v.gbs } : null);

export function summarizeBench(record) {
  const ceilingOf = (mode) => record.ceiling.filter((c) => c.mode === mode).reduce((a, c) => (c.gbs > a.gbs ? c : a));
  const ceiling = { read: { gbs: ceilingOf('read').gbs, config: ceilingOf('read').id }, copy: { gbs: ceilingOf('copy').gbs, config: ceilingOf('copy').id },
    all: record.ceiling.map((c) => ({ id: c.id, gbs: c.gbs })) };
  const rows = record.workloads.map((w) => {
    const v = w.variants;
    const out = { id: w.id, family: w.family, params: w.params, k: w.k ?? null, failedGate: w.failedGate ?? null,
      violations: w.violations ?? null, variants: Object.fromEntries(Object.entries(v).map(([k, x]) => [k, briefTiming(x)])) };
    if (v.unfused && v.fused) out.fused = verdict(v.unfused, v.fused);
    if (v.fused && v.recompute) out.recomputeVsFused = verdict(v.fused, v.recompute);
    if (v.unfused && v.recompute) out.recompute = verdict(v.unfused, v.recompute);
    if (v.hand) {
      const compiler = ['fused', 'unfused'].filter((k) => v[k]).map((k) => [k, v[k]]).sort((a, b) => a[1].gpuUs.median - b[1].gpuUs.median)[0];
      out.handVsCompiler = { compiler: compiler[0], ...verdict(compiler[1], v.hand) };
      if (v.unfused) out.handVsUnfused = verdict(v.unfused, v.hand);
      if (v.fused) out.handVsFused = verdict(v.fused, v.hand);
      if (v['hand-f32']) out.handF32VsUnfused = verdict(v.unfused, v['hand-f32']);
    }
    if (v['matmul-only']) out.matmulShare = v['matmul-only'].gpuUs.median / v.unfused.gpuUs.median;
    if (w.gate?.fusedVsUnfused) out.fusedVsUnfusedOutputs = w.gate.fusedVsUnfused;
    return out;
  });
  const get = (id) => rows.find((r) => r.id === id);
  const fam = (f) => rows.filter((r) => r.family === f && r.fused);
  // Bounds on a speedup from the ranges, for "beyond noise" against a fixed number.
  const range = (r) => ({ lo: r.variants.unfused.min / r.variants.fused.max, hi: r.variants.unfused.max / r.variants.fused.min });

  const P = {};
  {
    const big = fam('chain-unary').filter((r) => r.params.n >= TH.p1MinN);
    const sizes = [...new Set(big.map((r) => r.params.n))];
    const per = sizes.map((n) => {
      const rs = big.filter((r) => r.params.n === n).sort((a, b) => a.params.L - b.params.L);
      return { n, speedups: rs.map((r) => ({ L: r.params.L, speedup: r.fused.speedup, verdict: r.fused.verdict })),
        increases: rs.at(-1).fused.speedup > rs[0].fused.speedup, monotone: rs.every((r, i) => i === 0 || r.fused.speedup >= rs[i - 1].fused.speedup),
        belowHalfL: rs.filter((r) => r.params.L >= 2 && r.fused.speedup < r.params.L * TH.p1Fraction).map((r) => r.params.L) };
    });
    const small = fam('chain-unary').filter((r) => r.params.n === 1000).sort((a, b) => a.params.L - b.params.L);
    P.P1 = { per, small: small.map((r) => ({ L: r.params.L, speedup: r.fused.speedup, verdict: r.fused.verdict })),
      held: per.every((p) => p.increases && p.belowHalfL.length === 0) };
  }
  {
    const rs = fam('chain-binary').filter((r) => r.params.n === TH.bigN).sort((a, b) => a.params.L - b.params.L);
    const at = (L) => rs.find((r) => r.params.L === L);
    const over3 = rs.filter((r) => range(r).lo > TH.p2Ceiling).map((r) => r.params.L);
    const falls = at(4).fused.speedup < at(1).fused.speedup;
    P.P2 = { speedups: rs.map((r) => ({ L: r.params.L, speedup: r.fused.speedup, verdict: r.fused.verdict, launchesFused: r.variants.fused.launches })),
      over3BeyondNoise: over3, fallsFrom1To4: falls, held: over3.length === 0 && !falls };
  }
  {
    const rs = fam('matmul-epilogue');
    const gelu = rs.filter((r) => r.params.act === 'gelu');
    const reluBigK = rs.filter((r) => r.params.act === 'relu' && r.params.K >= TH.p3ReluK);
    const neutral = (r) => r.fused.verdict === 'noise' || Math.abs(r.fused.speedup - 1) <= TH.p3Neutral;
    P.P3 = { gelu: gelu.map((r) => ({ id: r.id, speedup: r.fused.speedup, verdict: r.fused.verdict })),
      reluK512: reluBigK.map((r) => ({ id: r.id, speedup: r.fused.speedup, verdict: r.fused.verdict })),
      losses: rs.filter((r) => r.fused.verdict === 'loss').map((r) => ({ id: r.id, speedup: r.fused.speedup })),
      geluAllWin: gelu.every((r) => r.fused.verdict === 'win'), reluNeutral: reluBigK.every(neutral) };
    P.P3.held = P.P3.geluAllWin && P.P3.reluNeutral;
  }
  {
    const rs = [...fam('layernorm'), ...fam('softmax')];
    const wide = rs.filter((r) => r.params.C >= TH.p4WideC && r.params.R >= TH.p4WideR);
    const narrow = rs.filter((r) => r.params.C <= TH.p4NarrowC && r.params.R === TH.p4NarrowR);
    P.P4 = { wide: wide.map((r) => ({ id: r.id, speedup: r.fused.speedup, verdict: r.fused.verdict })),
      narrow: narrow.map((r) => ({ id: r.id, speedup: r.fused.speedup, verdict: r.fused.verdict })),
      wideAbove2: wide.filter((r) => r.fused.speedup > TH.p4Speedup).length, wideCount: wide.length,
      narrowLoss: narrow.filter((r) => r.fused.verdict === 'loss').length, narrowWin: narrow.filter((r) => r.fused.verdict === 'win').length,
      losses: rs.filter((r) => r.fused.verdict === 'loss').map((r) => ({ id: r.id, speedup: r.fused.speedup })) };
    P.P4.winHeld = P.P4.wideAbove2 === P.P4.wideCount;
    // The loss prediction is falsified if fused still wins beyond noise at both C = 1 and C = 7.
    const winsAt = (C) => narrow.filter((r) => r.params.C === C).every((r) => r.fused.verdict === 'win');
    P.P4.lossHeld = !([...new Set(narrow.map((r) => r.params.C))].every(winsAt));
  }
  {
    const rs = rows.filter((r) => r.family === 'broadcast-recompute').sort((a, b) => a.params.R - b.params.R);
    const ratio = rs.map((r) => ({ R: r.params.R, recomputeOverFused: r.variants.recompute.median / r.variants.fused.median, verdict: r.recomputeVsFused.verdict,
      fused: r.fused, recompute: r.recompute }));
    const big = ratio.filter((x) => x.R >= TH.p5MinR);
    P.P5 = { ratio, held: big.every((x) => x.verdict === 'loss') && big.every((x, i) => i === 0 || x.recomputeOverFused > big[i - 1].recomputeOverFused) };
  }
  {
    const one = get('lost-reuse-1op'), many = get('lost-reuse-16op');
    P.P6 = { op1: { recomputeVsFused: one.recomputeVsFused, fused: one.fused, recompute: one.recompute },
      op16: { recomputeVsFused: many.recomputeVsFused, fused: many.fused, recompute: many.recompute },
      held: many.recomputeVsFused.verdict === 'loss' && one.recomputeVsFused.verdict !== 'loss' };
  }
  {
    const rs = rows.filter((r) => r.family === 'training-step');
    P.P7 = { steps: rs.map((r) => {
      const s = r.matmulShare, bound = 1 / s;
      return { id: r.id, matmulShare: s, amdahlBound: bound, speedup: r.fused.speedup, verdict: r.fused.verdict,
        exceedsBoundBeyondNoise: range(r).lo > bound, matmulDominated: s >= TH.p7Dominated, within125: r.fused.speedup <= TH.p7Within };
    }) };
    P.P7.held = P.P7.steps.every((x) => !x.exceedsBoundBeyondNoise && (!x.matmulDominated || x.within125));
  }
  {
    const mv = rows.filter((r) => r.family === 'baseline-matvec');
    const add = rows.filter((r) => r.family === 'baseline-add');
    const sw = rows.filter((r) => r.family === 'baseline-swiglu');
    const rms = rows.filter((r) => r.family === 'baseline-rmsnorm');
    const shapes = [...new Set(mv.map((r) => r.params.rows))].map((rowsN) => {
      const both = mv.filter((r) => r.params.rows === rowsN);
      const best = both.map((r) => r.variants.unfused).sort((a, b) => a.median - b.median)[0];
      const hand = both[0].variants.hand, handF32 = both[0].variants['hand-f32'];
      return { rows: rowsN, cols: both[0].params.cols, compilerBest: best.median, naive: both.find((r) => r.params.matmul === 'naive').variants.unfused.median,
        tiled: both.find((r) => r.params.matmul === 'tiled').variants.unfused.median, hand: hand.median, handF32: handF32.median,
        compilerOverHand: best.median / hand.median, compilerOverHandF32: best.median / handF32.median };
    });
    P.P8 = {
      matvec: shapes, matvecHeld: shapes.every((s) => s.compilerOverHand >= TH.p8Matvec),
      add: add.map((r) => ({ n: r.params.n, ...r.handVsUnfused, compilerOverHand: r.variants.unfused.median / r.variants.hand.median })),
      addHeld: add.every((r) => r.handVsUnfused.verdict === 'noise' || Math.abs(r.variants.unfused.median / r.variants.hand.median - 1) <= TH.p8Add),
      swiglu: sw.map((r) => ({ n: r.params.n, fusedOverHand: r.variants.fused.median / r.variants.hand.median, unfusedOverHand: r.variants.unfused.median / r.variants.hand.median })),
      rmsnorm: rms.map((r) => ({ rows: r.params.rows, fusedOverHand: r.variants.fused.median / r.variants.hand.median, unfusedOverHand: r.variants.unfused.median / r.variants.hand.median })),
    };
    const within2 = (xs) => xs.every((x) => x.fusedOverHand <= TH.p8Within) && xs.every((x) => x.unfusedOverHand > TH.p8Within);
    P.P8.swigluHeld = within2(P.P8.swiglu);
    P.P8.rmsnormHeld = within2(P.P8.rmsnorm);
    P.P8.held = P.P8.matvecHeld && P.P8.addHeld && P.P8.swigluHeld && P.P8.rmsnormHeld;
  }
  {
    const rs = [...fam('chain-unary'), ...fam('chain-binary')].filter((r) => r.params.n === TH.bigN);
    const copy = ceiling.copy.gbs;
    P.P9 = { copyCeiling: copy, fused: rs.map((r) => ({ id: r.id, gbs: r.variants.fused.gbs, fractionOfCopy: r.variants.fused.gbs / copy })),
      held: rs.every((r) => r.variants.fused.gbs >= copy * TH.p9Fraction) };
  }
  // Every loss, wherever it fell.
  const losses = [];
  for (const r of rows) {
    if (r.fused?.verdict === 'loss') losses.push({ id: r.id, comparison: 'fused vs unfused', speedup: r.fused.speedup });
    if (r.recompute?.verdict === 'loss') losses.push({ id: r.id, comparison: 'recompute vs unfused', speedup: r.recompute.speedup });
    if (r.recomputeVsFused?.verdict === 'loss') losses.push({ id: r.id, comparison: 'recompute vs fused', speedup: r.recomputeVsFused.speedup });
    if (r.handVsCompiler?.verdict === 'win') losses.push({ id: r.id, comparison: `compiler (${r.handVsCompiler.compiler}) vs hand-written`, speedup: 1 / r.handVsCompiler.speedup });
  }
  const counts = { workloads: rows.length, timed: rows.filter((r) => !r.failedGate).length, failedGate: rows.filter((r) => r.failedGate).length,
    withViolations: rows.filter((r) => r.violations && Object.keys(r.violations).length).length };
  const fusedVerdicts = { win: 0, loss: 0, noise: 0 };
  for (const r of rows) if (r.fused) fusedVerdicts[r.fused.verdict]++;
  const start = record.power.start;
  return { sourceHash: record.sourceHash, adapter: { vendor: record.adapter.vendor, architecture: record.adapter.architecture, timestampQuery: record.adapter.timestampQuery },
    power: { start: start?.activeMode ?? null, onAC: start?.onAC ?? null, acOverlay: start?.ActiveOverlayAcPowerScheme?.name ?? null,
      checks: record.power.checks.length, changed: record.power.checks.some((c) => c.activeMode !== start?.activeMode || c.onAC !== start?.onAC),
      recorded: record.recorded?.power?.activeMode ?? null },
    llm: { commit: record.llmKernels.commit, sha256: record.llmKernels.sha256 }, counterSelfTest: record.counterSelfTest, ceiling, counts, fusedVerdicts, predictions: P, losses, rows };
}
