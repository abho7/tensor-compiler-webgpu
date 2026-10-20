import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Graph } from '../src/ir/graph.js';
import { lower } from '../src/gpu/lower.js';
import { referenceWithBounds, unjudgeable } from '../src/gpu/bounds.js';
import { fuse, checkSemantics, bodyOf, MAX_OPERANDS, scalarConst } from '../src/fusion/fuse.js';
import { lowerFused } from '../src/fusion/lower-fused.js';
import { fusedKernel, FUSED_SABOTAGE, literal } from '../src/fusion/codegen-fused.js';
import { fusedDirectCases, layerNormGraph } from '../src/fusion/direct-cases-m4.js';
import { workloads, buildVariants, streamingReference, toF16, f16ToNumber } from '../src/fusion/workloads.js';
import { corpusGraphs, modelGraphs } from '../src/fusion/measure-m4.js';
import * as llm from '../src/fusion/llm-kernels.js';

const spec = JSON.parse(readFileSync(new URL('../golden/spec-m3b.json', import.meta.url)));

/** Every fused node of a fusion result. */
const fusedNodes = (f) => f.graph.nodes.filter((n) => n.op === 'fused');

/**
 * Default fusion never recomputes (amendment 30): inside a body, a broadcast or
 * transpose reads a body input or a constant, except a rowwise group's row
 * value broadcast back over its own row.
 */
function recomputes(node) {
  const b = bodyOf(node);
  return b.nodes.some((n) => {
    if (n.op !== 'broadcast' && n.op !== 'transpose') return false;
    const src = b.nodes[n.inputs[0]];
    if (src.op === 'input' || src.op === 'const') return false;
    const rowBroadcast = node.attrs.kind === 'rowwise' && n.op === 'broadcast' && src.shape.at(-1) === 1 && src.shape.length === n.shape.length;
    return !rowBroadcast;
  });
}

test('fusion preserves semantics bitwise on corpus programs and the model, and default fusion never recomputes (amendments 29, 30)', () => {
  for (let i = 1; i <= 300; i++) {
    for (const c of corpusGraphs(i)) {
      const f = fuse(c.graph, { keep: c.keep });
      assert.ok(checkSemantics(c.graph, f, c.inputs).identical, `program ${i} ${c.label}`);
      for (const n of fusedNodes(f)) {
        assert.ok(!recomputes(n), `program ${i}: a default-fused body recomputes across a move`);
        assert.ok(n.inputs.length <= MAX_OPERANDS);
      }
    }
  }
  for (const c of modelGraphs(spec)) {
    const f = fuse(c.graph, { keep: c.keep });
    assert.ok(checkSemantics(c.graph, f, c.inputs).identical, c.label);
    for (const n of fusedNodes(f)) assert.ok(!recomputes(n), `${c.label}: recomputes`);
  }
});

test('the recompute option does recompute, and still preserves semantics', () => {
  const w = workloads().find((x) => x.id === 'broadcast-recompute-R16');
  const { graph, outputs, inputs } = w.make();
  const plain = fuse(graph, { keep: outputs }), re = fuse(graph, { keep: outputs, recompute: true });
  assert.equal(fusedNodes(plain).length, 2, 'the producer stored once, then a load through the broadcast');
  assert.equal(fusedNodes(re).length, 1);
  assert.ok(recomputes(fusedNodes(re)[0]));
  assert.ok(checkSemantics(graph, re, inputs).identical);
});

test('a producer used outside its group is never absorbed; the sabotaged pass absorbs it, and semantics catch that', () => {
  const g = new Graph();
  const x = g.input('x', [8]);
  const t = g.exp(g.neg(x));
  g.tanh(g.mul(t, g.input('w', [8])));
  g.matmul(g.reshape(t, [1, 8]), g.input('m', [8, 2]));
  const inputs = { x: [...Array(8).keys()].map((i) => i / 4 - 1), w: [...Array(8).keys()].map((i) => 1 - i / 8), m: [...Array(16).keys()].map((i) => i / 16) };
  const honest = fuse(g);
  assert.ok(checkSemantics(g, honest, inputs).identical);
  // exp(neg(x)) is read by the tanh's group and by the matmul, so it is stored.
  assert.ok(honest.index.has(t), 'the shared producer survives as its own value');
  let caught;
  try {
    caught = !checkSemantics(g, fuse(g, { sabotage: 'fusion-absorbs-shared' }), inputs).identical;
  } catch { caught = true; }
  assert.ok(caught, 'the pass sabotage must be visible');
});

test('lowering a graph with no fused nodes gives milestone 2\'s launches exactly', () => {
  for (let i = 1; i <= 60; i++) {
    for (const c of corpusGraphs(i)) {
      for (const matmul of ['naive', 'tiled']) {
        const a = lower(c.graph, { matmul }).launches, b = lowerFused(c.graph, { matmul }).launches;
        assert.equal(a.length, b.length);
        a.forEach((l, k) => {
          assert.equal(b[k].code, l.code);
          assert.deepEqual([...b[k].params], [...l.params]);
          assert.deepEqual(b[k].grid, l.grid);
          assert.deepEqual(b[k].inputs, l.inputs);
          assert.equal(b[k].node, l.node);
        });
      }
    }
  }
});

test('every direct case fuses into exactly the kernel it is meant to test, and is judgeable', () => {
  const cases = fusedDirectCases();
  assert.ok(cases.length >= 90);
  for (const c of cases) {
    const f = fuse(c.graph);
    const launched = f.graph.nodes.filter((n) => n.op !== 'input' && n.op !== 'const');
    assert.equal(launched.length, 1, c.name);
    assert.equal(launched[0].op, 'fused', c.name);
    assert.equal(launched[0].attrs.kind, c.kind, c.name);
    assert.equal(launched[0].attrs.pattern, c.pattern, c.name);
    const ref = referenceWithBounds(c.graph, c.inputs);
    assert.equal(unjudgeable(ref.values, ref.bounds), 0, c.name);
  }
  const kinds = new Set(cases.map((c) => c.pattern));
  for (const p of ['elementwise', 'matmul-epilogue', 'layernorm', 'softmax', 'log-softmax']) assert.ok(kinds.has(p), p);
});

test('each kernel sabotage changes the kernels it targets and no others', () => {
  const cases = fusedDirectCases();
  const changed = Object.fromEntries(FUSED_SABOTAGE.map((s) => [s, new Set()]));
  for (const c of cases) {
    const [node] = fusedNodes(fuse(c.graph));
    const honest = fusedKernel(node, { matmul: c.matmul ?? 'naive' }).code;
    for (const s of FUSED_SABOTAGE) if (fusedKernel(node, { matmul: c.matmul ?? 'naive', sabotage: s }).code !== honest) changed[s].add(c.pattern);
  }
  assert.deepEqual([...changed['epilogue-before-bias']], ['matmul-epilogue']);
  assert.deepEqual([...changed['layernorm-tree-short']], ['layernorm']);
  assert.deepEqual([...changed['softmax-no-max']].sort(), ['log-softmax', 'softmax']);
  assert.deepEqual([...changed['fused-bounds-off-by-one']], ['elementwise']);
});

test('generated kernels: constants are exact float32 bits, tanh is clamped as in the unfused kernel, sizes come from P', () => {
  assert.equal(literal(1), 'bitcast<f32>(0x3f800000u)');
  assert.equal(literal(-1e4), 'bitcast<f32>(0xc61c4000u)');
  const g = new Graph();
  g.tanh(g.mul(g.input('x', [70]), g.const(0.1)));
  const [n] = fusedNodes(fuse(g));
  const k = fusedKernel(n);
  assert.match(k.code, /tanh\(clamp\(.*-20\.0, 20\.0\)\)/);
  assert.match(k.code, /bitcast<f32>\(0x3dcccccdu\)/, 'fround(0.1)');
  assert.deepEqual([...k.params], [70]);
  // One text for every size of the same structure (amendment 30).
  const g2 = new Graph();
  g2.tanh(g2.mul(g2.input('x', [100003]), g2.const(0.1)));
  assert.equal(fusedKernel(fusedNodes(fuse(g2))[0]).code, k.code);
  // A non-finite constant is never inlined.
  assert.equal(scalarConst({ op: 'const', attrs: { value: Infinity } }), false);
});

test('LayerNorm, softmax and log-softmax fuse whole when nothing else reads their insides', () => {
  const g = new Graph();
  layerNormGraph(g, g.input('x', [5, 64]), g.input('g', [64]), g.input('b', [64]));
  assert.equal(fusedNodes(fuse(g))[0].attrs.pattern, 'layernorm');
  const s = new Graph(); s.softmax(s.input('x', [3, 65]), 1);
  assert.equal(fusedNodes(fuse(s))[0].attrs.pattern, 'softmax');
  // Size-1 rows are left to elementwise fusion (amendment 30 needs C > 1).
  const one = new Graph(); one.softmax(one.input('x', [3, 1]), 1);
  assert.ok(fusedNodes(fuse(one)).every((n) => n.attrs.kind !== 'rowwise'));
});

test('revision R5: the training step has no full LayerNorm, softmax or activation epilogue; the forward pass has them all', () => {
  const [fwd, step] = modelGraphs(spec);
  const pf = fuse(fwd.graph, { keep: fwd.keep }), ps = fuse(step.graph, { keep: step.keep });
  const patterns = (f) => new Set(fusedNodes(f).map((n) => n.attrs.pattern));
  for (const p of ['layernorm', 'softmax', 'log-softmax', 'matmul-epilogue']) assert.ok(patterns(pf).has(p), p);
  for (const p of ['layernorm', 'softmax', 'log-softmax']) assert.ok(!patterns(ps).has(p), p);
  const act = (f, g) => f.groups.filter((x) => x.kind === 'epilogue' && [...x.members].some((id) => ['tanh', 'relu'].includes(g.nodes[id].op))).length;
  assert.equal(act(pf, fwd.graph), 2, 'one GELU epilogue per block');
  assert.equal(act(ps, step.graph), 0);
});

test('streamingReference gives exactly referenceWithBounds, holding only live values (revision R6)', () => {
  const ws = workloads();
  for (const id of ['layernorm-R64-C65', 'softmax-R64-C7', 'matmul-tiled-gelu-512x64x64', 'chain-unary-L16-n1000', 'chain-binary-L8-n65537', 'lost-reuse-1op', 'baseline-rmsnorm-rows52']) {
    const w = ws.find((x) => x.id === id);
    const { graph, outputs, inputs } = w.make();
    const s = streamingReference(graph, inputs, outputs);
    const r = referenceWithBounds(graph, inputs);
    for (const o of outputs) {
      const a = s.get(o);
      assert.ok(a.value.every((v, i) => Object.is(v, r.values[o][i])), `${id} values`);
      assert.ok(a.bound.every((v, i) => Object.is(v, r.bounds[o][i])), `${id} bounds`);
    }
  }
});

test('the sweep is what amendment 34 and revisions R5 and R6 list', () => {
  const ws = workloads();
  const ids = new Set(ws.map((w) => w.id));
  assert.equal(ids.size, ws.length);
  const by = (f) => ws.filter((w) => w.family === f);
  assert.equal(by('chain-unary').length, 20);
  assert.equal(by('chain-binary').length, 8);
  assert.equal(by('matmul-epilogue').length, 28);
  for (const f of ['layernorm', 'softmax']) {
    assert.ok(by(f).every((w) => w.params.R * w.params.C <= 2 ** 22));
    assert.equal(by(f).length, 23);
  }
  assert.deepEqual(by('broadcast-recompute').map((w) => w.params.R), [16, 256, 4096]);
  assert.deepEqual(by('forward').map((w) => w.id), ['forward-V16-B8', 'forward-V75-B16']);
  assert.deepEqual(by('training-step').map((w) => w.id), ['step-V16-B8', 'step-V16-B16', 'step-V75-B16']);
  // Sizes off the workgroup and tile grids are included.
  assert.ok(by('chain-unary').some((w) => w.params.n % 64 !== 0));
  assert.ok(by('matmul-epilogue').some((w) => w.params.M % 16 !== 0 && w.params.N % 16 !== 0));
});

test('the hand-written kernels are llm-inference-webgpu\'s, unedited (amendment 37)', () => {
  const sha = (s) => createHash('sha256').update(s).digest('hex');
  for (const name of ['ADD', 'SWIGLU', 'RMSNORM', 'MATVEC']) assert.equal(sha(llm[name]), llm.SHA256[name], name);
  assert.equal(llm.SOURCE.commit, '22e2d5b');
  // The matvec workloads build an f16 copy and an f32 copy of the same weights.
  const w = workloads().find((x) => x.id === 'baseline-matvec-896x896-naive');
  const b = buildVariants(w);
  assert.equal(b.programs.hand.buffers[0].bytes, 896 * 896 * 2);
  assert.equal(b.programs['hand-f32'].buffers[0].bytes, 896 * 896 * 4);
  assert.ok(!b.programs['hand-f32'].dispatches[0].code.includes('f16>'));
  // f16 rounding: round to nearest even, and back exactly.
  assert.deepEqual([...toF16([1, 65504, 1 + 2 ** -11, 1 + 3 * 2 ** -11, 2 ** -24, 1e6])], [0x3c00, 0x7bff, 0x3c00, 0x3c02, 0x0001, 0x7c00]);
  assert.equal(f16ToNumber(0x3c01), 1 + 2 ** -10);
});
