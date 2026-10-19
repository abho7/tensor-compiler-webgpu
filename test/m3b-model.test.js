import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Graph } from '../src/ir/graph.js';
import { grad } from '../src/autodiff.js';
import { buildStep, stateSpec, mlpLogits, PARAMS, ADAM } from '../src/train/step.js';
import { adamUpdate, adamStateSpec } from '../src/m3b/adam.js';
import { paramCount, buildForward, CONFIG, causalMask, MODEL_BUGS } from '../src/m3b/model.js';
import { buildStep3b } from '../src/m3b/step.js';
import { buildSpec3b } from '../src/m3b/spec.js';
import { TEXT, encodeText, periodicCopy, periodicCopyFloor } from '../src/m3b/data.js';
import { OPS } from '../src/ir/shapes.js';

const root = new URL('../', import.meta.url);
const alice = readFileSync(new URL('data/alice/alice.txt', root));

test("the generalised Adam builder produces milestone 3a's step graph node for node (DESIGN.md amendment 25)", () => {
  const dims = { inputs: 8, hidden: 16, classes: 4 }, batch = 32;
  for (const sabotage of [undefined, 'adam-bias-correction', 'weight-decay-sign']) {
    const ref = buildStep(dims, batch, ADAM, { sabotage });
    // step.js's construction up to the optimizer, then the generalised builder.
    const g = new Graph();
    const shapes = { W1: [8, 16], b1: [16], W2: [16, 4], b2: [4] };
    const ids = {};
    for (const p of PARAMS) ids[p] = g.input(p, shapes[p]);
    ids.x = g.input('x', [batch, 8]);
    ids.y = g.input('y', [batch, 4]);
    const logp = g.logSoftmax(mlpLogits(g, ids), 1);
    const loss = g.div(g.neg(g.sum(g.sum(g.mul(ids.y, logp), 1), 0)), g.const(batch));
    const G = grad(g, [loss], PARAMS.map((p) => ids[p]));
    const state = {};
    assert.deepEqual(adamStateSpec(PARAMS, shapes).map(([n]) => n), Object.keys(stateSpec(ADAM, dims)));
    for (const [name, shape] of adamStateSpec(PARAMS, shapes)) state[name] = G.graph.input(name, shape);
    const outputs = adamUpdate(G.graph, PARAMS, ids, Object.fromEntries(PARAMS.map((p, k) => [p, G.grads[k]])), state, ADAM, { sabotage });
    assert.deepEqual(G.graph.toJSON(), ref.graph.toJSON(), `graphs differ (sabotage ${sabotage})`);
    assert.deepEqual(outputs, ref.outputs);
  }
});

test('the model has the parameter counts DESIGN.md amendment 20 states, and uses only existing ops', () => {
  assert.equal(paramCount(16), 105872);
  assert.equal(paramCount(75), 113483);
  assert.deepEqual([CONFIG.T, CONFIG.d, CONFIG.H, CONFIG.dh, CONFIG.L, CONFIG.ff, CONFIG.eps, CONFIG.mask], [64, 64, 4, 16, 2, 256, 1e-5, -1e4]);
  const s = buildStep3b(16, 2);
  const used = new Set(s.graph.nodes.map((n) => n.op));
  for (const op of used) assert.ok(OPS.includes(op), op);
  for (const bug of Object.keys(MODEL_BUGS)) assert.ok(buildForward(16, 2, { bug }).graph.nodes.length > 0);
});

test('the causal mask: zero on and below the diagonal, -1e4 above; the bug reaches one further', () => {
  const m = causalMask(4), b = causalMask(4, 'mask-off-by-one');
  assert.deepEqual(m.slice(0, 4), [0, -1e4, -1e4, -1e4]);
  assert.deepEqual(m.slice(12, 16), [0, 0, 0, 0]);
  assert.deepEqual(b.slice(0, 4), [0, 0, -1e4, -1e4]);
});

test('the demo text is the one DESIGN.md amendment 24 recorded, and the data are what the code builds', () => {
  assert.equal(createHash('sha256').update(alice).digest('hex'), TEXT.sha256);
  const text = alice.toString('utf8');
  const enc = encodeText(text);
  assert.equal(enc.vocab.length, 75);
  assert.equal(enc.ids.length, 144604);
  assert.ok(!/gutenberg/i.test(text), 'no Project Gutenberg licence or trademark text');
  const d = periodicCopy();
  assert.equal(d.train.length, 512);
  assert.equal(d.val.length, 128);
  assert.ok(d.train.every((s) => s.length === 65 && s.every((v, i) => v === s[i % 16])));
  assert.ok(Math.abs(periodicCopyFloor() - (15 / 64) * Math.log(16)) < 1e-15);
});

test('golden/spec-m3b.json is exactly what the code builds', () => {
  const committed = readFileSync(new URL('golden/spec-m3b.json', root), 'utf8');
  assert.equal(committed, JSON.stringify(buildSpec3b(alice.toString('utf8'))) + '\n');
});
