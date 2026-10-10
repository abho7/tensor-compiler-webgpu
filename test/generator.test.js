import { test } from 'node:test';
import assert from 'node:assert/strict';
import { randomProgram, corpusProgram, TARGETS, MAX_SIZE } from '../src/testing/random-graph.js';
import { checkGradients } from '../src/testing/gradcheck.js';
import { evaluate } from '../src/interp.js';
import { Rng } from '../src/testing/rng.js';
import { size } from '../src/ir/shapes.js';

const corpus = Array.from({ length: 300 }, (_, k) => corpusProgram(k + 1));

test('the interval analysis is sound: every value lies inside its interval', () => {
  // Valid-by-construction rests on this, so it is checked at random inputs
  // across the whole input range, endpoints included.
  for (const p of corpus) {
    const rng = new Rng(p.seed ^ 0x5bd1e995);
    for (let t = 0; t < 5; t++) {
      const inputs = {};
      for (const n of p.graph.inputs()) {
        inputs[n.attrs.name] = Array.from({ length: size(n.shape) }, () => (t === 0 ? rng.pick([-2, 2]) : rng.uniform(-2, 2)));
      }
      const values = evaluate(p.graph, inputs);
      p.graph.nodes.forEach((n, id) => {
        const [lo, hi] = p.bounds[id];
        const slack = 1e-9 * Math.max(1, Math.abs(lo), Math.abs(hi));
        for (const v of values[id]) {
          assert.ok(v >= lo - slack && v <= hi + slack, `seed ${p.seed} node ${id} (${n.op}): ${v} outside [${lo}, ${hi}]`);
        }
      });
    }
  }
});

test('v2 programs are valid by construction: none is skipped', () => {
  for (let k = 0; k < 150; k++) {
    const r = checkGradients(corpus[k], { seed: k + 1 });
    assert.equal(r.status, 'agree', `program ${k + 1}: ${r.status} ${r.detail}`);
  }
});

test('every op is live: each computed node reaches an output', () => {
  for (const p of corpus) {
    const live = new Set(p.outputs);
    for (let id = p.graph.nodes.length - 1; id >= 0; id--) {
      if (live.has(id)) for (const i of p.graph.nodes[id].inputs) live.add(i);
    }
    for (const n of p.graph.nodes) {
      if (n.op === 'input' || n.op === 'const') continue;
      assert.ok(live.has(n.id), `seed ${p.seed}: node ${n.id} (${n.op}) is dead code`);
    }
  }
});

test('the shapes the brief asked for appear', () => {
  const nodes = corpus.flatMap((p) => p.graph.nodes);
  assert.ok(nodes.some((n) => n.shape.length >= 3), 'no rank-3 value');
  assert.ok(nodes.some((n) => n.shape.length === 4), 'no rank-4 value');
  assert.ok(nodes.some((n) => n.shape.includes(1) && n.shape.length >= 2), 'no size-1 dimension');
  assert.ok(nodes.some((n) => n.op === 'matmul' && n.shape.at(-1) !== n.shape.at(-2)), 'no non-square matmul');
  assert.ok(nodes.every((n) => size(n.shape) <= MAX_SIZE), `a tensor over ${MAX_SIZE} elements`);
});

test('a targeted program contains its op', () => {
  for (const op of TARGETS) {
    for (let seed = 1; seed <= 40; seed++) {
      const p = randomProgram(seed, { focus: op });
      assert.ok(p.graph.nodes.some((n) => n.op === op), `${op} seed ${seed} has no ${op}`);
    }
  }
});

test('the same seed gives the same program', () => {
  assert.deepEqual(corpusProgram(17).graph.toJSON(), corpusProgram(17).graph.toJSON());
  assert.deepEqual(randomProgram(5, { focus: 'matmul' }).graph.toJSON(), randomProgram(5, { focus: 'matmul' }).graph.toJSON());
});
