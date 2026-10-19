import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const text = (p) => readFileSync(new URL(p, import.meta.url), 'utf8');
const m = JSON.parse(text('../golden/measurements.json')).m3b;
const readme = text('../README.md');
const design = text('../DESIGN.md');
const loose = (s) => new RegExp(s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/ /g, '\\s+'));
const sig = (v) => v.toExponential(1).replace('e-', 'e-');

test("the README's milestone 3b figures match golden/measurements.json", () => {
  const c = m.cpu, g = m.gpu, tf = g.teacherForced, st = g.statistical;
  assert.match(readme, loose(`${m.model.params.synthetic.toLocaleString('en-US')} parameters on the synthetic task, ${m.model.params.text.toLocaleString('en-US')} on the demo text`));
  assert.match(readme, loose(`gradient, ${c.init.filter((x) => x.pass).length} of ${c.init.length} configs within ${c.tau1.toExponential().replace('e-', 'e-')}`.replace('1e-10', '1e-10')));
  assert.match(readme, loose(`${c.training.filter((x) => x.pass).length} of ${c.training.length} runs, ${c.training[0].steps} steps each, within 1e-8`));
  assert.equal(c.tau2, 1e-8);
  assert.match(readme, loose(`infinite for ${Math.round(100 * tf.whole.unjudgeable / tf.whole.elements)}% of a step's elements, so all ${tf.whole.unjudged} of ${tf.steps} steps are unjudged`));
  assert.equal(tf.whole.passed, 0);
  assert.equal(tf.whole.judgedFailures, 0, 'the README says none failed on the elements it could judge');
  assert.match(readme, loose(`budget: ${tf.local.passed} of ${tf.steps} steps pass`));
  const causalCpu = c.properties.causality, orderCpu = c.properties.batchOrder;
  assert.match(readme, loose(`bitwise, ${causalCpu.length} of ${causalCpu.length} and ${orderCpu.length} of ${orderCpu.length} on the interpreter and on the GPU`));
  assert.ok(causalCpu.every((x) => x.pass) && orderCpu.every((x) => x.pass));
  assert.equal(g.properties.causality.passed, causalCpu.length);
  assert.equal(g.properties.batchOrder.passed, orderCpu.length);
  assert.equal(g.demo.causalityGpu.passed, g.demo.causalityGpu.trials, 'causality also holds on the trained demo model');
  assert.match(readme, loose(`${g.demo.greedy.agree} of ${g.demo.greedy.k} characters agree`));
  assert.match(readme, loose(`${st.seeds} seeds, final validation loss equivalent`));
  assert.ok(st.gpuVsTorch64.pass);
  assert.match(readme, loose(`is ${sig(st.gpuVsTorch64.lower)} to ${sig(st.gpuVsTorch64.upper)}`));
  assert.match(readme, loose(`about ${Math.round(st.gpuVsTorch64.sd / st.torch32VsTorch64.sd)} times PyTorch-float32's`));
  const bugs = g.sabotage.bugs;
  assert.match(readme, loose(`all ${bugs.length} caught`));
  assert.ok(bugs.every((b) => b.caughtFirstBy));
  assert.equal(bugs.filter((b) => b.checks.torchInit.caught > 0).length, bugs.length);
  assert.ok(bugs.every((b) => Object.values(b.asPredicted).every(Boolean)), 'the README says every prediction held');
  assert.deepEqual(bugs.filter((b) => b.checks.causality.caught > 0).map((b) => b.name), ['mask-off-by-one', 'softmax-wrong-axis', 'merge-no-transpose']);
});

test("DESIGN.md's milestone 3b revision and findings match golden/measurements.json", () => {
  const v = m.vacuity;
  assert.match(design, loose(`${v.unjudgeable.toLocaleString('en-US')} of ${v.elements.toLocaleString('en-US')} elements (${(100 * v.unjudgeable / v.elements).toFixed(1)}%)`));
  assert.match(design, loose(`the first at node ${v.firstUnbounded.node} of ${v.nodes.toLocaleString('en-US')}`));
  assert.match(design, loose(`finite bound on ${v.logits[0].finite} of its ${v.logits[0].logits.toLocaleString('en-US')} logits (V = ${v.logits[0].V}) and ${v.logits[1].finite} of ${v.logits[1].logits.toLocaleString('en-US')} (V = ${v.logits[1].V})`));
  const sb = v.streamMaxBound;
  assert.match(design, loose(`goes from ${sig(sb.embedding)} after the embedding to ${sig(sb.afterAttention0)} after block 0's attention, ${sb.afterMlp0.toFixed(2)} after its MLP`));
  const g = m.gpu, tf = g.teacherForced, st = g.statistical, d = g.demo;
  assert.match(design, loose(`0 of ${tf.steps} steps pass and ${tf.whole.unjudged} are unjudged`));
  assert.match(design, loose(`The node-local check: ${tf.local.passed} of ${tf.steps} steps`));
  assert.match(design, loose(`is ${sig(st.gpuVsTorch64.lower)} to ${sig(st.gpuVsTorch64.upper)}, against a margin of ±${st.margin}`));
  const worstSeed = Math.max(Math.abs(st.gpuVsTorch64.min), Math.abs(st.gpuVsTorch64.max));
  assert.match(design, loose(`differ by more than ${sig(worstSeed)}`));
  assert.match(design, loose(`about ${Math.round(st.gpuVsTorch64.sd / st.torch32VsTorch64.sd)} times larger`));
  assert.match(design, loose(`about ${Math.round(d.maxLossDifferenceFromTorch64 / d.torch32VsTorch64MaxLossDiff)} times larger`));
  assert.match(design, loose(`agreed on ${d.greedy.agree} of ${d.greedy.k} characters`));
});
