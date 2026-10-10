# tensor-compiler-webgpu

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![tests](https://github.com/abho7/tensor-compiler-webgpu/actions/workflows/tests.yml/badge.svg)](https://github.com/abho7/tensor-compiler-webgpu/actions/workflows/tests.yml)

A small tensor compiler, written from scratch: a static-shape tensor IR, reverse-mode
autodiff as an IR-to-IR transform, a float64 CPU interpreter that defines what every
program means, and, from milestone 2, lowering to WGSL compute kernels so that a small
model can train in the browser. No ML framework in the engine. PyTorch is used offline,
once, to write golden files; they are committed, and CI never needs it.

**Report:** https://abho7.github.io/tensor-compiler-webgpu/ (every figure rendered from
[`golden/measurements.json`](golden/measurements.json))

## The question

When does kernel fusion pay off on an integrated GPU, and when does it lose? Answering it
needs measured numbers, including the cases where fusion is neutral or slower. And a fused
kernel is only worth timing once it is known to compute what the unfused program computes.
So the first milestone is the reference that everything after it is checked against.

| milestone | what | status |
|---|---|---|
| 1 | tensor IR, autodiff, float64 reference; fuzzed, and checked against PyTorch | **done** |
| 1.5 | hardening: comparison metric v2, generator v2, per-bug detection rates | **done** |
| 2 | lowering to WGSL, fuzzed against the interpreter under the float32 budget | not started |
| 3 | train an MLP, then a small transformer, in the browser; loss curves against PyTorch | not started |
| 4 | fusion and autotuning, timed against llm-inference-webgpu's hand-written kernels | not started |

## Design

The full design, with the error budget, is [DESIGN.md](DESIGN.md). It was committed before
any code, and changes made after measuring are appended as dated amendments rather than
edited in, so the history shows what was fixed in advance. In short:

**The IR.** A list of nodes in SSA form, each with a fully static shape inferred when it is
created. Ops: `add sub mul div neg exp log relu tanh matmul sum broadcast reshape transpose`,
plus `max` and `stopGradient` (so softmax can subtract its maximum and not overflow float32)
and `step` (relu's gradient mask). Binary ops need identical shapes. **Broadcasting is
explicit**: the builder accepts NumPy-style shapes and inserts `broadcast` nodes, so autodiff
has exactly one place to sum a gradient back down and the code generator has one op to
account for. `softmax` and `logSoftmax` are built from the ops, not added as ops.

**Autodiff.** Reverse mode, as a transform that returns a larger graph: the forward nodes,
an input per output for its cotangent, and backward nodes built through the same builder.
The result is an ordinary graph, so it runs on the same interpreter and can be
differentiated again. `max` has no gradient rule; reaching it throws rather than silently
contributing zero. The rules are one table, and the transform accepts a replacement table,
which is how the sabotage tests swap in a wrong rule without editing the source.

**The interpreter.** `Float64Array`s, plain loops in index order, no cleverness. It is the
definition later backends are judged against, and it is itself judged against PyTorch.

**The error budget** (fixed before measuring):

| comparison | budget |
|---|---|
| autodiff vs finite differences, per element, `abs(g - fd) / max(1, abs(g), abs(fd))` | 1e-6 |
| values and gradients vs PyTorch float64, per tensor, `max abs diff / max abs(torch)` (v1, milestone 1) | 1e-12 |
| the same, per element, `abs diff <= atol + rtol * abs(torch)`, `atol = 1e-12 * max(1, max abs(torch))` (v2, from milestone 1.5) | rtol 1e-12 |
| from M2, GPU float32 vs this interpreter | per op class, from the WGSL spec's accuracy: see DESIGN.md |

## How milestone 1 is checked

Every check has to be able to fail, and every checker was tested by injecting a bug and
confirming it was reported.

1. **Gradients against finite differences on random programs.** A seeded generator builds
   well-typed graphs over every op. Points where finite differences mean nothing (a value
   past 10⁴, a log or denominator near zero, a perturbation that crosses a relu kink) are
   redrawn, and the redraws are counted.
2. **Second order:** the gradient graph differentiated again, against finite differences of
   the gradient.
3. **PyTorch golden files.** Random programs plus hand-built softmax, log-softmax, MLP,
   attention and axis-permutation graphs, rebuilt op for op in PyTorch float64 by
   [`oracle/torch_golden.py`](oracle/torch_golden.py), gradients from autograd. Softmax is
   compared with `torch.softmax` itself, including logits near 800, where an unshifted softmax
   overflows float64. CI rebuilds the cases from their seeds and fails if they no longer match
   the committed ones, so the goldens cannot go stale silently.
4. **Sabotage.** Every gradient rule replaced, one at a time, by a plausible wrong one; the
   fuzzer runs until it catches it, and the program count is recorded. Four interpreter kernels
   are broken the same way to test the PyTorch comparison, which finite differences cannot
   replace: they differentiate whatever the interpreter computes.

## What it found

The numbers are on the [report](https://abho7.github.io/tensor-compiler-webgpu/); these are
the findings.

- **A bug in the builder.** Node ids are numbers, and the builder accepted numbers as scalar
  literals, so every `add(x, y)` added two constants equal to the ids. The fuzzer's first run
  failed on it at once. Literals are gone; scalars are `const` nodes.
- **A false alarm in my own checker.** Plain central differences reported a wrong gradient on
  program 496, which divides by a sum near zero. Shrinking the step showed autodiff was right
  and the estimate's truncation error was not. The checker now uses Richardson extrapolation.
  The budget did not move, and the old method's false alarm stays in the record.
- **A blind spot in my PyTorch metric.** Program 9 computes `x0 / x0`, whose gradient is exactly
  zero; both sides return different rounding residue near 1e-16, and dividing by PyTorch's
  largest value scores that as an error of 2. It is recorded as a failure of the budget as
  written, and the test pins it to exactly that.
- **Bugs the random set cannot see.** All 15 sabotaged gradient rules were caught, but a
  reshape-for-transpose mix-up and a wrong permutation inverse took over a hundred and nearly
  a hundred programs, because they only show on particular shapes. A transpose *kernel* bug
  failed none of the random golden cases; a hand-built case was added for it. And a uniform
  relative error in `exp` passes every softmax case, because it cancels in the ratio.

## Milestone 1.5: hardening before the GPU

Decided before any GPU measurement exists, and recorded in DESIGN.md as dated amendments.
Every figure here is in [`golden/measurements.json`](golden/measurements.json) and on the report;
test/docs.test.js checks the ones quoted below.

- **Metric v2.** Per element, `abs(ours - ref) <= atol + rtol * abs(ref)` with
  `atol = 1e-12 * max(1, max abs(ref))` and `rtol = 1e-12`. Every case is scored under both
  metrics: v1 passes 111 of 112, v2 passes 112 of 112, and the difference is exactly
  `random-9`. To show v2 is not too loose, every gradient was multiplied by 1 + 1e-4: v2 flags
  101 of the 102 cases whose gradient changed, and the one it misses is `random-9`, whose true
  gradient is zero (the "error" is residue times 1e-4). Each rule scaled by 1 + 1e-4 on its own:
  every miss is again `random-9`.
- **Skipped programs: 114 of 3,000 (3.8%) before, 0 of 2,000 after.** The skips were the
  generator's fault: 94 of 114 applied `log` to a value that could be negative, 18 divided by
  one that could reach zero. Generator v2 carries a sound interval for every value and emits
  `log`, `div` and `exp` only where the interval proves them safe; a test checks every value
  of 300 programs against its interval.
- **Detection rates**, the fraction of programs containing a sabotaged op that catch it,
  before and after over the whole v2 corpus: matmul 14.0% to 49.3%, transpose 5.4% to 53.9%,
  reshape 3.6% to 42.1%. **The 50% target is met for transpose and missed for matmul and
  reshape.** Targeted programs alone reach 98.5%, 94.0% and 73.0%; general programs alone
  27.9%, 30.2% and 30.1%. The corpus mix was not changed to reach the target.
- **What moved the rates most was not shapes.** Programs used to output only their last value,
  so most ops were dead code, and no checker catches a bug no output depends on. Every
  unconsumed value is now an output. The remaining misses in general programs are shapes where
  the bug cannot show (an inner dimension of 1, where a reshape and a transpose coincide) and
  constant operands with no gradient to get wrong. The targeted builders were shaped after
  seeing that, so their rate measures reach for these known bugs; the general rate is the less
  biased one.

## Running it

```
node --test                # everything, including recomputing measurements.json (minutes)
node tools/measure.js      # rewrite golden/measurements.json
node tools/serve.js        # the report at http://localhost:8099/
```

Regenerating the PyTorch goldens (only needed if the cases change):

```
py -3 -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
node tools/export-cases.js
.venv/Scripts/python oracle/torch_golden.py
```

## Layout

```
src/ir/         graph, shape rules, builder (broadcasting, softmax, log-softmax)
src/autodiff.js the transform and its rule table
src/interp.js   the float64 reference interpreter
src/testing/    RNG, random programs (v1 frozen, v2), gradient checker, sabotage, golden comparison, measurement
oracle/         torch_golden.py (offline only)
golden/         cases.json, torch.json, measurements.json
tools/          export-cases.js, measure.js, serve.js
test/           node --test
```

## Licence

MIT
