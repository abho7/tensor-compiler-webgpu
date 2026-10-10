# Design

A small tensor compiler: a static-shape tensor IR, reverse-mode autodiff as an
IR-to-IR transform, a float64 CPU interpreter that defines what every program
means, and (from milestone 2) lowering to WGSL compute kernels.

The question the project exists to answer: **when does kernel fusion pay off on
an integrated GPU, and when does it lose?** Everything before milestone 4 is
there so that the answer can be trusted: a fused kernel is only worth timing if
it is known to compute the same thing as the unfused one.

This document was written and committed before any code or measurement. The
tolerances in [Error budget](#error-budget) are fixed here so that no result can
move them after the fact. If a result breaks one, the result is recorded as a
failure; the budget is not widened to fit it.

## The IR

A program is a `Graph`: a list of nodes in SSA form, each defined once, each
naming its inputs by id, so the list is always in topological order. Every node
carries a fully static shape (an array of positive integers; `[]` is a scalar)
computed when the node is created. There is one element type, a real number:
float64 in the interpreter, float32 on the GPU. Tensors are dense and
row-major.

| op | inputs | attributes | result shape | rule |
|---|---|---|---|---|
| `input` | | `name`, `shape` | `shape` | a value supplied at run time |
| `const` | | `shape`, `value` (one number or a full array) | `shape` | |
| `add` `sub` `mul` `div` | a, b | | shape of a | a and b must have **identical** shapes |
| `neg` `exp` `log` `relu` `tanh` | x | | shape of x | |
| `step` | x | | shape of x | 1 where x > 0, else 0. Exists for relu's gradient |
| `matmul` | a, b | | `[...B, m, n]` | a is `[...B, m, k]`, b is `[...B, k, n]`, rank >= 2, batch dims identical |
| `sum` | x | `axis`, `keepDims` | x without `axis` (or with it set to 1) | |
| `max` | x | `axis`, `keepDims` | as `sum` | not differentiable; see below |
| `stopGradient` | x | | shape of x | identity forward, zero gradient |
| `broadcast` | x | `shape` | `shape` | NumPy rules: x's dims right-aligned, each 1 or equal |
| `reshape` | x | `shape` | `shape` | same element count |
| `transpose` | x | `perm` | permuted | `perm` is a permutation of the axes |

**Broadcasting is explicit.** The binary ops require identical shapes; the
builder accepts NumPy-style mismatched shapes and inserts `broadcast` nodes
itself. So every implicit expansion in a program is a node, which gives the
autodiff transform exactly one place to reduce a gradient back down (the
`broadcast` rule), and gives the code generator one op whose memory traffic
it has to account for, rather than an expansion hidden inside every binary op.

**Composites are built from the ops above, not added as ops.** The builder
provides `softmax(x, axis)` and `logSoftmax(x, axis)`:

```
m        = stopGradient(max(x, axis, keepDims))
z        = x - broadcast(m)
softmax  = exp(z) / broadcast(sum(exp(z), axis, keepDims))
logSoftmax = z - broadcast(log(sum(exp(z), axis, keepDims)))
```

Subtracting the maximum is what keeps `exp` from overflowing: float32 overflows
at exp(88.7), and attention scores reach that. `max` and `stopGradient` are
the two ops added beyond the softmax-from-the-others brief, for this reason.
The shift is exact: softmax is invariant to it, so its true gradient is zero,
and `stopGradient` states that rather than approximating it. `max` itself has
no gradient rule. Autodiff refuses (throws) if a gradient reaches a `max`
node, so a missing rule can never be silently treated as zero.

`step` exists because relu's gradient needs a mask and nothing else in the set
can produce one. Its gradient is zero, which is correct everywhere except at 0,
where nothing is differentiable anyway.

## Autodiff

Reverse mode, as a transform from a graph to a larger graph. `grad(graph,
outputs, wrt)` returns a new graph containing the forward nodes, one new
`input` per output for its cotangent, and backward nodes built from the same
ops through the same builder. Walk the forward nodes in reverse order, apply
each op's rule to its accumulated cotangent, and sum contributions with `add`
where a value is used more than once.

Because the result is an ordinary graph, it runs on the same interpreter,
lowers through the same code generator, and can be differentiated again. Rules:

| op | contribution to each input, given cotangent `g` and result `y` |
|---|---|
| `add` | `g`, `g` |
| `sub` | `g`, `-g` |
| `mul` | `g*b`, `g*a` |
| `div` | `g/b`, `-g*a/(b*b)` |
| `neg` | `-g` |
| `exp` | `g*y` |
| `log` | `g/x` |
| `relu` | `g*step(x)` |
| `tanh` | `g*(1 - y*y)` |
| `matmul` | `g @ bᵀ`, `aᵀ @ g` (transposing the last two axes) |
| `sum` | `broadcast(g)` back to x's shape (reshaped first if `keepDims` was false) |
| `broadcast` | `sum` over every expanded axis, then `reshape` to x's shape |
| `reshape` | `reshape(g, x.shape)` |
| `transpose` | `transpose(g, inverse(perm))` |
| `step`, `stopGradient`, `const`, `input` | nothing |
| `max` | throws |

The rules live in one table that the transform looks up, and the transform
accepts a replacement table. That lets the sabotage tests swap one rule for a
wrong one without editing the source.

## The reference interpreter

Evaluates a graph on `Float64Array`s, node by node, with no cleverness: no
fusion, no reuse, plain loops in index order. It is the semantic definition
every later backend is checked against. Its own check is PyTorch (below), which
it shares no code with.

## How M1 is checked

Each check has to be able to fail, and each checker is itself tested by
injecting a real bug and confirming that the checker reports it.

1. **Finite differences on random graphs (fuzzed).** A seeded generator builds
   random well-typed graphs over every op, with random shapes, and random
   inputs. The loss is `L = Σ w ⊙ f(x)` with a fixed random `w`. Autodiff's
   gradient with cotangent `w` is compared to central differences of `L` for
   every input element. Points too close to a kink or a domain edge (a relu
   input or a near-tie in `max` within δ of 0, a `log` argument or a `div`
   denominator within δ of 0, any non-finite value) are rejected and the inputs
   redrawn, and the number of rejections is reported rather than hidden.
   (Amended: see [Amendments](#amendments-after-measurement) for how kinks and
   near-ties are actually handled, and the change to the difference formula.)
2. **PyTorch golden files.** A fixed set of seeded random graphs, plus
   hand-built softmax, log-softmax, MLP and attention-shaped graphs, is
   exported as JSON. `oracle/torch_golden.py` rebuilds each one op for op in
   PyTorch float64 and records the outputs, and autograd's gradients for the
   same cotangent. The composites are compared against `torch.softmax` and
   `torch.log_softmax` themselves, not against a re-translation of our
   decomposition. The output files are committed, so CI never needs PyTorch.
   CI also regenerates the graphs from their seeds and checks they still match
   the committed ones, so a change to the generator cannot leave stale goldens
   passing.
3. **Sabotage.** Each gradient rule is replaced, one at a time, by a plausible
   wrong one (a dropped sign, a missing factor, a transposed operand, a missed
   broadcast axis). The fuzzer runs until it reports a mismatch, and the number
   of programs that took is recorded per rule. A rule whose sabotage is never
   caught within the program limit is recorded as a gap in the fuzzer.
4. **Second order.** Gradients of gradients on a subset of random graphs, also
   against finite differences.

## Error budget

Unit roundoff: float64 `u64 = 2^-53 ≈ 1.1e-16`, float32 `u32 = 2^-24 ≈ 6.0e-8`.

### M1: what the CPU reference must meet

| comparison | metric | budget |
|---|---|---|
| autodiff vs central differences (float64) | per element, `abs(g - fd) / max(1, abs(g), abs(fd))` | **1e-6** |
| interpreter outputs vs PyTorch float64 | per tensor, `max abs(ours - torch) / max(max abs(torch), 1e-300)` | **1e-12** |
| autodiff gradients vs PyTorch autograd float64 | same per-tensor metric | **1e-12** |
| our softmax / log-softmax vs `torch.softmax` / `torch.log_softmax` | same per-tensor metric | **1e-12** |

Why these numbers. Central differences with step `h = 1e-5 · max(1, abs(x))` carry
truncation error of order `h² · f'''` and rounding error of order
`u64 · abs(L) / h`, about 1e-11 · abs(L). The generator keeps values in a range where
`f'''` is modest, so a correct gradient sits several orders of magnitude inside
1e-6, while a wrong rule is typically wrong by order one. The PyTorch
comparison differs only by summation order and by a few ulp of `exp`, `log` and
`tanh` between JavaScript's libm and PyTorch's. Over graphs of a dozen nodes and
reductions of at most a few hundred terms that is around 1e-14, and 1e-12
leaves room for modest cancellation without hiding a real error.

### M2 onward: what the GPU (float32) must meet against the CPU (float64)

The reference is evaluated on the same inputs rounded to float32, so input
rounding is not counted against the kernel. Bounds follow the WGSL
specification's stated accuracy for each operation.

| op class | budget per element |
|---|---|
| `add` `sub` `mul` `neg` `relu` `step`; data movement (`broadcast` `reshape` `transpose`) | `abs(err) <= u32 · abs(ref)` (correctly rounded); data movement must be exact |
| `div` | `2.5 ulp`: `abs(err) <= 2.5 · 2u32 · abs(ref)` |
| `exp` | `(3 + 2·abs(x))` ulp |
| `log` | 3 ulp outside [0.5, 2]; absolute `2^-21` inside |
| `tanh` | absolute `2^-20` (WGSL inherits it from sinh/cosh; adapters have been measured missing transcendental bounds before, so this one is checked, not assumed) |
| `sum` over n, `max` over n, `matmul` with inner size k | `abs(err) <= γ_n · Σ abs(term)`, `γ_n = n·u32 / (1 - n·u32)` (k in place of n for matmul); `max` must be exact |
| a whole fused or unfused program | each node's budget composed along the graph; a fused kernel must meet the **same** budget as the unfused program it replaces |

M3 (training loss curves against PyTorch float32) will have its tolerance set
and committed before its first measurement, from these per-op bounds.

## Amendments after measurement

Recorded rather than edited in place, so the text above still shows what was
fixed before measuring. One metric has been replaced, and it was replaced
because of a result: random-9's failure under v1 (amendment 3) is why v2
(amendment 4) exists. v1's verdicts stay in the record beside v2's, and v2's
constants were set before any case was scored under it.

1. **The finite-difference formula** (2026-10-09). Plain central differences
   at `h = 1e-5 · max(1, abs(x))` raised a false alarm: near a pole their
   truncation error alone was 1.4e-6 relative, against an autodiff gradient
   that shrinking `h` showed to be right. The checker now uses Richardson
   extrapolation, `(4·D(h/2) - D(h)) / 3`, which cancels the `h²` term. The
   1e-6 budget is unchanged. The old method stays selectable and its false
   alarms are part of the record.
2. **Kinks and `max`.** Rather than a margin δ around relu inputs, the checker
   rejects a point if any perturbed evaluation changes the sign of any relu or
   step input, which is exact. `max` never needs a margin: the generator only
   emits it inside softmax and log-softmax, which are smooth across a tie in
   their maximum. A bare `stopGradient` is deliberately not the derivative,
   so the generator never emits one outside those composites.
3. **A blind spot in the PyTorch metric.** Normalising by `max abs(torch)` fails
   when the true value is exactly zero and both sides return rounding residue:
   `x / x` has gradient 0, and ours and PyTorch's are different values around
   1e-16, a "relative error" of 2. The case (`random-9`) is recorded as failing
   the M1 budget as stated, with its absolute figures. The M2 reduction and
   matmul budgets scale by the sum of the terms' absolute values rather than by
   the result, which should avoid this; whether the composed whole-program
   budget does is untested and is to be settled before M2 measures anything.

### Milestone 1.5 (2026-10-09), before any GPU measurement exists

4. **Metric v2 for whole-tensor comparisons.** Decided now, while no GPU
   result exists to tune it against. Per element:

   ```
   abs(ours - ref) <= atol + rtol · abs(ref)
   atol = atolScale · max(1, max abs(ref) over the tensor)
   ```

   with `rtol = atolScale = 1e-12` for the CPU (float64) against PyTorch
   (float64). The constants are the ones the M1.5 brief proposed, adopted before
   any case was scored under v2. A tensor's score is the largest ratio of the
   left side to the right; it passes at <= 1.

   Why: v1 divided by the largest reference value, so a tensor whose true value
   is exactly zero, computed by both sides as rounding residue, was judged
   against its own residue. v2's `atol` is floored at `atolScale · 1`, so
   residue around an exact zero is measured against a fixed absolute scale,
   and `rtol` keeps the large entries relative.

   What it gives up, stated before measuring: because `atol` is set by the
   tensor's largest entry, it is never smaller than `rtol · abs(ref)` for any
   entry, so v2 is never more than twice as loose as `max abs diff <=
   1e-12 · max(1, max abs ref)`. A relative error ε in one entry of magnitude r
   escapes when `ε · r <= atol + rtol · r`: for ε = 1e-4 that is entries below
   about 1e-8 · max(1, largest entry). Small entries beside large ones, and
   tensors whose values are all far below 1, are checked only absolutely.
   The (1 + 1e-4) injection measures what this misses in practice.

   v1 stays in the code and in the record: every case is scored under both, and
   the M1 figures are still v1's. M2 uses the same form for whole-tensor
   checks; its float32 constants are still to be fixed, from the per-op table
   above, before M2's first measurement.

5. **Generator v2.** The M1 generator (`random-graph-v1.js`) is frozen so M1's
   record and golden cases reproduce. v2 builds programs valid by
   construction: an interval for every value, sound for inputs in [-2, 2], and
   log, div and exp only where the interval proves them safe. v1's 3.8% of
   programs without a usable point were almost all `log` of a possibly
   negative value (94 of 114) or a denominator that could reach zero (18).
   v2 also adds larger and non-square shapes, size-1 dimensions, ranks up to 4,
   and targeted programs for matmul, transpose, reshape, broadcast and sum.

6. **Detection rate**, per sabotaged rule: the fraction of corpus programs
   containing the op that catch the bug. The v2 generator was revised twice
   after seeing rates, and both revisions are recorded here because they are
   exactly the kind of tuning a reader should know about:
   - First v2 (larger shapes, valid by construction): matmul, transpose and
     reshape still caught at 26.6%, 23.4% and 23.7% of a 1,000-program mixed
     corpus. Cause: v1 and first-v2 programs had a single output, the last
     value, and most earlier ops were dead code, which no checker can catch a
     bug in. Fix: every unconsumed value is an output.
   - Second: 43.1%, 42.0%, 39.2%. Cause: degenerate shapes (an inner
     dimension of 1, where a reshape and a transpose coincide; a 3-cycle over a
     size-1 axis, which moves data like a swap) and a matmul right operand that
     was a constant. Fix, in targeted programs only: dimensions >= 2, an input
     as matmul's right operand, permutations favouring non-involutions.
   The general generator keeps size-1 dims and constants on purpose, so its
   rate is the less biased number. The targeted builders were shaped after
   seeing which shapes hid these known bugs, so their rate measures reach for
   these bugs, not for unknown ones. The corpus mix (half general, half
   targeted) was fixed before the final measurement and not changed to reach
   the 50% target.

## Fusion (milestone 4, criteria fixed in advance)

The hypothesis to test, not a conclusion: fusion saves the memory round trip of
every intermediate it removes, and costs whatever it does to parallelism,
register pressure and recomputation. The candidates:

- **Elementwise chains** into one kernel. Expected to win when the chain is
  memory-bound, which on an integrated GPU sharing system memory is nearly
  always.
- **Elementwise prologue/epilogue into a reduction or matmul** (bias + relu
  after a matmul, exp before a sum). Expected to win unless it raises register
  use enough to cut occupancy.
- **Across a broadcast**, where fusing means recomputing the producer once per
  consumer element. Expected to lose when the producer is expensive and the
  broadcast is wide. This is the case most likely to show fusion losing, so it
  is measured on purpose.
- **Never** across a reduction whose output is consumed by more than one
  iteration space, without recomputation being counted as a cost.

Every fused kernel is fuzzed against the interpreter under the M2 budget before
it is timed. Timings record the Windows power mode beside every number,
because on this laptop the power plan moves CPU baselines by about 3.3x, and
only comparisons made in one session are reported as comparisons. The baseline
to beat or lose to is the hand-written matrix-vector kernel in
llm-inference-webgpu, which runs at roughly half the adapter's read-bandwidth
ceiling.

## Layout

```
src/ir/        graph, shapes, builder (broadcasting, composites), printer
src/autodiff.js  the transform and its rule table
src/interp.js    float64 reference interpreter
src/testing/   seeded RNG, random graph generator, gradient checker
oracle/        torch_golden.py (offline only; never run in CI)
golden/        torch cases, measurements.json (the report's only data source)
tools/         case export, measurement, local server
test/          node --test
index.html     the report, rendered from golden/measurements.json
```
