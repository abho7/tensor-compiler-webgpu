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

### Milestone 2 preconditions (2026-10-11), committed before the first GPU comparison

Milestone 2 lowers every op to WGSL, one kernel per op, and compares the GPU
(float32) with the interpreter (float64). Correctness only: nothing is timed.
Everything below was written, and the per-op accuracies checked against the
WGSL specification's own table (W3C Candidate Recommendation, section 15.7.4,
read 2026-10-11), before any kernel ran on a GPU.

7. **Revision R1: the tanh budget in the M2 table above was wrong.** It says
   absolute `2^-20`. The specification says tanh's f32 accuracy is "the worse
   of: absolute error 1.0×10^-5, inherited from sinh(x) / cosh(x)", which is
   about ten times looser at small x. Before: `2^-20`. After: the
   specification's bound, derived in amendment 8. Corrected before any
   measurement; the table above is left as written so the mistake stays visible.

8. **The float32 budget (metric v2 for float32).** Every comparison is per
   element: `abs(gpu - ref) <= S · E`, where `ref` is the float64 interpreter's
   value and `E` a first-order bound on the float32 error, propagated node by
   node through the program alongside the reference. `S = 2` is the safety
   factor: it covers the second-order terms dropped below and the bounds being
   evaluated at reference rather than exact values. It was fixed here, not
   fitted to any observed error.

   Constants: `u = 2^-24` (unit roundoff); `ε = 2^-23`, so `ULP(v) <= ε·|v|` for
   any normal v (the specification's definition of ULP); `τ = 2^-126`, the
   smallest normal float32, because the specification lets every input and
   output of these operations be flushed to zero when subnormal;
   `γ_n = n·u / (1 - n·u)`.

   Each node's bound is what it inherits from its operands' bounds (`e_a`,
   `e_b`, `e_x`) plus its own error, plus `ftz(r) = τ` when `|r| < τ` (it may
   have been flushed) and 0 otherwise:

   | op | inherited | own (from the specification) |
   |---|---|---|
   | input, const | 0 (both sides start from the same float32 value) | ftz |
   | add, sub | `e_a + e_b` | `u·|r|` (correctly rounded) |
   | mul | `|b|·e_a + |a|·e_b + e_a·e_b` | `u·|r|` |
   | div | `(e_a + |r|·e_b) / (|b| - e_b)`; unbounded if `e_b >= |b|` or `|b|` outside [2^-126, 2^126] | `2.5·ε·|r|` |
   | neg, relu, stopGradient | `e_x` | 0 |
   | exp | `|r|·(e^(e_x) - 1)` | `(3 + 2·(|x| + e_x))·ε·|r|·e^(e_x)` |
   | log | `-log(1 - e_x / x)`; unbounded if `e_x >= x` | `max(2^-21, 3·ε·(|r| + inherited))`: the worse of the two cases, since x may straddle [0.5, 2] |
   | tanh | `e_x` (`|tanh'| <= 1`) | `max(1e-5, (9.5 + 4·(|x| + e_x))·ε)`, below |
   | step | 1 if `0 < |x| <= e_x` (the sign may flip), else 0 | 0 |
   | max over n | `max_i e_i` (max is 1-Lipschitz in the largest-error norm) | 0 (correctly rounded, so exact) |
   | sum over n | `Σ e_i` | `γ_n · Σ |x_i|` |
   | matmul, inner size K | `Σ_k (|a_ik|·e_b,kj + |b_kj|·e_a,ik + e_a,ik·e_b,kj)` | `γ_K · Σ_k |a_ik·b_kj|` |
   | broadcast, reshape, transpose | the source element's bound | 0 (data movement is exact) |

   The sum and matmul bounds hold for any summation order, so they also cover
   the specification's permission to reassociate, and an FMA, which it allows
   only where at least as accurate.

   tanh, inherited from `sinh(x)/cosh(x)` with `sinh = (exp(x) - exp(-x))·0.5`
   and `cosh = (exp(x) + exp(-x))·0.5`: each exp carries
   `(3 + 2|x|)·ε` relative error, the sum and difference `u` each, the halving
   is exact, the final division `2.5·ε`. Relative to `cosh`, the numerator's
   error is at most `(3 + 2|x|)·ε + u·|t|` and the denominator's
   `(3 + 2|x|)·ε + u`, so with `|t| <= 1`:
   `|Δt| <= 2·(3 + 2|x|)·ε + 2u + 2.5·ε ≈ (9.5 + 4|x|)·ε` to first order. The
   specification takes the worse of that and `1e-5`; the `1e-5` dominates for
   `|x|` below about 18.6.

   For a single op on exact inputs, `E` reduces to the per-op class budget, so
   the direct tests use these constants: correctly rounded ops `2·u·|r|`; div
   `2·2.5·ε·|r|`; exp `2·(3 + 2|x|)·ε·|r|`; log `2·max(2^-21, 3·ε·|r|)`; tanh
   `2·max(1e-5, (9.5 + 4|x|)·ε)`; sum `2·γ_n·Σ|x_i|`; matmul `2·γ_K·Σ|a·b|`;
   data movement, max, neg, relu and step exact. Each also allows `2·τ` where
   a subnormal may flush.

   A whole program is judged on every node, not only its outputs, so a failure
   names the first kernel that went wrong.

   Why not the M1.5 form, `atol + rtol·|ref|` with `atol` set by the tensor's
   largest entry: amendment 4 conceded that it checks small entries beside
   large ones only absolutely, and in float32 that would hide exactly the kernel
   errors this milestone exists to find. The composed bound is what the M2 table
   above already specified for whole programs ("each node's budget composed
   along the graph"). The M1.5 CPU-vs-PyTorch metric is unchanged.

   An element is **skipped**, and counted, when its bound is unbounded (a
   division or log whose operand's error reaches zero) or its reference value
   lies outside the float32 range. A program with any skipped element is
   counted as skipped as a whole.

   If a later result forces a change to any of this, it is logged as a further
   revision with before and after, and nothing above is overwritten.

9. **Inputs are rounded to float32 first.** Every input, and every constant in
   the graph, is rounded with `Math.fround` before the float64 interpreter runs.
   Both sides start from identical values, and the comparison measures kernel
   error only.

10. **The exact-zero case under float32.** `random-9` computes `x0 / x0`,
    whose gradient is exactly zero; in float32 both contributions round at
    around 1e-7 of their size, not 1e-16. It is judged like everything else,
    by the propagated bound. That bound is built from the magnitudes of the two
    gradient contributions (`g/x0` and `-g·x0/(x0·x0)`) that cancel, not from the
    zero they cancel to: about `(2.5 + 1 + 2.5)·ε·|g/x0| ≈ 7e-7·|g/x0|` (each
    contribution's division, the products and the final add), against residue
    of order `u·|g/x0|`. No special case. Under the M1.5 form the same residue would have been
    judged against a fixed floor instead.

11. **Targets, stated before anything runs.**
    - Direct per-op tests: every injected kernel bug is caught by them. 100%.
    - Differential fuzzer: zero unexplained failures under the float32 budget.
      Every failure is investigated and recorded, whatever it turns out to be.
    - Fuzzer detection rate per injected bug: reported separately for targeted
      and general programs. No target. The general-program rate is the
      unbiased number, and the generator mix is not tuned to change it.
    - The correctness claim for each kernel rests on the direct tests. The
      fuzzer is supporting evidence, and the report says so.

12. **Generator float32 mode.** GPU fuzzing uses generator v2 (every
    unconsumed value an output, valid by construction), never the frozen M1
    generator. A `f32` option tightens its interval guards so float32 cannot
    overflow or flush where float64 would not: an `exp` argument must lie in
    [-40, 4] or is wrapped in tanh first (`e^-40 ≈ 4e-18` is far above the
    smallest normal float32; without this, `log(exp(x))` with `x` near -1000
    is fine in float64 and `log(0)` in float32). Denominators and log arguments
    keep v2's guard of at least 0.1 away from zero, and every value v2's bound
    of 1e3 in magnitude. With `f32` off the generator is unchanged, so the M1
    and M1.5 records and golden cases stay byte-identical, which a test checks.
    (Clarified 2026-10-11, before any GPU run: the guard is for the exps the
    generator emits. The exps inside softmax and log-softmax are built by the
    composite and are exempt: their argument is `x - stopGradient(max(x))`,
    at most 0 by construction, so the largest term is exactly 1 and the sum
    at least 1; anything that underflows below the smallest normal float32 is
    covered by the flush allowance. Interval analysis cannot see the
    correlation, so their intervals can look as wide as [-20, 20]. Measured on
    the corpus: the guard changed 0 of the first 2,000 corpus programs, and
    first fires at general seed 9604.)

### Milestone 2 revisions, forced by results

R2. **tanh overflows on this adapter, and my derivation missed that the
    specification allows it** (2026-10-11, first GPU run). The direct test
    `tanh large [1000]` returned NaN where the answer is -1: on this adapter
    (Intel gen-12lp) tanh is accurate up to |x| = 88.5 and NaN from |x| >= 89,
    exactly where exp overflows float32 (88.72). The specification's tanh is
    "the worse of" 1e-5 and the bound inherited from sinh/cosh, and that
    expression overflows there too, so NaN is a permitted result. Amendment 8's
    first-order derivation assumed exp stays finite and does not say so.
    Before: the kernel computed `tanh(x)`. After: `tanh(clamp(x, -20, 20))`;
    `tanh(20) = 1 - 8.5e-18` is exactly 1 in float32, so nothing is lost and
    the kernel never enters the region where the builtin may return anything.
    The budget is unchanged. The failing run is kept as
    `golden/gpu-m2-run1-direct.json` (251 of 252 direct cases passed).

### Milestone 3a preconditions (2026-10-14), committed before any training run

Milestone 3a trains an MLP classifier in the browser on the GPU. The
transformer is 3b. Nothing is timed. Everything below was written before any
training code existed or ran; it is amended only by dated revisions with
before and after.

13. **The new op: `sqrt`, and only `sqrt`.** Adam's update divides by
    `sqrt(v̂) + eps`. Nothing else in the MLP, softmax cross-entropy, SGD with
    momentum or Adam needs an op the IR lacks: products, quotients, sums,
    broadcasts and `log`/`exp` already exist, and the step-dependent bias
    corrections are carried as state (below), so no `pow` is needed. `sqrt` gets
    the full pipeline of every other op:
    - IR: a unary op, result shape of its input, domain x >= 0.
    - Interpreter: `Math.sqrt`.
    - Autodiff: `d sqrt(x) = g / (2·sqrt(x))`, written with the result `y`
      as `g / (2·y)`.
    - Checks of the rule: finite differences on random programs containing it
      (the M1 checker and budget, 1e-6), and PyTorch float64 golden cases (the
      metric v2 budget, 1e-12), in new files so the M1 goldens stay byte-identical.
    - Bound (amendment 8 extended): the WGSL specification gives sqrt as
      "inherited from 1.0 / inverseSqrt(x)", with inverseSqrt to 2 ULP and the
      division to 2.5 ULP, so its own error is `4.5·ε·|r|` to first order. It
      inherits `min(sqrt(e_x), e_x / sqrt(x))` from its operand (sqrt is
      1/2-Hölder, so this holds even when `e_x` reaches `x`), plus the flush
      allowance. A negative operand is unbounded and skipped. At exactly 0 the
      specification leaves `inverseSqrt(0)` undefined, so `sqrt(0)` may come
      back as 0 or as a flushed tiny value; the flush allowance covers both,
      and the direct tests check 0 explicitly because Adam meets it whenever a
      gradient element is exactly zero (a dead ReLU unit).
    - WGSL: one elementwise kernel, with direct adversarial tests under the
      float32 budget (sizes off the workgroup grid, size-1 dims, rank 3, huge,
      tiny, subnormal and exactly zero inputs) and sabotage.

14. **Optimizers are IR.** A training step is one graph: the model's forward
    pass, its loss, autodiff's backward pass, and the optimizer's update, all
    built from IR ops and compiled through the same interpreter, lowering and
    kernels as everything else. Its inputs are the parameters, the optimizer
    state and the batch; its outputs are the loss and the new parameters and
    state. Nothing about an optimizer is special-cased in a runtime.
    - SGD with momentum, as PyTorch defines it (dampening 0, not Nesterov):
      `g' = g + wd·p`, `buf = μ·buf + g'`, `p = p - lr·buf`, with `buf` starting
      at 0 (which equals PyTorch's first step, `buf = g'`).
    - Adam, as PyTorch defines it (L2 weight decay, not AdamW):
      `g' = g + wd·p`, `m = β1·m + (1-β1)·g'`, `v = β2·v + (1-β2)·g'²`,
      `p = p - lr·(m / (1-β1^t)) / (sqrt(v / (1-β2^t)) + eps)`. The powers
      `β1^t` and `β2^t` are state scalars, multiplied by β each step, so the
      graph never needs the step number.
    - Hyperparameters are constants in the graph: SGD lr 0.1, momentum 0.9,
      weight decay 5e-4; Adam lr 0.01, betas (0.9, 0.999), eps 1e-8, weight
      decay 1e-4 (for the synthetic set; the demo's are below).

15. **The model and the data.**
    - MLP: `x → relu(x·W1 + b1) → ·W2 + b2 → log-softmax`, loss the mean
      negative log-likelihood of the true class (softmax cross-entropy), all
      from existing ops. Initialisation as PyTorch's `nn.Linear` default,
      uniform in ±1/sqrt(fan_in) for weights and biases, drawn from this
      project's seeded RNG and handed to PyTorch explicitly, so both sides
      start from identical numbers.
    - **Synthetic set (golden comparisons):** generated in JS from a fixed
      seed and committed: four interleaved spiral arms in two dimensions,
      lifted to 8 features by a fixed seeded linear map plus noise; 256 training
      and 256 test points, balanced across 4 classes. The MLP is 8 → 16 → 4.
      Batches of 32 from a seeded permutation each epoch, so the batch order is
      a function of the seed alone and both sides see the same batches.
      CI regenerates the set from its seed and fails if it differs.
    - **Real set (the demo): UCI Optical Recognition of Handwritten Digits**
      (Alpaydin & Kaynak, 1998; doi:10.24432/C50P49), licensed CC BY 4.0,
      downloaded 2026-10-14 from
      `https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/`
      and committed byte for byte in `data/optdigits/`: `optdigits.tra`, 3,823
      rows, SHA-256 `e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd`;
      `optdigits.tes`, 1,797 rows, SHA-256
      `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8`.
      64 features (8x8 counts, 0 to 16, divided by 16) and a digit label. The
      demo trains 64 → 64 → 10 with Adam (lr 1e-3, other settings as above,
      no weight decay), batch 64, 10 epochs, and is overlaid on a PyTorch
      float64 run from the same initialisation and batch order. CI checks the
      checksums.

16. **What "matches PyTorch" means for training: the float64 interpreter
    against PyTorch float64.** Same data, initialisation and batch order. Four
    seeds, each trained 50 steps with SGD and with Adam: eight golden runs,
    produced offline by `oracle/torch_train.py` and committed. Every step, the
    loss and every parameter tensor are compared in the metric v2 form,
    `abs(ours - torch) <= atol + rtol·abs(torch)`, `atol = 1e-8·max(1, max abs(torch))`,
    `rtol = 1e-8`.
    Why 1e-8: per op, this interpreter and PyTorch differ by summation order and
    libm rounding, which M1 measured at under 1e-15 relative per tensor. A step
    composes a few dozen ops, so one step's disagreement is of order 1e-15 to
    1e-14 of the parameters' scale. Training then carries it forward, amplified
    each step by at most the step map's Lipschitz constant; for these small
    learning rates and small initial weights, that is taken to be at most 1.3,
    so 50 steps amplify by at most 1.3^50 ≈ 5e5. That gives up to about 5e-9,
    inside 1e-8. The budget is not loosened if this assumption fails; a failure
    is logged. A ReLU input within rounding of zero can send the two sides down
    different branches, which this tolerance does not cover; that would also be
    recorded as a failure, not explained away.

17. **What "agrees" means for the GPU: float32 against the float64
    interpreter**, on identical float32-rounded inputs (amendment 9).
    - **Per step, teacher-forced, every step of the eight golden runs:** at each
      step both sides are given the same float32-rounded state and batch and run
      one step, judged on every node by the float32 budget (amendment 8, S = 2).
      This is a single-step check, so the budget is exactly milestone 2's.
    - **Free-running, the first K = 5 steps:** each side trains on its own
      outputs, and the bound is carried across steps: the next step's inputs
      enter with the previous step's output bounds instead of zero. K = 5 because
      worst-case bounds compound multiplicatively from step to step (and a ReLU
      within its bound of zero contributes a whole unit), so they lose meaning
      after a handful of steps; five steps are enough to feed every state tensor
      back four times, including Adam's bias-correction powers and both moment
      buffers. The ratio of each bound to the value it bounds is reported, so a
      bound that has become vacuous is visible rather than counted as a pass.
    - **Statistically, after the trajectories have diverged:** 24 seeds, each
      trained 200 steps (25 epochs) on the synthetic set with SGD and with Adam,
      once on the GPU in float32 and once by the interpreter in float64, from the
      same initialisation and batch order. For each run, the final full-training
      loss and the test accuracy are computed by the float64 interpreter from
      that run's final parameters, so only the training trajectory differs. For
      each optimizer and each metric, the paired difference (GPU minus CPU) must
      pass a two one-sided t-test of equivalence at α = 0.05 (equivalently, its
      90% confidence interval, 23 degrees of freedom, must lie within the
      margin). Margins: 0.02 nats of loss, and 2 percentage points of accuracy
      (about 5 of the 256 test points). Differences below these are taken to be
      of no practical consequence for a classifier of this size; that is a
      judgement, stated here so it cannot be adjusted afterwards.

18. **Sabotage, and how detection is counted.** Each bug is injected alone:
    - Adam's second-moment bias correction computed with β1 instead of β2;
    - SGD momentum off by one: the parameter updated with the previous step's
      buffer;
    - the weight-decay term's sign flipped (`g - wd·p`), in both optimizers;
    - `sqrt`'s gradient missing its factor of one half (`g / y`);
    - a bounds bug in the `sqrt` kernel: its guard lets index `n` through.
    The checks are the PyTorch training goldens, finite differences, the PyTorch
    op goldens, the direct `sqrt` kernel tests, and the per-step GPU check. A
    detection rate is the fraction of a check's independent trials that catch
    the bug: golden runs to which the bug applies, programs containing `sqrt`,
    golden cases containing `sqrt`, direct cases, and teacher-forced runs. Which
    check catches a bug first is recorded in the order the checks run. The only
    expectation set in advance is that every bug is caught by at least one
    check. One prediction, so it can be held to: a bug in an optimizer's graph
    is invisible to the GPU-against-CPU checks by construction, since both run
    the same graph; only the PyTorch goldens can see it.

19. **Milestone 2's record under the new op.** Adding `sqrt` edits files inside
    milestone 2's source hash, so the check that the M2 recording came from the
    checked-in sources is pinned to commit `c261763` (`8f45285` before the
    2026-10-21 history rewrite), where it was recorded: CI
    reads those files from git history and hashes them. M2's kernel snapshot
    and kernel list are frozen as they were; new kernels get their own
    snapshot. The M2 harness is then run again on the extended compiler, into
    a new file, and must reproduce M2's results exactly; that is what shows the
    extension changed nothing M2 measured.

### Milestone 3a findings and revisions

F1. **A limitation of generator v2's interval analysis, present since
    milestone 1.5** (found 2026-10-14, while adding sqrt). Plain interval
    arithmetic treats the two factors of `x*x` as independent, so when `x`'s
    interval straddles zero, `x*x` gets a negative lower bound. Generator v2
    makes a `log` argument safe as `x*x + c`; that argument then looks possibly
    negative, the `log`'s interval comes out unbounded, and the op is
    discarded. It survives only when `x`'s interval is already non-negative.
    Measured on the milestone 1.5 and 2 fuzz corpus (`corpusProgram` 1 to
    2,000), against the same corpus with `x*x` given its true interval: 564
    `log` nodes in 451 programs as run, against 738 in 556; and the `x*x + c`
    form 18 times against 229. So that corpus had about 24% fewer `log` nodes
    and 19% fewer programs containing `log` than intended, and the `x*x + c`
    form at under 8% of its intended rate; `log` reached the corpus mostly
    through its `exp(...)` branch. Milestone 1's generator (v1) does not use
    intervals and is unaffected. Nothing measured is wrong: every `log` that
    was generated was checked. Coverage was thinner than assumed. The default
    generator is not changed, because milestones 1.5 and 2's records must stay
    byte-identical; the true-interval version exists only behind an option
    (`squareAware`), to measure this. `sqrt` (amendment 13) makes its argument
    safe as `exp(x) + c` instead, whose interval really is positive. The
    figures are recomputed by CI (`m3.intervalLimitation` in
    `golden/measurements.json`).

F2. **What the sqrt goldens check, and what the wrong sqrt rule escapes.** Of
    the 34 PyTorch golden cases for sqrt, 15 have at least one input whose
    gradient is exactly zero: 19 such inputs feed no output (the generator
    creates inputs it may not use), 2 reach an output through an exact zero,
    and one case (`sqrt-random-50`, a log-softmax over a size-1 axis) has no
    nonzero gradient at all. The wrong rule (`g / y`) still changes 31 of the
    34 cases beyond metric v2, and 126 of the 146 finite-difference programs
    containing sqrt. Every miss, in both, has a specific cause: sqrt depending
    on no input; its argument constant in the inputs (an `x - x`, or a
    descendant of a softmax over a size-1 axis, which is constant); the gradient
    arriving at it exactly zero (an `x - x` downstream, or such a softmax); or
    sqrt entering only as `y / y`, whose true gradient is zero. None is a
    weakness of the check; each is a program in which no rule for sqrt could
    show. Recorded per program in `m3.sqrt`.

F3. **An oracle run of mine overwrote milestone 1's PyTorch golden file,
    twice.** `oracle/torch_golden.py` wrote to a hard-coded
    `golden/torch.json`. Run for the sqrt cases before its paths were
    parameterised, it rewrote that file: the first time with identical bytes
    (the same cases), the second time with the 34 sqrt cases. Both times it was
    restored from git and its SHA-256 checked against the recorded value before
    anything else ran; nothing was committed in between. The script now takes
    explicit input and output paths and refuses to overwrite an existing file
    without `--force`.

F4. **The free-running bound lost its meaning for Adam well before K = 5**
    (2026-10-15, first GPU run of milestone 3a). Amendment 17 expected the
    carried bound to stay informative for five steps. For SGD it did: all four
    golden seeds were judged through all five steps, with the median bound under
    1e-5 of the value it bounds. For Adam it did not: of the four seeds, one was
    judged through all five steps (with some elements' bounds already larger
    than their values), and three became unjudgeable, at steps 2, 3 and 5,
    because a carried bound reached the size of the denominator `sqrt(v̂) + eps`
    it divides by, where a second moment is tiny. No GPU value fell outside its
    bound at any judged step. Recorded as measured; K is not changed and the
    unjudgeable steps are not counted as passes. The per-step teacher-forced
    check, which does not carry bounds, judged every step of every run.

F5. **What each check can see, measured.** As amendment 18 predicted, none
    of the GPU-against-interpreter checks caught a bug in an optimizer's graph:
    both sides run the same wrong graph. Only the PyTorch training goldens did,
    every time. The wrong sqrt gradient was invisible to every training check,
    PyTorch's included, for a reason not predicted in advance but simple: the
    MLP never differentiates sqrt (it appears only in Adam's update, which is
    not differentiated), so the rule is never used; only the op-level checks
    see it. The kernel bounds bug was caught by every direct sqrt case except
    the one whose size, 64, fills its workgroups exactly, so that no thread past
    the end exists to slip through the faulty guard; it was invisible to the
    teacher-forced runs, whose buffers are exact-size and whose out-of-bounds
    writes are dropped on this adapter (milestone 2's finding). Figures in
    `m3.training.gpu.sabotage`.

F6. **F1 understated the limitation: `div` lost more coverage than `log`**
    (2026-10-18). Before: F1 described the defect as thinning `log` alone.
    After: generator v2 also makes a `div` denominator safe as `x*x + c` or
    `-(x*x + c)`, which the same interval defect discards whenever `x` can be
    negative and `c` does not cover the spurious negative lower bound. On the same
    corpus of 2,000 programs: 438 `div` nodes in 380 programs as run, against
    706 in 568 with `x*x` given its true interval, and an `x*x + c` denominator
    10 times against 215. So the corpus had about 38% fewer `div` nodes and 33%
    fewer programs containing `div` than intended, and the `x*x + c`
    denominator at under 5% of its intended rate; `div` reached the corpus
    mostly with denominators already bounded away from zero or through `exp`.
    The counts are identical with the float32 guards on, which is how
    milestone 2 drew the corpus. As with F1, nothing measured is wrong and
    nothing is changed: every `div` that was generated was checked, and the
    default generator stays as it was. The figures are recomputed by CI
    (`m3.intervalLimitation`).

### Milestone 3b preconditions (2026-10-18), committed before any model code exists or runs

Milestone 3b trains a small character-level transformer in the browser on the
GPU. Nothing is timed and nothing is fused (that is milestone 4). Everything
below was written before any transformer code existed or ran; it is amended
only by dated revisions with before and after.

20. **The model.** Decoder-only, pre-LayerNorm, learned position embeddings,
    causal multi-head attention, a GELU (tanh form) MLP, a final LayerNorm and
    an untied output head. Exactly:
    - context `T = 64`, width `d = 64`, `H = 4` heads of `d_h = 16`, `L = 2`
      blocks, MLP hidden width 256; vocabulary `V = 16` for the synthetic task
      and `V = 75` for the demo text (amendment 24);
    - `h = x + attn(LN1(x))`, `x' = h + mlp(LN2(h))`; logits `LNf(x_L)·W_head + b_head`;
    - attention `softmax(Q·Kᵀ·(1/4) + M)·V` per head, with `Q, K, V = LN1(x)·W_q,k,v`
      **without biases**, then the heads merged and `·W_o + b_o`; the MLP is
      `GELU(·W_1 + b_1)·W_2 + b_2`;
    - LayerNorm with the biased variance (divisor `d`, as PyTorch), `eps = 1e-5`,
      a gain and a bias per block and for `LNf`;
    - loss: the mean over all `B·T` positions of the negative log-likelihood
      of the next character.
    Parameters: 105,872 for `V = 16`, 113,483 for `V = 75`.
    **Why no Q, K, V biases.** A bias on K adds the same `q·b_k` to every score in a
    row, which softmax ignores, so that bias's gradient is exactly zero
    mathematically and pure rounding residue numerically. Adam divides by
    the root of the gradient's square, so it would turn residue of 1e-17 into
    updates of about `±lr` whose sign differs between this interpreter and
    PyTorch: the per-step comparison (amendment 23) would fail with no bug
    present. Q and V lose theirs too, for uniformity; `W_o`, the MLP and the
    head keep theirs.
    **Why untied.** Amendment 23 compares every parameter's gradient. Tied,
    the embedding's gradient would be the sum of two paths (lookup and head),
    and a bug in either would show only in the sum. The cost is `64·V`
    parameters.
    **Initialisation:** `nn.Linear`'s default for every weight and bias (uniform in
    `±1/sqrt(fan_in)`), embeddings uniform in [-1, 1], LayerNorm gains 1 and
    biases 0, drawn from this project's seeded RNG in a fixed parameter order.
    PyTorch regenerates them with a Python port of that RNG and refuses to run
    unless the SHA-256 of its float64 bytes equals the one the JavaScript
    side wrote into the spec, so the same numbers start both sides without
    being stored.

21. **Composed from existing ops; no new op.** Every piece is built from ops
    already verified in milestones 1 to 3a:
    - token embedding: a one-hot `[B·T, V]` input times the `[V, d]` table (exact
      in any precision: one product by 1 and additions of exact zeros);
    - position embedding: the `[T, d]` table broadcast-added (positions are
      always 0 to T-1, so no lookup is needed);
    - LayerNorm: `sum`, `sub`, `mul`, `div` and `sqrt`, two-pass (mean, then the
      mean of squared deviations), so no `E[x²] - E[x]²` cancellation;
    - GELU: `0.5·x·(1 + tanh(sqrt(2/π)·(x + 0.044715·x³)))`;
    - causal mask: a constant `M` added before softmax, 0 on and below the
      diagonal and `-1e4` above. Scores are expected to be well under 1e3 in
      size, so `exp` of a masked entry minus its row's max underflows to exactly
      0 in float32 and float64 alike. A masked position then contributes
      exactly nothing, which is what makes amendment 26's causality test
      bitwise. (The size of the largest score is recorded.)
    - head split and merge: `reshape` and `transpose` (`[B,T,d] → [B,T,H,d_h] →
      [B,H,T,d_h]` and back); batched `matmul` for `Q·Kᵀ` and `P·V`; projections
      as `[B·T, d]` matmuls;
    - softmax and log-softmax: the existing stable composites.
    None of these is numerically unsafe in this form, so no op is added. Two
    consequences: there is no new kernel, so the brief's "bounds bug in any new
    kernel" has no target (amendment 27 says so rather than inventing one).
    And no file in `src/gpu/sources.js`'s `SOURCE_FILES` or `M2_SOURCE_FILES`
    is edited, so milestone 3a's GPU recordings stay valid against their
    sources as they are. Milestone 3b's own files get their own source list and hash.

22. **The float32 budget for this model** is amendment 8's per-element
    propagated bound with `S = 2`, applied to the whole training step (forward,
    loss, backward, Adam). It is computed from the float64 reference and its
    propagated operand errors, never from GPU output. The accumulation lengths
    that set its `γ_n` terms, at `u = 2^-24`:

    | where | n | γ_n |
    |---|---|---|
    | attention scores `Q·Kᵀ`; backward `dP = dO·Vᵀ` | d_h = 16 | 9.5e-7 |
    | LayerNorm sums; Q, K, V, O, MLP-up and head matmuls; their input gradients | d = 64 | 3.8e-6 |
    | softmax sum over keys; `P·V`; backward `dQ`, `dK`, `dV` | T = 64 | 3.8e-6 |
    | one-hot embedding (K = V); log-softmax sum over the vocabulary | V = 16 or 75 | 9.5e-7 / 4.5e-6 |
    | MLP-down matmul; its input gradient | 256 | 1.5e-5 |
    | every weight gradient and bias-gradient sum (over `B·T` positions) | 256 (B = 4), 512 (B = 8), 1024 (B = 16) | 1.5e-5 / 3.1e-5 / 6.1e-5 |

    Safety factor `S = 2`, unchanged from amendment 8 and for the same reason.
    What the bound looks like is predicted, not fitted: it is dominated by the
    long weight-gradient sums, and relative to an element it is small except
    where the element is a near-cancellation of larger terms; it is absolute, so
    there it is large relative to the value, and that is correct.
    **One deviation from amendment 8, stated before running.** Amendment 8
    counts a program with any unjudgeable element (unbounded error, or a
    reference outside float32's range) as skipped as a whole. A training step
    here has about a million elements; one unbounded division (for example,
    Adam's denominator for a nearly-zero second moment) would discard the
    step. Before: the whole step skipped. After: every judgeable element is
    judged, a step passes only if all of them pass, and the number of
    unjudgeable elements is reported beside it. A step in which more than 1%
    of elements are unjudgeable is reported as unjudged, not as a pass.

23. **What "matches PyTorch" means.** The PyTorch side
    (`oracle/torch_transformer.py`) is written with PyTorch's own primitives,
    not as a translation of this project's graph: `F.embedding`, `F.layer_norm`,
    `F.scaled_dot_product_attention(is_causal=True)` (which masks with -inf),
    `F.gelu(approximate='tanh')`, `F.cross_entropy` and `torch.optim.Adam`.
    - **(i) At initialisation, float64 vs float64.** Four synthetic configs
      (seeds 1-4, batch 4) and one text config (seed 1, batch 4): the logits
      `[B,T,V]`, the loss and every parameter's gradient, in metric v2:
      `abs(ours - torch) <= atol + rtol·abs(torch)`,
      `atol = τ1·max(1, max abs(torch))` per tensor, `rtol = τ1`, with
      **`τ1 = 1e-10`**. Why: each side rounds at `u = 1.1e-16` per op. The longest
      dependency chain from an input to a first-block gradient passes through
      about 200 ops (2 blocks of about 40 forward ops, the loss, and the
      backward pass back through them). A sum or matmul of length n contributes at most
      `γ_n·Σ|terms|`, with n at most 256 here (`γ_256 ≈ 2.8e-14`). Summed along the
      chain, that is about `6e-12` relative to the terms' absolute mass; the
      assumption is that mass exceeds a tensor's largest entry by at most a factor of
      10 (cancellation), giving about `6e-11`, inside `1e-10`. If that
      assumption fails, the failure is logged, not explained away.
    - **(ii) Training, float64 vs float64**, the same four synthetic configs, Adam,
      **N = 20 steps**. Every step: the loss, and for every parameter tensor four
      seeded random projections `⟨r_k, p⟩` (r uniform in [-1, 1]), checked
      against the projected allowance `Σ_i |r_k,i|·(atol + rtol·|p_i|)`, which is
      implied by the elementwise check, so a necessary condition of it. After
      step N, every element of every parameter in full. Tolerance **`τ2 = 1e-8`**
      (amendment 16's): one step's disagreement starts near 1e-13 of scale, and 20
      steps of amplification by at most 1.5 each (`1.5^20 ≈ 3.3e3`) stay
      inside 1e-8. Projections rather than full tensors every step, because 20
      steps of 106k parameters for four runs would be about 70 MB of golden
      data; a localised error persists under Adam and is caught in full at step N.
    - **(iii) GPU float32, teacher-forced**, the same four runs, all N = 20 steps:
      at each step the GPU is given the interpreter's float32-rounded
      parameters, Adam state and batch from before that step, runs one step,
      and is judged on every node by amendment 22.
    - **(iv) GPU float32, free-running: compared statistically, not per step.**
      Milestone 3a found (F4) that a bound carried across steps stops meaning
      anything within a few Adam steps; nothing here pretends otherwise. 24
      seeds, the synthetic task, batch 8, 300 Adam steps, trained on the GPU in
      float32 and by PyTorch in float64 from the same initialisation and batches.
      The final validation loss (the mean over all 128 validation sequences) is computed
      for the GPU run by the float64 interpreter from its final parameters,
      and for PyTorch by PyTorch. Over the paired differences (GPU minus
      PyTorch): a two one-sided t-test of equivalence at `α = 0.05`, that is, the 90%
      confidence interval of the mean difference (23 degrees of freedom) must
      lie within **±0.02 nats per character**. For context only, PyTorch
      float32 is compared with PyTorch float64 the same way on the same seeds.

24. **Data.**
    - **Synthetic (golden comparisons), "periodic copy":** vocabulary of 16
      symbols; each sequence is a block of 16 symbols drawn uniformly, repeated
      to 65 tokens (64 inputs, 64 next-token targets). 512 training and 128
      validation sequences from seed 3101, committed and regenerated by CI.
      The first 15 targets of a sequence are unpredictable, so the best
      possible loss is `(15/64)·ln 16 ≈ 0.650` nats. Batches: a seeded
      permutation of the training sequences each epoch, cut into batches
      (milestone 3a's schedule).
    - **Text (demo): Lewis Carroll, *Alice's Adventures in Wonderland***,
      Project Gutenberg eBook #11, public domain in the United States.
      Downloaded 2026-10-18 from `https://www.gutenberg.org/cache/epub/11/pg11.txt`,
      174,311 bytes, SHA-256
      `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`.
      Committed as `data/alice/alice.txt`: the text strictly between the
      `*** START OF THE PROJECT GUTENBERG EBOOK` line and the `*** END` line, line endings
      changed from CRLF to LF, so no Project Gutenberg licence or trademark
      text is included: 144,604 characters, 151,099 bytes of UTF-8, SHA-256
      `e3917589f48b4e9f4b731a6b7e127cf9ddd6e1b3819c78be02a2702ee73bfdb1`,
      which CI checks. 75 distinct characters,
      in code-point order, are the vocabulary. Split: the first 90% of
      characters are training, the rest validation. A training batch is 16
      windows of 65 characters at seeded uniform offsets in the training part;
      the demo's validation loss is the mean over 64 windows evenly spaced
      across the validation part. The demo trains 600 steps.

25. **Optimizer: milestone 3a's Adam graph, generalised, not rewritten.** The
    update is the same IR construction as `src/train/step.js` (amendment 14),
    moved into a builder that takes any parameter set. A test checks that for
    milestone 3a's MLP the builder produces node-for-node the graph `step.js`
    produces. Hyperparameters: lr `2e-3`, betas (0.9, 0.999), eps `1e-8`, no
    weight decay, for every run in this milestone.

26. **Property tests, stated now.**
    - **Causality.** For each of four synthetic configs at initialisation, positions
      `t ∈ {1, 17, 40, 63}`: the token at t is replaced by a different seeded token, and
      the logits at every position before t must be **bitwise identical**. That is
      exact rather than approximate because a masked score's `exp` is exactly 0
      (amendment 21) and every other op is per position. The same test must
      also see some logit at position t or later change, so it cannot pass
      vacuously. Run by the interpreter (CI) and on the GPU, at initialisation
      and on the demo's trained parameters.
    - **Batch-order independence.** Permuting the sequences in a batch must permute the
      logits **bitwise** (interpreter and GPU), and leave the loss and every gradient
      equal within metric v2 at `τ1` (interpreter; the batch sums reassociate).
    - **Greedy generation, GPU against the interpreter, K = 58 tokens.** From
      the demo's GPU-trained parameters (float32 values, given to both sides),
      prompt `"Alice "` (6 characters), each side generates greedily from its own
      previous tokens until the context of 64 is full: 58 tokens. They must
      agree on all 58. One exception, defined now: if they first disagree at a
      position where the interpreter's top two logits are closer than the sum
      of their float32 bounds (`S·(E_1 + E_2)`), that is recorded as a near-tie
      and the comparison stops there; any other disagreement is a failure.
      The demo page shows this agreement as its correctness evidence. The
      generated text is illustrative and no claim about its quality is made.

27. **Sabotage, each bug injected alone, in the model graph:**
    - causal mask off by one: row i may see position i+1 (leaks the future);
    - attention scale `1/sqrt(d_h)` missing;
    - softmax over the query axis instead of the key axis;
    - LayerNorm variance divided by `d - 1` instead of `d`;
    - LayerNorm `eps` of `1e-3` instead of `1e-5`;
    - head merge without its transpose (`[B,H,T,d_h]` reshaped straight to `[B,T,d]`);
    - position embedding off by one: position p gets row `(p + 1) mod T`;
    - a bounds bug in a new kernel: **not applicable**, since amendment 21 adds no
      kernel. Milestones 2 and 3a's kernel sabotages still cover the existing ones.

    Checks, in the order they run: causality (16 trials: 4 configs x 4
    positions), batch order (4 configs), PyTorch at initialisation (5 configs),
    PyTorch training (4 runs, first failing step recorded), GPU teacher-forced
    (the first golden run, its first 2 steps). The detection rate is caught
    trials over trials, per check. Expectations, fixed now so they can be held
    to: every bug is caught by at least one check; **causality must catch the
    mask bug**; causality is also predicted to catch the wrong softmax axis (a
    column normalisation mixes in later queries) and the missing merge
    transpose (it mixes positions), and none of the others; batch order is
    predicted to catch none (no bug mixes sequences); PyTorch at initialisation
    is predicted to catch all seven; GPU teacher-forced is predicted to catch
    none, because both sides run the same wrong graph (milestone 3a's F5).

28. **The demo** (`web/train-transformer.html`): trains the text model live on
    the visitor's GPU, 600 steps, and overlays its loss on the recorded PyTorch
    float64 and float32 curves from the same initialisation and batches. It
    records the adapter and, on this project's local server only, the power
    mode, and shows text generated greedily from the trained model with
    amendment 26's agreement count beside it. It reports no timings. CI cannot run the
    GPU: as in milestones 2 and 3a, it checks that every recording came from
    the sources checked in, recomputes everything that needs no GPU, and
    checks the report against the recordings.

### Milestone 3b revisions, forced by results

R3. **The composed float32 bound is vacuous inside one transformer step**
    (2026-10-18, first GPU runs, before any recording). Amendment 22 judged a
    teacher-forced step by amendment 8's bound propagated through the whole
    step. Measured on golden run 1, step 0: 9,488,470 of 14,844,201 elements
    (63.9%) have an infinite bound, the first at node 153 of 1,641 (block 1's
    first LayerNorm division). Even the forward pass alone, at
    initialisation, has a finite bound on 0 of its 1,024 logits (V = 16) and 0 of
    4,800 (V = 75). The mechanism: the bound is a first-order worst case, and
    each matmul multiplies an incoming bound by the L1 norm of a weight column
    (about 4 for 64 inputs at this initialisation, 8 for the 256-wide MLP-down),
    where real errors partly cancel. On top of that, the row max in softmax inherits the bound of
    the masked entries (the rounding of `score - 1e4`, about 6e-4). The largest
    absolute bound goes from 1.1e-6 after the embedding to 5.9e-3 after block
    0's attention, 0.95 after its MLP, and infinity in block 1's first
    LayerNorm, against values of order 1. Amendment 22's prediction ("small
    except near cancellation") was wrong.
    Consequences, decided before any further GPU run:
    - **Check (iii) is kept exactly as registered and recorded as it comes
      out.** By its own 1% rule every step is unjudged, which is not a pass.
      The elements it can judge are still judged and reported.
    - **Accounting bug found in my harness.** The first GPU sabotage run
      counted an unjudged step as a detection, so all seven bugs looked
      "caught" by the GPU check. An unjudged step detects nothing. After:
      a detection needs a judged element over its bound.
    - **Added: (iii-b), node-local teacher-forcing.** Before: none. After: at
      every step of the four golden runs, every kernel launch of the real step
      is given the float32-rounded reference values of its own operands, not
      the GPU's, and its output is judged against the float64 result on those
      operands by amendment 8's bound for that single op (operand errors zero,
      so the bound is the per-op budget of the direct tests), `S = 2`.
      Unjudgeable elements are counted, and the 1% rule applies as in amendment 22. A step
      passes if every judgeable element of every node passes. What it can see: every
      kernel, on this model's real shapes and data, at its own budget. What it cannot
      see: how error accumulates from op to op. That is left to the statistical
      comparison (iv), plus an observed per-step difference between the GPU's and
      the interpreter's loss and updated parameters, reported for information
      with no tolerance attached.
    - **Greedy generation: the near-tie exception is removed.** Before: a
      first disagreement where the top two logits differ by less than
      `S·(E_1 + E_2)` was a near-tie, not a failure. After: those bounds are
      infinite at every logit, so the exception would excuse any disagreement.
      A disagreement is reported as a failure, with the gap between the
      interpreter's top two logits beside it. Decided before any greedy run.

R4. **Two more bugs in my own checks, found before any recording**
    (2026-10-18). Neither changed a rule; both are recorded because the
    checks are what this milestone claims.
    - The demo page's greedy generation handed the GPU path a single sequence where
      it expected a batch, so the 65 tokens became a "batch" of 65. The run
      needed a 335 MB read-back buffer, WebGPU refused it, and the error vanished,
      because the page's click handler sat outside its error handling. The page
      showed "generating…" forever. Fixed: the call is wrapped, and the handler reports
      any error. Found on the first unrecorded run.
    - My first draft of the projection allowance (amendment 23 (ii)) took the size reference
      from PyTorch's step-N parameters at every step. That is not what amendment 23 says.
      Fixed before any comparison ran: it uses the parameters of the step being checked.

### Milestone 3b findings

F7. **What came out, against what was registered** (2026-10-19; figures in
    `m3b` of `golden/measurements.json`). PyTorch at initialisation: 5 of 5
    configs within `τ1`. Training: 4 of 4 runs within `τ2`. The whole-step
    teacher-forced check: 0 of 80 steps pass and 80 are unjudged, as R3 said
    they would be; none of its judgeable elements failed. The node-local
    check: 80 of 80 steps. Causality and batch order held on both sides.
    Greedy generation agreed on 58 of 58 characters. The statistical
    comparison passed. All seven bugs were caught, and every prediction of
    amendment 27 held, including that the GPU checks would catch none.

F8. **The statistical margin was far looser than what it ended up measuring.**
    The 90% interval of GPU minus PyTorch-float64 validation loss over 24 seeds
    is -2.4e-7 to 1.6e-7, against a margin of ±0.02. The trajectories barely
    diverged: in 300 Adam steps on this task, no seed's float32 and float64
    validation losses differ by more than 2.6e-6. So the test passes, but it says only that the
    difference is under 0.02 nats; it could not have told apart anything finer.
    One thing the data do show, not predicted: the GPU's float32 differs from
    float64 more than PyTorch's own CPU float32 does. Over the seeds, the spread of
    the differences is about 22 times larger; on the demo, the largest per-step loss
    difference is about 11 times larger. Both are inside every tolerance. Which kernels
    contribute (naive summation order, or WGSL's looser per-op accuracy) was not isolated.

### Milestone 4 preconditions (2026-10-19), committed before any fusion code exists or any timing runs

Milestone 4 asks the project's question: when does kernel fusion pay off on
this integrated GPU, and when does it lose? It is the first milestone with
timings. Everything below was written before any fusion code existed and
before any kernel was timed. Changes after this point are dated revisions
with before and after. The section "Fusion (milestone 4, criteria fixed in
advance)" below was written at milestone 1 and still stands; where this is
more specific, this governs.

29. **Fusion is an IR-to-IR pass.** `fuse(graph, options)` returns a new graph
    in which some groups of nodes are replaced by one `fused` node, plus a map
    from each surviving node to the node of the original graph whose value it
    carries. A fused node's attributes hold its **body**: an ordinary graph of
    the existing ops, whose inputs are the fused node's operands. The
    interpreter evaluates a fused node by evaluating its body. So semantic
    preservation is checkable in pure JavaScript. The fused graph, run on the float64
    interpreter, must give **bitwise identical** values to the original at
    every surviving node: same ops, same operands, same order, the same summation order inside the body. No
    tolerance. Checked by CI on milestone 2's fuzz corpus (`corpusProgram` 1 to
    2,000, forward and gradient graphs) and on milestone 3b's training steps.
    No file in `SOURCE_FILES` or `M3B_SOURCE_FILES` is edited, so milestones
    3a's and 3b's recordings stay valid against their sources. The pass, its
    code generator and its harnesses are new files with their own source list
    and hash (`M4_SOURCE_FILES`), and milestone 4's GPU records are made from
    those sources.

30. **The fusion patterns, a fixed set.**
    - **Elementwise.** A connected group of elementwise ops (`add` `sub` `mul`
      `div` `neg` `exp` `log` `relu` `tanh` `sqrt` `step` `stopGradient`) of
      one element count, with one output: its sink. A producer joins its
      consumer's group only if every one of its consumers is in that group, so
      nothing is computed twice. A `broadcast`, `reshape` or `transpose` whose
      source is outside the group becomes an indexed load in the fused kernel
      (it reads the source; no copy is made). A `reshape` of a value inside the
      group is free, because elementwise kernels index by flat position. A scalar
      constant, or one broadcast from a scalar, is inlined into the WGSL as its
      exact float32 bit pattern. Any other constant is an operand.
    - **Matmul epilogue.** A `matmul` whose only consumer is an elementwise
      group of its own shape: the matmul kernel computes its sum, then
      evaluates the group on it before storing, with the group's other
      operands loaded at the output's index. This covers bias plus ReLU and bias plus
      GELU (milestone 3b's tanh form). It uses the same naive or tiled matmul
      as the unfused lowering, so the pair differs only in the epilogue.
    - **Row-wise: LayerNorm, softmax, log-softmax.** A group over one
      `[R, C]` iteration space, with C the last axis, containing reductions
      (`sum`, `max`) over that axis with `keepDims`, whose results come back
      into the group through `broadcast`, plus elementwise ops on whole rows
      and on per-row scalars. One workgroup of 64 threads per row. Each
      reduction is a per-thread strided partial, then a tree through
      workgroup memory. Values over the row are recomputed in each phase that
      needs them rather than stored. The group is labelled `layernorm` (two
      sums and a `sqrt`), `softmax` (`max`, `exp`, `sum`, `div`),
      `log-softmax` (`max`, `exp`, `sum`, `log`), or else `rowwise`. The
      label is for reporting and sabotage only; the code generator is the same
      for every label.
    - **Attention (stretch).** Attempted only if everything else in this
      milestone is recorded. If it is not attempted, the report says so.
    Fixed limits: at most 6 operands per fused node, so that it binds within
    WebGPU's default of 8 storage buffers per shader stage (6, the output and
    the parameter block). A group that would exceed this stops growing. Every
    kernel reads its sizes and strides from its parameter block at run time,
    as the unfused kernels do, so a fused kernel is not specialised to a shape
    the unfused one is not. Reductions not over the last axis, and the
    loss's two plain sums, are not fused.
    **Option `recompute`, off by default, for measuring losses.** With it on, an
    elementwise producer may also join a group across a `broadcast` or
    `transpose`, which recomputes it once per consumer element, and a producer
    with several consumers is copied into each of their groups instead of
    being stored once. This is the fusion the milestone-1 section above
    expects to lose. It is measured on purpose and never used for the
    training step.

31. **Correctness first, before any timing.** Every fused kernel kind must
    pass three checks before anything is timed.
    - **Direct adversarial tests** (the correctness claim per kernel kind).
      Built from small graphs that fuse into the intended kind. Element counts
      1, 63, 64, 65, 1,000 and 100,003, and shapes `[3,5,7]` and `[1,7,1]`.
      Matmul shapes off the 16-wide tile (17x33x19, 15x16x17, 1x64x1, batched
      3x5x7x2). Rows of length C in {1, 7, 63, 64, 65, 257, 1000}. Large and
      small magnitudes. Softmax rows at [400, 600], [-600, -400] and with -1e4
      masks. LayerNorm rows that are constant (zero variance) or far from
      zero. Every buffer has milestone 2's canary padding. Judged at the sink
      by amendment 8's bound composed over the original small graph, `S = 2`.
      Target: every case passes.
    - **Differential fuzzer.** Generator v2 in float32 mode, `corpusProgram` 1
      to 2,000, forward and gradient graphs, naive and tiled matmul, as in
      milestone 2. Each run executes the unfused graph and the fused graph on the
      GPU. Both are judged against the float64 interpreter, the fused one at
      every surviving node, with amendment 8's composed bound and milestone 2's skip
      rule (fuzz programs are small, so the composed bound is meaningful there,
      unlike in a whole transformer step, R3). How often the fused and
      unfused GPU values are bitwise identical is reported, not required.
      Target: no unexplained failure.
    - **The float32 budget on the real model**, node-local as in R3: the fused
      training step of each of milestone 3b's four golden runs, all 20 steps.
      Every launch is given the float32-rounded reference values of its own
      operands. A fused node is judged against its body's float64 result by
      the bound composed over its body. That is exactly the budget of the
      unfused ops it replaces, as the milestone-1 section above requires.
      Target: every step passes.
    - **Sabotage, each bug injected alone**, and which check catches it first.
      The checks run in this order: semantic preservation, direct tests, fuzzer (programs 1 to
      500, to bound the run time), node-local training steps (golden run 1, first 2 steps).
      1. Matmul epilogue: the activation applied before the bias is added.
      2. Fused LayerNorm: its reductions' workgroup tree stops one level early,
         so each mean and variance loses the second half of its last combine.
      3. Fused softmax and log-softmax: the row maximum never subtracted.
      4. A bounds bug at the last workgroup: the fused elementwise kernel's guard lets
         index `n` through.
      5. In the pass itself, not a kernel: a producer that still has a consumer
         outside the group is absorbed, and that consumer is rewired to the
         group's output.
      Predictions, fixed now. Bugs 1 to 4 are invisible to semantic preservation,
      because it never runs a kernel. Bug 5 is caught by it. The direct tests catch 1 to 4,
      bug 3 through the large-input rows and bug 4 through the canaries. The fuzzer
      cannot catch bug 2, because generator v2 never builds a LayerNorm, nor
      bug 4, because its buffers are exact-size (milestone 2's finding). The
      node-local training step catches 1 and 2 (the model has GELU epilogues and LayerNorms),
      not 4 (exact-size buffers), and probably not 3, since the model's softmax inputs stay far
      below exp's overflow. Every bug is caught by at least one check.

32. **The timing protocol.**
    - **Clock.** GPU timestamp queries. This adapter reports `timestamp-query`
      (checked before writing this, along with its other limits: no kernel
      was timed). Wall-clock time from `submit` to `onSubmittedWorkDone` is
      recorded beside every timestamp result, never in place of it. If an
      adapter lacks timestamps, the page falls back to wall-clock and says so
      in the record.
    - **One timed repetition** is one command buffer with one compute pass,
      timestamped at its beginning and end. The pass contains the program's
      dispatches k times over. k is fixed per workload during warm-up, so that a
      repetition of the unfused variant takes at least 2 ms, and the fused
      variant uses the same k. The time per program run is the pass time divided by k. This is
      llm-inference-webgpu's method. It makes the browser's 65.5 µs timestamp
      quantum (measured there) irrelevant, and it keeps per-submit costs out
      of kernel time.
    - **Order.** For each workload: 5 warm-up repetitions of each variant,
      discarded. Then 32 timed repetitions of each, alternated in ABBA order,
      so a slow drift affects both alike. Reported: the median time per
      program run, the minimum and maximum, and every repetition.
    - **What counts as a difference.** Fused is said to win (or lose) only if
      the two variants' min-to-max ranges do not overlap. Otherwise the result
      is "within noise", whatever the medians say. The speedup is the unfused median
      over the fused median.
    - **Power.** The machine's own setting: AC power, Windows overlay "Best
      power efficiency" (milestone 3a's memory of this machine: it moves CPU timings
      about 3.3x and the iGPU little, and nothing here is a CPU-against-GPU comparison).
      The local server reports the power mode. The page reads it at the
      start, after every workload and at the end. A session in which it
      changes, or in which the machine is ever off AC, is discarded: logged
      in the record and not reported as a result. A session is one load of
      the benchmark page, and only results from one session are compared with
      each other.
    - **No GPU object creation inside the timed region.** Every buffer, bind
      group, parameter block, pipeline and query set is made before timing
      starts. The harness wraps the device's `create*` methods and
      `queue.writeBuffer` with counters and records, per workload, how many
      were called inside the timed region. Required: zero, except one command
      encoder per repetition, which WebGPU requires (a command buffer cannot be
      resubmitted). The counter is tested by breaking it: a deliberately
      wrong mode that makes a bind group per repetition must be flagged.
      This follows llm-inference-webgpu's lesson that creating objects per
      step cost about half a decode step.
    - **Correctness gates the timing.** The benchmark page refuses to run unless
      `golden/gpu-m4.json` exists, was made from the current sources, and
      passed every direct test and the fuzzer. Before timing a workload it also runs both
      variants once and checks their outputs against the float64 interpreter
      within amendment 8's budget. A workload that fails is recorded as failed
      and not timed.

33. **Achieved bandwidth.** For each memory-bound workload, the bytes a
    variant must move are counted from the graph: for every kernel, each
    operand buffer it reads (at its own size, once, as if every reuse hit
    cache) plus its output. Achieved GB/s is bytes over median time. The
    ceiling is measured on this adapter in the same session: llm-inference-webgpu's pure-read
    kernel (`u32`, `vec2` and `vec4` loads at 256, 1,024 and 4,096 workgroups of
    256 threads, over a 128 MiB buffer), plus a copy kernel (read and write, the
    same widths), which is the ceiling an elementwise kernel faces. The best
    configuration of each is reported as that ceiling. llm-inference-webgpu
    measured 57.3 to 60.4 GB/s for pure reads; that figure is quoted for
    comparison only, not used.

34. **The sweep.** Every size includes values that are not multiples of the
    64-thread workgroup or the 16-wide tile.
    - Unary elementwise chains of length L in {1, 2, 4, 8, 16} (exp, tanh,
      neg, sqrt-of-abs and similar, repeated) at n in {1,000, 65,537,
      1,048,573, 4,194,301}.
    - Binary chains (each op reads a fresh input) of length L in {1, 2, 4, 8}
      at n in {65,537, 4,194,301}. The 6-operand limit splits L = 8.
    - Matmul + bias + ReLU and + bias + GELU: `[M, K] x [K, N]` with
      (M, K, N) in {(512, 64, 64), (512, 64, 256), (512, 256, 64),
      (1000, 63, 257), (4096, 64, 256), (128, 1024, 128), (2048, 512, 512)},
      naive and tiled.
    - LayerNorm and softmax over R rows of length C: C in {1, 7, 64, 65, 256, 1000, 4096},
      R in {1, 64, 4096, 65536}, capped at 2^26 elements.
    - Losses on purpose: (a) across a broadcast. An expensive producer
      (16 transcendental ops on `[C]`, C = 256) broadcast to `[R, C]` and added,
      R in {16, 256, 4096}, in three variants: unfused, default fusion (the producer
      stored once) and `recompute`. (b) Lost reuse. A producer of 1 or 16 ops
      feeding 4 consumers, n = 1,048,573, with default fusion and with `recompute`.
    - The training step (amendment 36).

35. **Predictions, each with what would falsify it.**
    - **P1. Unary chains at large n (≥ 1,048,573) are memory-bound.** The
      unfused chain moves 2L·n floats, the fused one 2n, so the speedup should
      grow with L, and be at least L/2 at L ≥ 2. Falsified if it does not increase from
      L = 1 to L = 16, or is below L/2 at any L ≥ 2. At small n (1,000) the
      cost per dispatch dominates, so the speedup should also grow with L. No size is
      predicted there.
    - **P2. Binary chains.** Traffic falls from 3L·n to (L + 2)·n, so at
      n = 4,194,301 the speedup should rise with L but stay below 3. Falsified
      if it exceeds 3 beyond noise, or falls as L rises from 1 to 4.
    - **P3. Matmul epilogues help where the epilogue's traffic matters, and are
      neutral where the matmul dominates.** Predicted to win for GELU (about a
      dozen unfused kernels, some of them broadcasts of constants) at every
      shape. Predicted to be neutral, within 10% or within noise, for ReLU at K ≥ 512.
      A loss is possible from register pressure in the tiled kernel with a GELU
      epilogue. If fused is slower beyond noise anywhere, that is reported as
      the loss it is. Falsified as stated if GELU fusion does not win at some
      shape, or ReLU at K ≥ 512 moves more than 10% beyond noise.
    - **P4. Row-wise fusion wins big where rows are wide, and can lose where
      they are narrow.** The unfused LayerNorm is about ten kernels, and its
      reductions use one thread per row; the fused one is a single kernel, with
      a 64-thread tree per row. Predicted speedup above 2 for C ≥ 64 and R ≥ 64.
      Predicted loss for C ≤ 7 with R = 65,536, where most of each workgroup's
      64 threads have nothing to do. That loss prediction is falsified if fused
      still wins beyond noise at C = 1 and C = 7.
    - **P5. Recomputing across a broadcast loses.** The recompute variant
      evaluates the 16-op producer R times per element instead of once.
      Predicted slower than the default fusion beyond noise for R ≥ 256, with
      the gap growing with R. Falsified otherwise.
    - **P6. Lost reuse.** Copying the producer into 4 consumer groups is predicted to lose
      to storing it once when the producer is 16 ops, and to be neutral or
      win when it is 1 op, since it saves a store and 4 reads. Falsified if the 16-op
      case does not lose.
    - **P7. The training step (milestone 3b's model, synthetic, B = 8 and
      B = 16) is bounded by Amdahl's law.** Fusion removes almost only
      non-matmul work. In the same session, the unfused step is also timed with only its
      matmul dispatches (fraction s of the step). Predicted: speedup ≤ 1/s,
      and within 1.25 if s ≥ 0.8 ("matmul-dominated", the brief's neutral case).
      Falsified if the speedup exceeds 1/s beyond noise. (That would mean the
      fused matmul-with-epilogue kernel itself got faster, which nothing here
      intends.) I do not know s in advance and do not predict it.
    - **P8. Against the hand-written kernels** (amendment 37), the compiler is
      predicted to lose at matrix-vector shapes by at least 2x. Its matmuls run one
      thread per output with a serial loop over K, where the hand-written
      kernel gives each row a 256-thread tree and reads f16 weights. It is
      predicted to be within noise, or within 20%, of the hand-written `ADD`
      (the same one-thread-per-element design). Fused SwiGLU and RMSNorm are
      predicted within 2x of the hand-written ones, and unfused ones slower than 2x.
      Falsified per kernel as stated.
    - **P9. Bandwidth.** At n = 4,194,301, every fused elementwise kernel is
      predicted to reach at least half the measured copy ceiling. Falsified
      below half.

36. **Training, fused against unfused.**
    - **Step time.** The milestone 3b training step (synthetic, V = 16) at
      B = 8 and B = 16, and the text model (V = 75) at B = 16, timed per
      amendment 32. The fused step, the unfused step, and the unfused step's
      matmul dispatches alone, each beside the power mode.
    - **A new, tighter statistical comparison**, replacing nothing (milestone
      3b's result stands). The same 24 seeds, batch 8, 300 Adam steps,
      synthetic task and validation loss as amendment 23 (iv), trained with the
      **fused** step on the GPU. The paired differences, fused GPU minus PyTorch
      float64, must pass a two one-sided t-test of equivalence at α = 0.05 with margin
      **±1e-5 nats**. Justification, from milestone 3b's own spread (F8): there,
      GPU minus PyTorch float64 had standard deviation 5.7e-7 over the 24 seeds, and
      no seed differed by more than 2.6e-6. 1e-5 is about four times that
      worst seed. If fused kernels behaved like unfused ones, the 90% interval
      would be near ±2e-7, far inside. A tenfold wider spread would still pass.
      A fused kernel that shifted the trained model by a hundred-thousandth of a
      nat per character would fail. Second comparison, same test and margin:
      fused GPU minus milestone 3b's recorded unfused GPU, paired by seed.
      Whether any seed's fused trajectory is bitwise identical to the unfused
      one is reported, and not predicted. The row-wise kernels sum in a different
      order, so I expect it is not.

37. **Baselines: llm-inference-webgpu's hand-written kernels**, copied
    verbatim (with the commit and a SHA-256 of each) into
    `src/fusion/llm-kernels.js`, on the shapes the two projects share, in
    the same session as the compiler's versions:
    - `ADD`: n = 896, and 52·896 (a 52-token prefill);
    - `SWIGLU`: n = 4,864 and 52·4,864, against the IR's
      `g / (1 + exp(-g)) · u`, unfused and fused;
    - `RMSNORM`: rows 1 and 52 of 896, against the IR's
      `x / sqrt(mean(x²) + eps) · gain`, unfused and fused (row-wise);
    - `MATVEC`: 896x896 and 4,864x896, against the compiler's naive and tiled
      matmul of `[1, 896]` by `[896, N]`.
    Two differences are stated rather than hidden. The hand-written MATVEC and
    RMSNORM read f16 weights and gains, where this compiler only has f32. And the
    hand-written ADD and SWIGLU work in place. Byte counts for bandwidth use
    each kernel's real traffic. For MATVEC an f32-weight copy of the
    hand-written kernel (the same text, with `f16` replaced by `f32`) is also timed and labelled as
    such, so the comparison at equal bytes exists too. The report says where the compiler loses.

38. **What the report may claim.** One adapter (Intel Iris Xe, gen-12lp), one
    machine, one browser, an integrated GPU sharing system memory. Every
    timing carries its power mode. Nothing here is claimed to generalise to
    discrete GPUs, other drivers or other browsers. CI cannot run the GPU: as
    before, it checks that each recording came from the checked-in sources,
    recomputes everything that needs no GPU (semantic preservation included),
    and checks the report's figures against the recordings.

### Milestone 4 revisions, before any timing

R5. **The training step contains none of the full LayerNorm, softmax or
    GELU-epilogue patterns** (2026-10-19, the first fusion of the step, before
    any kernel was timed). Autodiff's backward pass reads values from inside
    the forward patterns: each LayerNorm's centred input and its standard
    deviation, each attention softmax's `exp`, and the GELU's input. So those
    values must be stored, and every group stops at them. The fused step
    keeps matmul epilogues that add a bias, or a bias and a residual, but none
    that applies an activation. It has row-wise kernels for parts of the
    softmax (the shift and the `exp`, or the normalisation), and none labelled
    `layernorm`, `softmax` or `log-softmax`. The figures are in `m4.fusion` of
    `golden/measurements.json`. Consequence: amendment 31's prediction that
    the node-local training step would catch bugs 1 and 2 cannot come true,
    because neither bug has a target there. That prediction stays as
    registered, and is reported as failed. Added, before any timing: the fused
    **forward pass** (logits and loss from parameters, as in inference), which
    does contain all three patterns. It is (a) a node-local check like R3's on
    the four golden configs at initialisation, run for every sabotage after the
    registered checks, and (b) a timed workload, synthetic `V = 16, B = 8` and
    text `V = 75, B = 16`.
    Also found and fixed before any timing, in my pass: default fusion took the
    producer behind a broadcast into the group. That is the recomputation that
    only the `recompute` option should do (amendment 30). Semantics were
    unaffected, since the interpreter evaluates bodies exactly, so only a
    structural test could see it, and one now does.

R6. **Two parts of amendments 32 and 34 could not be done as written, and are
    changed before any timing.**
    - The row workloads' cap of 2^26 elements. Unfused, a LayerNorm holds
      about 18 tensors of its input's size. At 2^26 float32 elements that is
      about 4.6 GB, on a machine whose GPU shares 16 GB of system memory with
      everything else. Before: 2^26. After: 2^22, so `R·C ≤ 4,194,304`.
    - The gate "checks their outputs against the float64 interpreter within
      amendment 8's budget". For the model workloads (forward pass and
      training step) that budget is vacuous (R3: the logits have no finite
      bound), so every output would be unjudged. Before: the composed budget
      for every workload. After: the composed budget for every workload except
      those. For those, the gate is the node-local check recorded in
      `golden/gpu-m4.json`, and in the session the fused and unfused outputs
      are compared, and their largest difference is reported, with no
      tolerance. The composed budget is computed one node at a time, with only
      live values held, which gives the same numbers as `referenceWithBounds`
      (a test checks this). For the hand-written kernels it uses their f16
      weights and gains rounded as the kernels store them.
    - Which session is the result. Short runs of the benchmark page on subsets
      of workloads (`?only=`) were and will be made to debug the harness. They
      are not recorded and not reported. The result is the first complete
      session recorded from the final sources in which the power mode did not
      change. A session discarded for power is kept in its own file and
      reported as discarded.

F9. **A probe of mine overwrote milestone 2's out-of-bounds record** (2026-10-19).
    To read this adapter's limits, and later to try harness functions by hand,
    I opened `web/oob.html` in the debugging browser as a convenient page to
    run scripts in. I had not checked that it runs its probe on load and saves
    `golden/gpu-oob.json`. It was opened three times. `git status` showed the
    change before anything was committed. The file was restored from git, and
    its SHA-256 matches the committed one
    (`a5c855a1b422331f9351af0710e50ac341967b6e88d15fa650c5626e99df841d`).
    `golden/measurements.json` was regenerated from the restored file
    and compared with the copy made while the overwritten file was in place.
    The overwritten record had carried new adapter metadata into it (a longer
    feature list, a newer browser version, a new recording time) but no
    different probe result. Regenerated, every figure of milestones 1 to 3b is
    byte-identical to the committed file. Hand probes now run in the report
    page, which saves nothing.

F10. **The fuzzer found a bug in my fusion pass** (2026-10-19, the first
    recording of `web/m4.html`, before any timing). 6 of its 5,318 runs could
    not be compiled. They were corpus programs 129, 240 and 1737, forward
    graphs, naive and tiled matmul. In each, a value fed both a matmul and
    that matmul's own epilogue (for example `matmul(x, W) + x`). The pass took
    the value into the group because every one of its consumers was in the
    group. But a matmul reads its operands at arbitrary indices, so they must
    be stored, and the code generator found no buffer for it. The semantic
    check could not see this: the interpreter evaluates the body exactly. The
    direct tests could not either: none of them reuses a matmul operand. Fixed:
    a matmul epilogue's operands are never group members. A new test lowers
    every corpus graph and fails on program 129 with the fix taken out. The
    failing recording is kept as `golden/gpu-m4-run1.json`. Everything else
    in it passed. The recording that counts was made again from the fixed
    sources.

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
