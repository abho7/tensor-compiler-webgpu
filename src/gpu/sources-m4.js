// The files that decide what milestone 4's GPU recordings compute
// (web/m4.html and web/m4-bench.html). They are hashed with sources.js's
// hashSources; CI recomputes the hash and fails if a recording is stale.
// Milestone 3a's and 3b's source lists are not edited (DESIGN.md amendment 29),
// so their recordings stay valid; this list includes the files 4 also runs.

export const M4_SOURCE_FILES = [
  'src/ir/shapes.js', 'src/ir/graph.js', 'src/interp.js', 'src/autodiff.js', 'src/testing/rng.js', 'src/testing/random-graph.js',
  'src/gpu/bounds.js', 'src/gpu/codegen.js', 'src/gpu/lower.js', 'src/gpu/runtime.js', 'src/gpu/harness-m3b.js',
  'src/train/data.js', 'src/train/step.js',
  'src/m3b/model.js', 'src/m3b/adam.js', 'src/m3b/step.js', 'src/m3b/data.js', 'src/m3b/config.js', 'src/m3b/cpu.js', 'src/m3b/verify-cpu.js',
  'src/fusion/fuse.js', 'src/fusion/codegen-fused.js', 'src/fusion/lower-fused.js', 'src/fusion/direct-cases-m4.js',
  'src/fusion/trainer-fused.js', 'src/gpu/harness-m4.js', 'src/gpu/sources-m4.js', 'web/m4.html',
];

/** The benchmark's own files, hashed with M4_SOURCE_FILES for the timing record. */
export const M4_BENCH_FILES = [
  'src/fusion/bench.js', 'src/fusion/workloads.js', 'src/fusion/llm-kernels.js', 'web/m4-bench.html',
];
