// The files that decide what milestone 3b's GPU recordings compute
// (web/m3b.html and web/train-transformer.html). They are hashed with
// sources.js's hashSources; CI recomputes the hash and fails if a recording is
// stale. Milestone 3a's SOURCE_FILES are not edited (DESIGN.md amendment 21),
// so its recordings stay valid; this list includes the ones 3b also runs.

export const M3B_SOURCE_FILES = [
  'src/ir/shapes.js', 'src/ir/graph.js', 'src/interp.js', 'src/autodiff.js', 'src/testing/rng.js',
  'src/gpu/bounds.js', 'src/gpu/codegen.js', 'src/gpu/lower.js', 'src/gpu/runtime.js', 'src/gpu/trainer.js',
  'src/train/data.js', 'src/train/step.js',
  'src/m3b/model.js', 'src/m3b/adam.js', 'src/m3b/step.js', 'src/m3b/data.js', 'src/m3b/config.js',
  'src/m3b/cpu.js', 'src/m3b/verify-cpu.js',
  'src/gpu/harness-m3b.js', 'src/gpu/sources-m3b.js', 'web/m3b.html', 'web/train-transformer.html',
];
