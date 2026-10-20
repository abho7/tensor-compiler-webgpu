// Milestone 4's committed inputs to measurement, read from the repository
// (Node only): the model spec, PyTorch's answers, and the GPU recordings.

import { readFileSync, existsSync, readdirSync } from 'node:fs';

export function m4Inputs(root) {
  const json = (p) => JSON.parse(readFileSync(new URL(p, root)));
  const optional = (p) => (existsSync(new URL(p, root)) ? json(p) : null);
  const discarded = readdirSync(new URL('golden/', root)).filter((f) => f.startsWith('gpu-m4-bench-discarded-')).sort().map((f) => json(`golden/${f}`));
  return {
    spec: json('golden/spec-m3b.json'), torch: json('golden/torch-m3b.json'), gpuM3b: optional('golden/gpu-m3b.json'),
    gpuRecord: optional('golden/gpu-m4.json'), gpuRun1: optional('golden/gpu-m4-run1.json'), benchRecord: optional('golden/gpu-m4-bench.json'), discarded,
  };
}
