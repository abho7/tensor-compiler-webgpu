// Milestone 3b's committed inputs to measurement, read from the repository
// (Node only): the spec, PyTorch's answers, the text and the GPU recordings.

import { readFileSync, existsSync } from 'node:fs';

export function m3bInputs(root) {
  const read = (p) => readFileSync(new URL(p, root));
  const json = (p) => JSON.parse(read(p));
  const optional = (p) => (existsSync(new URL(p, root)) ? json(p) : null);
  return {
    spec: json('golden/spec-m3b.json'), torch: json('golden/torch-m3b.json'), torchBytes: read('golden/torch-m3b.f64'),
    aliceText: read('data/alice/alice.txt').toString('utf8'),
    gpuRecord: optional('golden/gpu-m3b.json'), gpuDemo: optional('golden/gpu-m3b-demo.json'),
  };
}
