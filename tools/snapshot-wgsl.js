// Records the WGSL of every kernel added since milestone 2 in
// golden/wgsl-m3.json. Milestone 2's kernels stay in golden/wgsl.json, its
// record, which this never rewrites. test/gpu-codegen.test.js fails if any
// generated text no longer matches its snapshot.
//
//   node tools/snapshot-wgsl.js

import { readFileSync, writeFileSync } from 'node:fs';
import { kernelSource, KERNEL_KINDS } from '../src/gpu/codegen.js';

const m2 = JSON.parse(readFileSync(new URL('../golden/wgsl.json', import.meta.url))).kernels;
const later = KERNEL_KINDS.filter((k) => !(k in m2));
const kernels = Object.fromEntries(later.map((k) => [k, kernelSource(k)]));
writeFileSync(new URL('../golden/wgsl-m3.json', import.meta.url), JSON.stringify({ kernels }, null, 1) + '\n');
console.log(`wrote ${later.length} kernels added since milestone 2`);
