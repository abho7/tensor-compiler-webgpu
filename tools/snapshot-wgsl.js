// Records every kernel's WGSL in golden/wgsl.json. test/gpu-codegen.test.js
// fails if the generated text no longer matches, so a kernel cannot change
// without this being rerun deliberately (and the GPU results re-recorded).
//
//   node tools/snapshot-wgsl.js

import { writeFileSync } from 'node:fs';
import { kernelSource, KERNEL_KINDS } from '../src/gpu/codegen.js';

const kernels = Object.fromEntries(KERNEL_KINDS.map((k) => [k, kernelSource(k)]));
writeFileSync(new URL('../golden/wgsl.json', import.meta.url), JSON.stringify({ kernels }, null, 1) + '\n');
console.log(`wrote ${KERNEL_KINDS.length} kernels`);
