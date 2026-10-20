// Copies four hand-written kernels from llm-inference-webgpu verbatim into
// src/fusion/llm-kernels.js, with the commit they came from and the SHA-256 of
// each kernel's text (DESIGN.md amendment 37). Run once, by hand:
//
//   node tools/vendor-llm-kernels.js ../llm-inference-webgpu

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const repo = resolve(process.argv[2] ?? '../llm-inference-webgpu');
const commit = execFileSync('git', ['-C', repo, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
const dirty = execFileSync('git', ['-C', repo, 'status', '--porcelain', 'src/gpu/kernels.js'], { encoding: 'utf8' }).trim();
if (dirty) throw new Error('src/gpu/kernels.js has uncommitted changes; vendor from a commit');
const k = await import(pathToFileURL(join(repo, 'src/gpu/kernels.js')).href);
const NAMES = ['ADD', 'SWIGLU', 'RMSNORM', 'MATVEC'];
const sha = (s) => createHash('sha256').update(s).digest('hex');
let out = `// Four hand-written WGSL kernels from llm-inference-webgpu (src/gpu/kernels.js at
// commit ${commit}), copied verbatim by tools/vendor-llm-kernels.js as baselines for
// milestone 4 (DESIGN.md amendment 37). Not edited: test/m4-bench.test.js checks
// each text against the SHA-256 recorded here. GROUP is that file's workgroup size.

export const SOURCE = { repository: 'llm-inference-webgpu', file: 'src/gpu/kernels.js', commit: '${commit}' };
export const GROUP = ${k.GROUP};
`;
const hashes = {};
for (const n of NAMES) {
  out += `\nexport const ${n} = ${JSON.stringify(k[n])};\n`;
  hashes[n] = sha(k[n]);
}
out += `\nexport const SHA256 = ${JSON.stringify(hashes, null, 2)};\n`;
writeFileSync(new URL('../src/fusion/llm-kernels.js', import.meta.url), out);
console.log(`wrote src/fusion/llm-kernels.js from ${commit}`, hashes);
