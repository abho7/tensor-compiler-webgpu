// The files that determine what the GPU harness computes, and a hash over
// them. The harness records the hash with its results; CI recomputes it from
// the checked-out files and fails if they differ, so GPU results cannot
// silently outlive the code that produced them.

export const SOURCE_FILES = [
  'src/ir/shapes.js', 'src/ir/graph.js', 'src/interp.js', 'src/autodiff.js',
  'src/testing/rng.js', 'src/testing/random-graph.js',
  'src/gpu/bounds.js', 'src/gpu/codegen.js', 'src/gpu/lower.js', 'src/gpu/runtime.js',
  'src/gpu/direct-cases.js', 'src/gpu/harness.js', 'src/gpu/sources.js', 'web/gpu.html',
];

/** FNV-1a (32-bit) over each file's text, line endings normalised, in order. */
export function hashSources(texts) {
  let h = 0x811c9dc5;
  for (const text of texts) {
    const t = text.replace(/\r\n/g, '\n');
    for (let i = 0; i < t.length; i++) {
      h ^= t.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    h ^= 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}
