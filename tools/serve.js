// A static file server for the report and the browser harness pages, copied
// from llm-inference-webgpu, range support included. It also accepts one kind
// of write: a harness POSTs its results to /record/<name>.json and they are
// saved to golden/<name>.json, stamped with the machine's power mode. It
// listens on 127.0.0.1 only.
//
//   node tools/serve.js [port]

import { createReadStream, statSync, existsSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.argv[2] ?? 8099);

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.safetensors': 'application/octet-stream',
  '.bin': 'application/octet-stream',
};

// Windows power-mode overlays (memory of this machine: they move CPU timings
// several-fold). Recorded with every result even when nothing is timed.
const OVERLAYS = {
  '961cc777-2547-4f9d-8174-7d86181b8a7a': 'Best power efficiency',
  '00000000-0000-0000-0000-000000000000': 'Balanced',
  'ded574b5-45a0-4f42-8737-46345c09c238': 'Best performance',
};

function powerMode() {
  const out = { platform: process.platform };
  if (process.platform !== 'win32') return out;
  try {
    const reg = execFileSync('reg', ['query', 'HKLM\\SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes'], { encoding: 'utf8' });
    for (const key of ['ActiveOverlayAcPowerScheme', 'ActiveOverlayDcPowerScheme']) {
      const m = new RegExp(`${key}\\s+REG_SZ\\s+(\\S+)`).exec(reg);
      if (m) out[key] = { guid: m[1], name: OVERLAYS[m[1]] ?? 'unknown' };
    }
    const battery = execFileSync('powershell', ['-NoProfile', '-Command', '(Get-CimInstance Win32_Battery).BatteryStatus'], { encoding: 'utf8' }).trim();
    out.batteryStatus = battery;
    out.onAC = battery === '2';
    out.activeMode = (out.onAC ? out.ActiveOverlayAcPowerScheme : out.ActiveOverlayDcPowerScheme)?.name ?? 'unknown';
  } catch (error) {
    out.error = String(error.message ?? error);
  }
  return out;
}

function record(req, res, name) {
  const chunks = [];
  let total = 0;
  req.on('data', (c) => {
    total += c.length;
    if (total > 200e6) { res.writeHead(413).end('too large'); req.destroy(); return; }
    chunks.push(c);
  });
  req.on('end', () => {
    let body;
    try { body = JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { res.writeHead(400).end('not JSON'); return; }
    body.recorded = { at: new Date().toISOString(), power: powerMode() };
    writeFileSync(join(ROOT, 'golden', `${name}.json`), JSON.stringify(body) + '\n');
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ saved: `golden/${name}.json`, power: body.recorded.power }));
  });
}

const server = createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  if (req.method === 'POST') {
    const m = /^\/record\/([a-z0-9-]+)\.json$/.exec(url.pathname);
    if (!m) { res.writeHead(404).end('POST only to /record/<name>.json'); return; }
    record(req, res, m[1]);
    return;
  }
  // The power mode on its own, so a benchmark page can check it between
  // workloads and discard a session in which it changed (DESIGN.md amendment 32).
  if (url.pathname === '/power') {
    res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' }).end(JSON.stringify(powerMode()));
    return;
  }
  const requested = decodeURIComponent(url.pathname);
  const relative = normalize(requested).replace(/^([/\\])+/, '');
  let path = join(ROOT, relative);

  // Refuse anything that escapes the repository. The server exists to hand the
  // browser this project's files and nothing else. Checked before the
  // directory rule below, so no resolution can walk out of the tree.
  if (!path.startsWith(ROOT + sep) && path !== ROOT) {
    res.writeHead(403).end('outside the repository');
    return;
  }

  // A directory serves its index.html, as GitHub Pages does. The report is
  // index.html at the repository root, so without this the local server
  // disagrees at `/` with the deployment it stands in for -- which is how it
  // 404'd there while the published site was fine.
  if (existsSync(path) && statSync(path).isDirectory()) {
    path = join(path, 'index.html');
  }
  if (!existsSync(path) || statSync(path).isDirectory()) {
    res.writeHead(404).end(`not found: ${relative}`);
    return;
  }

  const size = statSync(path).size;
  const type = TYPES[extname(path)] ?? 'application/octet-stream';
  const headers = {
    'content-type': type,
    'accept-ranges': 'bytes',
    // The harness compares CPU and GPU in one tab and reads timings, so it
    // wants a high-resolution clock and SharedArrayBuffer available.
    'cross-origin-opener-policy': 'same-origin',
    'cross-origin-embedder-policy': 'require-corp',
    'cache-control': 'no-cache',
  };

  const range = req.headers.range;
  if (!range) {
    res.writeHead(200, { ...headers, 'content-length': size });
    createReadStream(path).pipe(res);
    return;
  }

  const match = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
  if (!match) {
    res.writeHead(416, { 'content-range': `bytes */${size}` }).end();
    return;
  }
  // A range may be open at either end: "bytes=100-" or "bytes=-500", the
  // second meaning the last 500 bytes rather than everything up to 500.
  let start;
  let end;
  if (match[1] === '') {
    const suffix = Number(match[2]);
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] === '' ? size - 1 : Math.min(Number(match[2]), size - 1);
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) {
    res.writeHead(416, { 'content-range': `bytes */${size}` }).end();
    return;
  }

  res.writeHead(206, {
    ...headers,
    'content-length': end - start + 1,
    'content-range': `bytes ${start}-${end}/${size}`,
  });
  createReadStream(path, { start, end }).pipe(res);
});

server.listen(port, '127.0.0.1', () => {
  console.log(`serving ${ROOT} at http://127.0.0.1:${port}/`);
  console.log(`  report:  http://127.0.0.1:${port}/`);
  console.log(`  harness: http://127.0.0.1:${port}/web/gpu.html`);
});
