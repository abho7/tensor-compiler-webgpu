// The equivalence test pre-registered in DESIGN.md amendment 17: for each
// optimizer and metric, the paired difference (GPU minus CPU) over the seeds
// passes a two one-sided t-test at alpha = 0.05, i.e. its 90% confidence
// interval lies inside +-margin.

export const MARGINS = { trainLoss: 0.02, testAccuracy: 0.02 };
export const ALPHA = 0.05;

/** Regularised incomplete beta I_x(a, b), by continued fraction (Numerical Recipes' betacf). */
function betacf(a, b, x) {
  let c = 1, d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < 1e-300) d = 1e-300;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}
function lnGamma(z) {
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (const c of g) ser += c / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
function incBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}
/** Student's t cumulative distribution with nu degrees of freedom. */
export function tCdf(t, nu) {
  const p = 0.5 * incBeta(nu / (nu + t * t), nu / 2, 0.5);
  return t >= 0 ? 1 - p : p;
}
/** The t quantile, by bisection on tCdf. */
export function tQuantile(p, nu) {
  let lo = -50, hi = 50;
  for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (tCdf(mid, nu) < p) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}

/** TOST on paired differences d (GPU minus CPU) with margin m. */
export function tost(d, margin, alpha = ALPHA) {
  const n = d.length;
  const mean = d.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(d.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  const se = sd / Math.sqrt(n);
  const t = tQuantile(1 - alpha, n - 1);
  const lower = mean - t * se, upper = mean + t * se;
  return { n, mean, sd, se, t, lower, upper, margin, pass: lower > -margin && upper < margin,
    min: Math.min(...d), max: Math.max(...d) };
}

/** Both optimizers x both metrics, from the recorded runs. */
export function equivalence(runs) {
  const out = {};
  for (const opt of ['sgd', 'adam']) {
    const rs = runs.filter((r) => r.optimizer === opt);
    out[opt] = Object.fromEntries(Object.entries(MARGINS).map(([metric, margin]) =>
      [metric, tost(rs.map((r) => r.gpu[metric] - r.cpu[metric]), margin)]));
  }
  return out;
}
