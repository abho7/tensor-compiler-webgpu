// A small seeded PRNG (mulberry32), so every random program and every random
// input is reproducible from one integer.

export class Rng {
  constructor(seed) { this.state = seed >>> 0; }

  /** Uniform in [0, 1). */
  next() {
    let t = (this.state = (this.state + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Integer in [lo, hi]. */
  int(lo, hi) { return lo + Math.floor(this.next() * (hi - lo + 1)); }
  uniform(lo, hi) { return lo + (hi - lo) * this.next(); }
  pick(list) { return list[Math.floor(this.next() * list.length)]; }
  chance(p) { return this.next() < p; }

  /** `n` numbers uniform in [lo, hi]. */
  array(n, lo = -1, hi = 1) { return Array.from({ length: n }, () => this.uniform(lo, hi)); }

  shuffle(list) {
    const a = [...list];
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
}
