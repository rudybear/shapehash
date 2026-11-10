/** sh encoder: cube-map radius/colour maps fitted with real spherical harmonics. */
import { BitWriter, METHOD_SH, SH_MAX_L, SH_MAX_LC, SH_MIN_RADIUS, SQRT4PI, coeffScale, realSH, shNorm, toBase64Url } from "../format.js";
import { Source, header } from "./source.js";

const SH_HEAD = 8 + 18 + 15 + 14;
const CUBE_R = 32;

function shBits(L: number, Ly: number, Lc: number): number {
  const grp = (l: number, b: number, dc: number): number => dc + (l > 0 ? 6 + ((l + 1) ** 2 - 1) * b : 0);
  return SH_HEAD + grp(L, 5, 7) + grp(Ly, 4, 6) + 2 * grp(Lc, 3, 5);
}

export function shDegrees(budget: number): [number, number, number] | null {
  const bits = budget * 8;
  if (shBits(0, 0, 0) > bits) return null;
  let best: [number, number, number] = [0, 0, 0];
  for (let L = 0; L <= SH_MAX_L; L++) {
    const Ly = Math.min(Math.floor(0.62 * L + 0.5), SH_MAX_L), Lc = Math.min(Math.floor(0.38 * L + 0.5), SH_MAX_LC);
    if (shBits(L, Ly, Lc) <= bits) best = [L, Ly, Lc];
  }
  let [L, Ly, Lc] = best;
  while (Ly < L && shBits(L, Ly + 1, Lc) <= bits) Ly++;
  while (Lc < Math.min(Ly, SH_MAX_LC) && shBits(L, Ly, Lc + 1) <= bits) Lc++;
  return [L, Ly, Lc];
}

/** Solve the symmetric positive definite system A x = b in place (Cholesky). A is n x n row-major. */
export function choleskySolve(A: Float64Array, b: Float64Array, n: number): Float64Array {
  for (let j = 0; j < n; j++) {
    let d = A[j * n + j];
    for (let k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    d = Math.sqrt(Math.max(d, 1e-300));
    A[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s / d;
    }
  }
  const x = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    let s = x[i];
    for (let k = 0; k < i; k++) s -= A[i * n + k] * x[k];
    x[i] = s / A[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < n; k++) s -= A[k * n + i] * x[k];
    x[i] = s / A[i * n + i];
  }
  return x;
}

function quantCoeffs(w: BitWriter, c: Float64Array, nbits: number): void {
  let m = 0;
  for (let i = 1; i < c.length; i++) m = Math.max(m, Math.abs(c[i]));
  let qs = 63;
  for (let q = 0; q < 64; q++) if (coeffScale(q) >= m) { qs = q; break; }
  const s = coeffScale(qs);
  const half = (1 << (nbits - 1)) - 1;
  w.write(qs, 6);
  for (let i = 1; i < c.length; i++) {
    const u = Math.sign(c[i]) * Math.sqrt(Math.min(Math.abs(c[i]) / s, 1));
    // round half to even, like the reference
    let r = Math.round(u * half);
    const x = u * half;
    if (Math.abs(x - Math.trunc(x)) === 0.5) r = 2 * Math.round(x / 2);
    w.write(Math.min(half, Math.max(-half, r)) + half, nbits);
  }
}

export function encodeSh(src: Source, budget: number): string | null {
  const deg = shDegrees(budget);
  if (!deg) return null;
  const [L, Ly, Lc] = deg;
  const e = src.e;
  // centre: centroid of the solid volume, in cube coordinates
  const occ = src.solid([48, 48, 48], e.map((v) => -v), e);
  const c = [0, 0, 0];
  let cnt = 0;
  for (let x = 0; x < 48; x++)
    for (let y = 0; y < 48; y++)
      for (let z = 0; z < 48; z++)
        if (occ[(x * 48 + y) * 48 + z]) { c[0] += x; c[1] += y; c[2] += z; cnt++; }
  const cq = c.map((v) => (cnt ? Math.min(31, Math.max(0, Math.floor((((v / cnt + 0.5) / 48) * 2 - 1 + 0.5) * 31 + 0.5))) : 15));
  const cc = cq.map((q) => q / 31 - 0.5);
  // cube-map bins: outermost radius and the colour found there
  const R = CUBE_R, nb = 6 * R * R;
  const rmax = new Float64Array(nb);
  const bin = new Int32Array(src.n);
  const rad = new Float32Array(src.n);
  const q = [0, 0, 0];
  for (let i = 0; i < src.n; i++) {
    for (let d = 0; d < 3; d++) q[d] = src.pts[i * 3 + d] / e[d] - cc[d];
    const ax = Math.abs(q[0]), ay = Math.abs(q[1]), az = Math.abs(q[2]);
    let a = 0;
    if (ay > ax && ay >= az) a = 1;
    else if (az > ax && az > ay) a = 2;
    const m = Math.max(Math.abs(q[a]), 1e-12);
    const u = q[a === 0 ? 1 : 0] / m, v = q[a === 2 ? 1 : 2] / m;
    let iu = Math.floor(((u + 1) / 2) * R), iv = Math.floor(((v + 1) / 2) * R);
    iu = iu < 0 ? 0 : iu >= R ? R - 1 : iu;
    iv = iv < 0 ? 0 : iv >= R ? R - 1 : iv;
    const b = ((a * 2 + (q[a] < 0 ? 1 : 0)) * R + iu) * R + iv;
    const r = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    bin[i] = b; rad[i] = r;
    if (r > rmax[b]) rmax[b] = r;
  }
  const acc = new Float64Array(nb * 4);
  for (let i = 0; i < src.n; i++) {
    const b = bin[i];
    if (rad[i] >= 0.9 * rmax[b]) {
      acc[b * 4] += src.col[i * 3]; acc[b * 4 + 1] += src.col[i * 3 + 1]; acc[b * 4 + 2] += src.col[i * 3 + 2]; acc[b * 4 + 3]++;
    }
  }
  const mean = src.meanColor();
  // targets and weights per bin
  const nc = (L + 1) * (L + 1);
  const B = new Float64Array(nb * nc);
  const wgt = new Float64Array(nb), wcol = new Float64Array(nb);
  const tgt = [new Float64Array(nb), new Float64Array(nb), new Float64Array(nb), new Float64Array(nb)];
  const norm = shNorm();
  const row = new Float64Array(nc);
  for (let face = 0; face < 6; face++)
    for (let iu = 0; iu < R; iu++)
      for (let iv = 0; iv < R; iv++) {
        const b = (face * R + iu) * R + iv;
        const u = ((iu + 0.5) / R) * 2 - 1, v = ((iv + 0.5) / R) * 2 - 1;
        const a = face >> 1, sg = face & 1 ? -1 : 1;
        const d = a === 0 ? [sg, u, v] : a === 1 ? [u, sg, v] : [u, v, sg];
        const l = Math.sqrt(1 + u * u + v * v);
        realSH(L, d[0] / l, d[1] / l, d[2] / l, row, norm);
        B.set(row, b * nc);
        wgt[b] = 1 / Math.pow(1 + u * u + v * v, 1.5);
        const hit = acc[b * 4 + 3] > 0;
        wcol[b] = wgt[b] * (hit ? 1 : 0.02);
        const col = hit ? [acc[b * 4] / acc[b * 4 + 3], acc[b * 4 + 1] / acc[b * 4 + 3], acc[b * 4 + 2] / acc[b * 4 + 3]] : mean;
        tgt[0][b] = Math.max(rmax[b], SH_MIN_RADIUS);
        tgt[1][b] = 0.25 * col[0] + 0.5 * col[1] + 0.25 * col[2];
        tgt[2][b] = 0.5 * col[0] - 0.5 * col[2];
        tgt[3][b] = -0.25 * col[0] + 0.5 * col[1] - 0.25 * col[2];
      }
  const ridge = (l: number, y: Float64Array, w: Float64Array): Float64Array => {
    const n = (l + 1) * (l + 1);
    const A = new Float64Array(n * n), rhs = new Float64Array(n);
    let ws = 0;
    for (let b = 0; b < nb; b++) {
      const wb = w[b], o = b * nc;
      ws += wb;
      for (let i = 0; i < n; i++) {
        const bi = B[o + i] * wb;
        rhs[i] += bi * y[b];
        for (let j = 0; j <= i; j++) A[i * n + j] += bi * B[o + j];
      }
    }
    const den = Math.max(1, (l * (l + 1)) ** 2);
    for (let ll = 0, i = 0; ll <= l; ll++)
      for (let m = 0; m < 2 * ll + 1; m++, i++) A[i * n + i] += (1e-4 * ws * (ll * (ll + 1)) ** 2) / den + 1e-9 * ws;
    return choleskySolve(A, rhs, n);
  };
  const w = header(METHOD_SH, 0, src.eq);
  for (const v of cq) w.write(v, 5);
  w.write(L, 5); w.write(Ly, 5); w.write(Lc, 4);
  const spec: [number, number, number, number, number, Float64Array][] = [[L, 5, 7, 0, 2, wgt], [Ly, 4, 6, 0, 1, wcol], [Lc, 3, 5, -0.5, 0.5, wcol], [Lc, 3, 5, -0.5, 0.5, wcol]];
  spec.forEach(([l, bits, dcb, lo, hi, wg], k) => {
    const cf = ridge(l, tgt[k], wg);
    const n = (1 << dcb) - 1;
    w.write(Math.min(n, Math.max(0, Math.floor(((cf[0] / SQRT4PI - lo) / (hi - lo)) * n + 0.5))), dcb);
    if (l > 0) quantCoeffs(w, cf, bits);
  });
  const data = w.toBytes();
  return data.length <= budget ? toBase64Url(data) : null;
}
