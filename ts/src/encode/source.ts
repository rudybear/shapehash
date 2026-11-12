/** Encoder input handling: normalisation, surface sampling, occupancy, scoring. Zero dependencies. */
import type { DecodedMesh } from "../decode.js";
import { BitWriter, VERSION } from "../format.js";

/** sRGB colour in [0,1] at barycentric coordinates (w0, w1, w2) of triangle `tri`. */
export type ColorSampler = (tri: number, w0: number, w1: number, w2: number, out: Float32Array | number[]) => void;

export interface MeshInput {
  /** xyz per vertex, any units / origin. */
  positions: ArrayLike<number>;
  /** Triangle list; omitted = non-indexed triangles. */
  indices?: ArrayLike<number>;
  /** Per-vertex sRGB colours, 3 per vertex, either 0..255 (Uint8Array) or 0..1 floats. */
  colors?: ArrayLike<number>;
  /** Alternative to `colors`: sample colour anywhere on the surface (textures). Takes precedence. */
  colorAt?: ColorSampler;
}

export const N_SAMPLES = 600_000;
export const N_FIT = 40_000;
export const N_SCORE = 8_000;
const F_TAU = 0.05;
const SIL_R = 24;

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function rgbToLab(r: number, g: number, b: number, out: Float64Array | number[], o = 0): void {
  const lin = (c: number): number => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const f = (t: number): number => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const R = lin(r), G = lin(g), B = lin(b);
  const fx = f((0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047);
  const fy = f(0.2126729 * R + 0.7151522 * G + 0.072175 * B);
  const fz = f((0.0193339 * R + 0.119192 * G + 0.9503041 * B) / 1.08883);
  out[o] = 116 * fy - 16;
  out[o + 1] = 500 * (fx - fy);
  out[o + 2] = 200 * (fy - fz);
}

/** Uniform-grid nearest-neighbour index over points in roughly [-1.2, 1.2]^3. */
export class PointGrid {
  private res: number;
  private start: Int32Array;
  private order: Int32Array;
  private lo = -1.25;
  private cell: number;
  constructor(private pts: ArrayLike<number>, n: number) {
    this.res = Math.max(4, Math.min(48, Math.round(Math.cbrt(n) * 1.2)));
    this.cell = 2.5 / this.res;
    const r = this.res;
    const key = new Int32Array(n);
    const count = new Int32Array(r * r * r + 1);
    for (let i = 0; i < n; i++) {
      key[i] = this.cellOf(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
      count[key[i] + 1]++;
    }
    for (let i = 0; i < r * r * r; i++) count[i + 1] += count[i];
    this.start = count.slice();
    this.order = new Int32Array(n);
    const fill = count.slice(0, r * r * r);
    for (let i = 0; i < n; i++) this.order[fill[key[i]]++] = i;
  }
  private clampCell(v: number): number {
    const c = Math.floor((v - this.lo) / this.cell);
    return c < 0 ? 0 : c >= this.res ? this.res - 1 : c;
  }
  private cellOf(x: number, y: number, z: number): number {
    return (this.clampCell(x) * this.res + this.clampCell(y)) * this.res + this.clampCell(z);
  }
  /** Index of the nearest point; squared distance in `this.d2`. */
  d2 = 0;
  nearest(x: number, y: number, z: number): number {
    const r = this.res, pts = this.pts;
    const cx = this.clampCell(x), cy = this.clampCell(y), cz = this.clampCell(z);
    let best = -1, bd = Infinity;
    for (let ring = 0; ring < r; ring++) {
      if (best >= 0) {
        const reach = (ring - 1) * this.cell; // everything within this distance has been visited
        if (reach > 0 && reach * reach >= bd) break;
      }
      const x0 = Math.max(0, cx - ring), x1 = Math.min(r - 1, cx + ring);
      const y0 = Math.max(0, cy - ring), y1 = Math.min(r - 1, cy + ring);
      const z0 = Math.max(0, cz - ring), z1 = Math.min(r - 1, cz + ring);
      for (let i = x0; i <= x1; i++)
        for (let j = y0; j <= y1; j++) {
          const shell = Math.abs(i - cx) === ring || Math.abs(j - cy) === ring;
          for (let k = z0; k <= z1; k++) {
            if (!shell && Math.abs(k - cz) !== ring) continue;
            const c = (i * r + j) * r + k;
            for (let q = this.start[c]; q < this.start[c + 1]; q++) {
              const p = this.order[q] * 3;
              const dx = pts[p] - x, dy = pts[p + 1] - y, dz = pts[p + 2] - z;
              const d = dx * dx + dy * dy + dz * dz;
              if (d < bd) { bd = d; best = this.order[q]; }
            }
          }
        }
    }
    this.d2 = bd;
    return best;
  }
}

export interface Sampled {
  pts: Float32Array;
  nrm: Float32Array;
  col: Float32Array;
}

/** Area-weighted samples of a coloured triangle mesh. */
export function sampleSurface(pos: ArrayLike<number>, idx: ArrayLike<number> | undefined, n: number, seed: number, color: ColorSampler): Sampled | null {
  const nt = idx ? Math.floor(idx.length / 3) : Math.floor(pos.length / 9);
  const cdf = new Float64Array(nt);
  let total = 0;
  const vi = (t: number, k: number): number => (idx ? idx[t * 3 + k] : t * 3 + k) * 3;
  for (let t = 0; t < nt; t++) {
    const a = vi(t, 0), b = vi(t, 1), c = vi(t, 2);
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const area = 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
    total += Number.isFinite(area) ? area : 0;
    cdf[t] = total;
  }
  if (!(total > 0)) return null;
  const rnd = mulberry32(seed);
  const pts = new Float32Array(n * 3), nrm = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const tmp = new Float32Array(3);
  for (let i = 0; i < n; i++) {
    const target = rnd() * total;
    let lo = 0, hi = nt - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    let u = rnd(), v = rnd();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    const w = 1 - u - v;
    const a = vi(lo, 0), b = vi(lo, 1), c = vi(lo, 2);
    for (let d = 0; d < 3; d++) pts[i * 3 + d] = w * pos[a + d] + u * pos[b + d] + v * pos[c + d];
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    nrm[i * 3] = nx / l; nrm[i * 3 + 1] = ny / l; nrm[i * 3 + 2] = nz / l;
    color(lo, w, u, v, tmp);
    for (let d = 0; d < 3; d++) col[i * 3 + d] = Math.min(1, Math.max(0, tmp[d]));
  }
  return { pts, nrm, col };
}

function dilate26(a: Uint8Array, X: number, Y: number, Z: number): Uint8Array {
  // separable 3x3x3 box maximum
  let src = a;
  const st = [Y * Z, Z, 1], n = [X, Y, Z];
  for (let ax = 0; ax < 3; ax++) {
    const dst = new Uint8Array(src.length);
    const s = st[ax];
    for (let x = 0; x < X; x++)
      for (let y = 0; y < Y; y++)
        for (let z = 0; z < Z; z++) {
          const i = (x * Y + y) * Z + z;
          const c = ax === 0 ? x : ax === 1 ? y : z;
          dst[i] = src[i] | (c > 0 ? src[i - s] : 0) | (c < n[ax] - 1 ? src[i + s] : 0);
        }
    src = dst;
  }
  return src;
}

/**
 * Solid occupancy of a point-sampled surface: splat, dilate the shell by one
 * voxel, flood-fill the outside from the padded border, grow the outside back.
 */
export function solidOccupancy(pts: Float32Array, n: number, dims: number[], lo: number[], hi: number[]): Uint8Array {
  const [nx, ny, nz] = dims;
  const X = nx + 4, Y = ny + 4, Z = nz + 4;
  const surf = new Uint8Array(X * Y * Z);
  for (let i = 0; i < n; i++) {
    let c = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      const v = Math.floor(((pts[i * 3 + d] - lo[d]) / (hi[d] - lo[d])) * dims[d]);
      c[d] = v < 0 ? 0 : v >= dims[d] ? dims[d] - 1 : v;
    }
    surf[((c[0] + 2) * Y + c[1] + 2) * Z + c[2] + 2] = 1;
  }
  const thick = dilate26(surf, X, Y, Z);
  const outside = new Uint8Array(X * Y * Z);
  const stack = new Int32Array(X * Y * Z);
  let sp = 0;
  outside[0] = 1;
  stack[sp++] = 0;
  const sx = Y * Z, sy = Z;
  while (sp > 0) {
    const i = stack[--sp];
    const z = i % Z, y = Math.floor(i / Z) % Y, x = Math.floor(i / sx);
    if (x > 0 && !outside[i - sx] && !thick[i - sx]) { outside[i - sx] = 1; stack[sp++] = i - sx; }
    if (x < X - 1 && !outside[i + sx] && !thick[i + sx]) { outside[i + sx] = 1; stack[sp++] = i + sx; }
    if (y > 0 && !outside[i - sy] && !thick[i - sy]) { outside[i - sy] = 1; stack[sp++] = i - sy; }
    if (y < Y - 1 && !outside[i + sy] && !thick[i + sy]) { outside[i + sy] = 1; stack[sp++] = i + sy; }
    if (z > 0 && !outside[i - 1] && !thick[i - 1]) { outside[i - 1] = 1; stack[sp++] = i - 1; }
    if (z < Z - 1 && !outside[i + 1] && !thick[i + 1]) { outside[i + 1] = 1; stack[sp++] = i + 1; }
  }
  const grown = dilate26(outside, X, Y, Z);
  const out = new Uint8Array(nx * ny * nz);
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++)
      for (let z = 0; z < nz; z++) {
        const i = ((x + 2) * Y + y + 2) * Z + z + 2;
        out[(x * ny + y) * nz + z] = !grown[i] || surf[i] ? 1 : 0;
      }
  return out;
}

function silhouettes(pts: ArrayLike<number>, n: number): Uint8Array[] {
  const out: Uint8Array[] = [];
  for (let ax = 0; ax < 3; ax++) {
    const o = [0, 1, 2].filter((k) => k !== ax);
    const m = new Uint8Array(SIL_R * SIL_R);
    for (let i = 0; i < n; i++) {
      let a = Math.floor(((pts[i * 3 + o[0]] + 1.05) / 2.1) * SIL_R), b = Math.floor(((pts[i * 3 + o[1]] + 1.05) / 2.1) * SIL_R);
      a = a < 0 ? 0 : a >= SIL_R ? SIL_R - 1 : a;
      b = b < 0 ? 0 : b >= SIL_R ? SIL_R - 1 : b;
      m[a * SIL_R + b] = 1;
    }
    out.push(m);
  }
  return out;
}

/** The asset as the encoders see it: a dense coloured surface sample in the normalised frame. */
export class Source {
  readonly pts: Float32Array;
  readonly nrm: Float32Array;
  readonly col: Float32Array;
  readonly n: number;
  readonly eq: [number, number, number];
  readonly e: [number, number, number];
  /** Translation and scale that map the input into the normalised frame: p_norm = (p - center) * scale. */
  readonly center: [number, number, number];
  readonly scale: number;
  readonly fitN: number;
  readonly scoreStart: number;
  readonly scoreN: number;
  private scoreGrid: PointGrid;
  private scorePts: Float32Array;
  private scoreLab: Float64Array;
  private scoreSil: Uint8Array[];
  private solids = new Map<string, Uint8Array>();
  cache = new Map<string, unknown>();

  constructor(input: MeshInput, samples = N_SAMPLES, seed = 2024) {
    const pos = input.positions, idx = input.indices;
    const nv = Math.floor(pos.length / 3);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    const count = idx ? idx.length : nv;
    if (count < 3) throw new Error("mesh has no triangles");
    for (let i = 0; i < count; i++) {
      const v = (idx ? idx[i] : i) * 3;
      if (!(v >= 0 && v + 2 < pos.length)) throw new Error("index out of range");
      for (let d = 0; d < 3; d++) {
        const c = pos[v + d];
        if (!Number.isFinite(c)) throw new Error("non-finite position");
        if (c < lo[d]) lo[d] = c;
        if (c > hi[d]) hi[d] = c;
      }
    }
    const half = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
    if (!(half > 0)) throw new Error("degenerate bounding box");
    this.center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    this.scale = 1 / half;
    const np = new Float32Array(nv * 3);
    for (let i = 0; i < nv * 3; i++) np[i] = (pos[i] - this.center[i % 3]) * this.scale;
    this.eq = [0, 0, 0];
    this.e = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      const ext = ((hi[d] - lo[d]) / 2) * this.scale;
      this.eq[d] = Math.min(63, Math.max(0, Math.ceil(ext * 64) - 1));
      this.e[d] = (this.eq[d] + 1) / 64;
    }
    let sampler: ColorSampler;
    if (input.colorAt) sampler = input.colorAt;
    else if (input.colors) {
      const c = input.colors;
      const k = c instanceof Uint8Array || c instanceof Uint8ClampedArray ? 1 / 255 : 1;
      sampler = (t, w0, w1, w2, out) => {
        const a = (idx ? idx[t * 3] : t * 3) * 3, b = (idx ? idx[t * 3 + 1] : t * 3 + 1) * 3, cc = (idx ? idx[t * 3 + 2] : t * 3 + 2) * 3;
        for (let d = 0; d < 3; d++) out[d] = (w0 * c[a + d] + w1 * c[b + d] + w2 * c[cc + d]) * k;
      };
    } else sampler = (_t, _a, _b, _c, out) => { out[0] = out[1] = out[2] = 0.8; };
    const s = sampleSurface(np, idx, samples, seed, sampler);
    if (!s) throw new Error("mesh has zero surface area");
    this.pts = s.pts; this.nrm = s.nrm; this.col = s.col; this.n = samples;
    this.fitN = Math.min(N_FIT, samples);
    this.scoreStart = Math.min(N_FIT, samples >> 1);
    this.scoreN = Math.min(N_SCORE, samples - this.scoreStart);
    this.scorePts = this.pts.subarray(this.scoreStart * 3, (this.scoreStart + this.scoreN) * 3);
    this.scoreGrid = new PointGrid(this.scorePts, this.scoreN);
    this.scoreLab = new Float64Array(this.scoreN * 3);
    for (let i = 0; i < this.scoreN; i++) {
      const o = (this.scoreStart + i) * 3;
      rgbToLab(this.col[o], this.col[o + 1], this.col[o + 2], this.scoreLab, i * 3);
    }
    this.scoreSil = silhouettes(this.scorePts, this.scoreN);
  }

  solid(dims: number[], lo: number[], hi: number[]): Uint8Array {
    const key = dims.join(",") + "|" + lo.map((v) => v.toFixed(6)).join(",") + "|" + hi.map((v) => v.toFixed(6)).join(",");
    let s = this.solids.get(key);
    if (!s) {
      s = solidOccupancy(this.pts, this.n, dims, lo, hi);
      this.solids.set(key, s);
    }
    return s;
  }

  meanColor(): [number, number, number] {
    const m: [number, number, number] = [0, 0, 0];
    for (let i = 0; i < this.fitN; i++) for (let d = 0; d < 3; d++) m[d] += this.col[i * 3 + d];
    return [m[0] / this.fitN, m[1] / this.fitN, m[2] / this.fitN];
  }

  /** F-score@0.05 of a point set against the score samples (geometry only). */
  fscore(p: Float32Array, n: number): number {
    const g = new PointGrid(p, n);
    let rc = 0, pr = 0;
    for (let i = 0; i < this.scoreN; i++) {
      g.nearest(this.scorePts[i * 3], this.scorePts[i * 3 + 1], this.scorePts[i * 3 + 2]);
      if (g.d2 < F_TAU * F_TAU) rc++;
    }
    for (let i = 0; i < n; i++) {
      this.scoreGrid.nearest(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
      if (this.scoreGrid.d2 < F_TAU * F_TAU) pr++;
    }
    rc /= this.scoreN; pr /= n;
    return (2 * pr * rc) / Math.max(pr + rc, 1e-9);
  }

  /**
   * Selection score in [0,1]. Inner searches use the mean of F-score@0.05 and a
   * colour term. With `full` (used to choose between methods) it is the mean of
   * F-score, colour term, a coarse silhouette IoU and normal consistency.
   */
  score(mesh: DecodedMesh, full = false): number {
    const colors = new Float32Array(mesh.colors.length);
    for (let i = 0; i < colors.length; i++) colors[i] = mesh.colors[i] / 255;
    const s = sampleSurface(mesh.positions, mesh.indices, N_SCORE, 7, (t, w0, w1, w2, out) => {
      const a = mesh.indices[t * 3] * 3, b = mesh.indices[t * 3 + 1] * 3, c = mesh.indices[t * 3 + 2] * 3;
      for (let d = 0; d < 3; d++) out[d] = w0 * colors[a + d] + w1 * colors[b + d] + w2 * colors[c + d];
    });
    if (!s) return 0;
    const n = N_SCORE, p = s.pts;
    const lab = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) rgbToLab(s.col[i * 3], s.col[i * 3 + 1], s.col[i * 3 + 2], lab, i * 3);
    const g = new PointGrid(p, n);
    let rc = 0, pr = 0, de1 = 0, de2 = 0, nc1 = 0, nc2 = 0;
    const gn = this.nrm, g0 = this.scoreStart * 3, dn = s.nrm;
    const dist = (a: Float64Array, i: number, b: Float64Array, j: number): number =>
      Math.sqrt((a[i] - b[j]) ** 2 + (a[i + 1] - b[j + 1]) ** 2 + (a[i + 2] - b[j + 2]) ** 2);
    for (let i = 0; i < this.scoreN; i++) {
      const j = g.nearest(this.scorePts[i * 3], this.scorePts[i * 3 + 1], this.scorePts[i * 3 + 2]);
      if (g.d2 < F_TAU * F_TAU) rc++;
      de1 += dist(this.scoreLab, i * 3, lab, j * 3);
      nc1 += Math.abs(gn[g0 + i * 3] * dn[j * 3] + gn[g0 + i * 3 + 1] * dn[j * 3 + 1] + gn[g0 + i * 3 + 2] * dn[j * 3 + 2]);
    }
    for (let i = 0; i < n; i++) {
      const j = this.scoreGrid.nearest(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
      if (this.scoreGrid.d2 < F_TAU * F_TAU) pr++;
      de2 += dist(lab, i * 3, this.scoreLab, j * 3);
      nc2 += Math.abs(dn[i * 3] * gn[g0 + j * 3] + dn[i * 3 + 1] * gn[g0 + j * 3 + 1] + dn[i * 3 + 2] * gn[g0 + j * 3 + 2]);
    }
    rc /= this.scoreN; pr /= n;
    const f = (2 * pr * rc) / Math.max(pr + rc, 1e-9);
    const de = 0.5 * (de1 / this.scoreN + de2 / n);
    const ct = 1 - Math.min(de, 40) / 40;
    if (!full) return 0.5 * (f + ct);
    const sil = silhouettes(p, n);
    let iou = 0;
    for (let k = 0; k < 3; k++) {
      let inter = 0, uni = 0;
      for (let i = 0; i < SIL_R * SIL_R; i++) {
        if (sil[k][i] & this.scoreSil[k][i]) inter++;
        if (sil[k][i] | this.scoreSil[k][i]) uni++;
      }
      iou += inter / Math.max(uni, 1);
    }
    const nc = 0.5 * (nc1 / this.scoreN + nc2 / n);
    return (f + ct + iou / 3 + nc) / 4;
  }
}

export function header(method: number, param: number, eq: number[]): BitWriter {
  const w = new BitWriter();
  w.write(VERSION, 3);
  w.write(method, 2);
  w.write(param, 3);
  for (const v of eq) w.write(v, 6);
  return w;
}
