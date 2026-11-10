/** vox encoder: resolution / colour-mode search with a range-coded occupancy grid. */
import { voxCubes, type VoxContent } from "../decode.js";
import { METHOD_VOX, OCC_PRIOR, RC_PROB_INIT, RangeEncoder, VOX_HEADER_BYTES, VOX_MAX_DIM, rgb555to8, toBase64Url } from "../format.js";
import { N_SCORE, Source, header, rgbToLab, sampleSurface } from "./source.js";

const RES_CANDIDATES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48, 56, 64];
const F_CANDIDATES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16];

function erode6(a: Uint8Array, X: number, Y: number, Z: number): Uint8Array {
  const out = new Uint8Array(a.length);
  const sx = Y * Z, sy = Z;
  for (let x = 1; x < X - 1; x++)
    for (let y = 1; y < Y - 1; y++)
      for (let z = 1; z < Z - 1; z++) {
        const i = (x * Y + y) * Z + z;
        out[i] = a[i] & a[i - sx] & a[i + sx] & a[i - sy] & a[i + sy] & a[i - 1] & a[i + 1];
      }
  return out;
}
function dilate6(a: Uint8Array, X: number, Y: number, Z: number): Uint8Array {
  const out = new Uint8Array(a.length);
  const sx = Y * Z, sy = Z;
  for (let x = 0; x < X; x++)
    for (let y = 0; y < Y; y++)
      for (let z = 0; z < Z; z++) {
        const i = (x * Y + y) * Z + z;
        out[i] = a[i] | (x > 0 ? a[i - sx] : 0) | (x < X - 1 ? a[i + sx] : 0) | (y > 0 ? a[i - sy] : 0) | (y < Y - 1 ? a[i + sy] : 0) | (z > 0 ? a[i - 1] : 0) | (z < Z - 1 ? a[i + 1] : 0);
      }
  return out;
}

/** Downsample the fine solid occupancy; thin sheets are boosted so shells survive. */
export function coarseOccupancy(src: Source, dims: number[]): Uint8Array {
  const s = Math.max(2, Math.ceil(64 / Math.max(dims[0], dims[1], dims[2])));
  const X = dims[0] * s, Y = dims[1] * s, Z = dims[2] * s;
  const fine = src.solid([X, Y, Z], src.e.map((v) => -v), src.e);
  const it = Math.max(1, Math.floor(s / 3));
  let thick = fine;
  for (let i = 0; i < it; i++) thick = erode6(thick, X, Y, Z);
  for (let i = 0; i < it; i++) thick = dilate6(thick, X, Y, Z);
  const n = dims[0] * dims[1] * dims[2];
  const sc = new Int32Array(n);
  for (let x = 0; x < X; x++)
    for (let y = 0; y < Y; y++)
      for (let z = 0; z < Z; z++) {
        const i = (x * Y + y) * Z + z;
        if (!fine[i]) continue;
        const c = (Math.floor(x / s) * dims[1] + Math.floor(y / s)) * dims[2] + Math.floor(z / s);
        sc[c] += 20 + (thick[i] ? 0 : 9 * s);
      }
  const occ = new Uint8Array(n);
  const thr = 6 * s * s * s;
  let any = false, max = 0;
  for (let i = 0; i < n; i++) {
    if (sc[i] >= thr) { occ[i] = 1; any = true; }
    if (sc[i] > max) max = sc[i];
  }
  if (!any) for (let i = 0; i < n; i++) occ[i] = max > 0 && 2 * sc[i] >= max ? 1 : 0;
  return occ;
}

/** Palettes of 1..32 colours (5-bit components) by k-means over per-region mean colours. */
export function makePalettes(src: Source): Uint8Array[] {
  const R = 24;
  const acc = new Map<number, number[]>();
  for (let i = 0; i < src.fitN; i++) {
    let key = 0;
    for (let d = 0; d < 3; d++) {
      const b = Math.floor(((src.pts[i * 3 + d] / src.e[d] + 1) / 2) * R);
      key = key * R + (b < 0 ? 0 : b >= R ? R - 1 : b);
    }
    let a = acc.get(key);
    if (!a) acc.set(key, (a = [0, 0, 0, 0]));
    a[0] += src.col[i * 3]; a[1] += src.col[i * 3 + 1]; a[2] += src.col[i * 3 + 2]; a[3]++;
  }
  const keys = [...acc.keys()].sort((a, b) => a - b);
  const na = keys.length;
  const atoms = new Float64Array(na * 3), wts = new Float64Array(na);
  keys.forEach((k, i) => {
    const a = acc.get(k)!;
    atoms[i * 3] = a[0] / a[3]; atoms[i * 3 + 1] = a[1] / a[3]; atoms[i * 3 + 2] = a[2] / a[3]; wts[i] = a[3];
  });
  const out: Uint8Array[] = [];
  for (let pb = 0; pb < 6; pb++) {
    const k = 1 << pb;
    const cen = new Float64Array(k * 3);
    let ws = 0;
    for (let i = 0; i < na; i++) { ws += wts[i]; for (let d = 0; d < 3; d++) cen[d] += atoms[i * 3 + d] * wts[i]; }
    for (let d = 0; d < 3; d++) cen[d] /= ws;
    const d2 = new Float64Array(na).fill(Infinity);
    const upd = (j: number): void => {
      for (let i = 0; i < na; i++) {
        const dd = (atoms[i * 3] - cen[j * 3]) ** 2 + (atoms[i * 3 + 1] - cen[j * 3 + 1]) ** 2 + (atoms[i * 3 + 2] - cen[j * 3 + 2]) ** 2;
        if (dd < d2[i]) d2[i] = dd;
      }
    };
    upd(0);
    for (let j = 1; j < k; j++) {
      let best = 0, bv = -1;
      for (let i = 0; i < na; i++) if (d2[i] * wts[i] > bv) { bv = d2[i] * wts[i]; best = i; }
      for (let d = 0; d < 3; d++) cen[j * 3 + d] = atoms[best * 3 + d];
      upd(j);
    }
    const lab = new Int32Array(na);
    for (let iter = 0; iter < 12; iter++) {
      for (let i = 0; i < na; i++) {
        let bj = 0, bd = Infinity;
        for (let j = 0; j < k; j++) {
          const dd = (atoms[i * 3] - cen[j * 3]) ** 2 + (atoms[i * 3 + 1] - cen[j * 3 + 1]) ** 2 + (atoms[i * 3 + 2] - cen[j * 3 + 2]) ** 2;
          if (dd < bd) { bd = dd; bj = j; }
        }
        lab[i] = bj;
      }
      const sum = new Float64Array(k * 4);
      for (let i = 0; i < na; i++) {
        const j = lab[i];
        for (let d = 0; d < 3; d++) sum[j * 4 + d] += atoms[i * 3 + d] * wts[i];
        sum[j * 4 + 3] += wts[i];
      }
      for (let j = 0; j < k; j++) if (sum[j * 4 + 3] > 0) for (let d = 0; d < 3; d++) cen[j * 3 + d] = sum[j * 4 + d] / sum[j * 4 + 3];
    }
    const q = new Uint8Array(k * 3);
    for (let i = 0; i < k * 3; i++) q[i] = Math.min(31, Math.max(0, Math.floor(cen[i] * 31 + 0.5)));
    out.push(q);
  }
  return out;
}

export function encodeVox(src: Source, budget: number): string | null {
  const e = src.e;
  let palettes = src.cache.get("palettes") as Uint8Array[] | undefined;
  if (!palettes) src.cache.set("palettes", (palettes = makePalettes(src)));
  const fitN = src.fitN;
  const flab = new Float64Array(fitN * 3);
  for (let i = 0; i < fitN; i++) rgbToLab(src.col[i * 3], src.col[i * 3 + 1], src.col[i * 3 + 2], flab, i * 3);
  const mean = src.meanColor();
  const body = budget - VOX_HEADER_BYTES;
  let best: { j: number; bytes: Uint8Array } | null = null;
  const seen = new Set<string>();
  let fails = 0;
  for (const n of RES_CANDIDATES) {
    const dims = e.map((v) => Math.min(VOX_MAX_DIM, Math.max(1, Math.round(n * v))));
    // Math.round rounds .5 up; numpy rint rounds half to even - n * e is a multiple of 1/64, ties are possible
    for (let d = 0; d < 3; d++) {
      const x = n * e[d];
      if (x - Math.floor(x) === 0.5) dims[d] = Math.min(VOX_MAX_DIM, Math.max(1, 2 * Math.round(x / 2)));
    }
    const key = dims.join(",");
    if (seen.has(key)) continue;
    seen.add(key);
    const [nx, ny, nz] = dims;
    const occ = coarseOccupancy(src, dims);
    const rc0 = new RangeEncoder();
    const probs = Uint16Array.from(OCC_PRIOR);
    const sx = ny * nz, sy = nz;
    let over = false;
    for (let x = 0; x < nx && !over; x++) {
      for (let y = 0; y < ny; y++)
        for (let z = 0; z < nz; z++) {
          const i = x * sx + y * sy + z;
          const a = x ? occ[i - sx] : 0, b = y ? occ[i - sy] : 0, c = z ? occ[i - 1] : 0;
          const d = x && y ? occ[i - sx - sy] : 0, ee = x && z ? occ[i - sx - 1] : 0, g = y && z ? occ[i - sy - 1] : 0;
          rc0.encode(probs, 4 * (a + b + c) + d + ee + g, occ[i]);
        }
      if (rc0.emitted > body) over = true;
    }
    if (over || rc0.finish().length > body) {
      if (++fails >= 2) break;
      continue;
    }
    // geometry term: F-score of the cube decode
    const geo: VoxContent = { dims: dims as [number, number, number], f: 1, colorMode: 0, palette: new Uint8Array(0), occ, cellDims: dims as [number, number, number], active: new Uint8Array(0), cellColors: new Uint8Array(nx * ny * nz * 3) };
    const cubes = voxCubes(geo, e);
    const gs = sampleSurface(cubes.positions, cubes.indices, N_SCORE, 7, (_t, _a, _b, _c, out) => { out[0] = out[1] = out[2] = 0; });
    const gscore = gs ? src.fscore(gs.pts, N_SCORE) : 0;
    // voxel index of every fit sample
    const vi = new Int32Array(fitN * 3);
    for (let i = 0; i < fitN; i++)
      for (let d = 0; d < 3; d++) {
        const v = Math.floor(((src.pts[i * 3 + d] / e[d] + 1) / 2) * dims[d]);
        vi[i * 3 + d] = v < 0 ? 0 : v >= dims[d] ? dims[d] - 1 : v;
      }
    for (const f of F_CANDIDATES) {
      if (f > 1 && f >= 2 * Math.max(nx, ny, nz)) break;
      const cx = Math.ceil(nx / f), cy = Math.ceil(ny / f), cz = Math.ceil(nz / f);
      const nc = cx * cy * cz;
      const active = new Uint8Array(nc);
      for (let x = 0; x < nx; x++)
        for (let y = 0; y < ny; y++)
          for (let z = 0; z < nz; z++) {
            const i = x * sx + y * sy + z;
            if (!occ[i]) continue;
            const inner = x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1 && occ[i - sx] && occ[i + sx] && occ[i - sy] && occ[i + sy] && occ[i - 1] && occ[i + 1];
            if (!inner) active[(Math.floor(x / f) * cy + Math.floor(y / f)) * cz + Math.floor(z / f)] = 1;
          }
      const acc = new Float64Array(nc * 4);
      const cellOf = new Int32Array(fitN);
      for (let i = 0; i < fitN; i++) {
        const c = (Math.floor(vi[i * 3] / f) * cy + Math.floor(vi[i * 3 + 1] / f)) * cz + Math.floor(vi[i * 3 + 2] / f);
        cellOf[i] = c;
        acc[c * 4] += src.col[i * 3]; acc[c * 4 + 1] += src.col[i * 3 + 1]; acc[c * 4 + 2] += src.col[i * 3 + 2]; acc[c * 4 + 3]++;
      }
      const cmean = new Float64Array(nc * 3);
      let ncell = 0;
      for (let c = 0; c < nc; c++) {
        if (active[c]) ncell++;
        for (let d = 0; d < 3; d++) cmean[c * 3 + d] = acc[c * 4 + 3] > 0 ? acc[c * 4 + d] / acc[c * 4 + 3] : mean[d];
      }
      const cell8 = new Uint8Array(nc * 3);
      const cellLab = new Float64Array(nc * 3);
      for (let cm = 0; cm < 7; cm++) {
        let rc: RangeEncoder;
        if (cm === 0) {
          if (rc0.emitted + Math.floor((ncell * 12) / 8) > body) continue;
          rc = rc0.copy();
          for (let c = 0; c < nc; c++) {
            if (!active[c]) continue;
            for (let d = 0; d < 3; d++) {
              const q = Math.min(15, Math.max(0, Math.floor(cmean[c * 3 + d] * 15 + 0.5)));
              cell8[c * 3 + d] = q * 17;
              rc.bypass(q, 4);
            }
          }
        } else {
          const pb = cm - 1, P = 1 << pb;
          const pal = palettes[pb];
          rc = rc0.copy();
          for (let i = 0; i < P * 3; i++) rc.bypass(pal[i], 5);
          const pr = new Uint16Array((P + 1) * P).fill(RC_PROB_INIT);
          const done = new Int16Array(nc).fill(-1);
          let overflow = false;
          for (let x = 0; x < cx && !overflow; x++)
            for (let y = 0; y < cy && !overflow; y++)
              for (let z = 0; z < cz; z++) {
                const c = (x * cy + y) * cz + z;
                if (!active[c]) continue;
                let bi = 0, bd = Infinity;
                for (let j = 0; j < P; j++) {
                  let dd = 0;
                  for (let d = 0; d < 3; d++) dd += (cmean[c * 3 + d] * 255.0 - rgb555to8(pal[j * 3 + d])) ** 2;
                  if (dd < bd) { bd = dd; bi = j; }
                }
                for (let d = 0; d < 3; d++) cell8[c * 3 + d] = rgb555to8(pal[bi * 3 + d]);
                let ctx = P;
                if (z && done[c - 1] >= 0) ctx = done[c - 1];
                else if (y && done[c - cz] >= 0) ctx = done[c - cz];
                else if (x && done[c - cy * cz] >= 0) ctx = done[c - cy * cz];
                let node = 1;
                for (let k = pb - 1; k >= 0; k--) {
                  const bit = (bi >> k) & 1;
                  rc.encode(pr, ctx * P + node, bit);
                  node = (node << 1) | bit;
                }
                done[c] = bi;
                if (rc.emitted > body) { overflow = true; break; }
              }
          if (overflow) continue;
        }
        const data = rc.finish();
        if (data.length > body) continue;
        let de = 0, cnt = 0;
        for (let c = 0; c < nc; c++) if (active[c]) rgbToLab(cell8[c * 3] / 255, cell8[c * 3 + 1] / 255, cell8[c * 3 + 2] / 255, cellLab, c * 3);
        for (let i = 0; i < fitN; i++) {
          const c = cellOf[i] * 3;
          if (!active[cellOf[i]]) continue;
          de += Math.sqrt((flab[i * 3] - cellLab[c]) ** 2 + (flab[i * 3 + 1] - cellLab[c + 1]) ** 2 + (flab[i * 3 + 2] - cellLab[c + 2]) ** 2);
          cnt++;
        }
        de = cnt ? de / cnt : 40;
        const j = 0.5 * (gscore + 1 - Math.min(de, 40) / 40);
        if (!best || j > best.j + 1e-12) {
          const w = header(METHOD_VOX, cm, src.eq);
          for (const d of dims) w.write(d - 1, 6);
          w.write(f - 1, 4);
          const head = w.toBytes();
          const bytes = new Uint8Array(head.length + data.length);
          bytes.set(head);
          bytes.set(data, head.length);
          best = { j, bytes };
        }
      }
    }
  }
  return best ? toBase64Url(best.bytes) : null;
}
