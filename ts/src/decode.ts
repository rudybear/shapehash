/** ShapeHash v1 decoder. Zero dependencies; works in browsers and Node. */
import {
  BitReader, METHOD_MESH, METHOD_NAMES, METHOD_SH, METHOD_VOX, MESH_MAX_TRIS, MESH_MAX_VERTS, OCC_PRIOR, RC_PROB_INIT, RangeDecoder,
  SH_MAX_L, SH_MIN_RADIUS, SQRT4PI, ShapeHashError, VERSION, VOX_HEADER_BYTES, coeffScale, dequantCoeff, fromBase64Url, icosphere,
  realSH, rgb555to8, shNorm, type Method,
} from "./format.js";

export { ShapeHashError } from "./format.js";
export type { Method } from "./format.js";

export interface DecodedMesh {
  /** Which representation the hash used. */
  method: Method;
  /** xyz per vertex, normalised frame: bounding box centred on the origin, longest half-extent = 1. */
  positions: Float32Array;
  /** Triangle list. */
  indices: Uint32Array;
  /** sRGB, 3 bytes per vertex. */
  colors: Uint8Array;
  /** Unit normals per vertex. */
  normals: Float32Array;
  /** True when the mesh is meant to be flat-shaded (vox cube faces). */
  flat: boolean;
  /** Half extents of the bounding box in the normalised frame (max = 1). */
  extents: [number, number, number];
}

export interface DecodeOptions {
  /** vox payloads: "smooth" = surface nets on a blurred 2x grid (default), "cubes" = exposed voxel faces. */
  vox?: "smooth" | "cubes";
  /** Refuse (ShapeHashError) to build meshes with more triangles than this. Default 500 000. */
  maxTriangles?: number;
}

export interface VoxContent {
  dims: [number, number, number];
  f: number;
  colorMode: number;
  palette: Uint8Array; // P * 3, 5-bit values
  occ: Uint8Array; // (x * ny + y) * nz + z
  cellDims: [number, number, number];
  active: Uint8Array;
  cellColors: Uint8Array; // 8-bit RGB per cell
}
export interface ShContent {
  centerQ: [number, number, number];
  L: number;
  Ly: number;
  Lc: number;
  groups: { dc: number; scaleQ: number; ac: Uint8Array }[];
}
export interface MeshContent {
  posBits: number;
  posQ: Uint16Array;
  colQ: Uint8Array;
  tris: Uint32Array;
}
/** Exact integer content of a hash (what the format actually stores). */
export interface Parsed {
  method: number;
  param: number;
  extentsQ: [number, number, number];
  vox?: VoxContent;
  sh?: ShContent;
  mesh?: MeshContent;
}

export function parse(hash: string): Parsed {
  const data = fromBase64Url(hash);
  const r = new BitReader(data);
  if (r.read(3) !== VERSION) throw new ShapeHashError("unsupported version");
  const method = r.read(2);
  const param = r.read(3);
  const out: Parsed = { method, param, extentsQ: [r.read(6), r.read(6), r.read(6)] };
  if (method === METHOD_VOX) out.vox = parseVox(data, r, param);
  else if (method === METHOD_SH) out.sh = parseSh(r, param);
  else if (method === METHOD_MESH) out.mesh = parseMesh(r, param);
  else throw new ShapeHashError("unknown method");
  return out;
}

function parseVox(data: Uint8Array, r: BitReader, cm: number): VoxContent {
  if (cm === 7) throw new ShapeHashError("reserved colour mode");
  const nx = r.read(6) + 1, ny = r.read(6) + 1, nz = r.read(6) + 1;
  const f = r.read(4) + 1;
  const rc = new RangeDecoder(data, VOX_HEADER_BYTES);
  const occ = new Uint8Array(nx * ny * nz);
  const probs = Uint16Array.from(OCC_PRIOR);
  const sy = nz, sx = ny * nz;
  let any = 0;
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++)
      for (let z = 0; z < nz; z++) {
        const i = x * sx + y * sy + z;
        const a = x ? occ[i - sx] : 0, b = y ? occ[i - sy] : 0, c = z ? occ[i - 1] : 0;
        const d = x && y ? occ[i - sx - sy] : 0, e = x && z ? occ[i - sx - 1] : 0, g = y && z ? occ[i - sy - 1] : 0;
        const bit = rc.decode(probs, 4 * (a + b + c) + d + e + g);
        occ[i] = bit;
        any |= bit;
      }
  if (!any) throw new ShapeHashError("empty occupancy grid");
  const pb = cm - 1;
  const P = cm ? 1 << pb : 0;
  const palette = new Uint8Array(P * 3);
  for (let i = 0; i < P * 3; i++) palette[i] = rc.bypass(5);
  // active colour cells: those containing a surface voxel
  const cx = Math.ceil(nx / f), cy = Math.ceil(ny / f), cz = Math.ceil(nz / f);
  const active = new Uint8Array(cx * cy * cz);
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++)
      for (let z = 0; z < nz; z++) {
        const i = x * sx + y * sy + z;
        if (!occ[i]) continue;
        const inner = x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1 &&
          occ[i - sx] && occ[i + sx] && occ[i - sy] && occ[i + sy] && occ[i - 1] && occ[i + 1];
        if (!inner) active[(Math.floor(x / f) * cy + Math.floor(y / f)) * cz + Math.floor(z / f)] = 1;
      }
  const cellColors = new Uint8Array(cx * cy * cz * 3);
  if (cm === 0) {
    for (let i = 0; i < active.length; i++)
      if (active[i]) for (let k = 0; k < 3; k++) cellColors[i * 3 + k] = rc.bypass(4) * 17;
  } else {
    const n = P;
    const pr = new Uint16Array((n + 1) * n).fill(RC_PROB_INIT);
    const idx = new Int16Array(active.length).fill(-1);
    for (let x = 0; x < cx; x++)
      for (let y = 0; y < cy; y++)
        for (let z = 0; z < cz; z++) {
          const i = (x * cy + y) * cz + z;
          if (!active[i]) continue;
          let ctx = n;
          if (z && idx[i - 1] >= 0) ctx = idx[i - 1];
          else if (y && idx[i - cz] >= 0) ctx = idx[i - cz];
          else if (x && idx[i - cy * cz] >= 0) ctx = idx[i - cy * cz];
          let node = 1;
          for (let k = 0; k < pb; k++) node = (node << 1) | rc.decode(pr, ctx * n + node);
          const v = node - n;
          idx[i] = v;
          for (let k = 0; k < 3; k++) cellColors[i * 3 + k] = rgb555to8(palette[v * 3 + k]);
        }
  }
  return { dims: [nx, ny, nz], f, colorMode: cm, palette, occ, cellDims: [cx, cy, cz], active, cellColors };
}

function parseSh(r: BitReader, param: number): ShContent {
  if (param !== 0) throw new ShapeHashError("reserved bits set");
  const centerQ: [number, number, number] = [r.read(5), r.read(5), r.read(5)];
  const L = r.read(5), Ly = r.read(5), Lc = r.read(4);
  if (L > SH_MAX_L || Ly > SH_MAX_L) throw new ShapeHashError("SH degree out of range");
  const groups: ShContent["groups"] = [];
  const spec: [number, number, number][] = [[L, 5, 7], [Ly, 4, 6], [Lc, 3, 5], [Lc, 3, 5]];
  for (const [l, bits, dcb] of spec) {
    const dc = r.read(dcb);
    let scaleQ = 0;
    const ac = new Uint8Array(l > 0 ? (l + 1) * (l + 1) - 1 : 0);
    if (l > 0) {
      scaleQ = r.read(6);
      for (let i = 0; i < ac.length; i++) {
        ac[i] = r.read(bits);
        if (ac[i] === (1 << bits) - 1) throw new ShapeHashError("reserved coefficient code");
      }
    }
    groups.push({ dc, scaleQ, ac });
  }
  return { centerQ, L, Ly, Lc, groups };
}

function refBits(seen: number): number {
  let n = 1;
  while (1 << n < seen) n++;
  return n;
}

function parseMesh(r: BitReader, param: number): MeshContent {
  if (param & 4) throw new ShapeHashError("reserved bits set");
  const pb = 5 + (param & 3);
  const nv = r.read(10), nt = r.read(11);
  if (nv < 3 || nt < 1 || nv > MESH_MAX_VERTS || nt > MESH_MAX_TRIS) throw new ShapeHashError("empty mesh");
  const posQ = new Uint16Array(nv * 3);
  const colQ = new Uint8Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    for (let k = 0; k < 3; k++) posQ[i * 3 + k] = r.read(pb);
    for (let k = 0; k < 3; k++) colQ[i * 3 + k] = r.read(4);
  }
  let seen = 0;
  const ref = (): number => {
    if (r.read(1)) {
      if (seen >= nv) throw new ShapeHashError("vertex reference out of range");
      return seen++;
    }
    if (seen === 0) throw new ShapeHashError("back reference before any vertex");
    const v = r.read(refBits(seen));
    if (v >= seen) throw new ShapeHashError("vertex reference out of range");
    return v;
  };
  const tris = new Uint32Array(nt * 3);
  let a = 0, b = 0, c = 0;
  for (let t = 0; t < nt; t++) {
    if (t === 0 || r.read(1) === 0) {
      a = ref(); b = ref(); c = ref();
    } else {
      const bit = r.read(1);
      const d = ref();
      if (bit === 0) { const na = c; a = na; c = d; } // (c, b, d)
      else { b = c; c = d; } // (a, c, d)
    }
    tris[t * 3] = a; tris[t * 3 + 1] = b; tris[t * 3 + 2] = c;
  }
  return { posBits: pb, posQ, colQ, tris };
}

// ---------------------------------------------------------------- geometry
export function vertexNormals(pos: ArrayLike<number>, idx: ArrayLike<number>): Float32Array {
  const acc = new Float64Array(pos.length);
  // three passes (one per triangle corner) so the summation order matches the reference bit for bit
  for (let k = 0; k < 3; k++)
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
      const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
      const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
      const o = idx[t + k] * 3;
      acc[o] += uy * vz - uz * vy;
      acc[o + 1] += uz * vx - ux * vz;
      acc[o + 2] += ux * vy - uy * vx;
    }
  const out = new Float32Array(pos.length);
  for (let i = 0; i < acc.length; i += 3) {
    const l = Math.sqrt(acc[i] * acc[i] + acc[i + 1] * acc[i + 1] + acc[i + 2] * acc[i + 2]);
    if (l > 1e-20) { out[i] = acc[i] / l; out[i + 1] = acc[i + 1] / l; out[i + 2] = acc[i + 2] / l; }
    else out[i + 2] = 1;
  }
  return out;
}

const QUAD: number[][][] = [
  [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
  [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
  [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
];

type Raw = { positions: Float64Array | Float32Array; indices: Uint32Array; colors: Uint8Array; normals?: Float32Array };

export function voxCubes(v: VoxContent, e: number[], maxTris = Infinity): Raw {
  const [nx, ny, nz] = v.dims;
  const sx = ny * nz, sy = nz;
  const [, cy, cz] = v.cellDims;
  const f = v.f;
  const size = [(2 * e[0]) / nx, (2 * e[1]) / ny, (2 * e[2]) / nz];
  const pos: number[] = [], idx: number[] = [], col: number[] = [], nrm: number[] = [];
  const step = [sx, sy, 1], dim = [nx, ny, nz];
  const p = [0, 0, 0];
  for (let ax = 0; ax < 3; ax++)
    for (let sg = 0; sg < 2; sg++)
      for (p[0] = 0; p[0] < nx; p[0]++)
        for (p[1] = 0; p[1] < ny; p[1]++)
          for (p[2] = 0; p[2] < nz; p[2]++) {
            const i = p[0] * sx + p[1] * sy + p[2];
            if (!v.occ[i]) continue;
            const edge = sg ? p[ax] === dim[ax] - 1 : p[ax] === 0;
            if (!edge && v.occ[sg ? i + step[ax] : i - step[ax]]) continue;
            const base = pos.length / 3;
            const ci = ((Math.floor(p[0] / f) * cy + Math.floor(p[1] / f)) * cz + Math.floor(p[2] / f)) * 3;
            for (let k = 0; k < 4; k++) {
              const q = QUAD[ax][sg ? k : 3 - k];
              for (let d = 0; d < 3; d++) pos.push((p[d] + (d === ax ? sg : q[d])) * size[d] - e[d]);
              col.push(v.cellColors[ci], v.cellColors[ci + 1], v.cellColors[ci + 2]);
              nrm.push(ax === 0 ? (sg ? 1 : -1) : 0, ax === 1 ? (sg ? 1 : -1) : 0, ax === 2 ? (sg ? 1 : -1) : 0);
            }
            idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
            if (idx.length > maxTris * 3) throw new ShapeHashError("mesh exceeds maxTriangles");
          }
  return { positions: Float64Array.from(pos), indices: Uint32Array.from(idx), colors: Uint8Array.from(col), normals: Float32Array.from(nrm) };
}

/** Integer field g = 2 * blur(upsampled occupancy) - 64; inside where g > 0. Dimensions are dims * up + 2 * pad. */
export function blurField(occ: Uint8Array, dims: number[], up = 2, pad = 2): { g: Int16Array; size: [number, number, number] } {
  const X = dims[0] * up + 2 * pad, Y = dims[1] * up + 2 * pad, Z = dims[2] * up + 2 * pad;
  let a = new Int16Array(X * Y * Z);
  for (let x = 0; x < dims[0]; x++)
    for (let y = 0; y < dims[1]; y++)
      for (let z = 0; z < dims[2]; z++) {
        if (!occ[(x * dims[1] + y) * dims[2] + z]) continue;
        for (let i = 0; i < up; i++)
          for (let j = 0; j < up; j++)
            for (let k = 0; k < up; k++) a[((x * up + pad + i) * Y + y * up + pad + j) * Z + z * up + pad + k] = 1;
      }
  const n = [X, Y, Z], st = [Y * Z, Z, 1];
  for (let ax = 0; ax < 3; ax++) {
    const b = new Int16Array(a.length);
    const s = st[ax];
    for (let x = 0; x < X; x++)
      for (let y = 0; y < Y; y++)
        for (let z = 0; z < Z; z++) {
          const i = (x * Y + y) * Z + z;
          const c = ax === 0 ? x : ax === 1 ? y : z;
          b[i] = 2 * a[i] + (c > 0 ? a[i - s] : 0) + (c < n[ax] - 1 ? a[i + s] : 0);
        }
    a = b;
  }
  for (let i = 0; i < a.length; i++) a[i] = 2 * a[i] - 64;
  return { g: a, size: [X, Y, Z] };
}

const CORNERS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
const EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];

/** Naive surface nets (SPEC.md appendix A). Vertices in sample coordinates. */
export function surfaceNets(g: Int16Array, size: number[], maxTris = Infinity): { verts: Float64Array; tris: Uint32Array } {
  const [X, Y, Z] = size;
  const st = [Y * Z, Z, 1];
  const off = CORNERS.map((c) => c[0] * st[0] + c[1] * st[1] + c[2]);
  const vid = new Int32Array(X * Y * Z).fill(-1); // indexed by the cell's min corner
  const verts: number[] = [];
  for (let x = 0; x < X - 1; x++)
    for (let y = 0; y < Y - 1; y++)
      for (let z = 0; z < Z - 1; z++) {
        const i = (x * Y + y) * Z + z;
        let cnt = 0;
        for (let k = 0; k < 8; k++) if (g[i + off[k]] > 0) cnt++;
        if (cnt === 0 || cnt === 8) continue;
        let ax = 0, ay = 0, az = 0, num = 0;
        for (const [a, b] of EDGES) {
          const ga = g[i + off[a]], gb = g[i + off[b]];
          if (ga > 0 === gb > 0) continue;
          const t = ga / (ga - gb);
          const pa = CORNERS[a], pb = CORNERS[b];
          ax += pa[0] + t * (pb[0] - pa[0]);
          ay += pa[1] + t * (pb[1] - pa[1]);
          az += pa[2] + t * (pb[2] - pa[2]);
          num++;
        }
        vid[i] = verts.length / 3;
        verts.push(x + ax / num, y + ay / num, z + az / num);
        if (verts.length > maxTris * 6) throw new ShapeHashError("mesh exceeds maxTriangles");
      }
  const tris: number[] = [];
  const p = [0, 0, 0];
  for (let ax = 0; ax < 3; ax++) {
    const b = (ax + 1) % 3, c = (ax + 2) % 3;
    for (p[0] = 0; p[0] < X; p[0]++)
      for (p[1] = 0; p[1] < Y; p[1]++)
        for (p[2] = 0; p[2] < Z; p[2]++) {
          if (p[ax] >= size[ax] - 1) continue;
          const i = (p[0] * Y + p[1]) * Z + p[2];
          const lo = g[i] > 0;
          if (lo === g[i + st[ax]] > 0) continue;
          if (p[b] < 1 || p[c] < 1 || p[b] > size[b] - 2 || p[c] > size[c] - 2) continue;
          const v0 = vid[i - st[b] - st[c]], v1 = vid[i - st[c]], v2 = vid[i], v3 = vid[i - st[b]];
          if (lo) tris.push(v0, v1, v2, v0, v2, v3);
          else tris.push(v0, v3, v2, v0, v2, v1);
          if (tris.length > maxTris * 3) throw new ShapeHashError("mesh exceeds maxTriangles");
        }
  }
  return { verts: Float64Array.from(verts), tris: Uint32Array.from(tris) };
}

export function voxSmooth(v: VoxContent, e: number[], maxTris = Infinity): Raw {
  const up = 2, pad = 2;
  const { g, size } = blurField(v.occ, v.dims, up, pad);
  const { verts, tris } = surfaceNets(g, size, maxTris);
  const n = verts.length / 3;
  const pos = new Float64Array(verts.length);
  const col = new Uint8Array(verts.length);
  const [cx, cy, cz] = v.cellDims;
  const cd = [cx, cy, cz];
  const f = v.f;
  // mean colour of the active cells (fallback)
  const mean = [0, 0, 0];
  let na = 0;
  for (let i = 0; i < v.active.length; i++)
    if (v.active[i]) { na++; for (let k = 0; k < 3; k++) mean[k] += v.cellColors[i * 3 + k]; }
  for (let k = 0; k < 3; k++) mean[k] /= Math.max(na, 1);
  const vox = [0, 0, 0], base = [0, 0, 0], fr = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < 3; d++) {
      vox[d] = (verts[i * 3 + d] - pad + 0.5) / up;
      pos[i * 3 + d] = vox[d] * ((2 * e[d]) / v.dims[d]) - e[d];
      const c = vox[d] / f - 0.5;
      base[d] = Math.floor(c);
      fr[d] = c - base[d];
    }
    let r = 0, gg = 0, b = 0, ws = 0;
    for (let dx = 0; dx < 2; dx++)
      for (let dy = 0; dy < 2; dy++)
        for (let dz = 0; dz < 2; dz++) {
          const x = base[0] + dx, y = base[1] + dy, z = base[2] + dz;
          if (x < 0 || y < 0 || z < 0 || x >= cd[0] || y >= cd[1] || z >= cd[2]) continue;
          const ci = (x * cy + y) * cz + z;
          if (!v.active[ci]) continue;
          const w = (dx ? fr[0] : 1 - fr[0]) * (dy ? fr[1] : 1 - fr[1]) * (dz ? fr[2] : 1 - fr[2]);
          r += w * v.cellColors[ci * 3]; gg += w * v.cellColors[ci * 3 + 1]; b += w * v.cellColors[ci * 3 + 2];
          ws += w;
        }
    if (ws > 1e-9) { r /= ws; gg /= ws; b /= ws; } else { r = mean[0]; gg = mean[1]; b = mean[2]; }
    col[i * 3] = Math.min(255, Math.max(0, Math.floor(r + 0.5)));
    col[i * 3 + 1] = Math.min(255, Math.max(0, Math.floor(gg + 0.5)));
    col[i * 3 + 2] = Math.min(255, Math.max(0, Math.floor(b + 0.5)));
  }
  return { positions: pos, indices: tris, colors: col };
}

function shMesh(s: ShContent, e: number[]): Raw {
  const { dirs, faces } = icosphere(s.L <= 8 ? 3 : 4);
  const n = dirs.length / 3;
  const Lmax = Math.max(s.L, s.Ly, s.Lc);
  const basis = new Float64Array((Lmax + 1) * (Lmax + 1));
  const norm = shNorm();
  const range: [number, number, number, number][] = [[5, 7, 0, 2], [4, 6, 0, 1], [3, 5, -0.5, 0.5], [3, 5, -0.5, 0.5]];
  const coef = s.groups.map((g, k) => {
    const [bits, dcb, lo, hi] = range[k];
    const c = new Float64Array(g.ac.length + 1);
    c[0] = (lo + (g.dc / ((1 << dcb) - 1)) * (hi - lo)) * SQRT4PI;
    const sc = coeffScale(g.scaleQ);
    for (let i = 0; i < g.ac.length; i++) c[i + 1] = dequantCoeff(g.ac[i], bits, sc);
    return c;
  });
  const cq = s.centerQ.map((q) => q / 31 - 0.5);
  const pos = new Float64Array(n * 3);
  const col = new Uint8Array(n * 3);
  const val = [0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    const x = dirs[i * 3], y = dirs[i * 3 + 1], z = dirs[i * 3 + 2];
    realSH(Lmax, x, y, z, basis, norm);
    for (let k = 0; k < 4; k++) {
      const c = coef[k];
      let a = 0;
      for (let j = 0; j < c.length; j++) a += basis[j] * c[j];
      val[k] = a;
    }
    const rad = Math.max(val[0], SH_MIN_RADIUS);
    pos[i * 3] = (cq[0] + x * rad) * e[0];
    pos[i * 3 + 1] = (cq[1] + y * rad) * e[1];
    pos[i * 3 + 2] = (cq[2] + z * rad) * e[2];
    const rgb = [val[1] + val[2] - val[3], val[1] + val[3], val[1] - val[2] - val[3]];
    for (let k = 0; k < 3; k++) col[i * 3 + k] = Math.min(255, Math.max(0, Math.floor(rgb[k] * 255 + 0.5)));
  }
  return { positions: pos, indices: faces.slice(), colors: col };
}

/** Build the triangle mesh for an already parsed hash. */
export function build(p: Parsed, options: DecodeOptions = {}): DecodedMesh {
  const e = p.extentsQ.map((q) => (q + 1) / 64) as [number, number, number];
  let raw: Raw;
  let flat = false;
  if (p.method === METHOD_VOX) {
    const cap = options.maxTriangles ?? 500_000;
    raw = options.vox === "cubes" ? voxCubes(p.vox!, e, cap) : voxSmooth(p.vox!, e, cap);
    if (raw.indices.length === 0) raw = voxCubes(p.vox!, e, cap);
    flat = raw.normals !== undefined;
  } else if (p.method === METHOD_SH) {
    raw = shMesh(p.sh!, e);
  } else {
    const m = p.mesh!;
    const n = (1 << m.posBits) - 1;
    const pos = new Float64Array(m.posQ.length);
    const col = new Uint8Array(m.colQ.length);
    for (let i = 0; i < pos.length; i++) pos[i] = ((m.posQ[i] * 2.0) / n - 1.0) * e[i % 3];
    for (let i = 0; i < col.length; i++) col[i] = m.colQ[i] * 17;
    raw = { positions: pos, indices: m.tris, colors: col };
  }
  const positions = Float32Array.from(raw.positions);
  return {
    method: METHOD_NAMES[p.method],
    positions,
    indices: raw.indices,
    colors: raw.colors,
    // computed from the float32 positions actually returned, so equal positions give equal normals
    normals: raw.normals ?? vertexNormals(positions, raw.indices),
    flat,
    extents: e,
  };
}

/** Decode a ShapeHash string into a coloured triangle mesh. Throws ShapeHashError on malformed input. */
export function decode(hash: string, options: DecodeOptions = {}): DecodedMesh {
  return build(parse(hash), options);
}
