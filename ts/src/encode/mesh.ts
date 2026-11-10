/** mesh encoder: remesh (solid -> surface nets) -> clustering -> quadric collapses -> strip code. */
import { blurField, decode, surfaceNets } from "../decode.js";
import { BitWriter, MESH_MAX_TRIS, MESH_MAX_VERTS, METHOD_MESH, toBase64Url } from "../format.js";
import { PointGrid, Source, header } from "./source.js";

interface TriMesh {
  v: Float64Array;
  t: Int32Array;
}

function dedupeFaces(t: number[]): number[] {
  const seen = new Set<string>();
  const out: number[] = [];
  for (let i = 0; i < t.length; i += 3) {
    const a = t[i], b = t[i + 1], c = t[i + 2];
    if (a === b || b === c || a === c) continue;
    const k = [a, b, c].sort((x, y) => x - y).join(",");
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(a, b, c);
  }
  return out;
}

export function clusterVertices(v: Float64Array, t: ArrayLike<number>, res = 32): TriMesh {
  const n = v.length / 3;
  const ids = new Map<number, number>();
  const map = new Int32Array(n);
  const acc: number[] = [];
  for (let i = 0; i < n; i++) {
    let key = 0;
    for (let d = 0; d < 3; d++) {
      const c = Math.floor(((v[i * 3 + d] + 1.05) / 2.1) * res);
      key = key * res + (c < 0 ? 0 : c >= res ? res - 1 : c);
    }
    let id = ids.get(key);
    if (id === undefined) {
      id = acc.length / 4;
      ids.set(key, id);
      acc.push(0, 0, 0, 0);
    }
    map[i] = id;
    acc[id * 4] += v[i * 3]; acc[id * 4 + 1] += v[i * 3 + 1]; acc[id * 4 + 2] += v[i * 3 + 2]; acc[id * 4 + 3]++;
  }
  const nv = acc.length / 4;
  const out = new Float64Array(nv * 3);
  for (let i = 0; i < nv; i++) for (let d = 0; d < 3; d++) out[i * 3 + d] = acc[i * 4 + d] / acc[i * 4 + 3];
  const tt: number[] = [];
  for (let i = 0; i < t.length; i++) tt.push(map[t[i]]);
  return { v: out, t: Int32Array.from(dedupeFaces(tt)) };
}

class Heap {
  private h: number[][] = [];
  get size(): number { return this.h.length; }
  push(e: number[]): void {
    const h = this.h;
    h.push(e);
    let i = h.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (h[p][0] <= h[i][0]) break;
      [h[p], h[i]] = [h[i], h[p]];
      i = p;
    }
  }
  pop(): number[] {
    const h = this.h;
    const top = h[0];
    const last = h.pop()!;
    if (h.length) {
      h[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < h.length && h[l][0] < h[m][0]) m = l;
        if (r < h.length && h[r][0] < h[m][0]) m = r;
        if (m === i) break;
        [h[m], h[i]] = [h[i], h[m]];
        i = m;
      }
    }
    return top;
  }
}

/** Heap-driven quadric-error edge collapses; snapshots for every vertex count <= keep. */
export function qemSequence(mesh: TriMesh, keep = 320, minVerts = 4): Map<number, TriMesh> {
  const nv = mesh.v.length / 3;
  const V = Float64Array.from(mesh.v);
  const faces: number[][] = [];
  for (let i = 0; i < mesh.t.length; i += 3) faces.push([mesh.t[i], mesh.t[i + 1], mesh.t[i + 2]]);
  const Q = new Float64Array(nv * 10); // symmetric 4x4: xx xy xz xw yy yz yw zz zw ww
  let areaSum = 0;
  for (const f of faces) {
    const a = f[0] * 3, b = f[1] * 3, c = f[2] * 3;
    const ux = V[b] - V[a], uy = V[b + 1] - V[a + 1], uz = V[b + 2] - V[a + 2];
    const vx = V[c] - V[a], vy = V[c + 1] - V[a + 1], vz = V[c + 2] - V[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const area = 0.5 * l;
    areaSum += area;
    if (l < 1e-20) continue;
    nx /= l; ny /= l; nz /= l;
    const d = -(nx * V[a] + ny * V[a + 1] + nz * V[a + 2]);
    const p = [nx, ny, nz, d];
    for (const vi of f) {
      let k = vi * 10;
      for (let i = 0; i < 4; i++) for (let j = i; j < 4; j++) Q[k++] += area * p[i] * p[j];
    }
  }
  const lam = (1e-3 * areaSum) / Math.max(nv, 1);
  const vf: Set<number>[] = Array.from({ length: nv }, () => new Set<number>());
  faces.forEach((f, fi) => f.forEach((v) => vf[v].add(fi)));
  const ver = new Int32Array(nv);
  const alive = new Uint8Array(faces.length).fill(1);
  const q = new Float64Array(10);
  const x = [0, 0, 0];
  const cost = (a: number, b: number): number => {
    for (let i = 0; i < 10; i++) q[i] = Q[a * 10 + i] + Q[b * 10 + i];
    const mx = 0.5 * (V[a * 3] + V[b * 3]), my = 0.5 * (V[a * 3 + 1] + V[b * 3 + 1]), mz = 0.5 * (V[a * 3 + 2] + V[b * 3 + 2]);
    const el2 = (V[a * 3] - V[b * 3]) ** 2 + (V[a * 3 + 1] - V[b * 3 + 1]) ** 2 + (V[a * 3 + 2] - V[b * 3 + 2]) ** 2;
    // solve (A + lam I) x = -b + lam mid
    const a00 = q[0] + lam, a01 = q[1], a02 = q[2], a11 = q[4] + lam, a12 = q[5], a22 = q[7] + lam;
    const r0 = -q[3] + lam * mx, r1 = -q[6] + lam * my, r2 = -q[8] + lam * mz;
    const c00 = a11 * a22 - a12 * a12, c01 = a02 * a12 - a01 * a22, c02 = a01 * a12 - a02 * a11;
    const det = a00 * c00 + a01 * c01 + a02 * c02;
    if (Math.abs(det) > 1e-30) {
      const c11 = a00 * a22 - a02 * a02, c12 = a01 * a02 - a00 * a12, c22 = a00 * a11 - a01 * a01;
      x[0] = (c00 * r0 + c01 * r1 + c02 * r2) / det;
      x[1] = (c01 * r0 + c11 * r1 + c12 * r2) / det;
      x[2] = (c02 * r0 + c12 * r1 + c22 * r2) / det;
    } else { x[0] = mx; x[1] = my; x[2] = mz; }
    if ((x[0] - mx) ** 2 + (x[1] - my) ** 2 + (x[2] - mz) ** 2 > 2.25 * el2) { x[0] = mx; x[1] = my; x[2] = mz; }
    const X = x[0], Y = x[1], Z = x[2];
    return q[0] * X * X + 2 * q[1] * X * Y + 2 * q[2] * X * Z + 2 * q[3] * X + q[4] * Y * Y + 2 * q[5] * Y * Z + 2 * q[6] * Y + q[7] * Z * Z + 2 * q[8] * Z + q[9] + lam * el2;
  };
  const heap = new Heap();
  const push = (a: number, b: number): void => {
    if (a > b) [a, b] = [b, a];
    const c = cost(a, b);
    heap.push([c, a, b, ver[a], ver[b], x[0], x[1], x[2]]);
  };
  const edges = new Set<number>();
  for (const f of faces)
    for (let k = 0; k < 3; k++) {
      const a = f[k], b = f[(k + 1) % 3];
      const key = a < b ? a * nv + b : b * nv + a;
      if (!edges.has(key)) { edges.add(key); push(a, b); }
    }
  let live = 0;
  for (const s of vf) if (s.size) live++;
  let nfaces = faces.length;
  const snaps = new Map<number, TriMesh>();
  const snap = (): void => {
    const rm = new Map<number, number>();
    const used: number[] = [];
    const t: number[] = [];
    for (let i = 0; i < faces.length; i++) if (alive[i]) for (const v of faces[i]) if (!rm.has(v)) { rm.set(v, 0); used.push(v); }
    used.sort((a, b) => a - b);
    used.forEach((v, i) => rm.set(v, i));
    for (let i = 0; i < faces.length; i++) if (alive[i]) for (const v of faces[i]) t.push(rm.get(v)!);
    const v = new Float64Array(used.length * 3);
    used.forEach((u, i) => { v[i * 3] = V[u * 3]; v[i * 3 + 1] = V[u * 3 + 1]; v[i * 3 + 2] = V[u * 3 + 2]; });
    snaps.set(used.length, { v, t: Int32Array.from(t) });
  };
  if (live <= keep) snap();
  let deferred: number[][] = [];
  let force = false;
  const removeFace = (fi: number, a: number, b: number): void => {
    alive[fi] = 0;
    nfaces--;
    for (const v of faces[fi]) {
      vf[v].delete(fi);
      if (v !== a && v !== b && vf[v].size === 0) live--;
    }
  };
  const P = [0, 0, 0, 0, 0, 0, 0, 0, 0], PN = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  while (live > minVerts && nfaces > 4) {
    if (heap.size === 0) {
      if (force || deferred.length === 0) break;
      force = true;
      for (const [a, b] of deferred) if (vf[a].size && vf[b].size) push(a, b);
      deferred = [];
      continue;
    }
    const e = heap.pop();
    const a = e[1], b = e[2];
    if (ver[a] !== e[3] || ver[b] !== e[4] || !vf[a].size || !vf[b].size) continue;
    if (!force) {
      let flip = false;
      const check = (fi: number): void => {
        if (flip) return;
        const f = faces[fi];
        const hasA = f.includes(a), hasB = f.includes(b);
        if (hasA && hasB) return;
        for (let k = 0; k < 3; k++) {
          const v = f[k];
          for (let d = 0; d < 3; d++) {
            P[k * 3 + d] = V[v * 3 + d];
            PN[k * 3 + d] = v === a || v === b ? e[5 + d] : V[v * 3 + d];
          }
        }
        const n0 = cross(P), n1 = cross(PN);
        const dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];
        const l0 = n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2], l1 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
        if (dot < 0.2 * Math.sqrt(l0 * l1)) flip = true;
      };
      vf[a].forEach(check);
      vf[b].forEach(check);
      if (flip) { deferred.push([a, b]); continue; }
    }
    V[a * 3] = e[5]; V[a * 3 + 1] = e[6]; V[a * 3 + 2] = e[7];
    for (let i = 0; i < 10; i++) Q[a * 10 + i] += Q[b * 10 + i];
    ver[a]++; ver[b]++;
    for (const fi of [...vf[b]]) {
      const f = faces[fi];
      if (f.includes(a)) removeFace(fi, a, b);
      else {
        faces[fi] = f.map((v) => (v === b ? a : v));
        vf[a].add(fi);
      }
    }
    vf[b].clear();
    live--;
    const seen = new Set<string>();
    for (const fi of [...vf[a]]) {
      const k = [...faces[fi]].sort((p, r) => p - r).join(",");
      if (seen.has(k)) removeFace(fi, a, a);
      else seen.add(k);
    }
    if (vf[a].size === 0) { live--; continue; }
    const nb = new Set<number>();
    for (const fi of vf[a]) for (const v of faces[fi]) if (v !== a) nb.add(v);
    for (const v of nb) push(a, v);
    if (live <= keep && nfaces >= 4) snap();
  }
  return snaps;
}

function cross(p: number[]): number[] {
  const ux = p[3] - p[0], uy = p[4] - p[1], uz = p[5] - p[2];
  const vx = p[6] - p[0], vy = p[7] - p[1], vz = p[8] - p[2];
  return [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
}

type Op = { restart: true; a: number; b: number; c: number } | { restart: false; bit: number; d: number };

/** Greedy generalised triangle strips. */
export function encodeStrips(t: ArrayLike<number>): Op[] {
  const nt = t.length / 3;
  const edge = new Map<string, number[]>();
  const ek = (u: number, v: number): string => (u < v ? u + "," + v : v + "," + u);
  for (let f = 0; f < nt; f++)
    for (let k = 0; k < 3; k++) {
      const key = ek(t[f * 3 + k], t[f * 3 + ((k + 1) % 3)]);
      const l = edge.get(key);
      if (l) l.push(f);
      else edge.set(key, [f]);
    }
  const visited = new Uint8Array(nt);
  const freeNb = (f: number): number => {
    let n = 0;
    for (let k = 0; k < 3; k++) for (const g of edge.get(ek(t[f * 3 + k], t[f * 3 + ((k + 1) % 3)]))!) if (!visited[g] && g !== f) n++;
    return n;
  };
  const ops: Op[] = [];
  for (let start = 0; start < nt; start++) {
    if (visited[start]) continue;
    let a = t[start * 3], b = t[start * 3 + 1], c = t[start * 3 + 2];
    visited[start] = 1;
    ops.push({ restart: true, a, b, c });
    for (;;) {
      let best: [number, number, number] | null = null; // (free neighbours, bit, face) lexicographic minimum
      for (let bit = 0; bit < 2; bit++) {
        const u = bit === 0 ? b : a, v = c;
        for (const g of edge.get(ek(u, v))!) {
          if (visited[g]) continue;
          const cand: [number, number, number] = [freeNb(g), bit, g];
          if (!best || cand[0] < best[0] || (cand[0] === best[0] && (cand[1] < best[1] || (cand[1] === best[1] && cand[2] < best[2])))) best = cand;
        }
      }
      if (!best) break;
      const [, bit, g] = best;
      const u = bit === 0 ? b : a, v = c;
      let d = -1;
      for (let k = 0; k < 3; k++) if (t[g * 3 + k] !== u && t[g * 3 + k] !== v) { d = t[g * 3 + k]; break; }
      visited[g] = 1;
      ops.push({ restart: false, bit, d });
      if (bit === 0) { a = c; c = d; } else { b = c; c = d; }
    }
  }
  return ops;
}

function refBits(seen: number): number {
  let n = 1;
  while (1 << n < seen) n++;
  return n;
}

/** Quantise, strip-encode and serialise; null if it does not fit. */
export function meshPayload(src: Source, m: TriMesh, pb: number, budget: number, colorFn: (pos: Float64Array) => Float64Array): string | null {
  const e = src.e;
  const n = (1 << pb) - 1;
  const nv = m.v.length / 3;
  // quantise and merge identical vertices (ordered lexicographically, like the reference)
  const qv = new Int32Array(nv * 3);
  const keys = new Float64Array(nv);
  for (let i = 0; i < nv; i++) {
    let key = 0;
    for (let d = 0; d < 3; d++) {
      const c = Math.min(1, Math.max(-1, m.v[i * 3 + d] / e[d]));
      const q = Math.min(n, Math.max(0, Math.floor(((c + 1) / 2) * n + 0.5)));
      qv[i * 3 + d] = q;
      key = key * 256 + q;
    }
    keys[i] = key;
  }
  const uniq = [...new Set(keys)].sort((a, b) => a - b);
  const idOf = new Map<number, number>();
  uniq.forEach((k, i) => idOf.set(k, i));
  const tt: number[] = [];
  for (let i = 0; i < m.t.length; i++) tt.push(idOf.get(keys[m.t[i]])!);
  // drop degenerate faces, then duplicates keeping the first of each vertex set, in sorted-key order (np.unique)
  const firstOf = new Map<string, number>();
  for (let i = 0; i < tt.length; i += 3) {
    const a = tt[i], b = tt[i + 1], c = tt[i + 2];
    if (a === b || b === c || a === c) continue;
    const k = [a, b, c].sort((x, y) => x - y).join(",");
    if (!firstOf.has(k)) firstOf.set(k, i);
  }
  const order2 = [...firstOf.values()].sort((a, b) => a - b);
  const T: number[] = [];
  for (const i of order2) T.push(tt[i], tt[i + 1], tt[i + 2]);
  if (T.length === 0 || T.length / 3 > MESH_MAX_TRIS) return null;
  const ops = encodeStrips(T);
  const order: number[] = [];
  const seen = new Map<number, number>();
  const bits = new BitWriter();
  ops.forEach((op, i) => {
    let refs: number[];
    if (op.restart) {
      if (i) bits.write(0, 1);
      refs = [op.a, op.b, op.c];
    } else {
      bits.write(1, 1);
      bits.write(op.bit, 1);
      refs = [op.d];
    }
    for (const v of refs) {
      const s = seen.get(v);
      if (s !== undefined) {
        bits.write(0, 1);
        bits.write(s, refBits(order.length));
      } else {
        bits.write(1, 1);
        seen.set(v, order.length);
        order.push(v);
      }
    }
  });
  if (order.length > MESH_MAX_VERTS || order.length < 3) return null;
  const total = 26 + 21 + order.length * (3 * pb + 12) + bits.length;
  if (total > budget * 8) return null;
  const pos = new Float64Array(order.length * 3);
  const uq = (id: number, d: number): number => Math.floor(uniq[id] / 256 ** (2 - d)) % 256;
  order.forEach((id, i) => { for (let d = 0; d < 3; d++) pos[i * 3 + d] = ((uq(id, d) * 2.0) / n - 1.0) * e[d]; });
  const cols = colorFn(pos);
  const w = header(METHOD_MESH, pb - 5, src.eq);
  w.write(order.length, 10);
  w.write(ops.length, 11);
  order.forEach((id, i) => {
    for (let d = 0; d < 3; d++) w.write(uq(id, d), pb);
    for (let d = 0; d < 3; d++) w.write(Math.min(15, Math.max(0, Math.floor(cols[i * 3 + d] * 15 + 0.5))), 4);
  });
  const sb = bits.toBytes();
  for (let i = 0; i < bits.length; i++) w.write((sb[i >> 3] >> (7 - (i & 7))) & 1, 1);
  return toBase64Url(w.toBytes());
}

export function encodeMesh(src: Source, budget: number): string | null {
  let snaps = src.cache.get("qem") as Map<number, TriMesh> | undefined;
  if (!snaps) {
    const R = 64, lim = 1.03;
    const occ = src.solid([R, R, R], [-lim, -lim, -lim], [lim, lim, lim]);
    const { g, size } = blurField(occ, [R, R, R], 2, 2);
    const nets = surfaceNets(g, size);
    const v = new Float64Array(nets.verts.length);
    for (let i = 0; i < v.length; i++) v[i] = ((nets.verts[i] - 2 + 0.5) / 2 / R) * 2 * lim - lim;
    const cl = clusterVertices(v, nets.tris);
    snaps = cl.t.length >= 12 ? qemSequence(cl) : new Map();
    src.cache.set("qem", snaps);
  }
  if (snaps.size === 0) return null;
  const mean = src.meanColor();
  const colorFn = (pos: Float64Array): Float64Array => {
    const n = pos.length / 3;
    const grid = new PointGrid(pos, n);
    const acc = new Float64Array(n * 4);
    for (let i = 0; i < src.fitN; i++) {
      const j = grid.nearest(src.pts[i * 3], src.pts[i * 3 + 1], src.pts[i * 3 + 2]);
      acc[j * 4] += src.col[i * 3]; acc[j * 4 + 1] += src.col[i * 3 + 1]; acc[j * 4 + 2] += src.col[i * 3 + 2]; acc[j * 4 + 3]++;
    }
    const out = new Float64Array(n * 3);
    for (let j = 0; j < n; j++) for (let d = 0; d < 3; d++) out[j * 3 + d] = acc[j * 4 + 3] > 0 ? acc[j * 4 + d] / acc[j * 4 + 3] : mean[d];
    return out;
  };
  const counts = [...snaps.keys()].sort((a, b) => a - b);
  let best: { j: number; s: string } | null = null;
  for (const pb of [5, 6, 7, 8]) {
    let lo = 0, hi = counts.length - 1;
    let found: string | null = null;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const s = meshPayload(src, snaps.get(counts[mid])!, pb, budget, colorFn);
      if (s !== null) { found = s; lo = mid + 1; } else hi = mid - 1;
    }
    if (found === null) continue;
    const j = src.score(decode(found));
    if (!best || j > best.j) best = { j, s: found };
  }
  return best ? best.s : null;
}
