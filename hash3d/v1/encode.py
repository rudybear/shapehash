"""ShapeHash v1 encoder (reference; mirrors ts/src/encode).

Everything is derived from a dense coloured surface point sample, so the
encoder needs no ray tracer, no marching cubes and no external simplifier:
    Source(points, normals, colors, extents)  ->  encode(src, budget, method)
"""
import heapq
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import format as F
from .decode import Decoded, _vox_cubes, active_cells, blur_field, decode, surface_nets, surface_voxels

N_FIT = 40000  # samples used for colour fitting
N_SCORE = 8000  # samples used by the selection score (disjoint from the fit set)
F_TAU = 0.05


class Source:
    """Dense coloured surface sample of the asset in the normalised frame."""

    def __init__(self, points, normals, colors, extents):
        self.pts = np.asarray(points, dtype=np.float64)
        self.nrm = np.asarray(normals, dtype=np.float64)
        self.col = np.clip(np.asarray(colors, dtype=np.float64), 0, 1)
        self.eq = F.quant_extents(extents)
        self.e = F.extents_from_q(self.eq)
        self.cache = {}
        n = len(self.pts)
        self.fit = slice(0, min(N_FIT, n))
        self.sc = slice(min(N_FIT, n // 2), min(N_FIT, n // 2) + N_SCORE)
        self.sc_tree = cKDTree(self.pts[self.sc])
        self.sc_lab = rgb_to_lab(self.col[self.sc])

    @classmethod
    def from_asset(cls, asset, n=600000, seed=2024):
        pts, nrm, col = asset.sample(n, seed=seed)
        return cls(pts, nrm, col, asset.extents)

    def solid(self, dims, lo, hi):
        key = (tuple(int(d) for d in np.broadcast_to(dims, (3,))), tuple(np.round(lo, 6)), tuple(np.round(hi, 6)))
        if key not in self.cache:
            self.cache[key] = solid_occupancy(self.pts, dims, lo, hi)
        return self.cache[key]


def solid_occupancy(points, dims, lo, hi):
    """Splat points, dilate the shell by one voxel (26-neighbourhood), flood
    fill the outside (6-connected) from the padded border, grow the outside
    back by one voxel; everything else is solid."""
    dims = np.broadcast_to(np.asarray(dims, dtype=np.int64), (3,))
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    ijk = np.clip(np.floor((points - lo) / (hi - lo) * dims).astype(np.int64), 0, dims - 1)
    surf = np.zeros(tuple(dims), dtype=bool)
    surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    full = np.ones((3, 3, 3), bool)
    thick = ndimage.binary_dilation(np.pad(surf, 2), structure=full)
    lab, _ = ndimage.label(~thick)
    outside = ndimage.binary_dilation(lab == lab[0, 0, 0], structure=full)
    return (~outside[2:-2, 2:-2, 2:-2]) | surf


def rgb_to_lab(rgb):
    """sRGB [0,1] -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    x = (0.4124564 * lin[..., 0] + 0.3575761 * lin[..., 1] + 0.1804375 * lin[..., 2]) / 0.95047
    y = 0.2126729 * lin[..., 0] + 0.7151522 * lin[..., 1] + 0.0721750 * lin[..., 2]
    z = (0.0193339 * lin[..., 0] + 0.1191920 * lin[..., 1] + 0.9503041 * lin[..., 2]) / 1.08883

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def sample_mesh(pos, idx, col, n, seed):
    pos = np.asarray(pos, dtype=np.float64)
    t = pos[idx]
    cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.sqrt((cr * cr).sum(1))
    if len(area) == 0 or area.sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(area), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    b = np.stack([1 - u - v, u, v], 1)
    p = np.einsum("nk,nkc->nc", b, t[fi])
    c = np.einsum("nk,nkc->nc", b, np.asarray(col, dtype=np.float64)[idx[fi]] / 255.0)
    nr = cr[fi] / np.maximum(2 * area[fi, None], 1e-20)
    return p, nr, c


SIL_R = 24


def silhouettes(p):
    """Three axis-aligned 24x24 point-splat silhouettes of a point set."""
    out = []
    for ax in range(3):
        o = [k for k in range(3) if k != ax]
        m = np.zeros((SIL_R, SIL_R), dtype=bool)
        ij = np.clip(np.floor((p[:, o] + 1.05) / 2.1 * SIL_R).astype(np.int64), 0, SIL_R - 1)
        m[ij[:, 0], ij[:, 1]] = True
        out.append(m)
    return out


def score(src, dec, full=False):
    """Selection score in [0,1]: mean of F-score@0.05 and a colour term, plus
    (full=True, used to pick between methods) a coarse silhouette IoU."""
    s = sample_mesh(dec.positions, dec.indices.astype(np.int64), dec.colors, N_SCORE, 7)
    if s is None:
        return 0.0
    p, n, c = s
    d1, i1 = cKDTree(p).query(src.pts[src.sc])
    d2, i2 = src.sc_tree.query(p)
    pr, rc = float((d2 < F_TAU).mean()), float((d1 < F_TAU).mean())
    f = 2 * pr * rc / max(pr + rc, 1e-9)
    lab = rgb_to_lab(c)
    de = 0.5 * (np.linalg.norm(src.sc_lab - lab[i1], axis=1).mean() + np.linalg.norm(lab - src.sc_lab[i2], axis=1).mean())
    ct = 1 - min(de, 40.0) / 40.0
    if not full:
        return 0.5 * (f + ct)
    if "sil" not in src.cache:
        src.cache["sil"] = silhouettes(src.pts[src.sc])
    sil = float(np.mean([(a & b).sum() / max((a | b).sum(), 1) for a, b in zip(src.cache["sil"], silhouettes(p))]))
    return (f + ct + sil) / 3.0


def header(method, param, eq):
    w = F.BitWriter()
    w.write(F.VERSION, 3)
    w.write(method, 2)
    w.write(param, 3)
    for v in eq:
        w.write(int(v), 6)
    return w


# ===================================================================== vox
RES_CANDIDATES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48, 56, 64]
F_CANDIDATES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]


def coarse_occupancy(src, dims):
    s = max(2, int(math.ceil(64 / dims.max())))
    fine = src.solid(dims * s, -src.e, src.e)
    thick = ndimage.binary_opening(fine, iterations=max(1, s // 3))
    thin = fine & ~thick

    def blk(a):
        return a.reshape(dims[0], s, dims[1], s, dims[2], s).sum(axis=(1, 3, 5))

    # score = fraction filled + 0.45 * s * fraction thin  >= 0.3, in integers (x 20 s^3)
    sc = 20 * blk(fine) + 9 * s * blk(thin)
    occ = sc >= 6 * s ** 3
    if not occ.any():
        occ = 2 * sc >= sc.max()
    return occ


def make_palettes(src):
    """Palettes of 1..32 colours (8-bit RGB555-representable) by k-means over
    per-region mean colours."""
    pts, col = src.pts[src.fit], src.col[src.fit]
    b = np.clip(np.floor((pts / src.e + 1) / 2 * 24).astype(np.int64), 0, 23)
    key = (b[:, 0] * 24 + b[:, 1]) * 24 + b[:, 2]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    atoms = np.zeros((len(uk), 3))
    np.add.at(atoms, inv, col)
    atoms /= cnt[:, None]
    wts = cnt.astype(np.float64)
    out = {}
    for pb in range(0, 6):
        k = 1 << pb
        cen = [np.average(atoms, axis=0, weights=wts)]
        d2 = ((atoms - cen[0]) ** 2).sum(1)
        for _ in range(k - 1):
            cen.append(atoms[int(np.argmax(d2 * wts))].copy())
            d2 = np.minimum(d2, ((atoms - cen[-1]) ** 2).sum(1))
        cen = np.array(cen)
        for _ in range(12):
            lab = np.argmin(((atoms[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
            for j in range(k):
                m = lab == j
                if m.any():
                    cen[j] = np.average(atoms[m], axis=0, weights=wts[m])
        q = np.clip(np.floor(cen * 31 + 0.5), 0, 31).astype(np.int64)
        out[pb] = q
    return out


def _encode_cells_palette(rc, act, idx, pb, limit):
    """Range-code palette indices of active cells; returns False on overflow."""
    n = 1 << pb
    probs = [F.RC_PROB_INIT] * ((n + 1) * n)
    done = np.full(act.shape, -1, dtype=np.int64).tolist()
    il = idx.tolist()
    for x, y, z in np.argwhere(act).tolist():
        if z and done[x][y][z - 1] >= 0:
            ctx = done[x][y][z - 1]
        elif y and done[x][y - 1][z] >= 0:
            ctx = done[x][y - 1][z]
        elif x and done[x - 1][y][z] >= 0:
            ctx = done[x - 1][y][z]
        else:
            ctx = n
        v = il[x][y][z]
        node = 1
        for k in range(pb - 1, -1, -1):
            bit = (v >> k) & 1
            rc.encode(probs, ctx * n + node, bit)
            node = (node << 1) | bit
        done[x][y][z] = v
        if len(rc.out) > limit:
            return False
    return True


def encode_vox(src, budget):
    e, eq = src.e, src.eq
    if "palettes" not in src.cache:
        src.cache["palettes"] = make_palettes(src)
    palettes = src.cache["palettes"]
    fpts, fcol = src.pts[src.fit], src.col[src.fit]
    flab = rgb_to_lab(fcol)
    body = budget - F.VOX_HEADER_BYTES
    best, seen, fails = None, set(), 0
    for n in RES_CANDIDATES:
        dims = np.clip(np.rint(n * e), 1, F.VOX_MAX_DIM).astype(int)
        if tuple(dims) in seen:
            continue
        seen.add(tuple(dims))
        occ = coarse_occupancy(src, dims)
        rc0 = F.RCEncoder()
        probs = list(F.OCC_PRIOR)
        o = occ.astype(int).tolist()
        over = False
        for x in range(dims[0]):
            for y in range(dims[1]):
                row = o[x][y]
                for z in range(dims[2]):
                    rc0.encode(probs, F.occ_ctx(o, x, y, z), row[z])
            if len(rc0.out) > body:
                over = True
                break
        if over or len(rc0.finish()) > body:
            fails += 1
            if fails >= 2:
                break
            continue
        # geometry term: F-score of the cube decode
        geo = {"occ": occ, "f": 1, "cell_colors": np.zeros(occ.shape + (3,), np.uint8)}
        pos, tris, col, nrm = _vox_cubes(geo, e)
        s = sample_mesh(pos, tris, col, N_SCORE, 7)
        d1 = cKDTree(s[0]).query(src.pts[src.sc])[0]
        d2 = src.sc_tree.query(s[0])[0]
        pr, rcl = float((d2 < F_TAU).mean()), float((d1 < F_TAU).mean())
        gscore = 2 * pr * rcl / max(pr + rcl, 1e-9)
        vi = np.clip(np.floor((fpts / e + 1) / 2 * dims).astype(np.int64), 0, dims - 1)
        for f in F_CANDIDATES:
            if f > 1 and f >= 2 * dims.max():
                break
            act = active_cells(occ, f)
            ci = vi // f
            acc = np.zeros(act.shape + (3,))
            cnt = np.zeros(act.shape)
            np.add.at(acc, (ci[:, 0], ci[:, 1], ci[:, 2]), fcol)
            np.add.at(cnt, (ci[:, 0], ci[:, 1], ci[:, 2]), 1)
            mean = np.where(cnt[..., None] > 0, acc / np.maximum(cnt[..., None], 1), fcol.mean(axis=0))
            ncell = int(act.sum())
            in_act = act[ci[:, 0], ci[:, 1], ci[:, 2]]
            for cm in range(0, 7):
                if cm == 0:
                    q = np.clip(np.floor(mean * 15 + 0.5), 0, 15).astype(np.int64)
                    cell8 = q * 17
                    if len(rc0.out) + (ncell * 12) // 8 > body:
                        continue
                    rc = rc0.copy()
                    for x, y, z in np.argwhere(act).tolist():
                        for k in range(3):
                            rc.bypass(int(q[x, y, z, k]), 4)
                else:
                    pb = cm - 1
                    pal = palettes[pb]
                    pal8 = F.rgb555_to_8(pal)
                    idx = np.argmin(((mean[..., None, :] * 255.0 - pal8) ** 2).sum(-1), axis=-1)
                    cell8 = pal8[idx]
                    rc = rc0.copy()
                    for c in pal:
                        for k in range(3):
                            rc.bypass(int(c[k]), 5)
                    if not _encode_cells_palette(rc, act, idx, pb, body):
                        continue
                data = rc.finish()
                if len(data) > body:
                    continue
                de = np.linalg.norm(flab[in_act] - rgb_to_lab(cell8[ci[in_act, 0], ci[in_act, 1], ci[in_act, 2]] / 255.0), axis=1).mean() if in_act.any() else 40.0
                j = 0.5 * (gscore + 1 - min(de, 40.0) / 40.0)
                if best is None or j > best[0] + 1e-12:
                    w = header(F.METHOD_VOX, cm, eq)
                    for d in dims:
                        w.write(int(d) - 1, 6)
                    w.write(f - 1, 4)
                    best = (j, w.to_bytes() + data)
    if best is None:
        return None
    return F.to_string(best[1])


# ====================================================================== sh
SH_HEAD = 8 + 18 + 15 + 14
CUBE_R = 32


def _sh_bits(L, Ly, Lc):
    def grp(l, b, dc):
        return dc + (6 + ((l + 1) ** 2 - 1) * b if l > 0 else 0)

    return SH_HEAD + grp(L, 5, 7) + grp(Ly, 4, 6) + 2 * grp(Lc, 3, 5)


def sh_degrees(budget):
    bits = budget * 8
    if _sh_bits(0, 0, 0) > bits:
        return None
    best = (0, 0, 0)
    for L in range(0, F.SH_MAX_L + 1):
        Ly, Lc = min(int(math.floor(0.62 * L + 0.5)), F.SH_MAX_L), min(int(math.floor(0.38 * L + 0.5)), F.SH_MAX_LC)
        if _sh_bits(L, Ly, Lc) <= bits:
            best = (L, Ly, Lc)
    L, Ly, Lc = best
    while Ly < L and _sh_bits(L, Ly + 1, Lc) <= bits:
        Ly += 1
    while Lc < min(Ly, F.SH_MAX_LC) and _sh_bits(L, Ly, Lc + 1) <= bits:
        Lc += 1
    return L, Ly, Lc


def cube_dirs(R=CUBE_R):
    """Bin-centre directions and solid-angle weights of a 6 x R x R cube map."""
    s = (np.arange(R) + 0.5) / R * 2 - 1
    u, v = np.meshgrid(s, s, indexing="ij")
    u, v = u.ravel(), v.ravel()
    one = np.ones_like(u)
    faces = [np.stack(a, 1) for a in ((one, u, v), (-one, u, v), (u, one, v), (u, -one, v), (u, v, one), (u, v, -one))]
    d = np.concatenate(faces)
    w = np.tile(1.0 / (1 + u * u + v * v) ** 1.5, 6)
    return d / np.sqrt((d * d).sum(1, keepdims=True)), w


def cube_bin(q, R=CUBE_R):
    a = np.abs(q)
    ax = np.argmax(a, axis=1)
    m = np.maximum(a[np.arange(len(q)), ax], 1e-12)
    sg = q[np.arange(len(q)), ax] < 0
    o1, o2 = np.array([1, 0, 0])[ax], np.array([2, 2, 1])[ax]
    u = q[np.arange(len(q)), o1] / m
    v = q[np.arange(len(q)), o2] / m
    iu = np.clip(np.floor((u + 1) / 2 * R).astype(np.int64), 0, R - 1)
    iv = np.clip(np.floor((v + 1) / 2 * R).astype(np.int64), 0, R - 1)
    return ((ax * 2 + sg) * R + iu) * R + iv


def _ridge(B, y, wgt, L, lam=1e-4):
    ls = np.concatenate([[l] * (2 * l + 1) for l in range(L + 1)]).astype(np.float64)
    A = B * wgt[:, None]
    reg = lam * wgt.sum() * (ls * (ls + 1)) ** 2 / max(1.0, (L * (L + 1.0)) ** 2)
    return np.linalg.solve(A.T @ B + np.diag(reg + 1e-9 * wgt.sum()), A.T @ y)


def encode_sh(src, budget):
    deg = sh_degrees(budget)
    if deg is None:
        return None
    L, Ly, Lc = deg
    e, eq = src.e, src.eq
    occ = src.solid(48, -e, e)
    idx = np.argwhere(occ)
    c = ((idx.mean(axis=0) + 0.5) / 48 * 2 - 1) if len(idx) else np.zeros(3)
    cq = np.clip(np.floor((c + 0.5) * 31 + 0.5), 0, 31).astype(int)
    c = cq / 31.0 - 0.5
    q = src.pts / e - c
    rad = np.sqrt((q * q).sum(1))
    b = cube_bin(q)
    nb = 6 * CUBE_R * CUBE_R
    rmax = np.zeros(nb)
    np.maximum.at(rmax, b, rad)
    top = rad >= 0.9 * rmax[b]
    acc = np.zeros((nb, 3))
    cnt = np.zeros(nb)
    np.add.at(acc, b[top], src.col[top])
    np.add.at(cnt, b[top], 1)
    hit = cnt > 0
    col = np.where(hit[:, None], acc / np.maximum(cnt[:, None], 1), src.col[src.fit].mean(axis=0))
    y = 0.25 * col[:, 0] + 0.5 * col[:, 1] + 0.25 * col[:, 2]
    co = 0.5 * col[:, 0] - 0.5 * col[:, 2]
    cg = -0.25 * col[:, 0] + 0.5 * col[:, 1] - 0.25 * col[:, 2]
    dirs, w = cube_dirs()
    B = F.real_sh(L, dirs)
    wc = w * np.where(hit, 1.0, 0.02)
    w_ = header(F.METHOD_SH, 0, eq)
    for v in cq:
        w_.write(int(v), 5)
    w_.write(L, 5)
    w_.write(Ly, 5)
    w_.write(Lc, 4)
    for (l, bits, dcb, lo, hi), tgt, wg in (((L, 5, 7, 0.0, 2.0), np.maximum(rmax, F.SH_MIN_RADIUS), w), ((Ly, 4, 6, 0.0, 1.0), y, wc),
                                            ((Lc, 3, 5, -0.5, 0.5), co, wc), ((Lc, 3, 5, -0.5, 0.5), cg, wc)):
        cf = _ridge(B[:, : (l + 1) ** 2], tgt, wg, l)
        n = (1 << dcb) - 1
        w_.write(int(np.clip(math.floor((cf[0] / F.SQRT4PI - lo) / (hi - lo) * n + 0.5), 0, n)), dcb)
        if l > 0:
            qs, codes = F.quant_coeffs(cf[1:], bits)
            w_.write(qs, 6)
            for v in codes:
                w_.write(int(v), bits)
    data = w_.to_bytes()
    return F.to_string(data) if len(data) <= budget else None


# ==================================================================== mesh
def cluster_vertices(V, T, res=32):
    key = np.clip(np.floor((V + 1.05) / 2.1 * res).astype(np.int64), 0, res - 1)
    key = (key[:, 0] * res + key[:, 1]) * res + key[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    acc = np.zeros((len(uk), 3))
    cnt = np.zeros(len(uk))
    np.add.at(acc, inv, V)
    np.add.at(cnt, inv, 1)
    T = inv[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    _, first = np.unique(np.sort(T, axis=1), axis=0, return_index=True)
    return acc / cnt[:, None], T[np.sort(first)]


def qem_sequence(V, T, keep=320, min_verts=4):
    """Heap-driven quadric-error edge collapses.  -> {n_vertices: (V, T)} for
    every vertex count <= keep reached on the way down."""
    V = [np.array(v, dtype=np.float64) for v in V]
    faces = [list(map(int, t)) for t in T]
    nv = len(V)
    Q = np.zeros((nv, 4, 4))
    tv = np.array(V)[np.array(faces)]
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    area = 0.5 * np.sqrt((n * n).sum(1))
    n = n / np.maximum(2 * area[:, None], 1e-20)
    pl = np.c_[n, -np.einsum("ij,ij->i", n, tv[:, 0])]
    Kf = area[:, None, None] * pl[:, :, None] * pl[:, None, :]
    for k in range(3):
        np.add.at(Q, np.array(faces)[:, k], Kf)
    Q = [q for q in Q]
    lam = 1e-3 * float(area.sum()) / max(nv, 1)
    vf = [set() for _ in range(nv)]
    for fi, t in enumerate(faces):
        for a in t:
            vf[a].add(fi)
    ver = [0] * nv
    alive_f = [True] * len(faces)
    eye = np.eye(3)

    def cost(a, b):
        q = Q[a] + Q[b]
        mid = 0.5 * (V[a] + V[b])
        d = V[a] - V[b]
        el2 = float(d @ d)
        try:
            x = np.linalg.solve(q[:3, :3] + lam * eye, -q[:3, 3] + lam * mid)
        except np.linalg.LinAlgError:
            x = mid
        dm = x - mid
        if float(dm @ dm) > 2.25 * el2:
            x = mid
        xh = np.append(x, 1.0)
        return float(xh @ q @ xh) + lam * el2, x

    heap = []

    def push(a, b):
        if a > b:
            a, b = b, a
        c, x = cost(a, b)
        heapq.heappush(heap, (c, a, b, ver[a], ver[b], tuple(x)))

    edges = set()
    for t in faces:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges.add((min(a, b), max(a, b)))
    for a, b in edges:
        push(a, b)
    live = sum(1 for s in vf if s)
    nfaces = len(faces)
    snaps = {}

    def snap():
        fl = [faces[i] for i in range(len(faces)) if alive_f[i]]
        used = sorted({v for t in fl for v in t})
        rm = {v: i for i, v in enumerate(used)}
        snaps[len(used)] = (np.array([V[v] for v in used]), np.array([[rm[v] for v in t] for t in fl], dtype=np.int64))

    if live <= keep:
        snap()
    deferred, force = [], False
    while live > min_verts and nfaces > 4:
        if not heap:
            if force or not deferred:
                break
            force = True
            for a, b in deferred:
                if vf[a] and vf[b]:
                    push(a, b)
            deferred = []
            continue
        c, a, b, sa, sb, x = heapq.heappop(heap)
        if ver[a] != sa or ver[b] != sb or not vf[a] or not vf[b]:
            continue
        x = np.array(x)
        if not force:
            flip = False
            for fi in vf[a] ^ vf[b]:
                t = faces[fi]
                p = [V[v] for v in t]
                pn = [x if v in (a, b) else V[v] for v in t]
                n0 = np.cross(p[1] - p[0], p[2] - p[0])
                n1 = np.cross(pn[1] - pn[0], pn[2] - pn[0])
                if float(n0 @ n1) < 0.2 * math.sqrt(float(n0 @ n0) * float(n1 @ n1)):
                    flip = True
                    break
            if flip:
                deferred.append((a, b))
                continue
        # collapse b into a
        V[a] = x
        Q[a] = Q[a] + Q[b]
        ver[a] += 1
        ver[b] += 1
        for fi in list(vf[b]):
            t = faces[fi]
            if a in t:  # degenerate
                alive_f[fi] = False
                nfaces -= 1
                for v in t:
                    vf[v].discard(fi)
                    if v not in (a, b) and not vf[v]:
                        live -= 1
            else:
                faces[fi] = [a if v == b else v for v in t]
                vf[a].add(fi)
        vf[b] = set()
        live -= 1
        # duplicate faces (same vertex set) around a
        seen = {}
        for fi in list(vf[a]):
            k = tuple(sorted(faces[fi]))
            if k in seen:
                alive_f[fi] = False
                nfaces -= 1
                for v in faces[fi]:
                    vf[v].discard(fi)
                    if v != a and not vf[v]:
                        live -= 1
            else:
                seen[k] = fi
        if not vf[a]:
            live -= 1
            continue
        nb = {v for fi in vf[a] for v in faces[fi]} - {a}
        for v in nb:
            ver[v] += 0
            push(a, v)
        if live <= keep and nfaces >= 4:
            snap()
    return snaps


def encode_strips(T):
    T = [tuple(int(x) for x in t) for t in T]
    edge = {}
    for fi, (a, b, c) in enumerate(T):
        for u, v in ((a, b), (b, c), (c, a)):
            edge.setdefault((min(u, v), max(u, v)), []).append(fi)
    visited = [False] * len(T)

    def free_nb(fi):
        a, b, c = T[fi]
        return sum(1 for u, v in ((a, b), (b, c), (c, a)) for g in edge[(min(u, v), max(u, v))] if not visited[g] and g != fi)

    ops = []
    for start in range(len(T)):
        if visited[start]:
            continue
        cur = T[start]
        visited[start] = True
        ops.append(("R",) + cur)
        while True:
            a, b, c = cur
            cands = []
            for bit, (u, v) in enumerate(((b, c), (a, c))):
                for g in edge[(min(u, v), max(u, v))]:
                    if not visited[g]:
                        cands.append((free_nb(g), bit, g))
            if not cands:
                break
            _, bit, g = min(cands)
            u, v = (b, c) if bit == 0 else (a, c)
            d = [x for x in T[g] if x != u and x != v][0]
            visited[g] = True
            ops.append(("C", bit, d))
            cur = (c, b, d) if bit == 0 else (a, c, d)
    return ops


def _nb(n):
    return max(1, (max(n, 2) - 1).bit_length())


def mesh_payload(src, V, T, pb, budget, color_fn):
    e = src.e
    n = (1 << pb) - 1
    q = np.clip(np.floor((np.clip(V / e, -1, 1) + 1) / 2 * n + 0.5), 0, n).astype(np.int64)
    uq, inv = np.unique(q, axis=0, return_inverse=True)
    T = inv.reshape(-1)[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    if len(T) == 0:
        return None
    _, first = np.unique(np.sort(T, axis=1), axis=0, return_index=True)
    T = T[np.sort(first)]
    if len(T) > F.MESH_MAX_TRIS:
        return None
    ops = encode_strips(T)
    order, seen, bits = [], {}, []
    for i, op in enumerate(ops):
        if op[0] == "R":
            if i:
                bits.append((0, 1))
            refs = op[1:]
        else:
            bits += [(1, 1), (op[1], 1)]
            refs = (op[2],)
        for v in refs:
            if v in seen:
                bits += [(0, 1), (seen[v], _nb(len(order)))]
            else:
                bits.append((1, 1))
                seen[v] = len(order)
                order.append(v)
    if len(order) > F.MESH_MAX_VERTS or len(order) < 3:
        return None
    total = 26 + 21 + len(order) * (3 * pb + 12) + sum(b for _, b in bits)
    if total > budget * 8:
        return None
    pos = (uq[order] * 2.0 / n - 1.0) * e
    cols = color_fn(pos)
    w = header(F.METHOD_MESH, pb - 5, src.eq)
    w.write(len(order), 10)
    w.write(len(ops), 11)
    for i, v in enumerate(order):
        for x in uq[v]:
            w.write(int(x), pb)
        for x in np.clip(np.floor(cols[i] * 15 + 0.5), 0, 15).astype(int):
            w.write(int(x), 4)
    for v, nb in bits:
        w.write(v, nb)
    return F.to_string(w.to_bytes())


def encode_mesh(src, budget):
    if "qem" not in src.cache:
        occ = src.solid(64, [-1.03] * 3, [1.03] * 3)
        v, t = surface_nets(blur_field(occ, 2, 2))
        v = ((v - 2 + 0.5) / 2) / 64 * 2.06 - 1.03
        v, t = cluster_vertices(v, t)
        src.cache["qem"] = qem_sequence(v, t) if len(t) >= 4 else {}
    snaps = src.cache["qem"]
    if not snaps:
        return None
    fpts, fcol = src.pts[src.fit], src.col[src.fit]
    mean = fcol.mean(axis=0)

    def color_fn(pos):
        _, idx = cKDTree(pos).query(fpts)
        acc = np.zeros((len(pos), 3))
        cnt = np.zeros(len(pos))
        np.add.at(acc, idx, fcol)
        np.add.at(cnt, idx, 1)
        return np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), mean)

    counts = sorted(snaps)
    best = None
    for pb in (5, 6, 7, 8):
        lo, hi, found = 0, len(counts) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            V, T = snaps[counts[mid]]
            s = mesh_payload(src, V, T, pb, budget, color_fn)
            if s is not None:
                found, lo = s, mid + 1
            else:
                hi = mid - 1
        if found is None:
            continue
        j = score(src, decode(found))
        if best is None or j > best[0]:
            best = (j, found)
    return best[1] if best else None


# ==================================================================== union
ENCODERS = {"vox": encode_vox, "sh": encode_sh, "mesh": encode_mesh}


def encode(src, budget, method="auto"):
    """-> (hash string, info dict).  method: auto | vox | sh | mesh."""
    if budget < 8 or budget > F.MAX_PAYLOAD_BYTES:
        raise ValueError("budget must be between 8 and 2048 bytes")
    names = list(ENCODERS) if method == "auto" else [method]
    cands = {}
    for name in names:
        s = ENCODERS[name](src, budget)
        if s is not None:
            cands[name] = s
    if not cands:
        raise ValueError(f"nothing fits in {budget} bytes")
    scores = {k: score(src, decode(v), full=True) for k, v in cands.items()} if len(cands) > 1 else {k: 0.0 for k in cands}
    pick = max(scores, key=scores.get)
    return cands[pick], {"method": pick, "scores": scores, "candidates": cands}
