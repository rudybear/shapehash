"""Aggressively decimated mesh: remesh (solid voxels -> marching cubes) ->
quadric-error decimation -> 6-bit positions, RGB444 vertex colours and a
generalised-triangle-strip connectivity code.  Decoder is a table read."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from ..bitio import BitWriter, dequant, quant, quant_extents, read_extents, read_rgb, write_extents, write_rgb
from ..mesh import ColorMesh
from .base import Codec, register

PB = 6  # bits per position component


def remesh(asset, res=64):
    """Clean closed surface of the solid occupancy (handles soups/multi-part)."""
    occ = asset.solid(res, [-1.03] * 3, [1.03] * 3)
    f = ndimage.gaussian_filter(np.pad(occ, 2).astype(np.float64), 0.6)
    v, faces, _, _ = marching_cubes(f, level=0.3)
    v = (v - 2 + 0.5) / res * 2.06 - 1.03
    return v, faces


def qem_sequence(V, F, min_verts=4):
    """Greedy quadric-error edge collapses all the way down; returns
    {vertex_count: (V, F)} snapshots.  No topology guards: at these sizes we
    only care that the blob keeps roughly the right shape."""
    V = np.asarray(V, dtype=np.float64).copy()
    F = np.asarray(F, dtype=np.int64).copy()
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2 * area[:, None], 1e-20)
    p = np.c_[n, -np.einsum("ij,ij->i", n, t[:, 0])]
    Kf = area[:, None, None] * p[:, :, None] * p[:, None, :]
    Q = np.zeros((len(V), 4, 4))
    for k in range(3):
        np.add.at(Q, F[:, k], Kf)
    lam = 1e-3 * area.sum() / max(len(V), 1)
    snaps = {}

    def snap():
        used, inv = np.unique(F, return_inverse=True)
        snaps[len(used)] = (V[used].copy(), inv.reshape(-1, 3).copy())
        return len(used)

    nv = snap()
    while nv > min_verts and len(F) > 4:
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        e = np.unique(e, axis=0)
        Qe = Q[e[:, 0]] + Q[e[:, 1]]
        mid = 0.5 * (V[e[:, 0]] + V[e[:, 1]])
        A = Qe[:, :3, :3] + lam * np.eye(3)
        b = -Qe[:, :3, 3] + lam * mid
        x = np.linalg.solve(A, b[..., None])[..., 0]
        # keep the new vertex near the edge (guards against wild extrapolation)
        el = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
        far = np.linalg.norm(x - mid, axis=1) > 1.5 * el
        x[far] = mid[far]
        xh = np.c_[x, np.ones(len(x))]
        cost = np.einsum("ni,nij,nj->n", xh, Qe, xh) + lam * el**2
        i = int(np.argmin(cost))
        a, bb = e[i]
        V[a] = x[i]
        Q[a] = Qe[i]
        F[F == bb] = a
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F = F[ok]
        _, first = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
        F = F[np.sort(first)]
        if len(F) < 4:
            break
        nv = snap()
    return snaps


def encode_strips(F):
    """Greedy generalised triangle strips.  -> list of ops:
    ('R', a, b, c) restart, ('C', edge_bit, d) continue across edge (b,c) [0] or (a,c) [1]."""
    F = [tuple(int(x) for x in f) for f in F]
    edge = {}
    for fi, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            edge.setdefault((min(u, v), max(u, v)), []).append(fi)
    visited = [False] * len(F)

    def free_nb(fi):
        a, b, c = F[fi]
        return sum(1 for u, v in ((a, b), (b, c), (c, a)) for g in edge[(min(u, v), max(u, v))] if not visited[g] and g != fi)

    def third(fi, u, v):
        return [x for x in F[fi] if x != u and x != v][0]

    ops = []
    for start in range(len(F)):
        if visited[start]:
            continue
        cur = F[start]
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
            d = third(g, *((b, c) if bit == 0 else (a, c)))
            visited[g] = True
            ops.append(("C", bit, d))
            cur = (c, b, d) if bit == 0 else (a, c, d)
    return ops


def _nb(n):
    return max(1, int(np.ceil(np.log2(max(n, 2)))))


def build_payload(codec, e, V, F, colors_fn, budget_bits):
    """Quantise, strip-encode and serialise; returns BitWriter or None if it does not fit."""
    q = quant(V / e, -1.0, 1.0, PB)
    uq, inv = np.unique(q, axis=0, return_inverse=True)
    F = inv.reshape(-1)[F]
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[ok]
    if len(F) == 0:
        return None
    _, first = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
    F = F[np.sort(first)]
    if len(uq) > 255 or len(F) > 1023:
        return None
    ops = encode_strips(F)
    # vertices are renumbered in order of first appearance
    order, seen = [], {}
    w = BitWriter()
    for op in ops:
        refs = op[1:] if op[0] == "R" else (op[2],)
        hdr = [(0, 1)] if op[0] == "R" else [(1, 1), (op[1], 1)]
        for v, nb in hdr:
            w.write(v, nb)
        for v in refs:
            if v in seen:
                w.write(0, 1)
                w.write(seen[v], _nb(len(order)))
            else:
                w.write(1, 1)
                seen[v] = len(order)
                order.append(v)
    conn_bits = len(w) - 1  # the first op is always a restart: its flag is dropped
    total = 8 + 18 + 8 + 10 + len(order) * (3 * PB + 12) + conn_bits
    if total > budget_bits:
        return None
    pos = dequant(uq[order], -1.0, 1.0, PB) * e
    cols = colors_fn(pos)
    out = codec.start()
    write_extents(out, e)
    out.write(len(order), 8)
    out.write(len(ops), 10)
    for i, v in enumerate(order):
        for x in uq[v]:
            out.write(int(x), PB)
        write_rgb(out, cols[i])
    for b in w.bits[1:]:
        out.write(b, 1)
    return out


@register
class MeshCodec(Codec):
    """Payload: tag(8) | extents 18 | N verts 8 | M tris 10 | N x [pos 3x6,
    RGB444] in order of first use | strip ops: per triangle [continue? 1]
    [edge 1] then per vertex ref [new? 1][index ceil(log2 seen)]."""

    name = "mesh"
    tag = 8
    decoder_cost = "trivial"
    decoder_needs = "bit reader only (table -> index buffer)"

    def encode(self, asset, budget_bytes):
        import fast_simplification

        if "qem" not in asset._cache:
            v, f = remesh(asset)
            if len(f) > 700:
                v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_count=600, agg=7)
                if len(f2) >= 8:
                    v, f = v2.astype(np.float64), f2.astype(np.int64)
            if len(f) > 3000:  # simplifier stalled badly: fall back to the convex hull
                import trimesh

                h = trimesh.Trimesh(v, f, process=False).convex_hull
                v, f = fast_simplification.simplify(h.vertices.astype(np.float32), h.faces.astype(np.int32), target_count=600, agg=7)
            asset._cache["qem"] = qem_sequence(v, f)
        snaps = asset._cache["qem"]
        e = quant_extents(asset.extents)
        spts, _, scol = asset.sample(40000, seed=11)
        mean = scol.mean(0)

        def colors_fn(pos):
            _, idx = cKDTree(pos).query(spts)
            acc = np.zeros((len(pos), 3))
            cnt = np.zeros(len(pos))
            np.add.at(acc, idx, scol)
            np.add.at(cnt, idx, 1)
            return np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), mean)

        counts = sorted(k for k in snaps if k <= 255)
        lo, hi, best = 0, len(counts) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            V, F = snaps[counts[mid]]
            out = build_payload(self, e, np.clip(V, -e, e), F, colors_fn, budget_bytes * 8)
            if out is not None:
                best, lo = out, mid + 1
            else:
                hi = mid - 1
        if best is None:  # nothing fits: emit an empty mesh (honest failure)
            best = self.start()
            write_extents(best, e)
            best.write(0, 8)
            best.write(0, 10)
        return self.finish(best, budget_bytes)

    def decode_payload(self, r, nibble):
        e = read_extents(r)
        n, m = r.read(8), r.read(10)
        pos = np.zeros((n, 3))
        col = np.zeros((n, 3))
        for i in range(n):
            pos[i] = dequant(np.array([r.read(PB) for _ in range(3)]), -1.0, 1.0, PB) * e
            col[i] = read_rgb(r)
        seen = 0

        def ref():
            nonlocal seen
            if r.read(1):
                seen += 1
                return seen - 1
            return r.read(_nb(seen))

        F, cur = [], None
        for t in range(m):
            if t == 0 or r.read(1) == 0:
                cur = (ref(), ref(), ref())
            else:
                bit = r.read(1)
                d = ref()
                a, b, c = cur
                cur = (c, b, d) if bit == 0 else (a, c, d)
            F.append(cur)
        F = np.array(F, dtype=np.int64).reshape(-1, 3)
        F = F[(F < n).all(axis=1)] if n else F[:0]
        return ColorMesh(pos, F, col)
