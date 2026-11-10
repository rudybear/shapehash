"""ShapeHash v1 decoder (reference).  parse() returns the exact integer
content of a hash; decode() turns it into a coloured triangle mesh."""
from dataclasses import dataclass

import numpy as np

from . import format as F
from .format import FormatError


@dataclass
class Decoded:
    method: str
    positions: np.ndarray  # (V,3) float32, normalised frame (bbox centred, longest half extent 1)
    indices: np.ndarray  # (T,3) uint32
    colors: np.ndarray  # (V,3) uint8 sRGB
    normals: np.ndarray  # (V,3) float32
    flat: bool = False


def vertex_normals(pos, idx):
    pos = np.asarray(pos, dtype=np.float64)
    t = pos[idx]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    vn = np.zeros_like(pos)
    for k in range(3):
        np.add.at(vn, idx[:, k], fn)
    ln = np.sqrt((vn * vn).sum(1, keepdims=True))
    return np.where(ln > 1e-20, vn / np.maximum(ln, 1e-20), np.array([0.0, 0.0, 1.0]))


# ------------------------------------------------------------------ parsing
def parse(s: str) -> dict:
    data = F.from_string(s)
    r = F.BitReader(data)
    if r.read(3) != F.VERSION:
        raise FormatError("unsupported version")
    method = r.read(2)
    param = r.read(3)
    out = {"method": method, "param": param, "extents_q": [r.read(6) for _ in range(3)]}
    if method == F.METHOD_VOX:
        _parse_vox(data, r, param, out)
    elif method == F.METHOD_SH:
        _parse_sh(r, param, out)
    elif method == F.METHOD_MESH:
        _parse_mesh(r, param, out)
    else:
        raise FormatError("unknown method")
    return out


def surface_voxels(occ):
    p = np.pad(occ, 1)
    inner = p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1] & p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1] & p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:]
    return occ & ~inner


def active_cells(occ, f):
    sv = np.argwhere(surface_voxels(occ))
    cd = -(-np.array(occ.shape) // f)
    act = np.zeros(tuple(cd), bool)
    c = sv // f
    act[c[:, 0], c[:, 1], c[:, 2]] = True
    return act


def _parse_vox(data, r, cm, out):
    if cm == 7:
        raise FormatError("reserved colour mode")
    dims = [r.read(6) + 1 for _ in range(3)]
    f = r.read(4) + 1
    rc = F.RCDecoder(data, F.VOX_HEADER_BYTES)
    pb = cm - 1
    probs = list(F.OCC_PRIOR)
    o = [[[0] * dims[2] for _ in range(dims[1])] for _ in range(dims[0])]
    any_occ = 0
    for x in range(dims[0]):
        for y in range(dims[1]):
            row = o[x][y]
            for z in range(dims[2]):
                b = rc.decode(probs, F.occ_ctx(o, x, y, z))
                row[z] = b
                any_occ |= b
    if not any_occ:
        raise FormatError("empty occupancy grid")
    occ = np.array(o, dtype=bool)
    palette = [[rc.bypass(5), rc.bypass(5), rc.bypass(5)] for _ in range(1 << pb)] if cm else []
    act = active_cells(occ, f)
    cols = np.zeros(act.shape + (3,), dtype=np.uint8)
    cells = np.argwhere(act)
    if cm == 0:
        for c in cells:
            cols[tuple(c)] = [rc.bypass(4) * 17 for _ in range(3)]
        idx = None
    else:
        pal8 = F.rgb555_to_8(np.array(palette)).astype(np.uint8)
        n = 1 << pb
        probs = [F.RC_PROB_INIT] * ((n + 1) * n)
        idx = np.full(act.shape, -1, dtype=np.int64)
        for c in cells:
            x, y, z = (int(v) for v in c)
            if z and idx[x, y, z - 1] >= 0:
                ctx = int(idx[x, y, z - 1])
            elif y and idx[x, y - 1, z] >= 0:
                ctx = int(idx[x, y - 1, z])
            elif x and idx[x - 1, y, z] >= 0:
                ctx = int(idx[x - 1, y, z])
            else:
                ctx = n
            node = 1
            for _ in range(pb):
                node = (node << 1) | rc.decode(probs, ctx * n + node)
            idx[x, y, z] = node - n
            cols[x, y, z] = pal8[node - n]
    out.update(dims=dims, f=f, color_mode=cm, palette=palette, occ=occ, active=act, cell_colors=cols)


def _parse_sh(r, param, out):
    if param != 0:
        raise FormatError("reserved bits set")
    center_q = [r.read(5) for _ in range(3)]
    L, Ly, Lc = r.read(5), r.read(5), r.read(4)
    if L > F.SH_MAX_L or Ly > F.SH_MAX_L:
        raise FormatError("SH degree out of range")
    groups = []
    for l, bits, dcb in ((L, 5, 7), (Ly, 4, 6), (Lc, 3, 5), (Lc, 3, 5)):
        dc = r.read(dcb)
        qs, ac = 0, []
        if l > 0:
            qs = r.read(6)
            ac = [r.read(bits) for _ in range((l + 1) ** 2 - 1)]
            if any(v == (1 << bits) - 1 for v in ac):
                raise FormatError("reserved coefficient code")
        groups.append({"dc": dc, "scale_q": qs, "ac": ac})
    out.update(center_q=center_q, L=L, Ly=Ly, Lc=Lc, groups=groups)


def _nb(n):
    return max(1, (max(n, 2) - 1).bit_length())


def _parse_mesh(r, param, out):
    if param & 4:
        raise FormatError("reserved bits set")
    pb = 5 + (param & 3)
    nv, nt = r.read(10), r.read(11)
    if nv < 3 or nt < 1:
        raise FormatError("empty mesh")
    pos_q = np.zeros((nv, 3), dtype=np.int64)
    col_q = np.zeros((nv, 3), dtype=np.int64)
    for i in range(nv):
        pos_q[i] = [r.read(pb) for _ in range(3)]
        col_q[i] = [r.read(4) for _ in range(3)]
    seen = 0

    def ref():
        nonlocal seen
        if r.read(1):
            if seen >= nv:
                raise FormatError("vertex reference out of range")
            seen += 1
            return seen - 1
        if seen == 0:
            raise FormatError("back reference before any vertex")
        v = r.read(_nb(seen))
        if v >= seen:
            raise FormatError("vertex reference out of range")
        return v

    tris, cur = [], None
    for t in range(nt):
        if t == 0 or r.read(1) == 0:
            cur = (ref(), ref(), ref())
        else:
            bit = r.read(1)
            d = ref()
            a, b, c = cur
            cur = (c, b, d) if bit == 0 else (a, c, d)
        tris.append(cur)
    out.update(pos_bits=pb, pos_q=pos_q, col_q=col_q, tris=np.array(tris, dtype=np.int64))


# ------------------------------------------------------------- mesh building
def _vox_cubes(p, e):
    occ, f, cols = p["occ"], p["f"], p["cell_colors"]
    dims = np.array(occ.shape)
    pad = np.pad(occ, 1)
    size = 2 * e / dims
    corners = {
        0: [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
        1: [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],
        2: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
    }
    V, T, C, N = [], [], [], []
    nv = 0
    for ax in range(3):
        for sg in (0, 1):
            sl = [slice(1, -1)] * 3
            sl[ax] = slice(2, None) if sg else slice(0, -2)
            idx = np.argwhere(occ & ~pad[tuple(sl)])
            if len(idx) == 0:
                continue
            q = np.array(corners[ax], dtype=np.float64)
            q[:, ax] = sg
            if not sg:
                q = q[::-1]
            V.append(((idx[:, None, :] + q[None]) * size - e).reshape(-1, 3))
            base = nv + 4 * np.arange(len(idx))
            T.append(np.stack([base, base + 1, base + 2, base, base + 2, base + 3], 1).reshape(-1, 3))
            c = idx // f
            C.append(np.repeat(cols[c[:, 0], c[:, 1], c[:, 2]], 4, axis=0))
            n = np.zeros(3)
            n[ax] = 1.0 if sg else -1.0
            N.append(np.tile(n, (4 * len(idx), 1)))
            nv += 4 * len(idx)
    return np.concatenate(V), np.concatenate(T), np.concatenate(C), np.concatenate(N)


def blur_field(occ, up=2, pad=2):
    """Integer field: nearest upsample by `up`, zero pad, one pass of the
    separable [1,2,1] kernel per axis (max value 64).  -> g = 2*F - 64,
    positive inside."""
    fld = np.kron(occ.astype(np.int64), np.ones((up, up, up), dtype=np.int64))
    fld = np.pad(fld, pad)
    for ax in range(3):
        z = np.zeros_like(fld)
        a = np.moveaxis(fld, ax, 0)
        b = np.moveaxis(z, ax, 0)
        b[...] = 2 * a
        b[1:] += a[:-1]
        b[:-1] += a[1:]
        fld = z
    return 2 * fld - 64


_CORNERS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def surface_nets(g):
    """Naive surface nets on integer field g (inside where g > 0).
    -> vertices in sample coordinates (N,3), triangles (M,3).  Deterministic
    order: vertices by cell scan order, quads by axis then edge scan order."""
    ins = g > 0
    X, Y, Z = g.shape
    cs = [ins[dx:X - 1 + dx, dy:Y - 1 + dy, dz:Z - 1 + dz] for dx, dy, dz in _CORNERS]
    cnt = sum(c.astype(np.int64) for c in cs)
    mixed = (cnt > 0) & (cnt < 8)
    cells = np.argwhere(mixed)
    vid = np.full(mixed.shape, -1, dtype=np.int64)
    vid[mixed] = np.arange(len(cells))
    acc = np.zeros((len(cells), 3))
    num = np.zeros(len(cells))
    ci, cj, ck = cells[:, 0], cells[:, 1], cells[:, 2]
    for a, b in _EDGES:
        pa, pb = np.array(_CORNERS[a]), np.array(_CORNERS[b])
        ga = g[ci + pa[0], cj + pa[1], ck + pa[2]].astype(np.float64)
        gb = g[ci + pb[0], cj + pb[1], ck + pb[2]].astype(np.float64)
        cross = (ga > 0) != (gb > 0)
        t = np.where(cross, ga / np.where(cross, ga - gb, 1.0), 0.0)
        acc += np.where(cross[:, None], pa[None] + t[:, None] * (pb - pa)[None], 0.0)
        num += cross
    verts = cells + acc / num[:, None]
    tris = []
    for ax in range(3):
        b, c = (ax + 1) % 3, (ax + 2) % 3
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        a0, a1 = ins[tuple(lo)], ins[tuple(hi)]
        ed = np.argwhere(a0 != a1)
        if len(ed) == 0:
            continue
        ed = ed[(ed[:, b] >= 1) & (ed[:, c] >= 1) & (ed[:, b] <= g.shape[b] - 2) & (ed[:, c] <= g.shape[c] - 2)]
        eb = np.zeros(3, dtype=np.int64)
        ec = np.zeros(3, dtype=np.int64)
        eb[b], ec[c] = 1, 1
        q = [ed - eb - ec, ed - ec, ed, ed - eb]
        q = [vid[p[:, 0], p[:, 1], p[:, 2]] for p in q]
        inside_lo = a0[ed[:, 0], ed[:, 1], ed[:, 2]]
        v0, v1, v2, v3 = q
        w1 = np.where(inside_lo, v1, v3)
        w3 = np.where(inside_lo, v3, v1)
        tris.append(np.stack([v0, w1, v2, v0, v2, w3], 1).reshape(-1, 3))
    tris = np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)
    return verts, tris


def _vox_smooth(p, e):
    occ, f, cols, act = p["occ"], p["f"], p["cell_colors"], p["active"]
    dims = np.array(occ.shape)
    up, pad = 2, 2
    verts, tris = surface_nets(blur_field(occ, up, pad))
    vox = (verts - pad + 0.5) / up  # continuous voxel coordinates
    pos = vox * (2 * e / dims) - e
    # colour: trilinear blend of the 8 nearest colour cells, active cells only
    cd = np.array(act.shape)
    c = vox / f - 0.5
    base = np.floor(c).astype(np.int64)
    fr = c - base
    acc = np.zeros((len(vox), 3))
    wsum = np.zeros(len(vox))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ci = base + (dx, dy, dz)
                ok = np.all((ci >= 0) & (ci < cd), axis=1)
                cc = np.clip(ci, 0, cd - 1)
                w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dz else 1 - fr[:, 2])
                w = np.where(ok & act[cc[:, 0], cc[:, 1], cc[:, 2]], w, 0.0)
                acc += w[:, None] * cols[cc[:, 0], cc[:, 1], cc[:, 2]]
                wsum += w
    mean = cols[act].astype(np.float64).mean(axis=0)
    col = np.where(wsum[:, None] > 1e-9, acc / np.maximum(wsum[:, None], 1e-9), mean)
    col = np.clip(np.floor(col + 0.5), 0, 255).astype(np.uint8)
    return pos, tris, col


def _sh_mesh(p, e):
    L, Ly, Lc = p["L"], p["Ly"], p["Lc"]
    c = np.array(p["center_q"]) / 31.0 - 0.5
    dirs, faces = F.icosphere(3 if L <= 8 else 4)
    B = F.real_sh(max(L, Ly, Lc), dirs)
    vals = []
    for (l, bits, dcb, lo, hi), g in zip(((L, 5, 7, 0.0, 2.0), (Ly, 4, 6, 0.0, 1.0), (Lc, 3, 5, -0.5, 0.5), (Lc, 3, 5, -0.5, 0.5)), p["groups"]):
        co = np.zeros((l + 1) ** 2)
        co[0] = (lo + g["dc"] / ((1 << dcb) - 1) * (hi - lo)) * F.SQRT4PI
        if l > 0:
            co[1:] = F.dequant_coeffs(g["ac"], bits, g["scale_q"])
        vals.append(B[:, : len(co)] @ co)
    rad = np.maximum(vals[0], F.SH_MIN_RADIUS)
    pos = (c + dirs * rad[:, None]) * e
    return pos, faces, F.ycc_to_rgb8(vals[1], vals[2], vals[3])


MAX_TRIANGLES = 500_000


def decode(s: str, vox: str = "smooth", max_triangles: int = MAX_TRIANGLES) -> Decoded:
    """vox: 'smooth' (surface nets) or 'cubes' (exposed voxel faces)."""
    d = _decode(s, vox)
    if len(d.indices) > max_triangles:
        raise FormatError("mesh exceeds max_triangles")
    return d


def _decode(s, vox):
    p = parse(s)
    e = F.extents_from_q(p["extents_q"])
    m = p["method"]
    if m == F.METHOD_VOX:
        if vox == "cubes":
            pos, tris, col, nrm = _vox_cubes(p, e)
            return Decoded("vox", pos.astype(np.float32), tris.astype(np.uint32), col.astype(np.uint8), nrm.astype(np.float32), True)
        pos, tris, col = _vox_smooth(p, e)
        if len(tris) == 0:  # degenerate: nothing survives the blur
            pos, tris, col, nrm = _vox_cubes(p, e)
            return Decoded("vox", pos.astype(np.float32), tris.astype(np.uint32), col.astype(np.uint8), nrm.astype(np.float32), True)
    elif m == F.METHOD_SH:
        pos, tris, col = _sh_mesh(p, e)
    else:
        pb = p["pos_bits"]
        pos = (p["pos_q"] * 2.0 / ((1 << pb) - 1) - 1.0) * e
        tris, col = p["tris"], (p["col_q"] * 17).astype(np.uint8)
    # computed from the float32 positions actually returned, so equal positions give equal normals
    nrm = vertex_normals(np.asarray(pos, dtype=np.float32).astype(np.float64), tris)
    return Decoded(F.METHOD_NAMES[m], pos.astype(np.float32), tris.astype(np.uint32), col.astype(np.uint8), nrm.astype(np.float32), False)
