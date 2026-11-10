"""Occupancy grid with context-adaptive binary arithmetic coding + coarse
colour cells.  Two decoders for the same payload layout: blocky cube faces
('vox', trivial decoder) and smoothed marching cubes ('voxs')."""
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from ..bitio import BitReader, quant_extents, read_extents, read_rgb, write_extents, write_rgb
from ..mesh import ColorMesh
from .base import Codec, register

SHIFT = 4
TOP = 1 << 24


class RCEncoder:
    """LZMA-style binary range coder (integer only, trivially portable)."""

    def __init__(self):
        self.low, self.range, self.cache, self.cache_size, self.out = 0, 0xFFFFFFFF, 0, 1, bytearray()

    def _shift_low(self):
        if self.low < 0xFF000000 or self.low >= (1 << 32):
            carry = self.low >> 32
            temp = self.cache
            while True:
                self.out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (self.low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (self.low & 0x00FFFFFF) << 8

    def encode(self, probs, ctx, bit):
        p = probs[ctx]
        bound = (self.range >> 11) * p
        if bit == 0:
            self.range = bound
            probs[ctx] = p + ((2048 - p) >> SHIFT)
        else:
            self.low += bound
            self.range -= bound
            probs[ctx] = p - (p >> SHIFT)
        while self.range < TOP:
            self.range = (self.range << 8) & 0xFFFFFFFF
            self._shift_low()

    def finish(self):
        for _ in range(5):
            self._shift_low()
        return bytes(self.out[1:])  # first byte is always 0


class RCDecoder:
    def __init__(self, data):
        self.data, self.pos, self.range, self.code = data, 0, 0xFFFFFFFF, 0
        for _ in range(4):
            self.code = (self.code << 8) | self._next()

    def _next(self):
        b = self.data[self.pos] if self.pos < len(self.data) else 0  # zero padding past the end
        self.pos += 1
        return b

    def decode(self, probs, ctx):
        p = probs[ctx]
        bound = (self.range >> 11) * p
        if self.code < bound:
            self.range = bound
            probs[ctx] = p + ((2048 - p) >> SHIFT)
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            probs[ctx] = p - (p >> SHIFT)
            bit = 1
        while self.range < TOP:
            self.range = (self.range << 8) & 0xFFFFFFFF
            self.code = ((self.code << 8) | self._next()) & 0xFFFFFFFF
        return bit


def _init_probs():
    # context = 4 * (#occupied face neighbours among -x,-y,-z) + (#occupied among the 3 edge-diagonal causal neighbours)
    p = []
    for n1 in range(4):
        for n2 in range(4):
            pocc = 0.08 + 0.84 * (2 * n1 + n2) / 9.0
            p.append(int(round((1 - pocc) * 2048)))
    return p


def _ctx(o, x, y, z):
    a = o[x - 1][y][z] if x else 0
    b = o[x][y - 1][z] if y else 0
    c = o[x][y][z - 1] if z else 0
    d = o[x - 1][y - 1][z] if x and y else 0
    e = o[x - 1][y][z - 1] if x and z else 0
    f = o[x][y - 1][z - 1] if y and z else 0
    return 4 * (a + b + c) + d + e + f


def rc_decode_occ(data, dims):
    dec, probs = RCDecoder(data), _init_probs()
    o = [[[0] * dims[2] for _ in range(dims[1])] for _ in range(dims[0])]
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                o[x][y][z] = dec.decode(probs, _ctx(o, x, y, z))
    return np.array(o, dtype=bool).reshape(dims)


def rc_encode_occ(occ):
    enc, probs = RCEncoder(), _init_probs()
    o = occ.astype(int).tolist()
    dims = occ.shape
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                enc.encode(probs, _ctx(o, x, y, z), o[x][y][z])
    data = enc.finish()
    # the 5-byte flush is conservative: strip trailing bytes while the decoder
    # (which zero-pads) still reproduces the grid exactly
    while len(data) > 0 and np.array_equal(rc_decode_occ(data[:-1], dims), occ):
        data = data[:-1]
    return data


def coarse_occupancy(asset, dims, e):
    """Downsample the fine solid occupancy to dims; thin sheets are boosted so
    shells do not vanish (they are what you see)."""
    s = max(2, int(np.ceil(64 / dims.max())))  # fine grid is >= 64 along the longest axis
    fine = asset.solid(dims * s, -e, e)
    thick = ndimage.binary_opening(fine, iterations=max(1, s // 3))
    thin = fine & ~thick

    def blk(a):
        return a.reshape(dims[0], s, dims[1], s, dims[2], s).mean(axis=(1, 3, 5))

    score = blk(fine) + 0.45 * s * blk(thin)
    occ = score >= 0.3
    if not occ.any():
        occ = score >= score.max() * 0.5
    return occ


def surface_voxels(occ):
    p = np.pad(occ, 1)
    inner = p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1] & p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1] & p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:]
    return occ & ~inner


def active_cells(occ, f):
    """Colour cells (blocks of f^3 voxels) containing surface voxels, in scan order."""
    sv = np.argwhere(surface_voxels(occ))
    cd = -(-np.array(occ.shape) // f)
    act = np.zeros(tuple(cd), bool)
    c = sv // f
    act[c[:, 0], c[:, 1], c[:, 2]] = True
    return act


RES_CANDIDATES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48]
HEAD = 8 + 18 + 18 + 5 + 10  # tag, extents, dims, colour factor, occupancy byte count


class _VoxBase(Codec):
    """Payload: tag(8) | extents 18 | dims 3x6 | colour-cell factor 5 | nbytes
    10 | per active colour cell RGB444 | (byte aligned) range-coded occupancy."""

    def encode(self, asset, budget_bytes):
        e = quant_extents(asset.extents)
        total = budget_bytes * 8
        best, seen = None, set()
        for n in RES_CANDIDATES:
            dims = np.clip(np.rint(n * e), 1, 64).astype(int)
            if tuple(dims) in seen:
                continue
            seen.add(tuple(dims))
            occ = coarse_occupancy(asset, dims, e)
            data = rc_encode_occ(occ)
            avail = total - HEAD - len(data) * 8
            if avail < 12 or len(data) > 1023:
                if best is not None:
                    break
                continue
            f = 1
            while f < 31:
                act = active_cells(occ, f)
                if act.sum() * 12 + (-(HEAD + int(act.sum()) * 12) % 8) <= avail:
                    break
                f += 1
            else:
                continue
            # trade geometry resolution against colour resolution (heuristic)
            score = dims.max() / f ** (1.0 / 3.0)
            if best is None or score > best[0]:
                best = (score, dims, occ, data, f, act)
        if best is None:
            occ = np.ones((1, 1, 1), bool)
            best = (0, np.array([1, 1, 1]), occ, rc_encode_occ(occ), 1, active_cells(occ, 1))
        _, dims, occ, data, f, act = best
        # mean colour per active cell
        pts, _, col = asset.sample(40000, seed=5)
        vi = np.clip(np.floor((pts / e + 1) / 2 * dims).astype(int), 0, dims - 1)
        ci = vi // f
        acc = np.zeros(act.shape + (3,))
        cnt = np.zeros(act.shape)
        np.add.at(acc, (ci[:, 0], ci[:, 1], ci[:, 2]), col)
        np.add.at(cnt, (ci[:, 0], ci[:, 1], ci[:, 2]), 1)
        mean = col.mean(axis=0)
        w = self.start()
        write_extents(w, asset.extents)
        for d in dims:
            w.write(int(d) - 1, 6)
        w.write(f, 5)
        w.write(len(data), 10)
        for idx in np.argwhere(act):
            t = tuple(idx)
            write_rgb(w, acc[t] / cnt[t] if cnt[t] > 0 else mean)
        w.write_bytes(b"")
        while len(w) % 8:
            w.write(0, 1)
        w.write_bytes(data)
        return self.finish(w, budget_bytes)

    def _read(self, r: BitReader):
        e = read_extents(r)
        dims = np.array([r.read(6) + 1 for _ in range(3)])
        f = r.read(5)
        nbytes = r.read(10)
        # occupancy sits after the colours, but colours need the occupancy: locate it from the end
        all_bits = r.bits
        data = bytes(np.packbits(np.array(all_bits[len(all_bits) - nbytes * 8:], dtype=np.uint8)).tolist()) if nbytes else b""
        occ = rc_decode_occ(data, tuple(dims))
        act = active_cells(occ, f)
        cols = np.zeros(act.shape + (3,))
        for idx in np.argwhere(act):
            cols[tuple(idx)] = read_rgb(r)
        return e, dims, f, occ, act, cols


@register
class VoxCodec(_VoxBase):
    name = "vox"
    tag = 4
    decoder_cost = "low"
    decoder_needs = "binary range decoder (~40 lines) + emit cube faces"

    def decode_payload(self, r, nibble):
        e, dims, f, occ, act, cols = self._read(r)
        p = np.pad(occ, 1)
        V, F, C = [], [], []
        size = 2 * e / dims
        corners = {  # per axis, quad corners in the other two axes
            0: [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
            1: [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],
            2: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        }
        for ax in range(3):
            for sg in (0, 1):
                sl = [slice(1, -1)] * 3
                sl[ax] = slice(2, None) if sg else slice(0, -2)
                idx = np.argwhere(occ & ~p[tuple(sl)])
                if len(idx) == 0:
                    continue
                q = np.array(corners[ax], dtype=np.float64)
                q[:, ax] = sg
                if not sg:
                    q = q[::-1]
                v = (idx[:, None, :] + q[None]) * size - e
                base = sum(len(x) for x in V) + 4 * np.arange(len(idx))
                V.append(v.reshape(-1, 3))
                F.append(np.stack([base, base + 1, base + 2], 1))
                F.append(np.stack([base, base + 2, base + 3], 1))
                c = idx // f
                C.append(np.repeat(cols[c[:, 0], c[:, 1], c[:, 2]], 4, axis=0))
        if not V:
            return ColorMesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
        return ColorMesh(np.concatenate(V), np.concatenate(F), np.concatenate(C), flat=True)


@register
class VoxSmoothCodec(_VoxBase):
    name = "voxs"
    tag = 5
    decoder_cost = "high"
    decoder_needs = "range decoder + upsample/blur + marching cubes"

    def decode_payload(self, r, nibble):
        e, dims, f, occ, act, cols = self._read(r)
        up = 3 if dims.max() <= 12 else (2 if dims.max() <= 24 else 1)  # keeps the MC grid <= ~48^3
        big = np.pad(np.kron(occ.astype(np.float64), np.ones((up, up, up))), up)
        fld = ndimage.gaussian_filter(big, max(0.45 * up, 0.6))
        if fld.max() <= 0.5:
            return ColorMesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
        v, faces, _, _ = marching_cubes(fld, level=0.5)
        vox = (v - up + 0.5) / up  # continuous voxel coordinates (0..dims)
        # colour volume: every voxel takes the colour of the nearest active cell, then trilinear lookup
        ind = ndimage.distance_transform_edt(~act, return_distances=False, return_indices=True)
        cellcol = cols[ind[0], ind[1], ind[2]]
        cc = np.clip(vox / f - 0.5, 0, None)
        col = np.stack([ndimage.map_coordinates(cellcol[..., k], cc.T, order=1, mode="nearest") for k in range(3)], 1)
        return ColorMesh(vox * (2 * e / dims) - e, faces, col)
