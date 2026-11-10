"""ShapeHash v1 constants, bit I/O, integer range coder, SH basis."""
import math

import numpy as np

VERSION = 1
METHOD_VOX, METHOD_SH, METHOD_MESH = 0, 1, 2
METHOD_NAMES = {0: "vox", 1: "sh", 2: "mesh"}
MAX_PAYLOAD_BYTES = 2048
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
_A2I = {c: i for i, c in enumerate(ALPHABET)}

# --- vox
VOX_MAX_DIM = 64
VOX_HEADER_BYTES = 6
RC_PROB_BITS = 11
RC_PROB_INIT = 1024
RC_SHIFT = 4
RC_TOP = 1 << 24
# P(bit == 0) * 2048 for occupancy context c = 4 * n1 + n2 (n1: occupied among the -x,-y,-z
# face neighbours; n2: occupied among the 3 causal edge-diagonal neighbours)
OCC_PRIOR = [1884, 1693, 1502, 1311, 1502, 1311, 1120, 928, 1120, 928, 737, 546, 737, 546, 355, 164]

# --- sh
SH_MAX_L = 24
SH_MAX_LC = 15
SH_BITS = (5, 4, 3)  # radius, luma, chroma AC bits
SCALE_MANT = (1.0, 1.189207115002721, 1.4142135623730951, 1.681792830507429)  # 2^(k/4)
SQRT4PI = 3.5449077018110318
SH_MIN_RADIUS = 0.02

# --- mesh
MESH_MAX_VERTS = 1023
MESH_MAX_TRIS = 2047


class FormatError(ValueError):
    """Raised for any malformed / unsupported hash string."""


def to_string(data: bytes) -> str:
    out = []
    for i in range(0, len(data), 3):
        chunk = data[i:i + 3]
        v = int.from_bytes(chunk + b"\0" * (3 - len(chunk)), "big")
        chars = [ALPHABET[(v >> s) & 63] for s in (18, 12, 6, 0)]
        out.append("".join(chars[: len(chunk) + 1]))
    return "".join(out)


def from_string(s: str) -> bytes:
    if not isinstance(s, str):
        raise FormatError("hash must be a string")
    if len(s) % 4 == 1 or len(s) > (MAX_PAYLOAD_BYTES * 4 + 2) // 3:
        raise FormatError("bad string length")
    out = bytearray()
    for i in range(0, len(s), 4):
        chunk = s[i:i + 4]
        v = 0
        for c in chunk:
            if c not in _A2I:
                raise FormatError("character outside the base64url alphabet")
            v = (v << 6) | _A2I[c]
        v <<= 6 * (4 - len(chunk))
        out += v.to_bytes(3, "big")[: len(chunk) - 1]
    return bytes(out)


class BitWriter:
    def __init__(self):
        self.bits = []

    def write(self, value, nbits):
        value = int(value)
        if value < 0 or value >= (1 << nbits):
            raise ValueError(f"value {value} does not fit in {nbits} bits")
        for i in range(nbits - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def __len__(self):
        return len(self.bits)

    def to_bytes(self):
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))


class BitReader:
    """Strict MSB-first reader: reading past the end is a FormatError."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.nbits = len(data) * 8

    def read(self, nbits):
        if self.pos + nbits > self.nbits:
            raise FormatError("truncated payload")
        v = 0
        for _ in range(nbits):
            v = (v << 1) | ((self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v


# ------------------------------------------------------------- range coder
class RCEncoder:
    """LZMA-style binary range coder; all state is integer."""

    def __init__(self):
        self.low, self.range, self.cache, self.cache_size, self.out = 0, 0xFFFFFFFF, 0, 1, bytearray()

    def copy(self):
        c = RCEncoder.__new__(RCEncoder)
        c.low, c.range, c.cache, c.cache_size, c.out = self.low, self.range, self.cache, self.cache_size, bytearray(self.out)
        return c

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

    def _norm(self):
        while self.range < RC_TOP:
            self.range = (self.range << 8) & 0xFFFFFFFF
            self._shift_low()

    def encode(self, probs, i, bit):
        p = probs[i]
        bound = (self.range >> RC_PROB_BITS) * p
        if bit == 0:
            self.range = bound
            probs[i] = p + ((2048 - p) >> RC_SHIFT)
        else:
            self.low += bound
            self.range -= bound
            probs[i] = p - (p >> RC_SHIFT)
        self._norm()

    def bypass(self, value, nbits):
        for k in range(nbits - 1, -1, -1):
            bound = self.range >> 1
            if (value >> k) & 1:
                self.low += bound
                self.range -= bound
            else:
                self.range = bound
            self._norm()

    def size_upper_bound(self):
        """Bytes the finished stream can take at most (cheap, for searches)."""
        return len(self.out) + self.cache_size + 3

    def finish(self) -> bytes:
        """Pick the value in [low, low+range) with the most trailing zero bits,
        flush it, and drop trailing zero bytes (the decoder zero-pads)."""
        c = self.copy()
        hi = c.low + c.range - 1
        for s in (32, 24, 16, 8, 0):
            v = ((c.low + (1 << s) - 1) >> s) << s
            if v <= hi:
                c.low = v
                break
        for _ in range(5):
            c._shift_low()
        data = bytes(c.out[1:])  # first byte is always 0
        return data.rstrip(b"\0")


class RCDecoder:
    def __init__(self, data: bytes, pos: int = 0):
        self.data, self.pos, self.range, self.code = data, pos, 0xFFFFFFFF, 0
        for _ in range(4):
            self.code = (self.code << 8) | self._next()

    def _next(self):
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def _norm(self):
        while self.range < RC_TOP:
            self.range = (self.range << 8) & 0xFFFFFFFF
            self.code = ((self.code << 8) | self._next()) & 0xFFFFFFFF

    def decode(self, probs, i):
        p = probs[i]
        bound = (self.range >> RC_PROB_BITS) * p
        if self.code < bound:
            self.range = bound
            probs[i] = p + ((2048 - p) >> RC_SHIFT)
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            probs[i] = p - (p >> RC_SHIFT)
            bit = 1
        self._norm()
        return bit

    def bypass(self, nbits):
        v = 0
        for _ in range(nbits):
            bound = self.range >> 1
            if self.code < bound:
                self.range = bound
                v <<= 1
            else:
                self.code -= bound
                self.range -= bound
                v = (v << 1) | 1
            self._norm()
        return v


def occ_ctx(o, x, y, z):
    a = o[x - 1][y][z] if x else 0
    b = o[x][y - 1][z] if y else 0
    c = o[x][y][z - 1] if z else 0
    d = o[x - 1][y - 1][z] if x and y else 0
    e = o[x - 1][y][z - 1] if x and z else 0
    f = o[x][y - 1][z - 1] if y and z else 0
    return 4 * (a + b + c) + d + e + f


# ----------------------------------------------------------------- helpers
def extents_from_q(q):
    return np.array([(v + 1) / 64.0 for v in q])


def quant_extents(e):
    return np.clip(np.ceil(np.asarray(e, dtype=np.float64) * 64) - 1, 0, 63).astype(int)


def rgb444_to_8(c):
    return np.asarray(c, dtype=np.int64) * 17


def rgb555_to_8(c):
    c = np.asarray(c, dtype=np.int64)
    return (c << 3) | (c >> 2)


def coeff_scale(qs):
    return SCALE_MANT[qs & 3] * 2.0 ** ((qs >> 2) - 13)


def quant_scale(m):
    """Smallest qs with coeff_scale(qs) >= m (63 if none)."""
    for qs in range(64):
        if coeff_scale(qs) >= m:
            return qs
    return 63


def dequant_coeffs(q, nbits, qs):
    half = (1 << (nbits - 1)) - 1
    u = np.clip((np.asarray(q, dtype=np.float64) - half) / half, -1.0, 1.0)
    return np.sign(u) * u * u * coeff_scale(qs)


def quant_coeffs(c, nbits):
    """-> (qs, integer codes).  Sign-sqrt companding, 2^nbits - 1 levels."""
    c = np.asarray(c, dtype=np.float64)
    m = float(np.max(np.abs(c))) if len(c) else 0.0
    qs = quant_scale(m)
    s = coeff_scale(qs)
    half = (1 << (nbits - 1)) - 1
    u = np.sign(c) * np.sqrt(np.minimum(np.abs(c) / s, 1.0))
    return qs, (np.clip(np.rint(u * half), -half, half).astype(np.int64) + half)


def ycc_to_rgb8(y, co, cg):
    r, g, b = y + co - cg, y + cg, y - co - cg
    return np.clip(np.floor(np.stack([r, g, b], -1) * 255.0 + 0.5), 0, 255).astype(np.uint8)


def real_sh(L, dirs):
    """Real SH basis (SPEC.md section 5.2). dirs (N,3) unit -> (N,(L+1)^2)."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    z = np.clip(z, -1.0, 1.0)
    s = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.arctan2(y, x)
    out = np.zeros((len(dirs), (L + 1) ** 2))
    pmm = np.ones(len(dirs))
    for m in range(L + 1):
        if m > 0:
            pmm = -pmm * (2 * m - 1) * s
        p2 = pmm
        p1 = None
        cm, sm = np.cos(m * phi), np.sin(m * phi)
        for l in range(m, L + 1):
            if l == m:
                p = pmm
            elif l == m + 1:
                p = z * (2 * m + 1) * pmm
            else:
                p = (z * (2 * l - 1) * p1 - (l + m - 1) * p2) / (l - m)
            p2, p1 = p1, p
            k = (2 * l + 1) / (4 * math.pi)
            for i in range(l - m + 1, l + m + 1):
                k /= i
            k = math.sqrt(k)
            if m == 0:
                out[:, l * l + l] = k * p
            else:
                out[:, l * l + l + m] = math.sqrt(2.0) * k * p * cm
                out[:, l * l + l - m] = math.sqrt(2.0) * k * p * sm
    return out


_ICO = {}


def icosphere(subdiv):
    if subdiv in _ICO:
        return _ICO[subdiv]
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, dtype=np.float64) / math.sqrt(1 + t * t) for p in v]
    for _ in range(subdiv):
        mid, nf = {}, []

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = v[a] + v[b]
                v.append(p / math.sqrt(float(p @ p)))
                mid[k] = len(v) - 1
            return mid[k]

        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    _ICO[subdiv] = (np.array(v), np.array(f, dtype=np.int64))
    return _ICO[subdiv]
