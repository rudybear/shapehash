"""Bit-level serialisation, string armour and small shared quantisers.

Every codec writes its payload through BitWriter, and the payload is turned
into a DB-friendly string with URL-safe base64 (no padding): 6 bits / char.
"""
import base64

import numpy as np


class BitWriter:
    def __init__(self):
        self.bits = []

    def write(self, value, nbits):
        value = int(value)
        if nbits == 0:
            return
        if value < 0 or value >= (1 << nbits):
            raise ValueError(f"value {value} does not fit in {nbits} bits")
        for i in range(nbits - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def write_bytes(self, data):
        for b in data:
            self.write(b, 8)

    def __len__(self):
        return len(self.bits)

    def to_bytes(self):
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(np.packbits(np.array(bits, dtype=np.uint8)).tolist()) if bits else b""


class BitReader:
    def __init__(self, data):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).tolist()
        self.pos = 0

    def read(self, nbits):
        v = 0
        for _ in range(nbits):
            # reading past the end yields zeros (keeps decoders total)
            b = self.bits[self.pos] if self.pos < len(self.bits) else 0
            v = (v << 1) | b
            self.pos += 1
        return v

    def remaining(self):
        return len(self.bits) - self.pos

    def read_rest_bytes(self):
        """Remaining whole bytes after aligning to a byte boundary."""
        self.pos += -self.pos % 8
        n = (len(self.bits) - self.pos) // 8
        return bytes(self.read(8) for _ in range(n))

    def align(self):
        self.pos += -self.pos % 8


def to_string(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def from_string(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ---------------------------------------------------------------- quantisers
def quant(x, lo, hi, nbits):
    """Uniform quantiser of x in [lo, hi] to nbits (bin centres at ends)."""
    n = (1 << nbits) - 1
    q = np.rint((np.asarray(x, dtype=np.float64) - lo) / (hi - lo) * n)
    return np.clip(q, 0, n).astype(np.int64)


def dequant(q, lo, hi, nbits):
    n = (1 << nbits) - 1
    return lo + np.asarray(q, dtype=np.float64) / n * (hi - lo)


def write_extents(w, e):
    """Half extents of the bbox in the normalised frame (max == 1): 3 x 6 bits."""
    for v in e:
        w.write(int(np.clip(np.ceil(v * 64) - 1, 0, 63)), 6)


def read_extents(r):
    return np.array([(r.read(6) + 1) / 64.0 for _ in range(3)])


def quant_extents(e):
    return (np.clip(np.ceil(np.asarray(e) * 64) - 1, 0, 63) + 1) / 64.0


def write_rgb(w, rgb, bits=(4, 4, 4)):
    for c, b in zip(rgb, bits):
        w.write(int(quant(c, 0.0, 1.0, b)), b)


def read_rgb(r, bits=(4, 4, 4)):
    return np.array([dequant(r.read(b), 0.0, 1.0, b) for b in bits])


def rgb_to_ycc(rgb):
    """YCoCg: Y in [0,1], Co/Cg in [-0.5, 0.5]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack([0.25 * r + 0.5 * g + 0.25 * b, 0.5 * r - 0.5 * b, -0.25 * r + 0.5 * g - 0.25 * b], -1)


def ycc_to_rgb(ycc):
    ycc = np.asarray(ycc, dtype=np.float64)
    y, co, cg = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    return np.clip(np.stack([y + co - cg, y + cg, y - co - cg], -1), 0.0, 1.0)


def compand(x, m):
    """BlurHash-style sign-sqrt companding of x in [-m, m] to [-1, 1]."""
    x = np.asarray(x, dtype=np.float64) / max(m, 1e-12)
    return np.sign(x) * np.sqrt(np.minimum(np.abs(x), 1.0))


def expand(u, m):
    u = np.asarray(u, dtype=np.float64)
    return np.sign(u) * u * u * m


def write_coeffs(w, c, nbits, scale_bits=6, scale_max=4.0):
    """Quantise a coefficient vector: shared log-scale + companded values."""
    c = np.asarray(c, dtype=np.float64)
    m = float(np.max(np.abs(c))) if len(c) else 0.0
    # log-quantised scale in [scale_max * 2^-10, scale_max]
    n = (1 << scale_bits) - 1
    if m <= 0:
        qs = 0
    else:
        qs = int(np.clip(np.ceil((np.log2(m / scale_max) + 10.0) / 10.0 * n), 0, n))
    mq = scale_max * 2.0 ** (qs / n * 10.0 - 10.0)
    w.write(qs, scale_bits)
    # odd number of levels (2^n - 1) so that zero is exactly representable
    half = (1 << (nbits - 1)) - 1
    q = np.clip(np.rint(compand(c, mq) * half), -half, half).astype(np.int64) + half
    for v in q:
        w.write(int(v), nbits)


def read_coeffs(r, count, nbits, scale_bits=6, scale_max=4.0):
    n = (1 << scale_bits) - 1
    qs = r.read(scale_bits)
    mq = scale_max * 2.0 ** (qs / n * 10.0 - 10.0)
    half = (1 << (nbits - 1)) - 1
    q = np.array([r.read(nbits) for _ in range(count)], dtype=np.float64)
    return expand(np.clip((q - half) / half, -1.0, 1.0), mq)


def qcoeffs(c, nbits, scale_bits=6, scale_max=4.0):
    """Round-trip helper (what the decoder will see)."""
    w = BitWriter()
    write_coeffs(w, c, nbits, scale_bits, scale_max)
    return read_coeffs(BitReader(w.to_bytes()), len(c), nbits, scale_bits, scale_max)
