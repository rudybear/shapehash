"""K coloured, oriented ellipsoids fitted to the solid volume (hard-EM
Gaussian mixture).  Decoder just instantiates low-poly spheres."""
import numpy as np

from ..bitio import dequant, quant, read_rgb, write_rgb
from ..mesh import ColorMesh, icosphere
from .base import Codec, register

PER = 18 + 15 + 14 + 12
RMIN, RMAX = 0.015, 1.5
Q = 1 / np.sqrt(2)


def quat_from_matrix(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12)) * 2
        q = [0, 0, 0, 0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def matrix_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def write_quat(w, q):
    i = int(np.argmax(np.abs(q)))
    if q[i] < 0:
        q = -q
    w.write(i, 2)
    for v in np.delete(q, i):
        w.write(int(quant(v, -Q, Q, 4)), 4)


def read_quat(r):
    i = r.read(2)
    rest = np.array([dequant(r.read(4), -Q, Q, 4) for _ in range(3)])
    big = np.sqrt(max(1 - (rest**2).sum(), 0.0))
    q = np.insert(rest, i, big)
    return q / np.linalg.norm(q)


def quant_radii(rad):
    q = quant(np.log(np.clip(rad, RMIN, RMAX)), np.log(RMIN), np.log(RMAX), 5)
    return q, np.exp(dequant(q, np.log(RMIN), np.log(RMAX), 5))


def mahal(pts, c, R, rad):
    """(N,K) squared ellipsoid-normalised distance (1 == on the surface)."""
    out = np.empty((len(pts), len(c)))
    for k in range(len(c)):
        y = (pts - c[k]) @ R[k] / rad[k]
        out[:, k] = (y * y).sum(1)
    return out


def fit_gmm(pts, K, rng, iters=12):
    n = len(pts)
    K = min(K, n)
    # k-means++ init
    cen = [pts[rng.integers(n)]]
    d2 = ((pts - cen[0]) ** 2).sum(1)
    for _ in range(K - 1):
        cen.append(pts[rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else pts[rng.integers(n)])
        d2 = np.minimum(d2, ((pts - cen[-1]) ** 2).sum(1))
    cen = np.array(cen)
    for _ in range(10):
        lab = np.argmin(((pts[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        for k in range(K):
            if (lab == k).any():
                cen[k] = pts[lab == k].mean(0)
    cov = np.tile(np.eye(3) * 0.05, (K, 1, 1))
    eps = 0.012**2
    for it in range(iters + 1):
        for k in range(K):
            sel = pts[lab == k]
            if len(sel) >= 5:
                cen[k] = sel.mean(0)
                cov[k] = np.cov(sel.T) + np.eye(3) * eps
        if it == iters:
            break
        cost = np.empty((n, K))
        for k in range(K):
            d = pts - cen[k]
            cost[:, k] = np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov[k]), d) + np.log(np.linalg.det(cov[k]))
        lab = np.argmin(cost, axis=1)
    return cen, cov


@register
class EllipsoidCodec(Codec):
    """Payload: tag(8) | K 7 | K x [centre 3x6, log-radii 3x5, quaternion
    (smallest-three, 2+3x4), RGB444] = 59 bits per ellipsoid."""

    name = "ellip"
    tag = 6
    encoder_note = "EM fit + 7 trial decodes"
    decoder_cost = "low"
    decoder_needs = "instanced unit spheres with an affine transform each (or splats / SDF union in a shader)"

    def encode(self, asset, budget_bytes):
        from ..metrics import quick_score

        K = int(min(127, max(1, (budget_bytes * 8 - 15) // PER)))
        occ = asset.solid(64, [-1.0] * 3, [1.0] * 3)
        vol = (np.argwhere(occ) + 0.5) / 64 * 2 - 1
        spts, _, scol = asset.sample(20000, seed=7)
        mean = scol.mean(0)
        best = None
        # two fitting modes: blobs filling the solid volume, or flat "surfels"
        # tiling the surface; pick by analysis-by-synthesis (mode is implicit)
        for mode, src, scales in (("vol", vol, (1.7, 1.9, 2.1, 2.3)), ("surf", spts, (2.0, 2.3, 2.6))):
            rng = np.random.default_rng(0)
            pts = src[rng.choice(len(src), size=min(len(src), 30000), replace=False)]
            cen, cov = fit_gmm(pts, K, rng)
            R, sig = [], []
            for k in range(len(cen)):
                ev, evec = np.linalg.eigh(cov[k])
                if np.linalg.det(evec) < 0:
                    evec[:, 0] *= -1
                # store the rotation as the decoder will see it
                R.append(matrix_from_quat(self._q_roundtrip(quat_from_matrix(evec))))
                sig.append(np.sqrt(np.maximum(ev, 1e-8)))
            R, sig = np.array(R), np.array(sig)
            cq = quant(cen, -1.0, 1.0, 6)
            cen_d = dequant(cq, -1.0, 1.0, 6)
            for s in scales:
                rq, rad_d = quant_radii(sig * s)
                lab = np.argmin(mahal(spts, cen_d, R, rad_d), axis=1)
                w = self.start()
                w.write(len(cen), 7)
                for k in range(len(cen)):
                    for v in cq[k]:
                        w.write(int(v), 6)
                    for v in rq[k]:
                        w.write(int(v), 5)
                    write_quat(w, quat_from_matrix(R[k]))
                    write_rgb(w, scol[lab == k].mean(0) if (lab == k).any() else mean)
                out = self.finish(w, budget_bytes)
                err = quick_score(asset, self.decode(out))
                if best is None or err < best[0]:
                    best = (err, out)
        return best[1]

    @staticmethod
    def _q_roundtrip(q):
        from ..bitio import BitReader, BitWriter

        w = BitWriter()
        write_quat(w, q)
        return read_quat(BitReader(w.to_bytes()))

    def decode_payload(self, r, nibble):
        K = r.read(7)
        cen, rad, R, col = [], [], [], []
        for _ in range(K):
            cen.append(dequant(np.array([r.read(6) for _ in range(3)]), -1.0, 1.0, 6))
            q = np.array([r.read(5) for _ in range(3)])
            rad.append(np.exp(dequant(q, np.log(RMIN), np.log(RMAX), 5)))
            R.append(matrix_from_quat(read_quat(r)))
            col.append(read_rgb(r))
        cen, rad, R = np.array(cen), np.array(rad), np.array(R)
        sv, sf = icosphere(2 if K <= 20 else 1)
        V, F, C = [], [], []
        for k in range(K):
            v = cen[k] + (sv * rad[k]) @ R[k].T
            # drop triangles completely buried inside another ellipsoid
            inside = np.zeros(len(v), bool)
            if K > 1:
                others = [j for j in range(K) if j != k]
                inside = mahal(v, cen[others], R[others], rad[others]).min(axis=1) < 0.97
            keep = ~inside[sf].all(axis=1)
            F.append(sf[keep] + k * len(sv))
            V.append(v)
            C.append(np.tile(col[k], (len(v), 1)))
        if not V:
            return ColorMesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
        return ColorMesh(np.concatenate(V), np.concatenate(F), np.concatenate(C))
