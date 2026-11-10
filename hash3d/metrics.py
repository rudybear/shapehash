"""Quality metrics between a ground-truth Asset and a decoded ColorMesh."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

from .asset import solid_occupancy
from .render import render_mesh

N_PTS = 30000
VOX = 64
FTAU = (0.05, 0.1)  # F-score thresholds, normalised units (2.5 % / 5 % of longest side)


class GroundTruth:
    """Everything about the original that the metrics need (computed once)."""

    def __init__(self, asset, renders):
        self.pts, self.nrm, self.col = asset.sample(N_PTS, seed=777)  # eval seed != encoder seeds
        self.tree = cKDTree(self.pts)
        self.lab = rgb2lab(self.col[None])[0]
        self.occ = asset.solid(VOX, [-1.05] * 3, [1.05] * 3)
        self.renders = renders


def _psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def evaluate(gt: GroundTruth, cm, renders=None):
    m = {}
    m["tris"] = int(len(cm.faces))
    m["verts"] = int(len(cm.vertices))
    if len(cm.faces) == 0:
        return m
    p, n, c = cm.sample(N_PTS, seed=4242)
    tree = cKDTree(p)
    d_gd, i_gd = tree.query(gt.pts)  # gt -> decoded (completeness)
    d_dg, i_dg = gt.tree.query(p)  # decoded -> gt (accuracy)
    m["chamfer"] = float(0.5 * (d_gd.mean() + d_dg.mean()))
    for tau in FTAU:
        pr, rc = float((d_dg < tau).mean()), float((d_gd < tau).mean())
        m[f"f{int(tau*100):02d}"] = 2 * pr * rc / max(pr + rc, 1e-9)
    nc1 = np.abs(np.einsum("ij,ij->i", gt.nrm, n[i_gd])).mean()
    nc2 = np.abs(np.einsum("ij,ij->i", n, gt.nrm[i_dg])).mean()
    m["normal"] = float(0.5 * (nc1 + nc2))
    lab = rgb2lab(c[None])[0]
    de1 = np.linalg.norm(gt.lab - lab[i_gd], axis=1).mean()
    de2 = np.linalg.norm(lab - gt.lab[i_dg], axis=1).mean()
    m["deltaE"] = float(0.5 * (de1 + de2))
    m["rgb_rmse"] = float(np.sqrt(0.5 * (np.mean((gt.col - c[i_gd]) ** 2) + np.mean((c - gt.col[i_dg]) ** 2))))
    occ = solid_occupancy(cm.sample(400000, seed=99)[0], VOX, [-1.05] * 3, [1.05] * 3)
    m["viou"] = float((occ & gt.occ).sum() / max((occ | gt.occ).sum(), 1))
    if renders is None:
        renders = render_mesh(cm, res=gt.renders[0][0].shape[0])
    sil, psnr, ssim, psnrb = [], [], [], []
    for (ia, ma), (ib, mb) in zip(gt.renders, renders):
        sil.append((ma & mb).sum() / max((ma | mb).sum(), 1))
        psnr.append(_psnr(ia, ib))
        ssim.append(structural_similarity(ia, ib, channel_axis=2, data_range=1.0))
        ga = ndimage.gaussian_filter(ia, (3, 3, 0))
        gb = ndimage.gaussian_filter(ib, (3, 3, 0))
        psnrb.append(_psnr(ga, gb))
    m["sil_iou"] = float(np.mean(sil))
    m["psnr"] = float(np.mean(psnr))
    m["psnr_blur"] = float(np.mean(psnrb))
    m["ssim"] = float(np.mean(ssim))
    return m


def quick_chamfer(asset, cm, n=6000):
    """Cheap encoder-side proxy (analysis-by-synthesis); uses its own samples."""
    if len(cm.faces) == 0:
        return 1e9
    a = asset.sample(n, seed=31)[0]
    b = cm.sample(n, seed=32)[0]
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def quick_score(asset, cm, res=48):
    """Encoder-side proxy that also sees holes: chamfer / silhouette-IoU^2 (lower is better)."""
    from .render import VIEWS, render_asset

    if len(cm.faces) == 0:
        return 1e9
    key = ("quick_masks", res)
    if key not in asset._cache:
        asset._cache[key] = [m for _, m in render_asset(asset, res=res, views=VIEWS[:4])]
    sil = []
    for ma, (_, mb) in zip(asset._cache[key], render_mesh(cm, res=res, views=VIEWS[:4])):
        sil.append((ma & mb).sum() / max((ma | mb).sum(), 1))
    return quick_chamfer(asset, cm) / max(float(np.mean(sil)), 0.05) ** 2
