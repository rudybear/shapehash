"""Loading glTF/GLB into a normalised, colour-sampleable ground-truth asset.

Normalised frame used everywhere in this project: the axis-aligned bounding
box is centred on the origin and scaled isotropically so that its longest
half-extent is 1 (the model fits in [-1, 1]^3).  Absolute scale/translation
are NOT part of the hash (the host application knows the node transform).

Colour is taken from (in order) base-colour texture x base-colour factor,
vertex colours, or the material base colour; everything is handled in sRGB.
"""
import numpy as np
import trimesh
from scipy import ndimage


def _material_info(visual):
    """-> (factor rgb float(3), texture float array HxWx3 or None)"""
    factor = np.ones(3)
    tex = None
    mat = getattr(visual, "material", None)
    if mat is None:
        return factor, tex
    img = None
    if isinstance(mat, trimesh.visual.material.PBRMaterial):
        f = mat.baseColorFactor
        if f is not None:
            f = np.asarray(f, dtype=np.float64).ravel()
            if f.max() > 1.0 or np.asarray(mat.baseColorFactor).dtype.kind in "ui":
                f = f / 255.0
            factor = f[:3]
        img = mat.baseColorTexture
    else:  # SimpleMaterial
        img = getattr(mat, "image", None)
        if img is None:
            d = getattr(mat, "diffuse", None)
            if d is not None:
                factor = np.asarray(d, dtype=np.float64).ravel()[:3] / 255.0
    if img is not None:
        try:
            a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            # cap texture size: plenty for sampling a placeholder ground truth
            while max(a.shape[:2]) > 1024:
                a = a[::2, ::2]
            tex = a
        except Exception:  # noqa
            tex = None
    return factor, tex


class Asset:
    """Triangle soup + everything needed to look up colour anywhere on it."""

    def __init__(self, name, vertices, faces, corner_uv, corner_rgb, corner_nrm, face_mat, materials):
        self.name = name
        self.vertices = vertices  # (V,3) normalised frame
        self.faces = faces  # (F,3)
        self.corner_uv = corner_uv  # (F,3,2)
        self.corner_rgb = corner_rgb  # (F,3,3) vertex colours (white if none)
        self.corner_nrm = corner_nrm  # (F,3,3) shading normals
        self.face_mat = face_mat  # (F,)
        self.materials = materials  # list of (factor, texture)
        tri = vertices[faces]
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.face_area = 0.5 * np.linalg.norm(cr, axis=1)
        self.face_normal = cr / np.maximum(2 * self.face_area[:, None], 1e-20)
        self.extents = np.abs(vertices).max(axis=0)  # half extents, max == 1
        self._mesh = None
        self._cache = {}

    # -- geometry helpers --------------------------------------------------
    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
        return self._mesh

    def color_at(self, face_idx, bary):
        """sRGB colour at barycentric positions on faces. bary: (N,3)."""
        face_idx = np.asarray(face_idx)
        out = np.einsum("nk,nkc->nc", bary, self.corner_rgb[face_idx])
        uv = np.einsum("nk,nkc->nc", bary, self.corner_uv[face_idx])
        fm = self.face_mat[face_idx]
        for mi, (factor, tex) in enumerate(self.materials):
            sel = np.nonzero(fm == mi)[0]
            if len(sel) == 0:
                continue
            c = np.broadcast_to(factor, (len(sel), 3)).astype(np.float64)
            if tex is not None:
                h, w = tex.shape[:2]
                x = np.floor((uv[sel, 0] % 1.0) * w).astype(np.int64) % w
                y = np.floor(((1.0 - uv[sel, 1]) % 1.0) * h).astype(np.int64) % h
                c = c * tex[y, x]
            out[sel] = out[sel] * c
        return np.clip(out, 0.0, 1.0)

    def sample(self, n, seed=0):
        """Area-weighted surface samples -> points, normals, colours."""
        key = ("sample", n, seed)
        if key not in self._cache:
            rng = np.random.default_rng(seed)
            p = self.face_area / self.face_area.sum()
            fi = rng.choice(len(self.faces), size=n, p=p)
            u, v = rng.random(n), rng.random(n)
            flip = u + v > 1
            u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
            bary = np.stack([1 - u - v, u, v], 1)
            pts = np.einsum("nk,nkc->nc", bary, self.vertices[self.faces[fi]])
            self._cache[key] = (pts, self.face_normal[fi], self.color_at(fi, bary))
        return self._cache[key]

    def solid(self, dims, lo, hi):
        """Solid occupancy on a grid (see solid_occupancy)."""
        key = ("solid", tuple(np.atleast_1d(dims).tolist()), tuple(np.round(lo, 6)), tuple(np.round(hi, 6)))
        if key not in self._cache:
            pts = self.sample(1_000_000, seed=12345)[0]
            self._cache[key] = solid_occupancy(pts, dims, lo, hi)
        return self._cache[key]

    def mean_color(self):
        return self.sample(20000, seed=1)[2].mean(axis=0)


def solid_occupancy(points, dims, lo, hi):
    """Robust 'solid' voxelisation that works for non-watertight soups.

    Surface points are splatted into the grid, then everything that cannot be
    reached from the grid boundary through empty voxels is declared inside.
    Open shells therefore stay thin shells; closed parts become solid; gaps
    narrower than ~2 voxels are sealed.
    """
    dims = np.broadcast_to(np.asarray(dims, dtype=np.int64), (3,))
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    # points outside the grid are clamped onto its boundary (keeps shells closed)
    ijk = np.clip(np.floor((points - lo) / (hi - lo) * dims).astype(np.int64), 0, dims - 1)
    surf = np.zeros(tuple(dims), dtype=bool)
    surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    # Point splatting is not guaranteed to be leak-free against a 6-connected
    # flood fill, so the fill runs against a 1-voxel dilated surface and the
    # outside region is then grown back by the same amount.
    full = np.ones((3, 3, 3), bool)
    thick = ndimage.binary_dilation(np.pad(surf, 2), structure=full)
    lab, _ = ndimage.label(~thick)
    outside = lab == lab[0, 0, 0]
    outside = ndimage.binary_dilation(outside, structure=full)
    return (~outside[2:-2, 2:-2, 2:-2]) | surf


def load_asset(path, name=None):
    scene = trimesh.load(path, force="scene", process=False)
    V, F, UV, RGB, NRM, FM, mats = [], [], [], [], [], [], []
    off = 0
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        v = trimesh.transformations.transform_points(g.vertices, T)
        f = np.asarray(g.faces, dtype=np.int64)
        nv = len(v)
        uv = np.zeros((nv, 2))
        rgb = np.ones((nv, 3))
        factor, tex = np.ones(3), None
        vis = g.visual
        if isinstance(vis, trimesh.visual.texture.TextureVisuals):
            if vis.uv is not None and len(vis.uv) == nv:
                uv = np.asarray(vis.uv, dtype=np.float64)
            factor, tex = _material_info(vis)
        elif isinstance(vis, trimesh.visual.color.ColorVisuals):
            try:
                if vis.kind == "vertex":
                    rgb = np.asarray(vis.vertex_colors, dtype=np.float64)[:, :3] / 255.0
                elif vis.kind == "face":
                    fc = np.asarray(vis.face_colors, dtype=np.float64)[:, :3] / 255.0
                    v = v[f].reshape(-1, 3)
                    rgb = np.repeat(fc, 3, axis=0)
                    f = np.arange(len(v)).reshape(-1, 3)
                    uv = np.zeros((len(v), 2))
                    nv = len(v)
                else:
                    rgb = np.broadcast_to(np.asarray(vis.main_color, dtype=np.float64)[:3] / 255.0, (nv, 3)).copy()
            except Exception:  # noqa
                pass
        try:
            gm = trimesh.Trimesh(v, f, process=False)
            n = np.asarray(gm.vertex_normals, dtype=np.float64)
        except Exception:  # noqa
            n = np.zeros((nv, 3))
        V.append(v)
        F.append(f + off)
        UV.append(uv[f])
        RGB.append(rgb[f])
        NRM.append(n[f])
        FM.append(np.full(len(f), len(mats), dtype=np.int64))
        mats.append((factor, tex))
        off += len(v)
    if not V:
        raise ValueError(f"no triangle geometry in {path}")
    V = np.concatenate(V)
    F = np.concatenate(F)
    UV, RGB, NRM, FM = np.concatenate(UV), np.concatenate(RGB), np.concatenate(NRM), np.concatenate(FM)
    # drop degenerate / non-finite faces
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = np.isfinite(area) & (area > 1e-14 * max(1.0, np.nanmax(area)))
    F, UV, RGB, NRM, FM = F[keep], UV[keep], RGB[keep], NRM[keep], FM[keep]
    used = np.unique(F)
    lo, hi = V[used].min(axis=0), V[used].max(axis=0)
    V = (V - (lo + hi) / 2) / ((hi - lo).max() / 2)
    if name is None:
        import os

        name = os.path.splitext(os.path.basename(path))[0]
    return Asset(name, V, F, UV, RGB, NRM, FM, mats)
