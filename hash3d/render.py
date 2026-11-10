"""Tiny orthographic ray-cast renderer (embree through trimesh).

Used for the perceptual metrics and for the comparison sheets.  Both the
original asset (per-pixel texture lookup) and decoded placeholders (vertex
colours) go through the same camera set and the same headlight shading.
"""
import numpy as np
import trimesh

# (azimuth, elevation) degrees; glTF is Y-up
VIEWS = [(30, 20), (120, 20), (210, 20), (300, 20), (75, 65), (255, -40)]
HALF = 1.6  # half width of the ortho window in normalised units
BG = 1.0


def _camera(az, el):
    az, el = np.radians(az), np.radians(el)
    d = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])  # towards camera
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, d)
    right /= np.linalg.norm(right)
    up2 = np.cross(d, right)
    return d, right, up2


def _rays(view, res):
    d, right, up = _camera(*view)
    s = (np.arange(res) + 0.5) / res * 2 - 1
    xx, yy = np.meshgrid(s * HALF, -s * HALF)
    o = d * 4.0 + xx[..., None] * right + yy[..., None] * up
    return o.reshape(-1, 3), np.broadcast_to(-d, (res * res, 3)).copy(), d


def cast(mesh, o, dirs):
    """-> tri index (-1 for miss), barycentric coords."""
    tri = mesh.ray.intersects_first(o, dirs)
    hit = tri >= 0
    bary = np.zeros((len(o), 3))
    if hit.any():
        t = mesh.triangles[tri[hit]]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        denom = np.einsum("ij,ij->i", n, dirs[hit])
        denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
        tt = np.einsum("ij,ij->i", n, t[:, 0] - o[hit]) / denom
        p = o[hit] + tt[:, None] * dirs[hit]
        b = trimesh.triangles.points_to_barycentric(t, p)
        bary[hit] = np.clip(b, 0, 1)
    return tri, bary


def _shade(albedo, normals, d):
    k = np.abs(normals @ d)
    return albedo * (0.35 + 0.65 * k)[:, None]


def render_asset(asset, res=160, views=VIEWS):
    """-> list of (rgb HxWx3 float, mask HxW bool)"""
    out = []
    for view in views:
        o, dirs, d = _rays(view, res)
        tri, bary = cast(asset.mesh, o, dirs)
        hit = tri >= 0
        img = np.full((res * res, 3), BG)
        if hit.any():
            alb = asset.color_at(tri[hit], bary[hit])
            n = np.einsum("nk,nkc->nc", bary[hit], asset.corner_nrm[tri[hit]])
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            n = np.where(ln > 1e-9, n / np.maximum(ln, 1e-9), asset.face_normal[tri[hit]])
            img[hit] = _shade(alb, n, d)
        out.append((img.reshape(res, res, 3), hit.reshape(res, res)))
    return out


def render_mesh(cm, res=160, views=VIEWS):
    out = []
    if len(cm.faces) == 0:
        return [(np.full((res, res, 3), BG), np.zeros((res, res), bool)) for _ in views]
    mesh = trimesh.Trimesh(cm.vertices, cm.faces, process=False)
    fn = cm.face_normals()
    vn = None if cm.flat else cm.vertex_normals()
    for view in views:
        o, dirs, d = _rays(view, res)
        tri, bary = cast(mesh, o, dirs)
        hit = tri >= 0
        img = np.full((res * res, 3), BG)
        if hit.any():
            f = cm.faces[tri[hit]]
            alb = np.einsum("nk,nkc->nc", bary[hit], cm.colors[f])
            if vn is None:
                n = fn[tri[hit]]
            else:
                n = np.einsum("nk,nkc->nc", bary[hit], vn[f])
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            img[hit] = _shade(alb, n, d)
        out.append((img.reshape(res, res, 3), hit.reshape(res, res)))
    return out


def outer_radius(mesh, center, dirs, far=8.0):
    """Distance from `center` to the outermost surface along each direction
    (0 where the ray from outside reaches the centre plane without a hit in
    front of it)."""
    o = center + dirs * far
    loc_tri = mesh.ray.intersects_first(o, -dirs)
    hit = loc_tri >= 0
    r = np.zeros(len(dirs))
    bary = np.zeros((len(dirs), 3))
    if hit.any():
        t = mesh.triangles[loc_tri[hit]]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        denom = np.einsum("ij,ij->i", n, -dirs[hit])
        denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
        tt = np.einsum("ij,ij->i", n, t[:, 0] - o[hit]) / denom
        p = o[hit] - tt[:, None] * dirs[hit]
        bary[hit] = np.clip(trimesh.triangles.points_to_barycentric(t, p), 0, 1)
        r[hit] = far - tt
    ok = hit & (r > 0)
    r[~ok] = 0.0
    return r, loc_tri, bary, ok
