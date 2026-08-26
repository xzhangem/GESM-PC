"""Neighbour graphs: Euclidean kNN or geodesic (heat-method) kNN."""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch
from scipy.spatial import KDTree

try:
    from pytorch3d.ops import knn_points as _p3d_knn
except Exception:
    _p3d_knn = None

try:
    import potpourri3d as pp3d
except Exception:
    pp3d = None

KNNOut = namedtuple("KNNOut", ["idx", "knn", "dists"])


def euclidean_knn(points: torch.Tensor, k: int) -> KNNOut:
    if points.dim() == 3:
        points = points[0]
    n, device = points.shape[0], points.device
    k = min(k, n - 1)
    if _p3d_knn is not None:
        out = _p3d_knn(points.unsqueeze(0), points.unsqueeze(0), K=k + 1, return_nn=True)
        return KNNOut(idx=out.idx[0, :, 1:], knn=out.knn[0, :, 1:], dists=out.dists[0, :, 1:])
    dist = torch.cdist(points, points)
    dist.fill_diagonal_(float("inf"))
    vals, idx = dist.topk(k, largest=False, dim=-1)
    return KNNOut(idx=idx, knn=points[idx], dists=vals)


def geodesic_knn(
    points: torch.Tensor,
    k: int,
    local_radius_factor: float = 5.0,
    subsample_ratio: float = 1.0,
) -> KNNOut:
    if pp3d is None:
        raise ImportError("geodesic kNN requires potpourri3d")
    if points.dim() == 3:
        points = points[0]
    pts = points.detach().cpu().numpy()
    n = len(pts)
    k = min(k, n - 1)
    tree = KDTree(pts)
    avg = float(np.mean(tree.query(pts[: min(100, n)], k=2)[0][:, 1]))
    solver = pp3d.PointCloudHeatSolver(pts)
    n_src = max(1, int(n * subsample_ratio)) if subsample_ratio < 1.0 else n
    sources = np.arange(n) if n_src >= n else np.random.choice(n, size=n_src, replace=False)
    geod = np.stack([solver.compute_distance(int(s)) for s in sources], axis=0)
    idx = np.full((n, k), -1, dtype=np.int64)
    dist = np.full((n, k), np.inf, dtype=np.float32)
    coord = np.zeros((n, k, 3), dtype=np.float32)
    for i in range(n):
        cand = np.array(tree.query_ball_point(pts[i], avg * local_radius_factor))
        if cand.size < k + 5:
            cand = np.arange(n)
        cand = cand[cand != i]
        src_i = int(np.argmin(geod[:, i]))
        d = geod[src_i, cand]
        order = np.argsort(d)[:k]
        take = cand[order]
        idx[i, : take.size] = take
        dist[i, : take.size] = d[order]
        coord[i, : take.size] = pts[take]
    device = points.device
    return KNNOut(
        idx=torch.from_numpy(idx).to(device=device),
        knn=torch.from_numpy(coord).to(device=device, dtype=points.dtype),
        dists=torch.from_numpy(dist).to(device=device, dtype=points.dtype),
    )


def build_knn(points: torch.Tensor, k: int, mode: str = "euclidean") -> KNNOut:
    mode = mode.lower()
    if mode in ("euclidean", "eucl", "l2"):
        return euclidean_knn(points, k)
    if mode in ("geodesic", "geo"):
        return geodesic_knn(points, k)
    raise ValueError(f"unknown knn mode {mode!r}")
