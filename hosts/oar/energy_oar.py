"""OAR-side GESM-PC + RQ-CD."""
from __future__ import annotations

import numpy as np
import torch
from pytorch3d.ops import estimate_pointcloud_normals

from gesm_pc import energy, knn_jacobian, operators, projectors, rq_cd

try:
    import robust_laplacian
except Exception:
    robust_laplacian = None


def estimate_normals(points: torch.Tensor) -> torch.Tensor:
    return estimate_pointcloud_normals(points.unsqueeze(0)).squeeze(0)


def point_laplacian(points: torch.Tensor):
    if robust_laplacian is None:
        n = points.shape[0]
        M = torch.full((n,), 1.0 / n, device=points.device, dtype=points.dtype)
        return M, None
    L_np, M_np = robust_laplacian.point_cloud_laplacian(points.detach().cpu().numpy())
    M = torch.from_numpy(np.diag(M_np.toarray())).to(device=points.device, dtype=points.dtype)
    L = torch.from_numpy(L_np.toarray()).to(device=points.device, dtype=points.dtype)
    return M, L


def gesm_pc_loss(
    points, vel, normals, mass, lap,
    weights=None, k=30, a1=1.0, b1=1.0, c1=1.0, a2=1.0,
    knn_idx=None, knn_pts=None,
):
    P, N = projectors(normals)
    J = knn_jacobian(points, vel, k=k, idx=knn_idx, neigh_p=knn_pts)
    DT, DN = operators(J, P, N)
    if weights is not None:
        a1, b1, c1, a2 = weights[:, 0], weights[:, 1], weights[:, 2], weights[:, 3]
    return energy(DT, DN, a1=a1, b1=b1, c1=c1, a2=a2, mass=mass, P=P, laplace=lap)


def fidelity_rq(deformed, target, sigma=1.0):
    return rq_cd(
        deformed.unsqueeze(0), target.unsqueeze(0),
        alpha=3.0, sigma=sigma, trunc=0.2, p=1, as_loss=True,
    )
