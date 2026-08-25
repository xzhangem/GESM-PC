"""GESM-PC building blocks matching the OAR host (pc_DDG.py + loss_functions.py).

Jacobian: δv = J δp, same layout as ``estimate_velocity_gradient_torch``.
Operators: paper D_T / D_N (two-sided P), which coincide with
``P @ J @ P`` / ``N @ J @ P`` after symmetrization.
RQ-CD: same kernel as ``correntropy_chamfer_distance`` (α=3, nearest-neighbour,
optional truncation, returned as a *negative* kernel sum so it is a loss).
"""
from __future__ import annotations

import torch
from torch import Tensor

try:
    from pytorch3d.ops import knn_points as _p3d_knn
except Exception:  # pragma: no cover
    _p3d_knn = None


def _as_cloud(x: Tensor) -> Tensor:
    """[N, 3] -> [1, N, 3]; leave [B, N, 3] unchanged."""
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"expected [N,3] or [B,N,3], got {tuple(x.shape)}")


def projectors(n: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """``P = I - nn^T``, ``N = nn^T`` (pc_DDG.LocalPCEnergy_Grass).

    Args:
        n: [N, 3] or [..., 3] normals.
    Returns:
        P, N: [..., 3, 3].
    """
    n = torch.nn.functional.normalize(n, dim=-1, eps=eps)
    N = torch.matmul(n.unsqueeze(-1), n.unsqueeze(-2))
    I = torch.eye(3, dtype=n.dtype, device=n.device)
    P = I - N
    return P, N


def knn_jacobian(
    p: Tensor,
    v: Tensor,
    k: int = 30,
    rcond: float = 1e-6,
    weighted: bool = True,
) -> Tensor:
    """Ambient WLS Jacobian, same residual as ``estimate_velocity_gradient_torch``.

        min_J  ∑_j w_j² ||δv_j - J δp_j||² ,   w_j = 1/||δp_j||
        J = (∑ w² δv ⊗ δp) (∑ w² δp ⊗ δp)⁺

    Neighbours from PyTorch3D ``knn_points`` when available (as in OAR),
    otherwise ``torch.cdist``. Self is excluded.

    Args:
        p, v: [N, 3] points and velocities.
        k: neighbourhood size.
        rcond: pinv cutoff (and Tikhonov floor).
        weighted: True = 1/||δp|| as in the paper; False = unweighted
            ``lstsq`` used in the default ``estimate_velocity_gradient_torch``.
    Returns:
        J: [N, 3, 3] with δv = J δp.
    """
    if p.dim() != 2 or p.shape[-1] != 3:
        raise ValueError("knn_jacobian expects p, v of shape [N, 3]")
    n_pts, device, dtype = p.shape[0], p.device, p.dtype
    k = min(int(k), n_pts - 1)

    if _p3d_knn is not None:
        knn = _p3d_knn(p.unsqueeze(0), p.unsqueeze(0), K=k + 1, return_nn=True)
        idx = knn.idx[0, :, 1:]
        neigh_p = knn.knn[0, :, 1:]
    else:
        dist = torch.cdist(p, p)
        dist.fill_diagonal_(float("inf"))
        idx = dist.topk(k, largest=False, dim=-1).indices
        neigh_p = p[idx]

    dp = neigh_p - p.unsqueeze(1)
    dv = v[idx] - v.unsqueeze(1)

    if weighted:
        w2 = 1.0 / dp.pow(2).sum(dim=-1).clamp_min(1e-12)
        G = torch.einsum("nk,nki,nkj->nij", w2, dp, dp)
        H = torch.einsum("nk,nki,nkj->nij", w2, dv, dp)
        J = torch.matmul(H, torch.linalg.pinv(G, rcond=rcond))
        return J

    # Unweighted normal equations, component-wise (OAR default).
    G = torch.bmm(dp.transpose(1, 2), dp) + rcond * torch.eye(
        3, device=device, dtype=dtype
    ).unsqueeze(0)
    J = torch.zeros(n_pts, 3, 3, device=device, dtype=dtype)
    for c in range(3):
        rhs = torch.bmm(dp.transpose(1, 2), dv[:, :, c].unsqueeze(-1))
        J[:, c, :] = torch.linalg.solve(G, rhs).squeeze(-1)
    return J


def operators(J: Tensor, P: Tensor, N: Tensor) -> tuple[Tensor, Tensor]:
    """Paper operators (and Grass ``P@J@P`` / ``N@J@P`` after symmetrization).

        D_T = P J P + P Jᵀ P
        D_N = N J P + P Jᵀ N
    """
    JP = torch.matmul(J, P)
    JT = J.transpose(-1, -2)
    DT = torch.matmul(P, JP) + torch.matmul(P, torch.matmul(JT, P))
    DN = torch.matmul(N, JP) + torch.matmul(P, torch.matmul(JT, N))
    return DT, DN


def _trace(A: Tensor) -> Tensor:
    return A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]


def _fro_sq(A: Tensor) -> Tensor:
    return (A * A).sum(dim=(-1, -2))


def energy(
    DT: Tensor,
    DN: Tensor,
    a1: float | Tensor = 1.0,
    b1: float | Tensor = 1.0,
    c1: float | Tensor = 1.0,
    mass: Tensor | None = None,
    P: Tensor | None = None,
    a2: float | Tensor = 0.0,
    laplace: Tensor | None = None,
) -> Tensor:
    """∑_i m_i (a1 g_shr + b1 g_scale + c1 g_bend) [+ a2 ||L v||²].

        g_shr   = ||D_T - (1/2) Tr(D_T) P||_F²
        g_scale = (1/2) Tr(D_T)²
        g_bend  = ||D_N||_F²

    ``a1,b1,c1,a2`` may be scalars or per-point [N] weights (ADW head).
    ``mass`` is the lumped mass ``M`` from robust_laplacian; default 1/N.
    ``laplace`` is the per-point ||L v||² if smoothing is used.
    """
    trT = _trace(DT)
    if P is None:
        eye = torch.eye(3, dtype=DT.dtype, device=DT.device)
        P = eye.expand(DT.shape[:-2] + (3, 3))
    g_shr = _fro_sq(DT - 0.5 * trT[..., None, None] * P)
    g_scale = 0.5 * trT * trT
    g_bend = _fro_sq(DN)
    dens = a1 * g_shr + b1 * g_scale + c1 * g_bend
    if laplace is not None and (
        not isinstance(a2, float) or a2 != 0.0
    ):
        dens = dens + a2 * laplace
    if mass is None:
        return dens.mean()
    return (mass * dens).mean()


def rq_cd(
    x: Tensor,
    y: Tensor,
    alpha: float = 3.0,
    sigma: float = 1.0,
    trunc: float | None = 0.2,
    p: int = 1,
    as_loss: bool = True,
) -> Tensor:
    """RQ Chamfer used in OAR ``correntropy_chamfer_distance``.

        K = (1 + d / (α σ²))^{-α}

    ``d`` is the nearest-neighbour distance with Minkowski order ``p``
    (OAR uses ``p=1``). Pairs with ``d >= trunc`` are dropped.
    If ``as_loss`` (default), returns ``-(mean_x K + mean_y K)`` so it
    can replace MMC as a training term; otherwise returns the paper
    ``mean(1-K_xy) + mean(1-K_yx)``.
    """
    x_b, y_b = _as_cloud(x), _as_cloud(y)
    B, N1, _ = x_b.shape
    N2 = y_b.shape[1]
    sigma2 = sigma ** 2 if sigma != 1.0 else 1.0

    if _p3d_knn is not None:
        x_nn = _p3d_knn(x_b, y_b, K=1, norm=p)
        y_nn = _p3d_knn(y_b, x_b, K=1, norm=p)
        cham_x = x_nn.dists[..., 0]
        cham_y = y_nn.dists[..., 0]
    else:
        d = torch.cdist(x_b, y_b, p=float(p))
        cham_x = d.min(dim=-1).values
        cham_y = d.min(dim=-2).values

    mask_x = torch.zeros_like(cham_x, dtype=torch.bool)
    mask_y = torch.zeros_like(cham_y, dtype=torch.bool)
    if trunc is not None:
        mask_x = cham_x >= trunc
        mask_y = cham_y >= trunc
        cham_x = cham_x.masked_fill(mask_x, 0.0)
        cham_y = cham_y.masked_fill(mask_y, 0.0)

    Kx = (1.0 + cham_x / (alpha * sigma2)).pow(-alpha)
    Ky = (1.0 + cham_y / (alpha * sigma2)).pow(-alpha)
    Kx = Kx.masked_fill(mask_x, 0.0)
    Ky = Ky.masked_fill(mask_y, 0.0)

    mean_x = Kx.sum(dim=-1) / float(N1)
    mean_y = Ky.sum(dim=-1) / float(N2)
    if as_loss:
        return -(mean_x + mean_y).sum()
    ones_x = (~mask_x).float()
    ones_y = (~mask_y).float()
    return ((ones_x - Kx).sum() / N1 + (ones_y - Ky).sum() / N2) / B
