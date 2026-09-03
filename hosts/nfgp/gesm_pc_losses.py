"""
GESM-PC losses for NFGP implicit deformation editing.

Reference (current manuscript):
  Def. 3.4, Def. 3.6 / Eq. (14), Thm. 3.7, Sec. 3.4.3 / Eq. (25)
  of "GESM-PC: General Elastic Shape Metric for Point Cloud Shape Analysis
  with Neural Deformation Representations".

---------------------------------------------------------------------------
Projectors (Eq. 11) — symmetric, NOT P_0 / N_0
---------------------------------------------------------------------------
    N = n n^T
    P = I - N

---------------------------------------------------------------------------
Symmetric first-order operators (Def. 3.4 / Eq. 12)
---------------------------------------------------------------------------
    D_T(v) = P (dv) P + P (dv)^T P
    D_N(v) = N (dv) P + P (dv)^T N

The paper omits the conventional 1/2 in the symmetrization (as in Eq. 4).

---------------------------------------------------------------------------
Orthogonal split (Eq. 13, Thm. 3.7); k_1 = 1/2
---------------------------------------------------------------------------
    (Dv)_shr   = D_T - (1/2) Tr(D_T) P
    (Dv)_scale = (1/2) Tr(D_T) P
    (Dv)_bend  = D_N          # no spherical correction; Tr(D_N)=0

Densities (Eq. 14 / App. A.2.3):
    g_shr   = <D_T, D_T> - (1/2) Tr(D_T)^2
            = ||D_T - (1/2) Tr(D_T) P||_F^2
    g_scale = (1/2) Tr(D_T)^2
    g_bend  = <D_N, D_N> = ||D_N||_F^2

---------------------------------------------------------------------------
NFGP Jacobian <-> displacement gradient
---------------------------------------------------------------------------
    D_θ(x) = x + u_θ(x)
    F = ∂D_θ/∂x = I + dv
    dv = L = F - I

Sec. 3.4.3 drops the Laplacian smoothness term on implicit surfaces.
"""

from __future__ import absolute_import, division, print_function

import torch
from trainers.utils.diff_ops import jacobian
from trainers.utils.igp_utils import sample_points, tangential_projection_matrix


def _frobenius_sq(mat):
    """Batched Frobenius squared norm: (..., 3, 3) -> (...)"""
    return (mat * mat).sum(dim=(-1, -2))


def _normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))


def build_P_N(normals, eps=1e-8):
    """
    Symmetric projectors (Eq. 11).

    Args:
        normals: (B, 3)
    Returns:
        N, P: each (B, 3, 3)
    """
    n = _normalize(normals, eps=eps).unsqueeze(-1)            # (B, 3, 1)
    N = torch.bmm(n, n.transpose(1, 2))                       # n n^T
    I = torch.eye(3, device=normals.device, dtype=normals.dtype)
    P = I.unsqueeze(0) - N
    return N, P


def gesm_pc_components(dv, normals, eps=1e-8):
    """
    GESM-PN components of Def. 3.4 / Eq. (13)–(14).

    Args:
        dv:      (B, 3, 3)  displacement gradient  L = F - I
        normals: (B, 3)     unit surface normals

    Returns:
        dict with per-point g_shr, g_scale, g_bend  (each (B,))
    """
    N, P = build_P_N(normals, eps=eps)                        # (B, 3, 3)
    dvT = dv.transpose(1, 2)

    # Eq. (12)
    P_dv = torch.bmm(P, dv)
    P_dvT = torch.bmm(P, dvT)
    DT = torch.bmm(P_dv, P) + torch.bmm(P_dvT, P)             # P dv P + P dv^T P
    DN = torch.bmm(torch.bmm(N, dv), P) + torch.bmm(P_dvT, N)  # N dv P + P dv^T N

    tr_DT = DT.diagonal(dim1=-2, dim2=-1).sum(-1)             # (B,)

    # Eq. (13) / Thm. 3.7: k1 = 1/2
    half_tr = (0.5 * tr_DT).view(-1, 1, 1)
    scale_mat = half_tr * P
    shr_mat = DT - scale_mat
    # bend = DN  (Tr(DN)=0, no spherical correction)

    # Eq. (14) / (34)–(35)
    g_shr = _frobenius_sq(shr_mat)                            # ||DT - 1/2 Tr P||_F^2
    g_scale = 0.5 * (tr_DT * tr_DT)                           # 1/2 Tr(DT)^2
    g_bend = _frobenius_sq(DN)                                # ||DN||_F^2

    return {
        'g_shr': g_shr,
        'g_scale': g_scale,
        'g_bend': g_bend,
        'shr_mat': shr_mat,
        'scale_mat': scale_mat,
        'bend_mat': DN,
        'DT': DT,
        'DN': DN,
        'N': N,
        'P': P,
    }


def gesm_pc_loss(
        inp_nf, out_nf, deform,
        x=None, weights=None,
        npoints=5000, dim=3,
        use_surf_points=True, invert_sampling=True,
        detach_weight=True, use_rejection=False,
        # Eq. (14) coefficients a1, b1, c1, a2
        weight_shear=1.0,
        weight_scale=0.0,
        weight_bend=1.0,
        weight_smooth=0.0,
        # Optional extra stretch: ||L^T L||_F
        weight_jtj=0.0,
        reduction='mean',
        return_components=False,
):
    """
    GESM-PN energy on the deformed neural surface (Eq. 14 / Eq. 25).

    dv = F - I, F = ∂D_θ/∂x.  Normals from ∇G_θ.
    """
    if x is None:
        x, weights = sample_points(
            npoints, dim=dim,
            sample_surf_points=use_surf_points,
            inp_nf=inp_nf, out_nf=out_nf, deform=deform,
            invert_sampling=invert_sampling,
            detach_weight=detach_weight,
            use_rejection=use_rejection,
        )
        bs, npoints = x.size(0), x.size(1)
    else:
        assert weights is not None
        if len(x.size()) == 2:
            bs, npoints = 1, x.size(0)
            x = x.view(1, npoints, dim)
        else:
            bs, npoints = x.size(0), x.size(1)

    x = x.view(bs, npoints, dim)
    if not torch.is_tensor(weights):
        weights = torch.ones(bs, npoints, device=x.device, dtype=x.dtype)
    weights = weights.view(bs, npoints)

    if x.is_leaf:
        x.requires_grad_(True)
    else:
        x.retain_grad()
    y_out = out_nf(x)
    normals, _ = tangential_projection_matrix(y_out, x)
    normals = normals.view(bs * npoints, dim)

    x_inp = deform(x).view(bs, npoints, dim)
    J, status = jacobian(x_inp, x)
    J = J.view(bs * npoints, dim, dim)

    I = torch.eye(dim, device=J.device, dtype=J.dtype).unsqueeze(0)
    L = J - I                                                 # dv = F - I

    comps = gesm_pc_components(L, normals)

    w = weights.view(bs * npoints)

    def _reduce(per_pt):
        val = (per_pt * w).view(bs, npoints)
        if reduction == 'mean':
            return val.mean()
        if reduction == 'sum':
            return val.sum()
        raise ValueError('Unknown reduction: %s' % reduction)

    loss_shear = _reduce(comps['g_shr']) * float(weight_shear)
    loss_scale = _reduce(comps['g_scale']) * float(weight_scale)
    loss_bend = _reduce(comps['g_bend']) * float(weight_bend)

    loss_jtj = J.new_zeros(())
    if float(weight_jtj) > 0.:
        LtL = torch.bmm(L.transpose(1, 2), L)
        loss_jtj = _reduce(_frobenius_sq(LtL)) * float(weight_jtj)

    # Implicit-surface setting (Sec. 3.4.3) drops Laplacian smoothness;
    # weight_smooth remains an optional proxy ||dv||_F^2.
    loss_smooth = J.new_zeros(())
    if float(weight_smooth) > 0.:
        loss_smooth = _reduce(_frobenius_sq(L)) * float(weight_smooth)

    loss_stretch = loss_shear + loss_scale + loss_jtj
    total = loss_stretch + loss_bend + loss_smooth

    if return_components:
        return total, {
            'loss_shear': loss_shear.detach(),
            'loss_scale': loss_scale.detach(),
            'loss_jtj': loss_jtj.detach() if torch.is_tensor(loss_jtj)
            else loss_jtj,
            'loss_stretch': loss_stretch.detach(),
            'loss_bend': loss_bend.detach(),
            'loss_smooth': loss_smooth.detach() if torch.is_tensor(loss_smooth)
            else loss_smooth,
        }
    return total


def stretch_loss_gesm(
        inp_nf, out_nf, deform,
        x=None, npoints=1000, dim=3, use_surf_points=False,
        invert_sampling=False, loss_type='l2', reduction='mean',
        weights=1, detach_weight=True, use_rejection=False,
        weight_shear=1.0, weight_scale=0.0, weight_jtj=0.0,
):
    """GESM-PC stretch group: shear (+ optional scale, ||L^T L||_F)."""
    return gesm_pc_loss(
        inp_nf, out_nf, deform,
        x=x, weights=weights,
        npoints=npoints, dim=dim,
        use_surf_points=use_surf_points,
        invert_sampling=invert_sampling,
        detach_weight=detach_weight,
        use_rejection=use_rejection,
        weight_shear=weight_shear,
        weight_scale=weight_scale,
        weight_jtj=weight_jtj,
        weight_bend=0.0,
        weight_smooth=0.0,
        reduction=reduction,
    )


def bending_loss_gesm(
        inp_nf, out_nf,
        x=None, weights=None,
        npoints=1000, dim=3, use_surf_points=False, deform=None,
        invert_sampling=False, detach_weight=True, use_rejection=False,
        loss_type='l2', reduction='mean',
        weight_bend=1.0,
):
    """GESM-PC bending (Eq. 14d): ||D_N||_F^2."""
    return gesm_pc_loss(
        inp_nf, out_nf, deform,
        x=x, weights=weights,
        npoints=npoints, dim=dim,
        use_surf_points=use_surf_points,
        invert_sampling=invert_sampling,
        detach_weight=detach_weight,
        use_rejection=use_rejection,
        weight_shear=0.0,
        weight_scale=0.0,
        weight_bend=weight_bend,
        weight_smooth=0.0,
        reduction=reduction,
    )
