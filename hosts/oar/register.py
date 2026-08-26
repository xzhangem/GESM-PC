"""Correspondence-free registration: OAR SIREN + GESM-PC + RQ-CD."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

_ROOT = Path(__file__).resolve().parents[2]
_HOST = Path(__file__).resolve().parent
for p in (_ROOT, _HOST):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from energy_oar import estimate_normals, fidelity_rq, gesm_pc_loss, point_laplacian
from io_ply import denormalize, load_xyz, nn_rmse, normalize_points, save_xyz, to_torch
from knn import build_knn
from model import SirenV, SirenVW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pair(
    src, tgt, *, steps=400, k=30, lambda_g=1e2, lambda_fid=1e4,
    lr=1e-4, adw=True, sigma=1.0, knn="euclidean",
):
    model = (SirenVW if adw else SirenV)(hidden_features=128, hidden_layers=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = ReduceLROnPlateau(opt, patience=1)
    normals = estimate_normals(src)
    mass, L = point_laplacian(src)
    print(f"  building {knn} kNN (k={k}) ...")
    nb = build_knn(src, k, mode=knn)

    model.train()
    running, n_r = 0.0, 0
    for i in range(steps):
        vel, weights = model(src)
        deformed = src + vel
        lap = None if L is None else (L @ vel).pow(2).sum(dim=-1)
        g = gesm_pc_loss(
            src, vel, normals, mass, lap, weights=weights, k=k,
            knn_idx=nb.idx, knn_pts=nb.knn,
        )
        fid = fidelity_rq(deformed, tgt, sigma=sigma)
        loss = lambda_g * g + lambda_fid * fid
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += float(loss)
        n_r += 1
        if i % 100 == 0:
            sched.step(running / max(n_r, 1))
            print(f"  step {i:4d}  loss={running / n_r:.4e}  G={float(g):.4e}  RQ={float(fid):.4e}")
            running, n_r = 0.0, 0

    model.eval()
    with torch.no_grad():
        vel, _ = model(src)
    return src + vel, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="deformed.ply")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--lambda-g", type=float, default=1e2)
    p.add_argument("--lambda-fid", type=float, default=1e4)
    p.add_argument("--knn", choices=("euclidean", "geodesic"), default="euclidean")
    p.add_argument("--no-adw", action="store_true")
    p.add_argument("--occ", action="store_true")
    args = p.parse_args()
    if args.occ:
        args.k, args.lambda_g, args.lambda_fid = 90, 1e3, 10.0

    src_n, _, _ = normalize_points(load_xyz(args.source))
    tgt_n, tgt_c, tgt_s = normalize_points(load_xyz(args.target))
    src, tgt = to_torch(src_n, DEVICE), to_torch(tgt_n, DEVICE)
    print(f"source {tuple(src.shape)}  target {tuple(tgt.shape)}  {DEVICE}")
    deformed, _ = train_pair(
        src, tgt, steps=args.steps, k=args.k,
        lambda_g=args.lambda_g, lambda_fid=args.lambda_fid,
        adw=not args.no_adw, knn=args.knn,
    )
    def_np = deformed.detach().cpu().numpy()
    print(f"NN-RMSE (normalized) = {nn_rmse(def_np, tgt_n):.6f}")
    save_xyz(args.out, denormalize(def_np, tgt_c, tgt_s))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
