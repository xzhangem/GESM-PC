"""
Compact GESM vs GESM-PC mesh check (no training).

Uses Hartman et al.'s own TestData/deformed_sphere (same connectivity):
template -> bend_X_theta and twist_X_theta.

Reports the three first-order components of face-wise GESM (getGabNorm)
and vertex-wise GESM-PC (_gesmpc_blocks with D_T = P J P + P J^T P,
D_N = N J P + P J^T N, (Dv)_bend = D_N), then Pearson correlation of
the two triples across samples. Identification experiment for the
current GESM-PC operators.

Usage (from H2_SurfaceMatch/, GPU as in the original code):
    python compare_gesm_pc.py
"""
import os
import sys
import zipfile
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enr.DDG import getMeshOneForms, getSurfMetric, getNormal
from enr.H2 import getGabNorm, _gesmpc_blocks, torchdeviceId, torchdtype
import utils.input_output as io

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(ROOT, "demo")
DATA = os.path.join(DEMO, "TestData", "deformed_sphere")
ZIP = os.path.join(DEMO, "TestData.zip")

PAIRS = [
    ("bend", 30), ("bend", 60), ("bend", 90), ("bend", 120),
    ("twist", 30), ("twist", 60), ("twist", 90),
]


def _ensure_data():
    if os.path.isfile(os.path.join(DATA, "template.ply")):
        return
    if not os.path.isfile(ZIP):
        raise FileNotFoundError("demo/TestData.zip not found")
    with zipfile.ZipFile(ZIP, "r") as z:
        z.extractall(DEMO)


def _to_torch(V, F):
    Vt = torch.from_numpy(np.asarray(V, dtype=np.float32)).to(
        dtype=torchdtype, device=torchdeviceId
    )
    Ft = torch.from_numpy(np.asarray(F, dtype=np.int64)).to(
        dtype=torch.long, device=torchdeviceId
    )
    return Vt, Ft


def gesm_face_components(V, h, F):
    """Face-wise GESM (a, b, c) of getGabNorm, no (b1-a1)/8 mix."""
    M1 = V + h
    alpha0 = getMeshOneForms(V, F)
    g0 = getSurfMetric(V, F)
    n0 = getNormal(F, V)
    xi = getMeshOneForms(M1, F) - alpha0
    dg = getSurfMetric(M1, F) - g0
    dn = getNormal(F, M1) - n0
    shear = getGabNorm(alpha0, xi, g0, dg, dn, 1.0, 0.0, 0.0, 0.0)
    scale = getGabNorm(alpha0, xi, g0, dg, dn, 0.0, 1.0, 0.0, 0.0)
    bend = getGabNorm(alpha0, xi, g0, dg, dn, 0.0, 0.0, 1.0, 0.0)
    return torch.stack([shear, scale, bend]).detach().cpu().numpy()


def gesmpc_vert_components(V, h, F):
    B = _gesmpc_blocks(V, h, F)
    A = B["areas"]
    shr = torch.sum(A * (B["DTshr"] * B["DTshr"]).sum(dim=(-1, -2)))
    scale = torch.sum(A * (0.5 * B["trT"] * B["trT"]))
    bend = torch.sum(A * (B["DNbend"] * B["DNbend"]).sum(dim=(-1, -2)))
    return torch.stack([shr, scale, bend]).detach().cpu().numpy()


def pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12))


def main():
    _ensure_data()
    V0, F0, _ = io.loadData(os.path.join(DATA, "template.ply"))
    V0, F0 = _to_torch(V0, F0)

    rows = []
    print("{:10s} {:>8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "type", "angle", "G-shr", "G-scale", "G-bend", "PC-shr", "PC-scale", "PC-bend"
    ))
    for kind, ang in PAIRS:
        path = os.path.join(DATA, "{}_X_{}.ply".format(kind, ang))
        V1, F1, _ = io.loadData(path)
        V1, F1 = _to_torch(V1, F1)
        if F1.shape[0] != F0.shape[0] or V1.shape[0] != V0.shape[0]:
            raise RuntimeError("connectivity mismatch: {}".format(path))
        h = V1 - V0
        g = gesm_face_components(V0, h, F0)
        pc = gesmpc_vert_components(V0, h, F0)
        rows.append((kind, ang, g, pc))
        print("{:10s} {:8d} {:10.4e} {:10.4e} {:10.4e} {:10.4e} {:10.4e} {:10.4e}".format(
            kind, ang, *g, *pc
        ))

    G = np.stack([r[2] for r in rows], axis=0)
    P = np.stack([r[3] for r in rows], axis=0)
    names = ("shear", "scale", "bend")
    print("\nPearson(GESM, GESM-PC) across the {} samples:".format(len(rows)))
    for k, name in enumerate(names):
        print("  {:6s}:  {:.3f}".format(name, pearson(G[:, k], P[:, k])))

    # Dominant component (row-normalized)
    print("\nArgmax component (row-normalized energies):")
    print("{:10s} {:>8s} {:>12s} {:>12s}".format("type", "angle", "GESM", "GESM-PC"))
    for (kind, ang, g, pc) in rows:
        gn = g / (g.sum() + 1e-12)
        pn = pc / (pc.sum() + 1e-12)
        print("{:10s} {:8d} {:>12s} {:>12s}".format(
            kind, ang, names[int(np.argmax(gn))], names[int(np.argmax(pn))]
        ))


if __name__ == "__main__":
    main()
