# GESM-PC

**General Elastic Shape Metric for Point Clouds** — an *extrinsic surrogate* of mesh [GESM](https://github.com/emmanuel-hartman/H2_SurfaceMatch) for unstructured points.

Mesh GESM splits first-order deformation into shear, scale and bend using the surface metric $g_q$ and mesh connectivity. Point clouds have neither. GESM-PC replaces $g_q$ by the projectors $\mathbf{P}=I-\mathbf{n}\mathbf{n}^{\top}$, $\mathbf{N}=\mathbf{n}\mathbf{n}^{\top}$ built from pointwise normals, and evaluates a first-order energy on a local Jacobian:

$$
D_T (\mathbf{v}) = \mathbf{P}\ d\mathbf{v}\ \mathbf{P} + \mathbf{P} (d\mathbf{v})^{\top} \mathbf{P},
\qquad
D_N (\mathbf{v}) = \mathbf{N}\ d\mathbf{v}\ \mathbf{P} + \mathbf{P} (d\mathbf{v})^{\top} \mathbf{N}.
$$

Shearing / scaling /bending component w.r.t. $d\mathbf{v}$ is 

$$
(D \mathbf{v})_{shr} = D_T \mathbf{v} - \frac{1}{2} \mathrm{Tr}(D_T \mathbf{v}) \mathbf{P},
\qquad
(D \mathbf{v})_{scale} = \frac{1}{2} \mathrm{Tr}(D_T \mathbf{v}) \mathbf{P},
\qquad
(D \mathbf{v})_{bend} = D_N \mathbf{v}.
$$

The elastic energy then is $<(D\mathbf{v})_{shr}, (D\mathbf{v})_{shr}>_F$

Paper: *GESM-PC: General Elastic Shape Metric for Point Cloud Shape Analysis with Neural Deformation Representations* (under review).

---

## Repository layout

```
GESM-PC/
  gesm_pc/              # standalone energy + WLS Jacobian + RQ-CD
  hosts/
    oar/                # correspondence-free registration (FAUST / TOSCA / Open-CAS)
    graphscnet/         # correspondence-based deformation graph (4DMatch)
    nfgp/               # implicit handle-based editing
    4deform/            # implicit interpolation
  examples/
    mesh_surrogate/     # face-wise GESM vs vertex-wise GESM-PC (Hartman sphere)
  third_party/          # unmodified upstream READMEs / licenses
```

Each `hosts/*` tree is the **upstream code with GESM-PC wired in**. Keep the original license files. Core operators live in `gesm_pc/` so hosts do not fork four copies of \(D_T,D_N\).

---

## Install

```bash
git clone https://github.com/xzhangem/GESM-PC.git
cd GESM-PC
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
pip install -e .
```

- Python ≥ 3.8, PyTorch (CUDA optional).
- Hosts may need extra packages; see each subsection.

Default deformation weights are **frozen and uniform** (the setting covered by the theorems). Adaptive Softmax weights, if enabled in a host, are a training convenience for matching only.

---

## Quick start (the energy only)

```python
import torch
from gesm_pc import projectors, operators, energy, knn_jacobian, rq_cd

p, v, n = ...          # [N,3] points, velocities, unit normals
P, N = projectors(n)   # P = I - nn^T, N = nn^T
J = knn_jacobian(p, v, k=30)          # ambient WLS; pinv on the 3x3 Gram
# D_T, D_N already contain *P; feeding J or J @ P gives the same G
DT, DN = operators(J, P, N)
G = energy(DT, DN, a1=1.0, b1=1.0, c1=1.0, mass=None)
fid = rq_cd(p_deformed, p_target, alpha=3.0, sigma=1.0)  # optional
```

`energy` returns \(\sum_i m_i(a_1\|(Dv)_{\mathrm{shr}}\|_F^2 + b_1\tfrac12\mathrm{Tr}(D_T)^2 + c_1\|D_N\|_F^2)\). Laplacian smoothing \(a_2\) is host-specific (cotangent on meshes, kNN on clouds).

---

## 1. OAR — correspondence-free registration

**Upstream:** [OAR](https://github.com/) *(fill official URL)*  
**Tasks:** FAUST, TOSCA (Tables II–III); Open-CAS occlusion (Table IV).  
**What changed:** LLR regularizer → GESM-P / GESM-PN; optional MMC → RQ-CD.

```bash
cd hosts/oar
# env: follow upstream OAR README, then
pip install -r requirements.txt
```

| Script | Setting | Notes |
|---|---|---|
| `scripts/run_faust.py` | \(K=30\), \(\lambda_1=10^2\), \(\lambda_2=10^4\) | intra / inter RMSE |
| `scripts/run_tosca.py` | same | Cat / Centaur / Dog / Gorilla |
| `scripts/run_opencas.py` | default \(K=30\); `--occ` sets \(K=90\), \(\lambda_1=10^3\), \(\lambda_2=10\) | EPE / AccS / AccR / N-CD |

Useful flags (names are placeholders — match your argparse):

```bash
python scripts/run_faust.py --energy pn          # GESM-PN (default)
python scripts/run_faust.py --energy p           # GESM-P, no normals
python scripts/run_faust.py --no-adw             # frozen uniform weights
python scripts/run_faust.py --fidelity mmc       # OAR MMC instead of RQ-CD
python scripts/run_opencas.py --occ --no-adw
```

Same-host protocol: SIREN, sampling and \(\lambda\) stay as in OAR; only the deformation energy (and optionally the fidelity) change. PN w/o ADW and PN w/o RQ-CD are the ablations in the paper.

**Data:** put FAUST / TOSCA / Open-CAS under `hosts/oar/data/` as required by upstream. We do not redistribute those datasets.

---

## 2. GraphSCNet — correspondence-based deformation graph

**Upstream:** [GraphSCNet](https://github.com/) *(fill URL)* + Lepard correspondences  
**Tasks:** 4DMatch / 4DLoMatch.  
**What changed:** ARAP deformation-graph energy → GESM-PC on node Jacobians.

```bash
cd hosts/graphscnet
pip install -r requirements.txt
python scripts/eval_4dmatch.py --energy pn     # or p
python scripts/eval_4dmatch.py --split 4dlomatch
```

Node head outputs a \(3\times 3\) Jacobian (not a 6D rigid motion). This is **not** a matched-architecture swap with the ARAP baseline; OAR is the matched-architecture experiment. Report inlier ratio / RMSE as in GraphSCNet.

---

## 3. NFGP — implicit handle editing

**Upstream:** [NFGP](https://github.com/) *(fill URL)*  
**Tasks:** handle-based edits (e.g. Jolteon).  
**What changed:** NFGP stretching / Hessian bending losses → GESM-PN on the residual-network Jacobian; no Laplacian term (level-set already smooth).

```bash
cd hosts/nfgp
python edit.py --shape jolteon --mode scale+bend --w-bend 1e-3 --w-scale 1e-1
python edit.py --shape jolteon --mode shear+bend --w-bend 1e-3 --w-shear 1e-1
```

Normals and Jacobians come from the SDF (`n = \nabla\phi/\|\nabla\phi\|`). Weights here are **frozen** (editing axes), not the matching ADW head.

---

## 4. 4Deform — implicit interpolation

**Upstream:** [4Deform](https://github.com/) *(fill URL)*  
**Tasks:** 4D-Dress / interpolation (SA\(\sigma\)).  
**What changed:** distortion + stretching energy → GESM-PN.

```bash
cd hosts/4deform
python interpolate.py --energy gesm-pn --dataset 4d-dress
```

---

## 5. Mesh surrogate (GESM vs GESM-PC)

**Upstream:** [H2_SurfaceMatch](https://github.com/emmanuel-hartman/H2_SurfaceMatch) (GPL-3.0)  
**Task:** Table of unweighted \((g_{\mathrm{shr}}, g_{\mathrm{scale}}, g_{\mathrm{bend}})\) on `deformed_sphere`.  
**What changed:** vertex-wise GESM-PC (`enr/H2.py`, `enr/DDG.py`) next to face-wise `getGabNorm`.

```bash
cd examples/mesh_surrogate   # or hosts/h2_surfacematch
python compare_gesm_pc.py
```

Uses Hartman’s `demo/TestData/deformed_sphere` (`template` → `bend_X_*`, `twist_X_*`). Prints the seven pairs and Pearson correlations. This supports *empirical tracking* of the three densities, not numerical equality of mesh GESM and GESM-PC.

Please keep Hartman’s `LICENSE` and cite [Hartman et al., IJCV 2023](https://github.com/emmanuel-hartman/H2_SurfaceMatch).

---

## Weights and fidelity (read this before training)

| Quantity | Role in the paper | Default in this repo |
|---|---|---|
| \(a_1,b_1,c_1,a_2\) | GESM family coefficients; theorems assume they are frozen | uniform after Softmax |
| ADW (Softmax head) | matching-stage convenience only | **off** unless `--adw` |
| RQ-CD \(\alpha=3,\sigma=1\) | empirical Chamfer kernel | OAR default; disable with `--fidelity mmc` |
| \(K\) | WLS stencil | 30; Open-CAS Occ uses 90 |

Karcher means and tangent PCA must use a **frozen global** \((a_1,b_1,c_1,a_2)\).

---

## Citation

```bibtex
@article{zhang2026gesmpc,
  title   = {GESM-PC: General Elastic Shape Metric for Point Cloud Shape Analysis with Neural Deformation Representations},
  year    = {2026},
  note    = {under review}
}
```

Please also cite the host methods you run (OAR, GraphSCNet, NFGP, 4Deform, Hartman GESM) and Su et al. / Bauer et al. for the GESM family.

---



## Acknowledgements

This implementation builds on mesh GESM ([Su et al.](https://arxiv.org/); [Hartman et al.](https://github.com/emmanuel-hartman/H2_SurfaceMatch)) and on the public neural deformation codebases listed above. GESM-PC is a drop-in energy, not a replacement for those systems.

