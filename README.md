# GESM-PC

**General Elastic Shape Metric for Point Clouds** — an *extrinsic surrogate* of mesh [GESM](https://github.com/emmanuel-hartman/H2_SurfaceMatch) for unstructured points.

Mesh GESM splits first-order deformation into shear, scale and bend using the surface metric $g_q$ and mesh connectivity. Point clouds have neither. GESM-PC replaces $g_q$ by the projectors
$\mathbf{P}=\mathbf{I}-\mathbf{n}\mathbf{n}^{\top}$, $\mathbf{N}=\mathbf{n}\mathbf{n}^{\top}$ built from pointwise normals, and evaluates a first-order energy on a local Jacobian:

$$
D_T=\mathbf{P} d\mathbf{v} \mathbf{P}+\mathbf{P}(d\mathbf{v})^{\top}\mathbf{P},\qquad
D_N=\mathbf{N} d\mathbf{v} \mathbf{P}+\mathbf{P}(d\mathbf{v})^{\top}\mathbf{N}.
$$

Shear / scale / bend components of Jacobian $d\mathbf{v}$ are the Frobenius-orthogonal blocks as:

$$
(D\mathbf{v})_{shr} = D_T\mathbf{v}-\frac{1}{2}\mathrm{Tr}(D_T\mathbf{v})\mathbf{P}, \qquad
(D\mathbf{v})_{scale}=\frac{1}{2}\mathrm{Tr}(D_T\mathbf{v})\mathbf{P}, \qquad
(D\mathbf{v})_{bend}=D_N\mathbf{v}.
$$

---

## Repository layout

```
GESM-PC/
  gesm_pc/              # standalone energy + WLS Jacobian + RQ-CD
    __init__.py
    core.py          # projectors / knn_jacobian / operators / energy / rq_cd
  hosts/
    oar/                                # correspondence-free registration (FAUST / TOSCA / Open-CAS)
    graphscnet/experiments_gesm         # correspondence-based deformation graph (4DMatch)
    nfgp/                               # implicit handle-based editing
    4deform/                            # implicit interpolation
    H2_SurfaceMatch/                    # GESM-PC in mesh form using GESM shape analysis framework
  examples/
    quickstart.py
```

Each `hosts/*` tree is the **upstream code with GESM-PC wired in**. Keep the original license files. Core operators live in `gesm_pc/` so hosts do not fork four copies of $D_T,D_N$. 

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

```bash
pip install -e .
python examples/quickstart.py
```

---

## 1. OAR — correspondence-free registration
**Upstream:** [OAR](https://github.com/zikai1/OAReg).
Correspondence-free test-time registration. OAR's SIREN is kept; LLR and MMC are replaced by GESM-PN and RQ-CD.

```
hosts/oar/
  register.py
  model.py        # SirenV / SirenVW
  energy_oar.py
  knn.py          # --knn euclidean | geodesic
  io_ply.py
```

From the repository root:

```bash
cd hosts/oar
pip install -e .
python register.py --source src.ply --target tgt.ply --out warped.ply
python register.py --source src.ply --target tgt.ply --knn geodesic
python register.py --source src.ply --target tgt.ply --no-adw
python register.py --source src.ply --target tgt.ply --occ
```


**Data:** put FAUST / TOSCA / Open-CAS under `hosts/oar/data/` as required by upstream. We do not redistribute those datasets.

---

## 2. GraphSCNet — correspondence-based deformation graph

**Upstream:** [GraphSCNet](https://github.com/qinzheng93/GraphSCNet)  
**What changed:** ARAP deformation-graph energy → GESM-PC on node Jacobians.
Here please download the original [GraphSCNet](https://github.com/qinzheng93/GraphSCNet) and set up the suggested enviornment. Then use graphscnet/experiments_gesm to replace its original experiments file. 
```bash
# 4DMatch
CUDA_VISIBLE_DEVICES=0 python test.py --test_epoch=EPOCH --benchmark=4DMatch-F
# 4DLoMatch
CUDA_VISIBLE_DEVICES=0 python test.py --test_epoch=EPOCH --benchmark=4DLoMatch-F
```


---

## 3. NFGP — implicit handle editing

**Upstream:** [NFGP](https://github.com/stevenygd/NFGP) 
**Tasks:** handle-based edits (e.g. Jolteon)
**What changed:** 
Add gesm_pc_losses.py to NFGP/trainers/losses/
Exchange implicit_deform.py with the original NFGP/trainers/implicit_deform.py
Exchange train.py with the original NFGP/train.py
Add jolteon_jump_gesm.yaml to NFGP/configs/deformation/

```bash
cd hosts/nfgp
python train.py configs/deformation/jolteon_jump_gesm.yaml
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
**What changed:** vertex-wise GESM-PC (`enr/H2.py`, `enr/DDG.py`) next to face-wise `getGabNorm`.

Direct comparsions between GESM-PC and GESM using deformed_sphere samples:
```bash
cd hosts/H2_SurfaceMatch
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
  note    = {under review}
}
```

Please also cite the host methods you run (OAR, GraphSCNet, NFGP, 4Deform, Hartman GESM) and Su et al. / Bauer et al. for the GESM family.

---

## Licenses


---

## Acknowledgements

This implementation builds on mesh GESM ([Su et al.](https://arxiv.org/); [Hartman et al.](https://github.com/emmanuel-hartman/H2_SurfaceMatch)) and on the public neural deformation codebases listed above. GESM-PC is a drop-in energy, not a replacement for those systems.
