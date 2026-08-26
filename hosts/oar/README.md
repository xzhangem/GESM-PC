# OAR + GESM-PC

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
pip install -e .
python hosts/oar/register.py --source src.ply --target tgt.ply --out warped.ply
python hosts/oar/register.py --source src.ply --target tgt.ply --knn geodesic
python hosts/oar/register.py --source src.ply --target tgt.ply --no-adw
python hosts/oar/register.py --source src.ply --target tgt.ply --occ
```
