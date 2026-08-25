"""Drop-in pattern for the OAR training loop."""
import torch
from gesm_pc import projectors, operators, energy, knn_jacobian, rq_cd

torch.manual_seed(0)
N = 512
p = torch.randn(N, 3)
n = torch.nn.functional.normalize(p, dim=-1)
v = 0.05 * torch.randn(N, 3)
p_def = p + v

P, Nmat = projectors(n)
J = knn_jacobian(p, v, k=30)          # same δv = J δp as pc_DDG
DT, DN = operators(J, P, Nmat)        # D_T, D_N (two-sided P)
G = energy(DT, DN, a1=1.0, b1=1.0, c1=1.0, mass=None, P=P)
fid = rq_cd(p_def.unsqueeze(0), p.unsqueeze(0), alpha=3.0, sigma=1.0)

print(f"G     = {float(G):.6f}")
print(f"RQ-CD = {float(fid):.6f}")
trN = (DN[..., 0, 0] + DN[..., 1, 1] + DN[..., 2, 2]).abs().mean()
print(f"mean |Tr(DN)| = {float(trN):.2e}")
