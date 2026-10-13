r"""Audit v2: same checks, but reference computed in float64 so the test
tolerance measures LOGIC error rather than float32 rounding."""
import sys, math, torch, torch.nn as nn, numpy as np
sys.path.insert(0, ".")
from xai.engine import _ln_linear_decomp, shapley_two_branch

torch.manual_seed(0)
FAIL = []
def chk(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond: FAIL.append(name)

print("=" * 78)
print("B1. decomposition vs FLOAT64 reference (logic test)")
print("=" * 78)
for D, K in [(1024, 4), (256, 24), (256, 256), (37, 5)]:
    ln = nn.LayerNorm(D); head = nn.Linear(D, 2)
    nn.init.normal_(ln.weight, 1.0, .3); nn.init.normal_(ln.bias, 0, .3)
    comp = torch.randn(K, D) * 2.0
    y = comp.sum(0).double()
    # float64 reference, replicating nn.LayerNorm exactly
    g = ln.weight.detach().double(); b = ln.bias.detach().double()
    mu = y.mean(); sig = torch.sqrt(y.var(unbiased=False) + ln.eps)
    yn = (y - mu) / sig * g + b
    W = head.weight.detach().double(); bb = head.bias.detach().double()
    lg = W @ yn + bb
    d_true = (lg[1] - lg[0]).item()
    contribs, const, d_tot = _ln_linear_decomp(comp, ln, head)
    err = abs(d_tot - d_true)
    chk(f"D={D:4d} K={K:3d} reconstruction", err < 1e-9,
        f"err={err:.2e}  (d={d_tot:+.12f})")
    chk(f"D={D:4d} K={K:3d} additivity", abs(contribs.sum() + const - d_tot) < 1e-10, "")

print()
print("=" * 78)
print("B2. diag trick vs float64 reference (per-patch temporal attribution)")
print("=" * 78)
for D in (256, 64):
    ln = nn.LayerNorm(D); head = nn.Linear(D, 2)
    nn.init.normal_(ln.weight, 1.0, .3); nn.init.normal_(ln.bias, 0, .3)
    y = (torch.randn(D) * 3).double()
    g = ln.weight.detach().double(); b = ln.bias.detach().double()
    mu = y.mean(); sig = torch.sqrt(y.var(unbiased=False) + ln.eps)
    yn = (y - mu) / sig * g + b
    W = head.weight.detach().double(); bb = head.bias.detach().double()
    lg = W @ yn + bb; d_true = (lg[1] - lg[0]).item()
    c, const, d_tot = _ln_linear_decomp(torch.diag(y), ln, head)
    chk(f"D={D} diag total", abs(d_tot - d_true) < 1e-9,
        f"err={abs(d_tot-d_true):.2e}")
    chk(f"D={D} diag count", len(c) == D, f"{len(c)} values")
    # exact closed form
    w = W[1] - W[0]
    manual = ((y * w * g) - (y / D) * (w * g).sum()) / sig
    chk(f"D={D} diag closed form", np.allclose(c, manual.numpy(), atol=1e-14),
        f"max dev {np.abs(c-manual.numpy()).max():.2e}")

print()
print("=" * 78)
print("B3. softmax == sigmoid(d) in float64")
print("=" * 78)
for _ in range(4):
    lg = (torch.randn(2) * 4).double()
    p_soft = lg.softmax(-1)[1].item()
    p_sig = 1 / (1 + math.exp(-(lg[1] - lg[0]).item()))
    chk("softmax==sigmoid(d)", abs(p_soft - p_sig) < 1e-14,
        f"err={abs(p_soft-p_sig):.2e}")

print()
print("=" * 78)
print("B4. mean-additivity premise: mu(y) == sum_k mu(y_k)")
print("=" * 78)
comp = torch.randn(7, 128).double()
lhs = comp.sum(0).mean().item()
rhs = comp.mean(dim=1).sum().item()
chk("mu additive over components", abs(lhs - rhs) < 1e-13,
    f"{lhs:.14f} vs {rhs:.14f}")

print()
print("=" * 78)
print("RESULT:", "ALL LOGIC CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}")
print("=" * 78)
