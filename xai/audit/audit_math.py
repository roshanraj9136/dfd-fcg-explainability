r"""Audit: exactness of the attribution algebra + faithfulness of claims."""
import sys, math, torch, torch.nn as nn, numpy as np
sys.path.insert(0, ".")
from xai.engine import _ln_linear_decomp, shapley_two_branch, FACE_PARTS

torch.manual_seed(0)
FAIL = []
def chk(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond: FAIL.append(name)

print("=" * 78)
print("A1. LayerNorm+Linear additive decomposition vs real nn modules")
print("=" * 78)
for D, K in [(1024, 4), (256, 24), (37, 5)]:
    ln = nn.LayerNorm(D); head = nn.Linear(D, 2)
    nn.init.normal_(ln.weight, 1.0, .3); nn.init.normal_(ln.bias, 0, .3)
    comp = torch.randn(K, D) * 2.0
    y = comp.sum(0)
    # ground truth through the REAL modules
    with torch.no_grad():
        lg = head(ln(y.float()))
    d_true = (lg[1] - lg[0]).item()
    contribs, const, d_tot = _ln_linear_decomp(comp, ln, head)
    chk(f"D={D},K={K} reconstruction", abs(d_tot - d_true) < 1e-8,
        f"decomp={d_tot:+.10f} true={d_true:+.10f} err={abs(d_tot-d_true):.2e}")
    chk(f"D={D},K={K} additivity", abs(contribs.sum() + const - d_tot) < 1e-10,
        f"sum+const={contribs.sum()+const:+.10f}")

print()
print("=" * 78)
print("A2. THE DIAG TRICK: per-dimension attribution (temporal per-patch)")
print("=" * 78)
D = 256
ln = nn.LayerNorm(D); head = nn.Linear(D, 2)
nn.init.normal_(ln.weight, 1.0, .3); nn.init.normal_(ln.bias, 0, .3)
y = torch.randn(D) * 3
with torch.no_grad():
    lg = head(ln(y.float()))
d_true = (lg[1] - lg[0]).item()
c_diag, const, d_tot = _ln_linear_decomp(torch.diag(y.double()), ln, head)
chk("diag() total reconstructs d", abs(d_tot - d_true) < 1e-8,
    f"{d_tot:+.10f} vs {d_true:+.10f}")
chk("diag() rows sum to y", torch.allclose(torch.diag(y.double()).sum(0), y.double()),
    "")
# closed form for per-dim contribution
g = ln.weight.detach().double(); b = ln.bias.detach().double()
W = head.weight.detach().double(); w = W[1] - W[0]
mu = y.double().mean(); sig = torch.sqrt(y.double().var(unbiased=False) + ln.eps)
manual = ((y.double() * w * g) - (y.double() / D) * (w * g).sum()) / sig
chk("diag() matches closed form", np.allclose(c_diag, manual.numpy(), atol=1e-12),
    f"max dev {np.abs(c_diag-manual.numpy()).max():.2e}")
# leave-one-out semantic cross-check: zeroing dim j changes d by ~contrib (not exact
# because mu/sigma shift), so only check RANK correlation is sensible
print("      note: per-patch values are an exact additive split, not LOO deltas.")

print()
print("=" * 78)
print("A3. sigma computed from TOTAL y (not per component) + eps placement")
print("=" * 78)
src = open("xai/engine.py").read()
chk("sigma uses total y", "var = y.var(unbiased=False)" in src and "sig = torch.sqrt(var + eps)" in src, "")
chk("biased variance (matches nn.LayerNorm)", "unbiased=False" in src, "")
ln2 = nn.LayerNorm(8); yv = torch.randn(8)
ref = ((yv - yv.mean()) / torch.sqrt(yv.var(unbiased=False) + ln2.eps)) * ln2.weight + ln2.bias
chk("manual LN == nn.LayerNorm", torch.allclose(ref, ln2(yv), atol=1e-6),
    f"max dev {(ref-ln2(yv)).abs().max().item():.2e}")

print()
print("=" * 78)
print("A4. softmax(2-class) == sigmoid(logit difference)")
print("=" * 78)
for _ in range(3):
    lg = torch.randn(2) * 4
    p_soft = lg.softmax(-1)[1].item()
    p_sig = 1 / (1 + math.exp(-(lg[1] - lg[0]).item()))
    chk("softmax==sigmoid(d)", abs(p_soft - p_sig) < 1e-12, f"{p_soft:.12f} vs {p_sig:.12f}")

print()
print("=" * 78)
print("A5. Shapley: 2-player formula + efficiency")
print("=" * 78)
for ps, pt, pf in [(.93, .90, .94), (.47, .41, .42), (.2, .9, .55)]:
    a, b = shapley_two_branch(ps, pt, pf)
    chk(f"efficiency (ps={ps},pt={pt},pf={pf})", abs((a + b) - (pf - .5)) < 1e-12,
        f"phi_s+phi_t={a+b:+.6f} vs pf-0.5={pf-.5:+.6f}")
# textbook check: phi_i = sum over orderings
def brute(ps, pt, pf, prior=.5):
    # orderings: (S,T) and (T,S)
    phi_s = .5 * ((ps - prior) + (pf - pt))
    phi_t = .5 * ((pt - prior) + (pf - ps))
    return phi_s, phi_t
a, b = shapley_two_branch(.93, .90, .94); c, d = brute(.93, .90, .94)
chk("matches brute-force ordering average", abs(a - c) < 1e-15 and abs(b - d) < 1e-15, "")

print()
print("=" * 78)
print("A6. spatial re-implementation equals original mean() reduction")
print("=" * 78)
b_, t_, q_, w_ = 2, 10, 4, 64
s_mix = torch.randn(b_, t_, q_, w_).double()
orig = s_mix.flatten(1, 2).mean(dim=1)
mine = (s_mix.sum(dim=1) / (t_ * q_)).sum(dim=1)
chk("flatten(1,2).mean == sum/(t*q) then sum_q",
    torch.allclose(orig, mine, atol=1e-12),
    f"max dev {(orig-mine).abs().max().item():.2e}")

print()
print("=" * 78)
print("RESULT:", "ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}")
print("=" * 78)
