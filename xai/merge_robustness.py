r"""Merge the robustness chunks into one file and build the robustness figure."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("xai/out"); FIG = Path("xai/figs"); FIG.mkdir(exist_ok=True, parents=True)
parts = ["robust_p1.json", "robust_p2.json", "robust_p3.json"]
merged = None
for p in parts:
    d = json.load(open(OUT / p))
    if merged is None:
        merged = d
    else:
        merged["conditions"].update(d["conditions"])
json.dump(merged, open(OUT / "robustness.json", "w"), indent=2)
print(f"merged {len(merged['conditions'])} conditions -> xai/out/robustness.json")

TYPES = ["CS", "CC", "BW", "GNC", "GB", "JPEG", "VC"]
NAMES = merged["names"]; LEVELS = [1, 3, 5]
clean = merged["clean"]

# ---------------- summary table ----------------
rows = []
print(f"\n{'cond':<6} {'param':>7} | {'REAL P':>7} {'verdict':>8} | {'FAKE P':>7} {'verdict':>8} | err")
print("-" * 74)
print(f"{'CLEAN':<6} {'-':>7} | {clean['000']['p_final']:7.3f} {clean['000']['verdict']:>8} | "
      f"{clean['000_003']['p_final']:7.3f} {clean['000_003']['verdict']:>8} | 0")
n_fp = n_fn = 0
for t in TYPES:
    for lv in LEVELS:
        k = f"{t}_{lv}"
        if k not in merged["conditions"]: continue
        c = merged["conditions"][k]; r = c["results"]
        er = not r["000"]["correct"]; ef = not r["000_003"]["correct"]
        n_fp += er; n_fn += ef
        rows.append(dict(type=t, level=lv, param=c["param"],
                         real_p=r["000"]["p_final"], fake_p=r["000_003"]["p_final"],
                         real_ok=r["000"]["correct"], fake_ok=r["000_003"]["correct"],
                         real_pct_s=r["000"]["pct_spatial"],
                         fake_pct_s=r["000_003"]["pct_spatial"]))
        bad = ("FP" if er else "") + ("FN" if ef else "")
        print(f"{k:<6} {str(c['param']):>7} | {r['000']['p_final']:7.3f} "
              f"{r['000']['verdict']:>8} | {r['000_003']['p_final']:7.3f} "
              f"{r['000_003']['verdict']:>8} | {bad}")
print("-" * 74)
print(f"false positives (REAL->FAKE): {n_fp}/21   false negatives (FAKE->REAL): {n_fn}/21")

summary = dict(n_conditions=len(rows), false_positives=int(n_fp), false_negatives=int(n_fn),
               clean_real_p=clean["000"]["p_final"], clean_fake_p=clean["000_003"]["p_final"],
               rows=rows)
json.dump(summary, open(OUT / "robustness_summary.json", "w"), indent=2)

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
x = np.arange(len(TYPES))

ax = axes[0]
for lv, mk, al in zip(LEVELS, ["o", "s", "^"], [.45, .72, 1.0]):
    vr = [next((r["real_p"] for r in rows if r["type"] == t and r["level"] == lv), np.nan) for t in TYPES]
    ax.plot(x, vr, mk + "-", color="#2E6FD9", alpha=al, label=f"severity {lv}", ms=6)
ax.axhline(clean["000"]["p_final"], color="#2E6FD9", ls=":", lw=2, label="clean")
ax.axhline(.5, color="k", ls="--", lw=1.4)
ax.axhspan(.5, 1.0, color="#D9622E", alpha=.08)
ax.set_xticks(x); ax.set_xticklabels(TYPES); ax.set_ylim(0, 1)
ax.set_ylabel("P(FAKE)"); ax.set_title("REAL sample: anything above the\ndashed line is a FALSE POSITIVE", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[1]
for lv, mk, al in zip(LEVELS, ["o", "s", "^"], [.45, .72, 1.0]):
    vf = [next((r["fake_p"] for r in rows if r["type"] == t and r["level"] == lv), np.nan) for t in TYPES]
    ax.plot(x, vf, mk + "-", color="#D9622E", alpha=al, label=f"severity {lv}", ms=6)
ax.axhline(clean["000_003"]["p_final"], color="#D9622E", ls=":", lw=2, label="clean")
ax.axhline(.5, color="k", ls="--", lw=1.4)
ax.axhspan(0, .5, color="#2E6FD9", alpha=.08)
ax.set_xticks(x); ax.set_xticklabels(TYPES); ax.set_ylim(0, 1)
ax.set_title("FAKE sample: anything below the\ndashed line is a FALSE NEGATIVE", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[2]
w = .35
for i, (key, c, lb) in enumerate([("real_pct_s", "#2E6FD9", "REAL sample"),
                                  ("fake_pct_s", "#D9622E", "FAKE sample")]):
    v = [np.nanmean([r[key] for r in rows if r["type"] == t]) for t in TYPES]
    ax.bar(x + (i - .5) * w, v, w, color=c, label=lb, edgecolor="white")
ax.axhline(clean["000"]["pct_spatial"], color="#2E6FD9", ls=":", lw=2)
ax.axhline(clean["000_003"]["pct_spatial"], color="#D9622E", ls=":", lw=2)
ax.axhline(50, color="k", ls="--", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(TYPES); ax.set_ylabel("spatial share (%)")
ax.set_title("Branch reliance shifts under\ndegradation (dotted = clean)", fontsize=10.5)
ax.legend(fontsize=8)

fig.suptitle("Robustness: the authors' own 7 perturbations x 3 severities, with branch attribution",
             fontsize=12.5, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "f10_robustness.png", dpi=170, bbox_inches="tight", facecolor="white")
print("\nwrote f10_robustness.png")
