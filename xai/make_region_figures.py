r"""Figures for the region-based (patch-defined) attribution."""
import json, sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, ".")
from xai.engine import FACE_PARTS

OUT, FIG = Path("xai/out"), Path("xai/figs")
A = np.load(OUT / "regions.npz", allow_pickle=True)
R = json.load(open(OUT / "regions.json"))
GRID, CROP = 16, 150
VIDS = ["000", "000_003"]
SHORT = {"000": "REAL sample", "000_003": "FAKE sample"}
REGIONS = ["eyes", "nose", "lips", "skin"]
RCOL = {"eyes": "#2E6FD9", "nose": "#D9622E", "lips": "#8E44AD",
        "skin": "#1E7A3C", "background": "#AAAAAA"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})


def face(v):
    cap = cv2.VideoCapture(f"resources/cropped/videos/{v}.avi")
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.mean(np.stack(fr[:100]), 0).astype(np.uint8)


def save(fig, n):
    fig.savefig(FIG / n, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  wrote", n)


# ---------------- F11: the patch regions + per-region spatial attribution ----
fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))
order = ["background", "skin", "lips", "nose", "eyes"]
cmap = ListedColormap([RCOL[o] for o in order])

ax = axes[0]
rmap = A["000/region_map"].astype(str)
idx = np.vectorize({o: i for i, o in enumerate(order)}.get)(rmap)
ax.imshow(face("000"), extent=[0, CROP, CROP, 0])
ax.imshow(idx, extent=[0, CROP, CROP, 0], cmap=cmap, alpha=.55,
          vmin=0, vmax=len(order) - 1)
lm = A["000/landmarks"]
ax.scatter(lm[:, 0], lm[:, 1], s=3, c="k", alpha=.8)
for r in range(GRID + 1):
    ax.axhline(r * CROP / GRID, color="w", lw=.3, alpha=.5)
    ax.axvline(r * CROP / GRID, color="w", lw=.3, alpha=.5)
ax.set_title("Step 1: cut the 16x16 patch grid\ninto landmark-defined regions", fontsize=10.5)
ax.axis("off"); ax.grid(False)
hs = [plt.Rectangle((0, 0), 1, 1, fc=RCOL[o]) for o in order]
ax.legend(hs, [f"{o} ({int((rmap==o).sum())}p)" for o in order],
          fontsize=7.5, loc="lower center", ncol=3, framealpha=.9)

for ax, v in zip(axes[1:3], VIDS):
    sp = A[f"{v}/sp_patch"].reshape(GRID, GRID)
    ax.imshow(face(v), extent=[0, CROP, CROP, 0])
    hm = cv2.resize(sp, (CROP, CROP), interpolation=cv2.INTER_CUBIC)
    m = np.abs(sp).max()
    im = ax.imshow(hm, extent=[0, CROP, CROP, 0], cmap="RdBu_r",
                   vmin=-m, vmax=m, alpha=.62)
    ax.set_title(f"Step 2: {SHORT[v]}\nSPATIAL contribution per patch", fontsize=10.5)
    ax.axis("off"); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.046)

ax = axes[3]
x = np.arange(len(REGIONS)); w = .38
for i, v in enumerate(VIDS):
    vals = [R[v]["spatial_region_share_pct"][r] for r in REGIONS]
    b = ax.bar(x + (i - .5) * w, vals, w,
               color=["#2E6FD9", "#D9622E"][i], label=SHORT[v], edgecolor="white")
    ax.bar_label(b, fmt="%.1f%%", fontsize=8, padding=1)
ax.set_xticks(x); ax.set_xticklabels(REGIONS)
ax.set_ylabel("share of spatial evidence (%)")
ax.set_title("Step 3: per-region share\n(exact, sums to 100%)", fontsize=10.5)
ax.legend(fontsize=8.5); ax.set_ylim(0, 68)
fig.suptitle("Region-based attribution: patches grouped by facial landmarks, "
             "then each region's exact contribution",
             fontsize=12.5, fontweight="bold")
fig.tight_layout()
save(fig, "f11_regions.png")

# ---------------- F12: cross-validation + spatial vs temporal per region ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
x = np.arange(len(REGIONS)); w = .38
qshare = {}
for v in VIDS:
    q = R[v]["query_contrib"]
    tot = sum(abs(q[p]) for p in FACE_PARTS)
    qshare[v] = {p: 100 * abs(q[p]) / tot for p in FACE_PARTS}
for i, v in enumerate(VIDS):
    rs = [R[v]["spatial_region_share_pct"][r] for r in REGIONS]
    qs = [qshare[v][r] for r in REGIONS]
    ax.plot(x, rs, "o-", color=["#2E6FD9", "#D9622E"][i], lw=2, ms=7,
            label=f"{SHORT[v]} - by PATCH region")
    ax.plot(x, qs, "s--", color=["#2E6FD9", "#D9622E"][i], lw=1.5, ms=6,
            alpha=.55, label=f"{SHORT[v]} - by model QUERY")
ax.set_xticks(x); ax.set_xticklabels(REGIONS)
ax.set_ylabel("share (%)")
ax.set_title("Cross-validation: two independent\ndecompositions agree on the ranking", fontsize=10.5)
ax.legend(fontsize=7.5)

ax = axes[1]
for i, (key, lb) in enumerate([("spatial_region_share_pct", "SPATIAL"),
                               ("temporal_region_share_pct", "TEMPORAL")]):
    vals = [R["000_003"][key][r] for r in REGIONS]
    b = ax.bar(x + (i - .5) * w, vals, w,
               color=["#2E6FD9", "#D9622E"][i], label=lb, edgecolor="white")
    ax.bar_label(b, fmt="%.1f", fontsize=8, padding=1)
ax.set_xticks(x); ax.set_xticklabels(REGIONS)
ax.set_ylabel("share within branch (%)")
ax.set_title("FAKE sample: the two branches read\nDIFFERENT regions", fontsize=10.5)
ax.legend(fontsize=8.5)

ax = axes[2]
for i, (key, lb) in enumerate([("spatial_region_share_pct", "SPATIAL"),
                               ("temporal_region_share_pct", "TEMPORAL")]):
    vals = [R["000"][key][r] for r in REGIONS]
    b = ax.bar(x + (i - .5) * w, vals, w,
               color=["#2E6FD9", "#D9622E"][i], label=lb, edgecolor="white")
    ax.bar_label(b, fmt="%.1f", fontsize=8, padding=1)
ax.set_xticks(x); ax.set_xticklabels(REGIONS)
ax.set_ylabel("share within branch (%)")
ax.set_title("REAL sample: nose carries appearance,\nskin carries motion", fontsize=10.5)
ax.legend(fontsize=8.5)
fig.suptitle("Validating the region attribution and comparing the two branches "
             "on identical region definitions", fontsize=12.5, fontweight="bold")
fig.tight_layout()
save(fig, "f12_region_validation.png")

# ---------------- summary table to stdout ----------------
print("\n--- SPATIAL branch, share of evidence by PATCH REGION ---")
print(f'{"region":<12}{"REAL":>10}{"FAKE":>10}   {"REAL signed":>13}{"FAKE signed":>13}')
for r in REGIONS:
    print(f'{r:<12}{R["000"]["spatial_region_share_pct"][r]:>9.1f}%'
          f'{R["000_003"]["spatial_region_share_pct"][r]:>9.1f}%'
          f'{R["000"]["spatial_region_signed"][r]:>14.4f}'
          f'{R["000_003"]["spatial_region_signed"][r]:>13.4f}')
print("\n--- TEMPORAL branch, same regions ---")
for r in REGIONS:
    print(f'{r:<12}{R["000"]["temporal_region_share_pct"][r]:>9.1f}%'
          f'{R["000_003"]["temporal_region_share_pct"][r]:>9.1f}%')
print("\n--- query x region cross-table (FAKE sample) ---")
print(f'{"query":<8}' + "".join(f'{r:>11}' for r in REGIONS + ["background"]))
for q in FACE_PARTS:
    c = R["000_003"]["query_region_cross"][q]
    print(f'{q:<8}' + "".join(f'{c[r]:>11.4f}' for r in REGIONS + ["background"]))
