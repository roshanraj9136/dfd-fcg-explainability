r"""
Build every figure for the deck plus a derived-metrics JSON
(zone densities, REAL->FAKE deltas, single-branch counterfactuals).

    python xai/make_figures.py
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, ".")
from xai.engine import FACE_PARTS

OUT = Path("xai/out")
FIG = Path("xai/figs")
FIG.mkdir(parents=True, exist_ok=True)
GRID, CROP = 16, 150
ZONES = ["eyes", "brows", "nose", "mouth", "jaw/contour", "cheek/forehead", "background"]
VIDS = ["000", "000_003"]
TITLE = {"000": "000.mp4  (ground truth REAL)", "000_003": "000_003.mp4  (ground truth FAKE)"}
SHORT = {"000": "REAL sample", "000_003": "FAKE sample"}
C_SP, C_TP, C_FU, C_FI = "#2E6FD9", "#D9622E", "#8E44AD", "#1C1C1C"

A = np.load(OUT / "arrays.npz", allow_pickle=True)
REP = json.load(open(OUT / "report.json"))
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})


def face_image(vid):
    cap = cv2.VideoCapture(f"resources/cropped/videos/{vid}.avi")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.mean(np.stack(frames[:100]), axis=0).astype(np.uint8)


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


# ------------------------------------------------------------------ F1 arch
def f1_architecture():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, txt, fc, ec="#333", fs=10, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.45)

    def arrow(x1, y1, x2, y2, c="#444", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, lw=1.5, color=c, ls=ls))

    box(.15, 2.45, 1.5, 1.3, "10 frames\naligned face\n224x224", "#F2F2F2", fs=9.5)
    box(2.0, 1.4, 1.95, 3.4,
        "FROZEN\nCLIP ViT-L/14\n24 layers\n\n303.2 M params\n(0 trained)",
        "#DDE8F7", fs=9.5, bold=True)
    ax.text(2.97, 1.05, "per layer: q, k, v, emb", ha="center", fontsize=8.5, style="italic")
    arrow(1.68, 3.1, 1.97, 3.1)

    box(4.35, 3.45, 3.0, 1.45,
        "SPATIAL branch  (S)\n4 component queries x cosine-attn\n-> y_s   (1024-d)", "#CFE0F8", fs=9.5)
    box(4.35, 1.3, 3.0, 1.45,
        "TEMPORAL branch  (T)\nframe-to-frame affinity + convs\n-> y_t   (256-d = 16x16)", "#F8DCCB", fs=9.5)
    arrow(3.97, 3.6, 4.33, 4.15); arrow(3.97, 2.6, 4.33, 2.05)
    ax.text(5.85, 5.03, "side network: 0.25 M trainable params (0.08%)",
            ha="center", fontsize=9, style="italic", color="#333")

    box(7.75, 4.0, 1.55, .85, "s_head\nlogits_s", "#CFE0F8", fs=9)
    box(7.75, 1.35, 1.55, .85, "t_head\nlogits_t", "#F8DCCB", fs=9)
    box(7.75, 2.65, 1.55, .85, "a_head\nlogits_a", "#E8D5F2", fs=9)
    arrow(7.37, 4.17, 7.73, 4.42); arrow(7.37, 2.02, 7.73, 1.77)
    arrow(7.37, 3.9, 7.73, 3.3, c="#777", ls="--"); arrow(7.37, 1.9, 7.73, 2.9, c="#777", ls="--")

    box(9.75, 2.5, 1.5, 1.2, "ENSEMBLE\nmean of the\n3 softmaxes", "#FFF3C4", fs=9.5, bold=True)
    for y in (4.42, 3.07, 1.77):
        arrow(9.32, y, 9.73, 3.1)
    box(11.5, 2.65, 1.35, .9, "P(FAKE)", "#D7F2D7", fs=10.5, bold=True)
    arrow(11.27, 3.1, 11.48, 3.1)

    ax.text(6.5, .35,
            r"$P_{final} = \frac{1}{3}\left[\sigma(logits_s)+\sigma(logits_t)+\sigma(logits_a)\right]$"
            "\nthe output is an explicit average of the three heads -> branch shares are exact, not estimated",
            ha="center", fontsize=10.5, linespacing=1.6)
    ax.set_title("DFD-FCG architecture and the three decision heads we decompose",
                 fontsize=13, fontweight="bold", pad=8)
    save(fig, "f1_architecture.png")


# ------------------------------------------------------- F2 verdict / time
def f2_verdicts():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, v in zip(axes, VIDS):
        st = A[f"{v}/clip_starts"]; t = st / REP["videos"][v]["fps"]
        ax.axhspan(.5, 1.0, color="#D9622E", alpha=.07)
        ax.axhspan(.0, .5, color="#2E6FD9", alpha=.07)
        ax.axhline(.5, color="k", ls="--", lw=1.2, label="decision threshold 0.5")
        ax.plot(t, A[f"{v}/p_final"], color=C_FI, lw=2.2, label="final  P(FAKE)")
        ax.plot(t, A[f"{v}/p_spatial"], color=C_SP, lw=1.2, alpha=.8, label="spatial head")
        ax.plot(t, A[f"{v}/p_temporal"], color=C_TP, lw=1.2, alpha=.8, label="temporal head")
        ax.plot(t, A[f"{v}/p_fusion"], color=C_FU, lw=1.2, alpha=.8, label="fusion head")
        r = REP["videos"][v]
        ax.set_title(f"{TITLE[v]}\nverdict {r['verdict']}   mean P(FAKE)="
                     f"{r['prob']['final_mean']:.3f}   clip agreement {r['clip_level_accuracy']:.0f}%",
                     fontsize=10.5)
        ax.set_xlabel("clip start time (s)"); ax.set_ylim(0, 1)
        ax.text(.02, .94, "FAKE", transform=ax.transAxes, color="#A2441E", fontweight="bold")
        ax.text(.02, .03, "REAL", transform=ax.transAxes, color="#1F4C96", fontweight="bold")
    axes[0].set_ylabel("P(FAKE)")
    axes[0].legend(fontsize=8.5, loc="center left", framealpha=.95)
    fig.suptitle("Per-clip decision over the whole video (60 overlapping 3 s clips each)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "f2_verdict_timeline.png")


# ------------------------------------------------------- F3 branch shares
def f3_branch_share():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    x = np.arange(2); w = .2
    ax = axes[0]
    for i, (k, c, lb) in enumerate([("p_spatial", C_SP, "spatial head"),
                                    ("p_temporal", C_TP, "temporal head"),
                                    ("p_fusion", C_FU, "fusion head"),
                                    ("p_final", C_FI, "FINAL (ensemble)")]):
        vals = [A[f"{v}/{k}"].mean() for v in VIDS]
        b = ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=lb,
                   edgecolor="white", lw=.8)
        ax.bar_label(b, fmt="%.3f", fontsize=7.5, padding=1)
    ax.axhline(.5, color="k", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[v] for v in VIDS])
    ax.set_ylabel("mean P(FAKE)"); ax.set_ylim(0, 1.12)
    ax.set_title("What each head decides", fontsize=11)
    ax.legend(fontsize=8, ncol=1, loc="upper left")

    ax = axes[1]
    sp = [REP["videos"][v]["branch_shapley"]["pct_spatial_mean"] for v in VIDS]
    tp = [REP["videos"][v]["branch_shapley"]["pct_temporal_mean"] for v in VIDS]
    b1 = ax.barh(x, sp, .45, color=C_SP, label="SPATIAL")
    b2 = ax.barh(x, tp, .45, left=sp, color=C_TP, label="TEMPORAL")
    for bb, vv in ((b1, sp), (b2, tp)):
        ax.bar_label(bb, labels=[f"{q:.1f}%" for q in vv], label_type="center",
                     color="white", fontweight="bold", fontsize=11)
    ax.set_yticks(x); ax.set_yticklabels([SHORT[v] for v in VIDS])
    ax.set_xlabel("share of the evidence that moved the decision away from chance (%)")
    ax.set_title("Branch contribution (Shapley, exact)", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower center", ncol=2); ax.set_xlim(0, 100)
    ax.grid(False)

    ax = axes[2]
    for v, c in zip(VIDS, [C_SP, C_TP]):
        ax.hist(A[f"{v}/pct_spatial"], bins=18, alpha=.6, label=SHORT[v],
                color=c, edgecolor="white")
    ax.axvline(50, color="k", ls="--", lw=1.2)
    ax.set_xlabel("spatial share per clip (%)"); ax.set_ylabel("number of clips")
    ax.set_title("Per-clip spread of the spatial share", fontsize=11)
    ax.legend(fontsize=8.5)
    fig.suptitle("Spatial vs temporal: how much each branch contributed",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "f3_branch_share.png")


# ------------------------------------------------------- F4 facial parts
def f4_parts(derived):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    x = np.arange(4)
    for ax, v in zip(axes[:2], VIDS):
        c = A[f"{v}/contrib_parts"]
        m, sd = c.mean(0), c.std(0)
        cols = ["#D9622E" if q > 0 else "#2E6FD9" for q in m]
        b = ax.bar(x, m, .6, yerr=sd, color=cols, capsize=3, edgecolor="white")
        ax.bar_label(b, fmt="%+.3f", fontsize=8.5, padding=2)
        ax.axhline(0, color="k", lw=1.1)
        ax.set_xticks(x); ax.set_xticklabels(FACE_PARTS)
        ax.set_title(f"{SHORT[v]}\nspatial logit contribution per component", fontsize=10.5)
        ax.set_ylabel("contribution to  logit(FAKE)-logit(REAL)")
        lo, hi = min(m - sd), max(m + sd)
        pad = .22 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(.98, .96, "+ pushes FAKE", transform=ax.transAxes, ha="right",
                color="#A2441E", fontsize=9, fontweight="bold")
        ax.text(.98, .03, "- pushes REAL", transform=ax.transAxes, ha="right",
                color="#1F4C96", fontsize=9, fontweight="bold")

    ax = axes[2]
    d = np.array([derived["delta_parts"][p] for p in FACE_PARTS])
    b = ax.bar(x, d, .6, color="#444", edgecolor="white")
    ax.bar_label(b, fmt="%+.3f", fontsize=8.5, padding=2)
    ax.axhline(0, color="k", lw=1.1)
    ax.set_xticks(x); ax.set_xticklabels(FACE_PARTS)
    ax.set_title("FAKE minus REAL\n(same actor, only the swap differs)", fontsize=10.5)
    ax.set_ylabel("change in contribution")
    fig.suptitle("Inside the spatial branch: which facial component carries the evidence",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "f4_facial_parts.png")


# ------------------------------------------------------- F5 temporal maps
def f5_temporal_maps():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    maps = {}
    for v in VIDS:
        maps[v] = A[f"{v}/contrib_patches"].mean(0).reshape(GRID, GRID)
    vmax = max(np.abs(m).max() for m in maps.values())

    for ax, v in zip(axes[:2], VIDS):
        img = face_image(v)
        ax.imshow(img, extent=[0, CROP, CROP, 0])
        hm = cv2.resize(maps[v], (CROP, CROP), interpolation=cv2.INTER_CUBIC)
        im = ax.imshow(hm, extent=[0, CROP, CROP, 0], cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, alpha=.62)
        lm = A[f"{v}/landmarks"]
        ax.scatter(lm[:, 0], lm[:, 1], s=2.5, c="#00FF66", alpha=.85)
        ax.set_title(f"{SHORT[v]}\ntemporal evidence per patch", fontsize=10.5)
        ax.axis("off"); ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=.046, label="logit contribution")

    ax = axes[2]
    diff = maps["000_003"] - maps["000"]
    img = face_image("000_003")
    ax.imshow(img, extent=[0, CROP, CROP, 0])
    hm = cv2.resize(diff, (CROP, CROP), interpolation=cv2.INTER_CUBIC)
    im = ax.imshow(hm, extent=[0, CROP, CROP, 0], cmap="RdBu_r",
                   vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(), alpha=.62)
    lm = A["000_003/landmarks"]
    ax.scatter(lm[:, 0], lm[:, 1], s=2.5, c="#00FF66", alpha=.85)
    ax.set_title("FAKE minus REAL\nwhere temporal evidence appeared", fontsize=10.5)
    ax.axis("off"); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.046, label="change")
    fig.suptitle("Inside the temporal branch: the 16x16 patch grid is directly interpretable",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "f5_temporal_maps.png")


# ------------------------------------------------------- F6 zones
def f6_zones(derived):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    zz = [z for z in ZONES if z != "background"]
    y = np.arange(len(zz)); w = .38
    ax = axes[0]
    for i, v in enumerate(VIDS):
        vals = [derived["zone_density"][v][z] for z in zz]
        b = ax.barh(y + (i - .5) * w, vals, w,
                    color=[C_SP, C_TP][i], label=SHORT[v], edgecolor="white")
        ax.bar_label(b, fmt="%.4f", fontsize=7.5, padding=2)
    ax.set_yticks(y); ax.set_yticklabels(zz)
    ax.set_xlabel("mean |contribution| per patch  (area-normalised)")
    ax.set_title("Temporal evidence density by anatomical zone", fontsize=11)
    ax.legend(fontsize=8.5); ax.invert_yaxis()

    ax = axes[1]
    vals = [derived["zone_density_delta"][z] for z in zz]
    cols = ["#D9622E" if q > 0 else "#2E6FD9" for q in vals]
    b = ax.barh(y, vals, .55, color=cols, edgecolor="white")
    ax.bar_label(b, fmt="%+.4f", fontsize=8, padding=2)
    ax.axvline(0, color="k", lw=1.1)
    ax.set_yticks(y); ax.set_yticklabels(zz)
    ax.set_xlabel("FAKE minus REAL, per patch")
    ax.set_title("Which zones became temporally unstable after the swap", fontsize=11)
    ax.invert_yaxis()
    fig.suptitle("Area-normalised temporal attribution (removes the bias of large zones)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "f6_zones.png")


# ------------------------------------------------------- F7 layers
def f7_layers():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    L = np.arange(24)
    for ax, (key, name, cmapc) in zip(
            axes, [("contrib_spatial_layers", "SPATIAL branch", C_SP),
                   ("contrib_temporal_layers", "TEMPORAL branch", C_TP)]):
        for v, ls in zip(VIDS, ["--", "-"]):
            m = A[f"{v}/{key}"].mean(0)
            ax.plot(L, m, ls, marker="o", ms=3.5, lw=1.8,
                    color=cmapc if v == "000_003" else "#888",
                    label=SHORT[v])
        ax.axhline(0, color="k", lw=1.1)
        ax.set_xlabel("CLIP ViT-L/14 layer index (0 = shallowest)")
        ax.set_ylabel("logit contribution")
        ax.set_title(f"{name}: where in the frozen encoder the evidence is read", fontsize=11)
        ax.legend(fontsize=8.5)
    fig.suptitle("Layer-wise decomposition (contributions sum exactly to the head logit)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "f7_layers.png")


# ------------------------------------------------------- F8 attention
def f8_attention():
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for r, v in enumerate(VIDS):
        attn = A[f"{v}/s_attention"]          # (L, Q, 256)
        img = face_image(v)
        for c, part in enumerate(FACE_PARTS):
            ax = axes[r, c]
            mp = attn[:, c].mean(0).reshape(GRID, GRID)
            ax.imshow(img, extent=[0, CROP, CROP, 0])
            hm = cv2.resize(mp, (CROP, CROP), interpolation=cv2.INTER_CUBIC)
            ax.imshow(hm, extent=[0, CROP, CROP, 0], cmap="inferno", alpha=.58)
            ax.set_title(f"'{part}' query", fontsize=10)
            ax.axis("off"); ax.grid(False)
            if c == 0:
                ax.text(-.08, .5, SHORT[v], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("Facial Component Guidance: where each of the 4 component queries looks\n"
                 "(cosine attention, averaged over 24 layers x 10 frames x 60 clips)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(h_pad=2.6)
    save(fig, "f8_attention.png")


# ------------------------------------------------------- F9 counterfactual
def f9_counterfactual(derived):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, v in zip(axes, VIDS):
        names = ["spatial\nonly", "temporal\nonly", "full model\n(S+T)"]
        vals = [A[f"{v}/p_spatial"].mean(), A[f"{v}/p_temporal"].mean(),
                A[f"{v}/p_final"].mean()]
        cols = [C_SP, C_TP, C_FI]
        b = ax.bar(names, vals, .55, color=cols, edgecolor="white")
        ax.bar_label(b, fmt="%.3f", fontsize=10, padding=2, fontweight="bold")
        ax.axhline(.5, color="k", ls="--", lw=1.3)
        ax.axhspan(.5, 1.05, color="#D9622E", alpha=.07)
        ax.axhspan(0, .5, color="#2E6FD9", alpha=.07)
        ax.set_ylim(0, 1.08); ax.set_ylabel("mean P(FAKE)")
        gt = REP["videos"][v]["ground_truth"]
        verd = ["FAKE" if q >= .5 else "REAL" for q in vals]
        ax.set_title(f"{SHORT[v]}  (truth {gt})\nverdict if a branch is removed: "
                     + " / ".join(verd), fontsize=10.5)
    fig.suptitle("Branch necessity: does the verdict survive deleting a branch?\n"
                 "(inference-time ablation of the released checkpoint; the spatial head is\n"
                 "provably independent of the temporal branch, so these are exact)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "f9_counterfactual.png")


# ------------------------------------------------------- derived metrics
def derive():
    d = {}
    zone_density, zone_abs_sum, zone_counts = {}, {}, {}
    for v in VIDS:
        zmap = A[f"{v}/zone_map"].astype(str)
        pm = A[f"{v}/contrib_patches"].mean(0).reshape(GRID, GRID)
        dens, asum, cnt = {}, {}, {}
        for z in ZONES:
            msk = zmap == z
            n = int(msk.sum())
            cnt[z] = n
            asum[z] = float(np.abs(pm[msk]).sum()) if n else 0.0
            dens[z] = float(np.abs(pm[msk]).mean()) if n else 0.0
        zone_density[v], zone_abs_sum[v], zone_counts[v] = dens, asum, cnt
    d["zone_density"] = zone_density
    d["zone_abs_sum"] = zone_abs_sum
    d["zone_patch_counts"] = zone_counts
    d["zone_density_delta"] = {z: zone_density["000_003"][z] - zone_density["000"][z]
                               for z in ZONES}
    face_only = {z: v for z, v in d["zone_density_delta"].items() if z != "background"}
    d["zone_density_delta_top"] = sorted(face_only, key=lambda z: -face_only[z])[:3]
    d["zone_density_dominant"] = {
        v: max((z for z in ZONES if z != "background"), key=lambda z: zone_density[v][z])
        for v in VIDS}

    pm_r = A["000/contrib_parts"].mean(0)
    pm_f = A["000_003/contrib_parts"].mean(0)
    d["delta_parts"] = {p: float(b - a) for p, a, b in zip(FACE_PARTS, pm_r, pm_f)}
    d["delta_parts_top"] = max(d["delta_parts"], key=lambda p: d["delta_parts"][p])

    for v in VIDS:
        ps, pt, pf = (A[f"{v}/p_spatial"], A[f"{v}/p_temporal"], A[f"{v}/p_final"])
        gt = 1 if v == "000_003" else 0
        d.setdefault("counterfactual", {})[v] = dict(
            spatial_only_mean=float(ps.mean()),
            temporal_only_mean=float(pt.mean()),
            full_mean=float(pf.mean()),
            spatial_only_verdict="FAKE" if ps.mean() >= .5 else "REAL",
            temporal_only_verdict="FAKE" if pt.mean() >= .5 else "REAL",
            full_verdict="FAKE" if pf.mean() >= .5 else "REAL",
            spatial_only_correct=bool((ps.mean() >= .5) == bool(gt)),
            temporal_only_correct=bool((pt.mean() >= .5) == bool(gt)),
            both_branches_individually_sufficient=bool(
                ((ps.mean() >= .5) == bool(gt)) and ((pt.mean() >= .5) == bool(gt))),
        )
    # agreement between the two branches per clip
    for v in VIDS:
        bs = A[f"{v}/p_spatial"] >= .5
        bt = A[f"{v}/p_temporal"] >= .5
        d["counterfactual"][v]["clip_branch_agreement_pct"] = float(100 * (bs == bt).mean())
    json.dump(d, open(OUT / "derived.json", "w"), indent=2)
    return d


if __name__ == "__main__":
    print("deriving metrics ...")
    derived = derive()
    print("building figures ...")
    f1_architecture()
    f2_verdicts()
    f3_branch_share()
    f4_parts(derived)
    f5_temporal_maps()
    f6_zones(derived)
    f7_layers()
    f8_attention()
    f9_counterfactual(derived)
    print("\nzone density dominant:", derived["zone_density_dominant"])
    print("zone delta top:", derived["zone_density_delta_top"])
    print("delta parts:", {k: round(v, 3) for k, v in derived["delta_parts"].items()})
    print("counterfactual:", json.dumps(derived["counterfactual"], indent=2))
