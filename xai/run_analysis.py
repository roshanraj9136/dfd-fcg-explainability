r"""
Run the full spatial/temporal explainability analysis on the bundled sample
videos and dump every quantity needed for the figures and the deck.

Usage:
    python xai/run_analysis.py [--max-clips N] [--batch B]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from xai.engine import (FACE_PARTS, Recorder, analyse_clips, clip_start_indices,
                        load_model, read_video_frames, verify_equivalence)

CKPT = "checkpoint/weights.ckpt"
CROP = 150          # aligned-crop pixel size produced by crop_main_face
GRID = 16           # ViT-L/14 patch grid at 224px

# the two videos shipped in the repository.
# FaceForensics++ naming: "000" is the pristine target actor, "000_003" is the
# same actor with identity 003 swapped in -> ground-truth FAKE.
SAMPLES = [
    dict(name="000", label=0, label_name="REAL",
         crop="resources/cropped/videos/000.avi",
         orig="resources/videos/000.mp4",
         fdata="resources/cropped/frame_data/000.pickle"),
    dict(name="000_003", label=1, label_name="FAKE",
         crop="resources/cropped/videos/000_003.avi",
         orig="resources/videos/000_003.mp4",
         fdata="resources/cropped/frame_data/000_003.pickle"),
]

# 68-point landmark groups -> anatomical zone
LM_GROUPS = [
    (range(0, 17), "jaw/contour"),
    (range(17, 27), "brows"),
    (range(27, 36), "nose"),
    (range(36, 48), "eyes"),
    (range(48, 68), "mouth"),
]
ZONES = ["eyes", "brows", "nose", "mouth", "jaw/contour", "cheek/forehead", "background"]


def mean_landmarks(fdata_path):
    """Average 68-point landmarks (crop coordinates) over the whole video."""
    data = pickle.load(open(fdata_path, "rb"))
    pts = []
    for fr in data:
        lm = np.asarray(fr["landmarks"])
        if lm.size:
            pts.append(lm.reshape(-1, 68, 2)[0])
    return np.mean(np.stack(pts), axis=0)      # (68,2) in 150x150 crop coords


def patch_zone_map(landmarks, grid=GRID, crop=CROP, near_px=11.0):
    """
    Assign each of the grid*grid ViT patches to an anatomical zone.

    A patch is labelled by the group of its nearest landmark when that
    landmark is within `near_px`; otherwise it is 'cheek/forehead' if it lies
    inside the convex hull of the face landmarks and 'background' if outside.
    """
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull

    step = crop / grid
    centres = np.array([[(c + .5) * step, (r + .5) * step]
                        for r in range(grid) for c in range(grid)])

    group_of = np.empty(68, dtype=object)
    for idxs, nm in LM_GROUPS:
        for i in idxs:
            group_of[i] = nm

    hull = ConvexHull(landmarks)
    hull_path = MplPath(landmarks[hull.vertices])

    labels = []
    for p in centres:
        d = np.linalg.norm(landmarks - p, axis=1)
        j = int(np.argmin(d))
        if d[j] <= near_px:
            labels.append(group_of[j])
        elif hull_path.contains_point(p):
            labels.append("cheek/forehead")
        else:
            labels.append("background")
    return np.array(labels).reshape(grid, grid), centres


def agg(values, labels):
    """Sum `values` (grid,grid) per zone label."""
    return {z: float(values[labels == z].sum()) for z in ZONES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", type=str, default="xai/out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading model ...", flush=True)
    model = load_model(CKPT, "cuda")
    rec = Recorder(model)
    tf = model.transform

    report = dict(
        model=dict(
            architecture=model.hparams.architecture,
            num_frames=model.hparams.num_frames,
            op_mode=list(model.hparams.op_mode),
            face_parts=list(model.hparams.face_parts),
            clip_layers=int(model.model.encoder.model.transformer.layers),
            spatial_dim=int(model.model.encoder.spatial_dim),
            temporal_dim=int(model.model.encoder.temporal_dim),
            params_total=int(sum(p.numel() for p in model.model.parameters())),
            params_trainable=int(sum(p.numel() for p in model.model.parameters()
                                     if p.requires_grad)),
            checkpoint="released FFG ViT-L/14 (trained on FaceForensics++ c23)",
        ),
        videos={},
    )
    arrays = {}

    for si, s in enumerate(SAMPLES):
        print(f"\n=== {s['name']}  (ground truth {s['label_name']}) ===", flush=True)
        frames, fps = read_video_frames(s["crop"])
        offsets, starts = clip_start_indices(frames.shape[0], fps,
                                             num_frames=model.hparams.num_frames,
                                             max_clips=args.max_clips)
        print(f"  frames={frames.shape[0]} fps={fps:.2f} clips_used={len(starts)}"
              f" (of {int(frames.shape[0]-offsets[-1].item())}) span={offsets[-1].item()} frames")

        # one-time faithfulness check on this video
        if si == 0:
            probe = torch.stack([tf(frames[offsets + starts[0]])])
            err, ok, _, _ = verify_equivalence(model, rec, probe)
            print(f"  instrumentation faithfulness: max abs logit error = {err:.3e} "
                  f"-> {'EXACT' if ok else 'MISMATCH'}")
            report["instrumentation_max_logit_error"] = float(err)
        rec.patch()

        lms = mean_landmarks(s["fdata"])
        zmap, _ = patch_zone_map(lms)

        res_all = []
        t0 = time.time()
        for i in range(0, len(starts), args.batch):
            chunk = starts[i:i + args.batch]
            clips = torch.stack([tf(frames[offsets + st]) for st in chunk])
            res_all.extend(analyse_clips(model, rec, clips))
            done = min(i + args.batch, len(starts))
            el = time.time() - t0
            print(f"\r  clips {done}/{len(starts)}  "
                  f"{el/done:.2f}s/clip  eta {(len(starts)-done)*el/done:5.0f}s",
                  end="", flush=True)
        print(f"\n  done in {time.time()-t0:.1f}s")

        # ---------------- aggregate over clips ----------------
        A = lambda k: np.array([r[k] for r in res_all])
        p_fin, p_sp, p_tp, p_fu = A("p_final"), A("p_spatial"), A("p_temporal"), A("p_fusion")
        phi_s, phi_t = A("shapley_spatial"), A("shapley_temporal")
        d_sp, d_tp, d_fu = A("d_spatial"), A("d_temporal"), A("d_fusion")
        f_sp, f_tp = A("fusion_spatial_term"), A("fusion_temporal_term")
        c_parts = np.stack([r["contrib_parts"] for r in res_all])            # (N,4)
        c_patch = np.stack([r["contrib_patches"] for r in res_all])          # (N,256)
        c_slay = np.stack([r["contrib_spatial_layers"] for r in res_all])    # (N,24)
        c_tlay = np.stack([r["contrib_temporal_layers"] for r in res_all])   # (N,24)
        c_plp = np.stack([r["contrib_parts_layers"] for r in res_all])       # (N,24,4)
        s_attn = np.stack([r["s_attention"] for r in res_all])               # (N,24,4,256)

        verdict = "FAKE" if p_fin.mean() >= 0.5 else "REAL"
        # branch-level percentage shares (|Shapley| normalised)
        tot_phi = np.abs(phi_s) + np.abs(phi_t)
        pct_sp = 100 * np.abs(phi_s) / tot_phi
        pct_tp = 100 * np.abs(phi_t) / tot_phi

        # logit-space totals, splitting the fusion head exactly
        tot_sp_logit = d_sp + f_sp
        tot_tp_logit = d_tp + f_tp
        den = np.abs(tot_sp_logit) + np.abs(tot_tp_logit)
        lpct_sp, lpct_tp = 100 * np.abs(tot_sp_logit) / den, 100 * np.abs(tot_tp_logit) / den

        # per-zone temporal attribution (mean over clips)
        patch_mean = c_patch.mean(0).reshape(GRID, GRID)
        zone_signed = agg(patch_mean, zmap)
        zone_abs = agg(np.abs(patch_mean), zmap)

        parts_mean = c_parts.mean(0)
        parts_share = 100 * np.abs(parts_mean) / np.abs(parts_mean).sum()

        agree = int(((p_fin >= .5).astype(int) == s["label"]).sum())

        vid = dict(
            ground_truth=s["label_name"],
            verdict=verdict,
            correct=bool((p_fin.mean() >= .5) == bool(s["label"])),
            n_clips=len(res_all),
            clip_level_accuracy=round(100 * agree / len(res_all), 2),
            fps=fps, n_frames=int(frames.shape[0]),
            prob=dict(
                final_mean=float(p_fin.mean()), final_std=float(p_fin.std()),
                final_min=float(p_fin.min()), final_max=float(p_fin.max()),
                final_median=float(np.median(p_fin)),
                spatial_mean=float(p_sp.mean()), temporal_mean=float(p_tp.mean()),
                fusion_mean=float(p_fu.mean()),
            ),
            branch_shapley=dict(
                phi_spatial_mean=float(phi_s.mean()),
                phi_temporal_mean=float(phi_t.mean()),
                pct_spatial_mean=float(pct_sp.mean()), pct_spatial_std=float(pct_sp.std()),
                pct_temporal_mean=float(pct_tp.mean()), pct_temporal_std=float(pct_tp.std()),
                clips_spatial_dominant=int((pct_sp > pct_tp).sum()),
                clips_temporal_dominant=int((pct_tp >= pct_sp).sum()),
            ),
            branch_logit=dict(
                d_spatial_mean=float(d_sp.mean()), d_temporal_mean=float(d_tp.mean()),
                d_fusion_mean=float(d_fu.mean()),
                fusion_spatial_term_mean=float(f_sp.mean()),
                fusion_temporal_term_mean=float(f_tp.mean()),
                pct_spatial_mean=float(lpct_sp.mean()), pct_temporal_mean=float(lpct_tp.mean()),
            ),
            spatial_parts=dict(
                contrib_mean={p: float(v) for p, v in zip(FACE_PARTS, parts_mean)},
                share_pct={p: float(v) for p, v in zip(FACE_PARTS, parts_share)},
                dominant=FACE_PARTS[int(np.argmax(np.abs(parts_mean)))],
            ),
            temporal_zones=dict(
                signed=zone_signed, abs=zone_abs,
                dominant=max(zone_abs, key=zone_abs.get),
                dominant_face_only=max(
                    {k: v for k, v in zone_abs.items() if k != "background"},
                    key=lambda k: zone_abs[k]),
            ),
            layers=dict(
                spatial_top=[int(i) for i in np.argsort(-np.abs(c_slay.mean(0)))[:5]],
                temporal_top=[int(i) for i in np.argsort(-np.abs(c_tlay.mean(0)))[:5]],
            ),
        )
        report["videos"][s["name"]] = vid

        arrays[f"{s['name']}/p_final"] = p_fin
        arrays[f"{s['name']}/p_spatial"] = p_sp
        arrays[f"{s['name']}/p_temporal"] = p_tp
        arrays[f"{s['name']}/p_fusion"] = p_fu
        arrays[f"{s['name']}/phi_spatial"] = phi_s
        arrays[f"{s['name']}/phi_temporal"] = phi_t
        arrays[f"{s['name']}/pct_spatial"] = pct_sp
        arrays[f"{s['name']}/pct_temporal"] = pct_tp
        arrays[f"{s['name']}/contrib_parts"] = c_parts
        arrays[f"{s['name']}/contrib_patches"] = c_patch
        arrays[f"{s['name']}/contrib_spatial_layers"] = c_slay
        arrays[f"{s['name']}/contrib_temporal_layers"] = c_tlay
        arrays[f"{s['name']}/contrib_parts_layers"] = c_plp
        arrays[f"{s['name']}/s_attention"] = s_attn.mean(0)
        arrays[f"{s['name']}/zone_map"] = zmap
        arrays[f"{s['name']}/landmarks"] = lms
        arrays[f"{s['name']}/clip_starts"] = np.array(starts)
        arrays[f"{s['name']}/d_spatial"] = d_sp
        arrays[f"{s['name']}/d_temporal"] = d_tp
        arrays[f"{s['name']}/fusion_spatial_term"] = f_sp
        arrays[f"{s['name']}/fusion_temporal_term"] = f_tp

        print(f"  VERDICT {verdict} (truth {s['label_name']}) "
              f"P_fake={p_fin.mean():.4f} +/- {p_fin.std():.4f}")
        print(f"  branch share  spatial {pct_sp.mean():.1f}%  temporal {pct_tp.mean():.1f}%")
        print(f"  spatial parts {dict((p, round(float(v),1)) for p,v in zip(FACE_PARTS, parts_share))}")
        print(f"  temporal zone dominant: {vid['temporal_zones']['dominant_face_only']}")

    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.savez_compressed(out / "arrays.npz", **arrays)
    print(f"\nwrote {out/'report.json'} and {out/'arrays.npz'}")


if __name__ == "__main__":
    main()
