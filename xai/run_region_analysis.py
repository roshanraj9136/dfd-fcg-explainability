r"""
Region-based attribution for the SPATIAL branch.

WHY THIS FILE EXISTS
--------------------
`run_analysis.py` decomposes the spatial branch over the model's four *learned
component queries* (lips / skin / eyes / nose).  That is a decomposition in
FEATURE space: each query is a 1024-d embedding that attends softly over ALL
256 patches, so "eyes = 45.6%" means "the query guided towards eye-like
features contributed 45.6%", not "the eye patches contributed 45.6%".

This file adds the missing SPATIAL-REGION decomposition: we cut the face into
patch regions defined by the 68 facial landmarks (eyes, nose, lips, skin,
background) and compute how much each REGION contributed.

It is exact, by the same algebra as before.  The spatial feature is

    y_s = (1/(T*Q)) * sum_t sum_q sum_l  s_aff[t,q,l] * v[t,l,:]
                                  ^^^^^
                            a sum over the 256 patches

so y_s is additive over patches, and also jointly additive over
(query, patch) pairs.  Feeding those components through the same
LayerNorm+Linear decomposition therefore yields exact per-region shares.

Outputs
-------
1. exact spatial contribution of each face REGION (patch-defined)
2. the query x region cross-table: where each named query actually collected
   its evidence -- this validates (or refutes) the feature-space numbers
3. the same for the temporal branch, on identical region definitions, so the
   two branches are finally comparable on a common footing

    python xai/run_region_analysis.py --max-clips 30
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from xai.engine import (FACE_PARTS, _ln_linear_decomp, clip_start_indices,
                        load_model, read_video_frames)

CKPT = "checkpoint/weights.ckpt"
CROP, GRID = 150, 16
FAKE, REAL = 1, 0

# Regions aligned 1:1 with the model's four component names, so the
# feature-space and region-space numbers can be compared directly.
# 68-point landmark indices (dlib/2D-FAN convention).
LM = {
    "eyes": list(range(36, 48)) + list(range(17, 27)),   # eyes + brows
    "nose": list(range(27, 36)),
    "lips": list(range(48, 68)),
}
REGIONS = ["eyes", "nose", "lips", "skin", "background"]

SAMPLES = [
    dict(name="000", label=0, label_name="REAL",
         crop="resources/cropped/videos/000.avi",
         fdata="resources/cropped/frame_data/000.pickle"),
    dict(name="000_003", label=1, label_name="FAKE",
         crop="resources/cropped/videos/000_003.avi",
         fdata="resources/cropped/frame_data/000_003.pickle"),
]


# --------------------------------------------------------------------- regions
def mean_landmarks(fdata_path):
    data = pickle.load(open(fdata_path, "rb"))
    pts = [np.asarray(f["landmarks"]).reshape(-1, 68, 2)[0]
           for f in data if np.asarray(f["landmarks"]).size]
    return np.mean(np.stack(pts), axis=0)


def region_map(landmarks, grid=GRID, crop=CROP, near=12.0):
    """Assign each of the 256 patches to one of the 5 regions."""
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull

    step = crop / grid
    centres = np.array([[(c + .5) * step, (r + .5) * step]
                        for r in range(grid) for c in range(grid)])
    hull = ConvexHull(landmarks)
    inside = MplPath(landmarks[hull.vertices])

    labels = []
    for p in centres:
        best, bestd = None, 1e9
        for name, idxs in LM.items():
            d = np.linalg.norm(landmarks[idxs] - p, axis=1).min()
            if d < bestd:
                best, bestd = name, d
        if bestd <= near:
            labels.append(best)
        elif inside.contains_point(p):
            labels.append("skin")
        else:
            labels.append("background")
    return np.array(labels).reshape(grid, grid)


# ------------------------------------------------------- spatial per-patch rec
class RegionRecorder:
    """
    Replaces spatial_detection so it also returns the per-(query, patch)
    breakdown of y_s, accumulated across the 24 layers.
    `y` itself is bit-identical to the original implementation.
    """

    def __init__(self, model):
        self.blocks = list(model.model.encoder.decoder.decoder_layers)
        self.acc_ql = None      # (b, Q, L_patch, W) accumulated over layers
        self.t_patch = []
        self._patched = False

    def reset(self):
        self.acc_ql = None
        self.t_patch = []

    def patch(self):
        if self._patched:
            return
        for blk in self.blocks:
            blk.spatial_detection = self._spatial(blk)
            blk.temporal_detection = self._temporal(blk)
        self._patched = True

    def unpatch(self):
        for blk in self.blocks:
            for n in ("spatial_detection", "temporal_detection"):
                if n in blk.__dict__:
                    del blk.__dict__[n]
        self._patched = False

    def _spatial(self, blk):
        rec = self

        def fn(attrs):
            _k = attrs[blk.s_k_attr][:, :, 1:]
            _v = attrs[blk.s_v_attr][:, :, 1:]
            b = _k.shape[0]
            s_q = blk.syno_embedding.unsqueeze(0).repeat(b, 1, 1)
            if len(_k.shape) == 5:
                _k = _k.flatten(-2).contiguous()
            if len(_v.shape) == 5:
                _v = _v.flatten(-2).contiguous()

            s_aff = torch.einsum(
                "nqw,ntkw->ntqk",
                s_q / (s_q.norm(dim=-1, keepdim=True) + 1e-4),
                _k / (_k.norm(dim=-1, keepdim=True) + 1e-4))
            s_aff = (s_aff * 100).softmax(dim=-1)

            T, Q = s_aff.shape[1], s_aff.shape[2]

            # y_ql[b,q,l,w] = (1/(T*Q)) * sum_t s_aff[b,t,q,l] * v[b,t,l,w]
            y_ql = torch.einsum("ntql,ntlw->nqlw", s_aff, _v) / (T * Q)

            rec.acc_ql = y_ql.detach().float() if rec.acc_ql is None \
                else rec.acc_ql + y_ql.detach().float()

            y = y_ql.sum(dim=(1, 2))        # == original s_mix.flatten(1,2).mean(1)
            if blk.attn_record:
                blk.aff = s_aff
            return dict(s_q=s_q, y=y)

        return fn

    def _temporal(self, blk):
        rec = self
        orig = type(blk).temporal_detection

        def fn(attrs):
            ret = orig(blk, attrs)
            rec.t_patch.append(ret["y"].detach().float().cpu())
            return ret

        return fn


# --------------------------------------------------------------------- analyse
@torch.inference_mode()
def analyse(model, rec, clips, device="cuda"):
    m = model.model
    rec.reset()
    out = m(clips.to(device))
    p_final = out["logits"].double().cpu().softmax(-1)[:, FAKE].numpy()

    ql = rec.acc_ql.cpu()                      # (b, Q, 256, 1024)
    tp = torch.stack(rec.t_patch)              # (24, b, 256)
    B, Q, LP, Wd = ql.shape

    res = []
    for i in range(B):
        comp_ql = ql[i]                                   # (Q, 256, 1024)
        # --- exact spatial contribution per PATCH (summed over queries) ---
        c_patch, const_s, d_s = _ln_linear_decomp(
            comp_ql.sum(0), m.s_ln, m.s_head)             # (256,)
        # --- exact joint (query x patch) ---
        c_ql, _, _ = _ln_linear_decomp(
            comp_ql.reshape(Q * LP, Wd), m.s_ln, m.s_head)
        c_ql = c_ql.reshape(Q, LP)
        # --- exact per-query (for cross-checking run_analysis) ---
        c_q, _, _ = _ln_linear_decomp(comp_ql.sum(1), m.s_ln, m.s_head)
        # --- temporal per patch ---
        tv = tp[:, i].sum(0).double()
        c_tpatch, const_t, d_t = _ln_linear_decomp(
            torch.diag(tv), m.t_ln, m.t_head)

        res.append(dict(p_final=float(p_final[i]),
                        d_spatial=float(d_s), d_temporal=float(d_t),
                        const_spatial=float(const_s), const_temporal=float(const_t),
                        c_patch_spatial=c_patch, c_patch_temporal=c_tpatch,
                        c_query=c_q, c_query_patch=c_ql))
    del out
    return res


def agg(vals, rmap):
    v = vals.reshape(GRID, GRID)
    return {r: float(v[rmap == r].sum()) for r in REGIONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", type=str, default="xai/out/regions.json")
    args = ap.parse_args()

    print("loading model ...", flush=True)
    model = load_model(CKPT, "cuda")
    rec = RegionRecorder(model)
    tf = model.transform

    report, arrays = {"regions": REGIONS}, {}

    for si, s in enumerate(SAMPLES):
        print(f"\n=== {s['name']} (truth {s['label_name']}) ===", flush=True)
        frames, fps = read_video_frames(s["crop"])
        offsets, starts = clip_start_indices(
            frames.shape[0], fps, num_frames=model.hparams.num_frames,
            max_clips=args.max_clips)

        # faithfulness: patched spatial must equal pristine
        if si == 0:
            probe = torch.stack([tf(frames[offsets + starts[0]])])
            rec.unpatch()
            clean = model.model(probe.to("cuda"))["logits"].double().cpu().clone()
            rec.patch(); rec.reset()
            pat = model.model(probe.to("cuda"))["logits"].double().cpu().clone()
            err = (clean - pat).abs().max().item()
            print(f"  per-patch instrumentation error: {err:.3e} "
                  f"-> {'EXACT' if err < 1e-4 else 'MISMATCH'}")
            report["instrumentation_max_logit_error"] = err
        rec.patch()

        lms = mean_landmarks(s["fdata"])
        rmap = region_map(lms)
        counts = {r: int((rmap == r).sum()) for r in REGIONS}
        print("  patches per region:", counts)

        allres = []
        for i in range(0, len(starts), args.batch):
            clips = torch.stack([tf(frames[offsets + st])
                                 for st in starts[i:i + args.batch]])
            allres.extend(analyse(model, rec, clips))
            print(f"\r  clips {min(i+args.batch,len(starts))}/{len(starts)}",
                  end="", flush=True)
        print()

        A = lambda k: np.stack([r[k] for r in allres])
        sp_patch = A("c_patch_spatial").mean(0)     # (256,)
        tp_patch = A("c_patch_temporal").mean(0)
        qp = A("c_query_patch").mean(0)             # (Q, 256)
        cq = A("c_query").mean(0)                   # (Q,)
        d_s = float(np.mean([r["d_spatial"] for r in allres]))
        const_s = float(np.mean([r["const_spatial"] for r in allres]))

        # region sums (signed) and shares
        sp_reg = agg(sp_patch, rmap)
        tp_reg = agg(tp_patch, rmap)
        face = {k: v for k, v in sp_reg.items() if k != "background"}
        tot = sum(abs(v) for v in face.values())
        sp_share = {k: 100 * abs(v) / tot for k, v in face.items()}
        facet = {k: v for k, v in tp_reg.items() if k != "background"}
        tott = sum(abs(v) for v in facet.values())
        tp_share = {k: 100 * abs(v) / tott for k, v in facet.items()}

        # density (per patch) so big regions do not win by area alone
        sp_dens = {r: (sp_reg[r] / counts[r] if counts[r] else 0.0) for r in REGIONS}

        # query x region cross-table
        cross = {}
        for qi, qn in enumerate(FACE_PARTS):
            cross[qn] = agg(qp[qi], rmap)

        report[s["name"]] = dict(
            ground_truth=s["label_name"], n_clips=len(allres),
            p_final_mean=float(A("p_final").mean()),
            verdict="FAKE" if A("p_final").mean() >= .5 else "REAL",
            d_spatial=d_s, const_spatial=const_s,
            patch_counts=counts,
            spatial_region_signed=sp_reg,
            spatial_region_share_pct=sp_share,
            spatial_region_density=sp_dens,
            spatial_region_dominant=max(sp_share, key=sp_share.get),
            temporal_region_signed=tp_reg,
            temporal_region_share_pct=tp_share,
            temporal_region_dominant=max(tp_share, key=tp_share.get),
            query_contrib={p: float(v) for p, v in zip(FACE_PARTS, cq)},
            query_region_cross=cross,
            # does the named query collect evidence from its own region?
            query_self_region_pct={
                qn: (100 * abs(cross[qn].get(qn if qn != "skin" else "skin", 0.0))
                     / max(sum(abs(x) for k, x in cross[qn].items()
                               if k != "background"), 1e-12))
                for qn in FACE_PARTS},
            additivity_check=dict(
                sum_patches_plus_const=float(sp_patch.sum() + const_s),
                d_spatial=d_s,
                abs_err=abs(float(sp_patch.sum() + const_s) - d_s)),
        )
        arrays[f"{s['name']}/sp_patch"] = sp_patch
        arrays[f"{s['name']}/tp_patch"] = tp_patch
        arrays[f"{s['name']}/query_patch"] = qp
        arrays[f"{s['name']}/region_map"] = rmap
        arrays[f"{s['name']}/landmarks"] = lms

        r = report[s["name"]]
        print(f"  verdict {r['verdict']}  P={r['p_final_mean']:.4f}")
        print(f"  additivity: sum(patches)+const={r['additivity_check']['sum_patches_plus_const']:+.6f}"
              f" vs d_spatial={d_s:+.6f} (err {r['additivity_check']['abs_err']:.2e})")
        print("  SPATIAL by region  :", {k: round(v, 1) for k, v in sp_share.items()})
        print("  TEMPORAL by region :", {k: round(v, 1) for k, v in tp_share.items()})
        print("  query-space (old)  :", {p: round(float(v), 3) for p, v in zip(FACE_PARTS, cq)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    np.savez_compressed("xai/out/regions.npz", **arrays)
    print(f"\nwrote {args.out} and xai/out/regions.npz")


if __name__ == "__main__":
    main()
