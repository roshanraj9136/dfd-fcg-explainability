r"""
Extend the explainability analysis to the authors' own robustness evaluation
protocol (the 7 perturbations x 5 severity levels used in the DFD-FCG paper).

The perturbation parameters and functions are taken verbatim from
src/preprocess/robustness/{distortions.py, phase1_apply_all_to_videos.py}.

Protocol (as in the repo readme): the perturbation is applied to the ORIGINAL
full-resolution video, and the face crop is then produced with the landmarks
extracted from the CLEAN video -- so the geometry is identical across
conditions and only the pixel statistics change.

    python xai/run_robustness.py --levels 1 3 5 --clips 12
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from src.preprocess.robustness.distortions import (block_wise, color_contrast,
                                                   color_saturation,
                                                   gaussian_blur,
                                                   gaussian_noise_color,
                                                   jpeg_compression)
from xai.engine import (FACE_PARTS, Recorder, analyse_clips, clip_start_indices,
                        load_model, read_video_frames)

PARAMS = {  # verbatim from phase1_apply_all_to_videos.get_distortion_parameter
    "CS":   [0.4, 0.3, 0.2, 0.1, 0.0],
    "CC":   [0.85, 0.725, 0.6, 0.475, 0.35],
    "BW":   [16, 32, 48, 64, 80],
    "GNC":  [0.001, 0.002, 0.005, 0.01, 0.05],
    "GB":   [7, 9, 13, 17, 21],
    "JPEG": [2, 3, 4, 5, 6],
    "VC":   [30, 32, 35, 38, 40],
}
FRAME_FN = {"CS": color_saturation, "CC": color_contrast, "BW": block_wise,
            "GNC": gaussian_noise_color, "GB": gaussian_blur,
            "JPEG": jpeg_compression}
NAMES = {"CS": "colour saturation", "CC": "colour contrast",
         "BW": "block-wise noise", "GNC": "Gaussian noise",
         "GB": "Gaussian blur", "JPEG": "down/up-sampling (JPEG-like)",
         "VC": "video compression (H.264 CRF)"}
SAMPLES = [("000", "REAL", 0), ("000_003", "FAKE", 1)]
ROOT = Path("resources/robust")


# ------------------------------------------------------------------ phase 1
def distort_video(src, dst, kind, level):
    """Apply a frame-level or whole-video perturbation to the original video."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    param = PARAMS[kind][level - 1]

    if kind == "VC":
        import av
        inp = av.open(str(src))
        ist = inp.streams.video[0]
        out = av.open(str(dst), "w")
        ost = out.add_stream("libx264", rate=ist.average_rate)
        ost.width, ost.height, ost.pix_fmt = ist.codec_context.width, ist.codec_context.height, "yuv420p"
        ost.options = {"crf": str(param)}          # == ffmpeg -crf <param>
        for frame in inp.decode(video=0):
            for pkt in ost.encode(frame):
                out.mux(pkt)
        for pkt in ost.encode():
            out.mux(pkt)
        out.close(); inp.close()
        return

    fn = FRAME_FN[kind]
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        wr.write(fn(fr.copy(), param))
    cap.release(); wr.release()


def prepare(kind, level):
    """Build resources/robust/<kind>/<level>/{videos,frame_data,cropped}."""
    base = ROOT / kind / str(level)
    (base / "videos").mkdir(parents=True, exist_ok=True)
    (base / "frame_data").mkdir(parents=True, exist_ok=True)
    ext = ".mp4"
    for name, _, _ in SAMPLES:
        dst = base / "videos" / f"{name}{ext}"
        if not dst.exists():
            distort_video(Path(f"resources/videos/{name}.mp4"), dst, kind, level)
        # reuse the CLEAN landmarks, exactly as the authors' phase2 does
        shutil.copyfile(f"resources/frame_data/{name}.pickle",
                        base / "frame_data" / f"{name}.pickle")
    need = any(not (base / "cropped" / "videos" / f"{n}.avi").exists()
               for n, _, _ in SAMPLES)
    if need:
        subprocess.run(
            [sys.executable, "-m", "src.preprocess.crop_main_face",
             f"--root-dir={base.as_posix()}", "--video-dir=videos",
             "--fdata-dir=frame_data", "--crop-dir=cropped",
             "--glob-exp=*", "--workers=1", "--replace"],
            check=True, capture_output=True)
    return base


# ------------------------------------------------------------------ phase 2
def analyse(model, rec, base, n_clips, batch=2):
    tf = model.transform
    res = {}
    for name, gt, lbl in SAMPLES:
        vid = base / "cropped" / "videos" / f"{name}.avi"
        frames, fps = read_video_frames(vid)
        offsets, starts = clip_start_indices(
            frames.shape[0], fps, num_frames=model.hparams.num_frames,
            max_clips=n_clips)
        out = []
        for i in range(0, len(starts), batch):
            clips = torch.stack([tf(frames[offsets + s]) for s in starts[i:i + batch]])
            out.extend(analyse_clips(model, rec, clips))
        g = lambda k: np.array([r[k] for r in out])
        phi_s, phi_t = g("shapley_spatial"), g("shapley_temporal")
        tot = np.abs(phi_s) + np.abs(phi_t)
        parts = np.stack([r["contrib_parts"] for r in out]).mean(0)
        res[name] = dict(
            ground_truth=gt,
            p_final=float(g("p_final").mean()),
            p_spatial=float(g("p_spatial").mean()),
            p_temporal=float(g("p_temporal").mean()),
            p_fusion=float(g("p_fusion").mean()),
            verdict="FAKE" if g("p_final").mean() >= .5 else "REAL",
            correct=bool((g("p_final").mean() >= .5) == bool(lbl)),
            clip_accuracy=float(100 * ((g("p_final") >= .5).astype(int) == lbl).mean()),
            pct_spatial=float((100 * np.abs(phi_s) / tot).mean()),
            pct_temporal=float((100 * np.abs(phi_t) / tot).mean()),
            parts={p: float(v) for p, v in zip(FACE_PARTS, parts)},
            parts_share={p: float(v) for p, v in zip(
                FACE_PARTS, 100 * np.abs(parts) / np.abs(parts).sum())},
            dominant_part=FACE_PARTS[int(np.argmax(np.abs(parts)))],
            n_clips=len(out),
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--types", type=str, nargs="+",
                    default=["CS", "CC", "BW", "GNC", "GB", "JPEG", "VC"])
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--out", type=str, default="xai/out/robustness.json")
    args = ap.parse_args()

    print("preparing perturbed videos (authors' protocol) ...", flush=True)
    bases = {}
    for k in args.types:
        for lv in args.levels:
            try:
                bases[(k, lv)] = prepare(k, lv)
                print(f"  {k}(level {lv}) param={PARAMS[k][lv-1]}  ok", flush=True)
            except Exception as e:
                print(f"  {k}(level {lv}) FAILED: {e}", flush=True)

    print("\nloading model ...", flush=True)
    model = load_model("checkpoint/weights.ckpt", "cuda")
    rec = Recorder(model)
    rec.patch()

    out = dict(protocol="DFD-FCG paper robustness protocol "
                        "(src/preprocess/robustness), applied to the bundled "
                        "FaceForensics++ sample pair",
               params=PARAMS, names=NAMES, clean=None, conditions={})

    # clean reference with the same clip budget
    print("analysing CLEAN reference ...", flush=True)
    out["clean"] = analyse(model, rec, Path("resources"), args.clips)
    for n, r in out["clean"].items():
        print(f"  {n}: P={r['p_final']:.3f} {r['verdict']} "
              f"S/T={r['pct_spatial']:.0f}/{r['pct_temporal']:.0f}%")

    for (k, lv), base in bases.items():
        print(f"analysing {k} level {lv} ...", flush=True)
        r = analyse(model, rec, base, args.clips)
        out["conditions"][f"{k}_{lv}"] = dict(type=k, level=lv,
                                              param=PARAMS[k][lv - 1], results=r)
        for n, rr in r.items():
            print(f"  {n}: P={rr['p_final']:.3f} {rr['verdict']} "
                  f"S/T={rr['pct_spatial']:.0f}/{rr['pct_temporal']:.0f}% "
                  f"dom={rr['dominant_part']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
