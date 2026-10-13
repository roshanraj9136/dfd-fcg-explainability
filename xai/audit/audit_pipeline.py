r"""Audit: pipeline logic, ground-truth labels, patch-coordinate mapping,
checkpoint load integrity, and the op_mode branch-ablation claim."""
import sys, json, pickle, math
import numpy as np, torch, cv2
sys.path.insert(0, ".")

FAIL, WARN = [], []
def chk(n, c, d=""):
    print(f"  [{'PASS' if c else 'FAIL'}] {n}  {d}")
    if not c: FAIL.append(n)
def warn(n, d=""):
    print(f"  [WARN] {n}  {d}"); WARN.append(n)

print("=" * 78); print("C1. transform geometry: is linear patch->crop mapping valid?")
print("=" * 78)
from src.clip.clip import _transform
import torchvision
t = _transform(224)
print("   transform:", [type(x).__name__ for x in t.transforms])
# the cropped videos are square 150x150
cap = cv2.VideoCapture("resources/cropped/videos/000.avi")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
print(f"   cropped frame size: {w}x{h}")
chk("cropped frames are square", w == h, f"{w}x{h}")
chk("Resize(224) on square -> CenterCrop(224) is a no-op (no spatial loss)",
    w == h, "so patch centres map linearly onto the crop: VALID")
# empirical: push a marker pixel through and see where it lands
probe = torch.zeros(3, 150, 150)
probe[:, 10:20, 100:110] = 1.0           # a square at a known location
outp = t(probe)
ys, xs = torch.where(outp[0] > outp[0].mean())
sy, sx = ys.float().mean().item() / 224 * 150, xs.float().mean().item() / 224 * 150
chk("marker centre preserved under transform",
    abs(sy - 15) < 2 and abs(sx - 105) < 2,
    f"expected ~(15,105) got ({sy:.1f},{sx:.1f})")

print(); print("=" * 78); print("C2. ground-truth labels for the bundled pair")
print("=" * 78)
# FF++ convention: NNN.mp4 = pristine original; TARGET_SOURCE.mp4 = manipulated
print("   FaceForensics++ naming: 'NNN.mp4' = original, 'TARGET_SOURCE.mp4' = manipulated")
print("   -> 000.mp4 = REAL ; 000_003.mp4 = FAKE (target 000, source 003)")
f = {}
for n in ("000", "000_003"):
    cap = cv2.VideoCapture(f"resources/videos/{n}.mp4")
    ok, fr = cap.read(); cap.release()
    f[n] = fr
    print(f"   {n}.mp4 first frame {fr.shape}")
chk("both videos same resolution", f["000"].shape == f["000_003"].shape, "")
d = np.abs(f["000"].astype(float) - f["000_003"].astype(float))
print(f"   mean|diff| over full frame = {d.mean():.2f}")
chk("videos differ (one is manipulated)", d.mean() > 0.5, f"mean abs diff {d.mean():.2f}")
# the difference should be concentrated in the FACE region, not the background
lm = pickle.load(open("resources/frame_data/000.pickle", "rb"))[0]["landmarks"][0]
lm = np.asarray(lm).reshape(68, 2)
x0, y0 = lm.min(0).astype(int); x1, y1 = lm.max(0).astype(int)
face_d = d[max(y0,0):y1, max(x0,0):x1].mean()
mask = np.ones(d.shape[:2], bool); mask[max(y0,0):y1, max(x0,0):x1] = False
bg_d = d[mask].mean()
print(f"   mean|diff| inside face bbox = {face_d:.2f} | outside = {bg_d:.2f}")
chk("difference concentrated in the FACE (consistent with a face swap)",
    face_d > 3 * bg_d, f"ratio {face_d/max(bg_d,1e-6):.1f}x")
cv2.imwrite("xai/audit/pair_real.png", f["000"])
cv2.imwrite("xai/audit/pair_fake.png", f["000_003"])
cv2.imwrite("xai/audit/pair_absdiff.png",
            (d / max(d.max(), 1e-6) * 255).astype(np.uint8))

print(); print("=" * 78); print("C3. clip index bounds")
print("=" * 78)
from xai.engine import read_video_frames, clip_start_indices
for n in ("000", "000_003"):
    fr, fps = read_video_frames(f"resources/cropped/videos/{n}.avi")
    N = fr.shape[0]
    off, starts = clip_start_indices(N, fps, num_frames=10, max_clips=60)
    mx = starts[-1] + int(off[-1])
    chk(f"{n}: max frame index {mx} < n_frames {N}", mx < N,
        f"n_clips={len(starts)} span={int(off[-1])}")
    chk(f"{n}: no duplicate starts", len(set(starts)) == len(starts), "")

print(); print("=" * 78); print("C4. checkpoint load integrity (strict)")
print("=" * 78)
from src.model.clip.svl import FFGSynoVideoLearner
ck = torch.load("checkpoint/weights.ckpt", map_location="cpu", weights_only=False)
sd = ck["state_dict"]
print(f"   checkpoint state_dict entries: {len(sd)}")
m = FFGSynoVideoLearner.load_from_checkpoint(
    "checkpoint/weights.ckpt", map_location="cpu", attn_record=True)
res = m.load_state_dict(sd, strict=False)
print(f"   missing keys   : {len(res.missing_keys)}")
print(f"   unexpected keys: {len(res.unexpected_keys)}")
# the only trainable parts are side-net + heads: make sure those are all present
trainable = [k for k, p in m.named_parameters() if p.requires_grad]
miss_tr = [k for k in trainable if k not in sd]
print(f"   trainable params: {len(trainable)}  | NOT in checkpoint: {len(miss_tr)}")
if miss_tr[:5]: print("     e.g.", miss_tr[:5])
chk("every TRAINABLE weight is loaded from the checkpoint", len(miss_tr) == 0,
    f"{len(miss_tr)} missing")
if res.missing_keys:
    frozen_only = all(("encoder.model." in k) for k in res.missing_keys)
    print(f"   all missing keys are frozen-CLIP keys? {frozen_only}")

print(); print("=" * 78)
print("C5. does attn_record=True change the numerics?")
print("=" * 78)
src = open("src/model/clip/svl.py").read()
i = src.find("if self.attn_record:")
print("   svl.py SynoBlock.spatial_detection tail:")
print("    ", src[i:i+70].replace("\n", " | "))
chk("attn_record only stores self.aff (no math change)",
    "self.aff = s_aff" in src[i:i+70], "store-only side effect")

print(); print("=" * 78)
print("C6. is the SPATIAL branch independent of the TEMPORAL branch?")
print("=" * 78)
import inspect
from src.model.clip.svl import SynoBlock, BinaryLinearClassifier
sfn = inspect.getsource(SynoBlock.spatial_detection)
uses_t = any(tok in sfn for tok in ("t_conv", "t_proj", "p_conv", "y_t", "temporal"))
chk("spatial_detection does not read any temporal module/output", not uses_t,
    "inputs are only attrs[k], attrs[emb], syno_embedding")
ffn = inspect.getsource(BinaryLinearClassifier.forward)
chk("logits_s computed from s_ln/s_head only",
    "logits_s = self.s_head(_s)" in ffn and "_s = self.s_ln(results[\"syno_s\"])" in ffn, "")
print("   => for FIXED weights, s_head output is identical under op_mode=[S] and [S,T].")
warn("but the released checkpoint was TRAINED jointly (op_mode=[S,T]); a model "
     "RE-TRAINED with op_mode=[S] would have different weights. Single-branch "
     "numbers are INFERENCE-TIME ablations, not retrained single-branch models.")

print(); print("=" * 78)
print("C7. reported numbers vs raw arrays")
print("=" * 78)
A = np.load("xai/out/arrays.npz", allow_pickle=True)
R = json.load(open("xai/out/report.json"))
for v in ("000", "000_003"):
    r = R["videos"][v]
    ps, pt = A[f"{v}/phi_spatial"], A[f"{v}/phi_temporal"]
    tot = np.abs(ps) + np.abs(pt)
    pct_s = (100 * np.abs(ps) / tot).mean()
    chk(f"{v} pct_spatial matches report",
        abs(pct_s - r["branch_shapley"]["pct_spatial_mean"]) < 1e-6,
        f"recomputed {pct_s:.4f} vs {r['branch_shapley']['pct_spatial_mean']:.4f}")
    pf = A[f"{v}/p_final"]
    chk(f"{v} mean P matches report",
        abs(pf.mean() - r["prob"]["final_mean"]) < 1e-9,
        f"{pf.mean():.6f}")
    lbl = 1 if v == "000_003" else 0
    acc = 100 * ((pf >= .5).astype(int) == lbl).mean()
    chk(f"{v} clip accuracy matches report",
        abs(acc - r["clip_level_accuracy"]) < 0.01, f"{acc:.2f}%")
    # ensemble identity on raw arrays
    ens = (A[f"{v}/p_spatial"] + A[f"{v}/p_temporal"] + A[f"{v}/p_fusion"]) / 3
    chk(f"{v} P_final == mean of 3 heads (max dev)",
        np.abs(ens - pf).max() < 2e-4, f"{np.abs(ens-pf).max():.2e}")

print(); print("=" * 78)
print("FAILURES:", FAIL if FAIL else "none")
print("WARNINGS:", len(WARN))
print("=" * 78)
