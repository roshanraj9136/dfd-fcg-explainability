"""Smoke test: load checkpoint, run one clip, verify attribution exactness."""
import sys, math, torch
sys.path.insert(0, ".")
from xai.engine import (load_model, read_video_frames, clip_start_indices,
                        Recorder, analyse_clips, verify_equivalence, FACE_PARTS)

CKPT = "checkpoint/weights.ckpt"
VID = "resources/cropped/videos/000_003.avi"

print("loading model ...", flush=True)
model = load_model(CKPT, "cuda")
print("  arch:", model.hparams.architecture,
      "| frames:", model.hparams.num_frames,
      "| op_mode:", model.hparams.op_mode,
      "| parts:", model.hparams.face_parts)
print("  spatial_dim:", model.model.encoder.spatial_dim,
      "temporal_dim:", model.model.encoder.temporal_dim,
      "| CLIP layers:", model.model.encoder.model.transformer.layers)
n_tr = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.model.parameters())
print(f"  params: {n_all/1e6:.1f}M total | {n_tr/1e6:.2f}M trainable(side-net+heads)")

frames, fps = read_video_frames(VID)
print("video:", tuple(frames.shape), "fps", round(fps, 3))
offsets, starts = clip_start_indices(frames.shape[0], fps, max_clips=None)
print("offsets:", offsets.tolist(), "| n_clips:", len(starts))

tf = model.transform
clip = torch.stack([tf(frames[offsets + starts[0]])])
print("clip tensor:", tuple(clip.shape), "range",
      round(clip.min().item(), 3), round(clip.max().item(), 3))

rec = Recorder(model)
err, ok, clean, patched = verify_equivalence(model, rec, clip)
print(f"\nEQUIVALENCE max|clean-instrumented| = {err:.3e}  -> {'PASS' if ok else 'FAIL'}")
print("  clean logits   :", clean.tolist())
print("  instrumented   :", patched.tolist())

res = analyse_clips(model, rec, clip)[0]
print("\n--- head probabilities (P of FAKE) ---")
print(f"  spatial {res['p_spatial']:.4f} | temporal {res['p_temporal']:.4f} "
      f"| fusion {res['p_fusion']:.4f} || FINAL {res['p_final']:.4f}")
ens = (res['p_spatial'] + res['p_temporal'] + res['p_fusion']) / 3
print(f"  ensemble check (P_s+P_t+P_a)/3 = {ens:.6f} vs P_final {res['p_final']:.6f} "
      f"-> diff {abs(ens-res['p_final']):.2e}")

print("\n--- exact logit-difference reconstruction (fake-real) ---")
for br in ("spatial", "temporal"):
    d = res[f"d_{br}"]
    p = 1/(1+math.exp(-d))
    print(f"  {br:9s}: d={d:+.4f} -> sigmoid {p:.4f} (head P {res['p_'+br]:.4f}) "
          f"diff {abs(p-res['p_'+br]):.2e}")
d_a = res["d_fusion"]; p_a = 1/(1+math.exp(-d_a))
print(f"  fusion   : d={d_a:+.4f} -> sigmoid {p_a:.4f} (head P {res['p_fusion']:.4f}) "
      f"diff {abs(p_a-res['p_fusion']):.2e}")

print("\n--- within-spatial additivity ---")
print("  parts:", dict(zip(FACE_PARTS, [round(float(v), 4) for v in res['contrib_parts']])))
print(f"  sum(parts)+const = {res['contrib_parts'].sum()+res['const_spatial']:+.4f} "
      f"vs d_spatial {res['d_spatial']:+.4f}")
print(f"  sum(spatial layers)+const = {res['contrib_spatial_layers'].sum()+res['const_spatial']:+.4f}")
print(f"  sum(patches)+const = {res['contrib_patches'].sum()+res['const_temporal']:+.4f} "
      f"vs d_temporal {res['d_temporal']:+.4f}")
print(f"  sum(temporal layers)+const = {res['contrib_temporal_layers'].sum()+res['const_temporal']:+.4f}")

print("\n--- shapley over branches ---")
ps, pt = res['shapley_spatial'], res['shapley_temporal']
print(f"  phi_spatial {ps:+.4f} | phi_temporal {pt:+.4f} | sum {ps+pt:+.4f} "
      f"| P_final-0.5 {res['p_final']-0.5:+.4f}")
tot = abs(ps) + abs(pt)
print(f"  -> spatial {100*abs(ps)/tot:.1f}% | temporal {100*abs(pt)/tot:.1f}%")
print("\nmem:", round(torch.cuda.max_memory_allocated()/1e9, 2), "GB")
