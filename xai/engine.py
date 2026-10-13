r"""
Explainability engine for DFD-FCG (CVPR'25).

The released FFG ViT-L/14 model produces its final probability as an explicit
average of three head probabilities:

    P_final = ( softmax(logits_s) + softmax(logits_t) + softmax(logits_a) ) / 3

where
    logits_s = s_head( s_ln( y_s ) )                 spatial branch  (y_s: 1024-d)
    logits_t = t_head( t_ln( y_t ) )                 temporal branch (y_t:  256-d = 16x16 patch grid)
    logits_a = a_head( cat[ s_ln(y_s), t_ln(y_t) ] ) fusion head

and y_s / y_t are *sums over the 24 CLIP layers* of per-layer side-network
outputs.  y_s is in addition a *sum over the 4 facial-component queries*
(lips, skin, eyes, nose) and y_t is indexed by patch position.

Both facts make an EXACT additive attribution possible (no approximation,
no gradient surrogate):

  Let  d = logit[FAKE] - logit[REAL],  w = W[1]-W[0],  and LN(y)_j = g_j (y_j-mu)/sig + b_j.
  Then for  y = sum_k y^(k):

      d = (1/sig) * sum_k sum_j w_j g_j ( y^(k)_j - mu_k )   +   ( sum_j w_j b_j + bias_diff )
          \________________  contribution of component k  ________/     \____ constant ____/

  because mu = sum_k mu_k is itself additive over the components.

This module therefore reports, for every sampled clip:
  * the three head probabilities and the exact ensemble decomposition
  * Shapley values for the {spatial, temporal} branch pair, using the repo's
    own op_mode semantics for the single-branch games
  * exact per-facial-part logit contributions inside the spatial branch
  * exact per-patch (16x16) logit contributions inside the temporal branch
  * exact per-CLIP-layer contributions for both branches
  * the spatial cosine-attention maps of the 4 component queries
"""

import math
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model.clip.svl import FFGSynoVideoLearner

FACE_PARTS = ["lips", "skin", "eyes", "nose"]
REAL, FAKE = 0, 1


# --------------------------------------------------------------------------- #
#  model
# --------------------------------------------------------------------------- #
def load_model(ckpt_path, device="cuda", attn_record=True):
    """Load the released checkpoint, forcing attention recording on."""
    try:
        model = FFGSynoVideoLearner.load_from_checkpoint(
            ckpt_path, map_location="cpu", attn_record=attn_record
        )
    except Exception:
        model = FFGSynoVideoLearner.load_from_checkpoint(
            ckpt_path, map_location="cpu", attn_record=attn_record, strict=False
        )
    model.eval()
    model.to(device)
    return model


# --------------------------------------------------------------------------- #
#  video IO
# --------------------------------------------------------------------------- #
def read_video_frames(path):
    """Read every frame of a video as a uint8 tensor (T, C, H, W) in RGB."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    arr = np.stack(frames)                       # (T, H, W, C)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous(), float(fps)


def clip_start_indices(n_frames, fps, num_frames=10, stride=0.333, max_clips=None):
    """Frame offsets used by the repo's demo: frame i*stride*fps for i in 0..9."""
    offsets = torch.tensor(
        [int(math.floor(i * stride * fps)) for i in range(num_frames)], dtype=torch.long
    )
    n_clips = int(n_frames - offsets[-1].item())
    if n_clips <= 0:
        raise RuntimeError("video too short for a 10-frame / 3s clip")
    starts = list(range(n_clips))
    if max_clips is not None and len(starts) > max_clips:
        # evenly spaced subset covering the whole video
        sel = np.linspace(0, len(starts) - 1, max_clips).round().astype(int)
        starts = [starts[i] for i in sorted(set(sel.tolist()))]
    return offsets, starts


# --------------------------------------------------------------------------- #
#  instrumentation
# --------------------------------------------------------------------------- #
class Recorder:
    """
    Replaces SynoBlock.spatial_detection / temporal_detection with versions
    that additionally expose the per-component decomposition.  The returned
    'y' is mathematically identical to the original implementation (verified
    in verify_equivalence()).
    """

    def __init__(self, model):
        self.model = model
        self.blocks = list(model.model.encoder.decoder.decoder_layers)
        self.store = {}
        self._patched = False

    def reset(self):
        self.store = {"s_parts": [], "t_patch": [], "s_aff": []}

    def patch(self):
        if self._patched:
            return
        for blk in self.blocks:
            blk.spatial_detection = self._make_spatial(blk)
            blk.temporal_detection = self._make_temporal(blk)
        self._patched = True

    def _make_spatial(self, blk):
        store = self

        def fn(attrs):
            # ---- faithful re-implementation of SynoBlock.spatial_detection ----
            _k = attrs[blk.s_k_attr][:, :, 1:]      # drop CLS
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
                _k / (_k.norm(dim=-1, keepdim=True) + 1e-4),
            )
            s_aff = (s_aff * 100).softmax(dim=-1)
            s_mix = torch.einsum("ntql,ntlw->ntqw", s_aff, _v)   # (b,t,q,w)

            # original: y = s_mix.flatten(1,2).mean(dim=1)  ==  sum_{t,q} s_mix /(t*q)
            t_n, q_n = s_mix.shape[1], s_mix.shape[2]
            y_parts = s_mix.sum(dim=1) / (t_n * q_n)             # (b,q,w); sums to y
            y = y_parts.sum(dim=1)

            store.store["s_parts"].append(y_parts.detach().float().cpu())
            store.store["s_aff"].append(s_aff.detach().float().cpu())
            if blk.attn_record:
                blk.aff = s_aff
            return dict(s_q=s_q, y=y)

        return fn

    def _make_temporal(self, blk):
        store = self
        orig = type(blk).temporal_detection

        def fn(attrs):
            ret = orig(blk, attrs)                               # exact original
            store.store["t_patch"].append(ret["y"].detach().float().cpu())
            return ret

        return fn


# --------------------------------------------------------------------------- #
#  exact attribution algebra
# --------------------------------------------------------------------------- #
def _ln_linear_decomp(y_components, ln, head):
    """
    Exact additive decomposition of  d = (head(ln(y))[FAKE] - head(ln(y))[REAL])
    over an additive split  y = sum_k y_components[k].

    y_components : (K, D) tensor, summing over K gives y
    Returns (contribs (K,), const, d_total)
    """
    y_components = y_components.double()
    y = y_components.sum(0)
    D = y.shape[0]

    g = ln.weight.detach().double().cpu()
    b = ln.bias.detach().double().cpu()
    eps = ln.eps

    mu = y.mean()
    var = y.var(unbiased=False)
    sig = torch.sqrt(var + eps)

    W = head.weight.detach().double().cpu()
    bias = head.bias.detach().double().cpu()
    w = W[FAKE] - W[REAL]
    bias_diff = (bias[FAKE] - bias[REAL]).item()

    mu_k = y_components.mean(dim=1, keepdim=True)                 # (K,1)
    centred = y_components - mu_k                                 # (K,D)
    contribs = (centred * (w * g).unsqueeze(0)).sum(dim=1) / sig  # (K,)
    const = (w * b).sum().item() + bias_diff
    d_total = contribs.sum().item() + const
    assert D == g.shape[0]
    return contribs.numpy(), const, d_total


def _softmax2(d):
    """P(FAKE) for a 2-class logit difference d = logit_fake - logit_real."""
    return 1.0 / (1.0 + math.exp(-d))


def shapley_two_branch(p_s, p_t, p_full, prior=0.5):
    """
    Shapley values of the {spatial, temporal} pair for the fake-probability.

    v(empty)  = prior (0.5, no evidence)
    v({S})    = p_s     spatial head alone  (INFERENCE-TIME branch ablation)
    v({T})    = p_t     temporal head alone (INFERENCE-TIME branch ablation)
    v({S,T})  = p_full  the released S+T ensemble

    Faithfulness note (verified in xai/audit/audit_pipeline.py, check C6):
    `spatial_detection` reads only the frozen CLIP attributes and its own
    syno_embedding, and logits_s depends only on s_ln/s_head.  So for THESE
    fixed weights the spatial head output is numerically identical whether
    op_mode is ["S"] or ["S","T"].  This is therefore an exact ablation of
    the released checkpoint.  It is NOT the same thing as a model RE-TRAINED
    with op_mode=["S"], which would converge to different weights.

    phi_S + phi_T = p_full - prior  (efficiency).
    """
    phi_s = 0.5 * ((p_s - prior) + (p_full - p_t))
    phi_t = 0.5 * ((p_t - prior) + (p_full - p_s))
    return phi_s, phi_t


# --------------------------------------------------------------------------- #
#  main analysis of one clip batch
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def analyse_clips(model, recorder, clips, device="cuda"):
    """
    clips : (B, T, 3, 224, 224) normalised input
    Returns a list of per-clip dictionaries with the full attribution.
    """
    m = model.model                      # BinaryLinearClassifier
    recorder.reset()
    out = m(clips.to(device))

    logits_s = out["logits_s"].double().cpu()
    logits_t = out["logits_t"].double().cpu()
    logits_a = out["logits_a"].double().cpu()
    logits_f = out["logits"].double().cpu()

    p_s = logits_s.softmax(-1)[:, FAKE].numpy()
    p_t = logits_t.softmax(-1)[:, FAKE].numpy()
    p_a = logits_a.softmax(-1)[:, FAKE].numpy()
    p_f = logits_f.softmax(-1)[:, FAKE].numpy()

    # recorded per-layer tensors
    s_parts = torch.stack(recorder.store["s_parts"])   # (L, B, Q, 1024)
    t_patch = torch.stack(recorder.store["t_patch"])   # (L, B, 256)
    s_aff = torch.stack(recorder.store["s_aff"])       # (L, B, T, Q, 256)

    L, B, Q, Dsp = s_parts.shape
    Dtp = t_patch.shape[-1]

    s_ln, s_head = m.s_ln, m.s_head
    t_ln, t_head = m.t_ln, m.t_head

    results = []
    for i in range(B):
        # ---------- spatial branch: parts, layers, layer x part ----------
        sp_lp = s_parts[:, i]                              # (L, Q, 1024)
        c_part, const_s, d_s = _ln_linear_decomp(
            sp_lp.sum(0), s_ln, s_head)                    # over Q parts
        c_slayer, _, _ = _ln_linear_decomp(
            sp_lp.sum(1), s_ln, s_head)                    # over L layers
        c_lp, _, _ = _ln_linear_decomp(
            sp_lp.reshape(L * Q, Dsp), s_ln, s_head)       # joint
        c_lp = c_lp.reshape(L, Q)

        # ---------- temporal branch: patches, layers ----------
        tp_l = t_patch[:, i]                               # (L, 256)
        c_patch, const_t, d_t = _ln_linear_decomp(
            torch.diag(tp_l.sum(0).double()), t_ln, t_head)  # per-dimension
        c_tlayer, _, _ = _ln_linear_decomp(tp_l, t_ln, t_head)

        # ---------- fusion head: exact spatial / temporal split ----------
        _s = s_ln(s_parts[:, i].sum((0, 1)).to(device).float()).double().cpu()
        _t = t_ln(t_patch[:, i].sum(0).to(device).float()).double().cpu()
        Wa = m.a_head.weight.detach().double().cpu()
        ba = m.a_head.bias.detach().double().cpu()
        wa = Wa[FAKE] - Wa[REAL]
        a_sp = float((wa[:Dsp] * _s).sum())
        a_tp = float((wa[Dsp:] * _t).sum())
        a_bias = float(ba[FAKE] - ba[REAL])
        d_a = a_sp + a_tp + a_bias

        phi_s, phi_t = shapley_two_branch(
            float(p_s[i]), float(p_t[i]), float(p_f[i]))

        results.append(dict(
            # ---- probabilities ----
            p_spatial=float(p_s[i]), p_temporal=float(p_t[i]),
            p_fusion=float(p_a[i]), p_final=float(p_f[i]),
            # ---- head logit differences (fake - real) ----
            d_spatial=float(d_s), d_temporal=float(d_t), d_fusion=float(d_a),
            const_spatial=float(const_s), const_temporal=float(const_t),
            # ---- fusion head split ----
            fusion_spatial_term=a_sp, fusion_temporal_term=a_tp,
            fusion_bias=a_bias,
            # ---- shapley over branches ----
            shapley_spatial=phi_s, shapley_temporal=phi_t,
            # ---- exact within-branch attribution ----
            contrib_parts=c_part,                 # (4,)  lips/skin/eyes/nose
            contrib_spatial_layers=c_slayer,      # (24,)
            contrib_parts_layers=c_lp,            # (24,4)
            contrib_patches=c_patch,              # (256,)
            contrib_temporal_layers=c_tlayer,     # (24,)
            # ---- attention ----
            s_attention=s_aff[:, i].mean(1).numpy(),   # (L, Q, 256) time-averaged
        ))

    # free
    del out
    return results


@torch.inference_mode()
def verify_equivalence(model, recorder, clips, device="cuda", tol=1e-4):
    """
    Confirm the instrumented spatial_detection reproduces the pristine model's
    logits bit-for-bit (within fp tolerance), so all attributions describe the
    real released model and not a modified one.
    """
    m = model.model
    # pristine
    for blk in recorder.blocks:
        for name in ("spatial_detection", "temporal_detection"):
            if name in blk.__dict__:
                del blk.__dict__[name]
    recorder._patched = False
    clean = m(clips.to(device))["logits"].double().cpu().clone()

    recorder.patch()
    recorder.reset()
    patched = m(clips.to(device))["logits"].double().cpu().clone()

    err = (clean - patched).abs().max().item()
    return err, err < tol, clean, patched
