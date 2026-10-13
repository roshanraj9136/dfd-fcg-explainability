# DFD-FCG Explainability

Exact spatial-vs-temporal attribution for **DFD-FCG** (CVPR 2025), a video deepfake
detector built on a frozen CLIP ViT-L/14 encoder.

This repository answers a specific question: **when the model says FAKE or REAL, what
produced that answer?** It reports, for every 3-second clip, how much of the decision came
from the spatial branch vs the temporal branch, which facial component carried the evidence,
and where on the face the temporal evidence sits.

Upstream model: [aiiu-lab/DFD-FCG](https://github.com/aiiu-lab/DFD-FCG) ·
paper: [arXiv:2404.05583](https://arxiv.org/abs/2404.05583)

---

## Why the attribution is exact

DFD-FCG happens to be unusually amenable to exact decomposition. Its final probability is an
explicit mean of three head softmaxes:

```
P_final = [ softmax(logits_s) + softmax(logits_t) + softmax(logits_a) ] / 3
```

and each head is `LayerNorm → Linear` applied to a feature that is itself a **sum** of parts
(24 CLIP layers; 4 facial components; 256 patches).

Because the decision is a 2-class softmax, the whole verdict reduces to one scalar
`d = logit(FAKE) − logit(REAL)`, with `P(FAKE) = σ(d)`. And because the LayerNorm mean is
additive over the parts, `d` splits with **no residual term**:

```
d = Σ_k contribution(part k) + constant
```

So the contributions are not gradient surrogates or sampling estimates — they sum back to the
model's own logit. Verified to **1e-16** in float64 (`xai/audit/audit_math2.py`), and the
instrumented model reproduces the untouched model's logits to **8.9e-08**.

Branch split uses Shapley values over `{S, T}`, which satisfies efficiency exactly:
`φ_S + φ_T = P_final − 0.5`.

---

## Results

Both videos bundled with the upstream repository are FaceForensics++ samples, classified
correctly. 60 overlapping 3-second clips per video.

| video | truth | verdict | mean P(FAKE) | clips correct | spatial | temporal |
|---|---|---|---|---|---|---|
| `000.mp4` | REAL | **REAL** ✓ | 0.421 | 88% (53/60) | 29.1% | **70.9%** |
| `000_003.mp4` | FAKE | **FAKE** ✓ | 0.932 | 100% (60/60) | **50.4%** | 49.6% |

### Why it said FAKE — the eyes

Spatial-branch contribution to `logit(FAKE) − logit(REAL)`:

| component | FAKE | share | REAL | share | change |
|---|---|---|---|---|---|
| **eyes** | **+1.039** | **45.6%** | +0.245 | 24.5% | +0.794 |
| skin | +0.617 | 27.1% | +0.079 | 8.0% | +0.538 |
| nose | +0.438 | 19.2% | **−0.601** | 60.3% | **+1.039** |
| lips | +0.183 | 8.0% | +0.071 | 7.2% | +0.112 |

The **eyes** carry the FAKE verdict. In the real video the **nose** is instead the strongest
REAL vote (−0.601); after the swap every component turns towards FAKE, the nose shifting most.

### Why it said REAL — motion consistency

The real verdict is **70.9% temporal**. The evidence for authenticity is that motion is
coherent across frames, not that individual frames look clean. This asymmetry is the main
qualitative finding: **the two verdicts are not mirror images of each other.**

### Layer depth

Spatial evidence concentrates in **deep** layers (19, 17, 20 — semantic/identity features);
temporal evidence also draws on **shallow** layers (4, 18, 8 — texture and motion). The two
branches genuinely read the frozen encoder at different depths, supporting the paper's design
claim.

### Branch redundancy

On clean video **either branch alone already reaches the correct verdict** (spatial-only 0.912,
temporal-only 0.908 for the fake). The branches are redundant rather than complementary here;
fusion adds confidence (0.91 → 0.93), not coverage.

### Robustness — the real weakness is quality, not forgery type

Using the authors' own perturbation protocol (7 types × 3 severities, parameters taken verbatim
from `src/preprocess/robustness/`):

- **12/21 false positives** — REAL wrongly called FAKE
- **2/21 false negatives** — FAKE wrongly called REAL

The failure is strongly **asymmetric**: degradation pushes scores **up**, so the model mistakes
low quality for manipulation. Blur, JPEG-style resampling and H.264 compression flip the real
video at *every* severity level. Gaussian noise does the reverse and hides the fake. Block-wise
noise is the only fully robust condition.

A quality-aware threshold or compression augmentation would likely help more than any
additional branch.

---

## Figures

| | |
|---|---|
| `f1_architecture.png` | model and the three decision heads |
| `f2_verdict_timeline.png` | per-clip decision across each video |
| `f3_branch_share.png` | spatial vs temporal share |
| `f4_facial_parts.png` | per-component contribution |
| `f5_temporal_maps.png` | 16×16 temporal evidence on the face |
| `f6_zones.png` | area-normalised zone attribution |
| `f7_layers.png` | per-layer contribution across 24 layers |
| `f8_attention.png` | where each of the 4 FCG queries looks |
| `f9_counterfactual.png` | branch-ablation verdicts |
| `f10_robustness.png` | robustness sweep with branch reliance |

---

## Reproducing

Requires the upstream repo, its released checkpoint, and a CUDA GPU
(peak 2.5 GB VRAM at batch size 2 — fits a 4 GB card).

```bash
git clone https://github.com/aiiu-lab/DFD-FCG && cd DFD-FCG
# download the authors' checkpoint into checkpoint/{setting.yaml,weights.ckpt}
cp -r /path/to/this/repo/xai .

# preprocessing (landmarks, then aligned face crops)
python -m src.preprocess.fetch_landmark_bbox \
  --root-dir=./resources --video-dir=videos --fdata-dir=frame_data --glob-exp="*"
python -m src.preprocess.crop_main_face \
  --root-dir=./resources --video-dir=videos --fdata-dir=frame_data \
  --crop-dir=cropped --glob-exp="*" --workers=1

python xai/smoke.py                                 # exactness self-test
python xai/run_analysis.py --max-clips 60           # main analysis
python xai/make_figures.py                          # figures + derived metrics
python xai/run_robustness.py --levels 1 3 5         # robustness sweep
python xai/merge_robustness.py
python xai/build_deck.py                            # presentation
python xai/audit/audit_math2.py                     # verify exactness
python xai/audit/audit_pipeline.py                  # verify pipeline
```

On Windows set `TORCHDYNAMO_DISABLE=1` (the `face_alignment` package tries to use Triton,
which is unavailable there).

### Files

```
xai/
  engine.py            attribution algebra + model instrumentation
  run_analysis.py      main per-clip analysis
  make_figures.py      figures and derived metrics
  run_robustness.py    authors' perturbation protocol
  merge_robustness.py  merge sweep chunks + robustness figure
  build_deck.py        presentation builder
  check_deck.py        deck layout validator
  smoke.py             end-to-end exactness self-test
  audit/               independent correctness audits
  out/                 report.json, derived.json, robustness.json, arrays.npz
  figs/                all figures
```

---

## Limitations

- **Two videos only.** These are the samples shipped with the upstream repository.
  FaceForensics++, Celeb-DF, DFDC and DeeperForensics are EULA-gated and cannot be obtained
  without a signed agreement, so **no cross-dataset generalisation is claimed here.** The
  robustness sweep is what extends the evaluation beyond a single clean condition.
- Single-branch numbers are **inference-time ablations** of the released checkpoint. The
  spatial head provably never reads the temporal branch, so the ablation is exact for these
  weights — but it is *not* the same as a model re-trained with `op_mode=["S"]`, which would
  converge to different weights.
- Patch-to-zone mapping uses nearest-landmark assignment on a 16×16 grid, so zone boundaries
  are approximate even though the per-patch contributions are exact.
- The robustness sweep uses 8 clips per condition; the main analysis uses 60.

## License

[MIT](LICENSE) for the explainability code in this repository. See [NOTICE](NOTICE) for third-party license scope.

The DFD-FCG model code and weights are by Han, Huang, Hua and Chen and are released for
**research purposes only** — see the [upstream repository](https://github.com/aiiu-lab/DFD-FCG).
The sample videos are governed by the FaceForensics++ Terms of Use.

```bibtex
@inproceedings{cvpr25_dfd_fcg,
  title={Towards More General Video-based Deepfake Detection through Facial
         Component Guided Adaptation for Foundation Model},
  author={Yue-Hua Han and Tai-Ming Huang and Kai-Lung Hua and Jun-Cheng Chen},
  booktitle={CVPR},
  year={2025}
}
```
