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

The spatial branch can be decomposed **two independent ways**, and they agree.

**View 1 — by the model's own component queries** (feature space; each query attends over all
256 patches):

| component query | FAKE | share | REAL | share | change |
|---|---|---|---|---|---|
| **eyes** | **+1.039** | **45.6%** | +0.245 | 24.5% | +0.794 |
| skin | +0.617 | 27.1% | +0.079 | 8.0% | +0.538 |
| nose | +0.438 | 19.3% | **−0.601** | 60.3% | **+1.039** |
| lips | +0.183 | 8.1% | +0.071 | 7.2% | +0.112 |

**View 2 — by patch region** (image space; the 16×16 grid cut into landmark-defined regions,
which is the spatially localised answer):

| face region | patches | FAKE share | FAKE signed | REAL share | REAL signed |
|---|---|---|---|---|---|
| **eyes** | 46–50 | **47.7%** | +1.1507 | 37.7% | +0.4331 |
| skin | 70–75 | 33.0% | +0.7955 | 2.9% | +0.0337 |
| lips | 22–27 | 13.7% | +0.3299 | 3.6% | +0.0416 |
| **nose** | 18 | 5.6% | −0.1347 | **55.7%** | **−0.6402** |

The **eyes** carry the FAKE verdict (45.6% by query, 47.7% by region — two different methods,
same conclusion). The **nose** is the strongest REAL vote in the authentic video (−0.640,
55.7% of its spatial evidence).

This works because the spatial feature is itself a sum over patches —
`y_s = (1/TQ) Σ_t Σ_q Σ_l s_aff[t,q,l]·v[t,l]` — so grouping patches into regions and adding
their contributions stays exact. Verified to **4e-16**.

### The two branches read different regions

Put on identical region definitions:

| region | SPATIAL (fake) | TEMPORAL (fake) | SPATIAL (real) | TEMPORAL (real) |
|---|---|---|---|---|
| eyes | **47.7%** | 30.8% | 37.7% | 0.9% |
| skin | 33.0% | **41.9%** | 2.9% | **66.6%** |
| lips | 13.7% | 18.5% | 3.6% | 23.4% |
| nose | 5.6% | 8.8% | **55.7%** | 9.1% |

Appearance evidence concentrates in the **eyes**, motion evidence in the **skin**. So although
either branch alone reaches the correct verdict on clean video, they are reading genuinely
different evidence rather than duplicating each other.

### One FCG query drifts off its own anatomy

Having both views allows a cross-check: *does the query named "eyes" actually collect its
evidence from eye patches?*

| query | own-region share (REAL) | own-region share (FAKE) | strongest region (FAKE) | |
|---|---|---|---|---|
| eyes | 69.2% | 88.6% | eyes | localised ✓ |
| skin | 79.4% | 79.5% | skin | localised ✓ |
| lips | 59.6% | 86.0% | lips | localised ✓ |
| **nose** | 86.2% | **9.1%** | **skin** (77.1%) | **drifted ✗** |

Three of four queries sit on their own anatomy. The **nose** query is well localised on the
authentic face (86.2%) but **collapses to 9.1% once the face is swapped**, drawing 77% of its
evidence from neighbouring skin instead.

Interpretation: FCG anchors each query to *real* facial features, so when a swap replaces the
nose it no longer matches that prototype and the attention slides onto adjacent skin. This is
a concrete, measured limitation of FCG, and it explains precisely why the two views disagree
about the nose (19.3% by query vs 5.6% by region).

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
| `f4_facial_parts.png` | per-component contribution (query view) |
| `f5_temporal_maps.png` | 16×16 temporal evidence on the face |
| `f6_zones.png` | area-normalised zone attribution |
| `f7_layers.png` | per-layer contribution across 24 layers |
| `f8_attention.png` | where each of the 4 FCG queries looks |
| `f9_counterfactual.png` | branch-ablation verdicts |
| `f10_robustness.png` | robustness sweep with branch reliance |
| `f11_regions.png` | **patch regions + per-region spatial attribution** |
| `f12_region_validation.png` | **cross-validation of the two views, branch comparison** |

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
python xai/run_region_analysis.py --max-clips 30    # patch-region attribution
python xai/make_region_figures.py                   # region figures
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
  run_region_analysis.py   exact per-patch / per-region spatial attribution
  make_region_figures.py   region figures + cross-validation
  run_robustness.py    authors' perturbation protocol
  merge_robustness.py  merge sweep chunks + robustness figure
  build_deck.py        presentation builder
  check_deck.py        deck layout validator
  smoke.py             end-to-end exactness self-test
  audit/               independent correctness audits
  out/                 report.json, derived.json, regions.json, robustness.json, *.npz
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
- Region boundaries come from nearest-landmark assignment on a 16×16 grid, so a patch
  straddling two regions is assigned to one. The per-patch contributions are exact; the
  grouping into named regions is the approximate step.
- The robustness sweep uses 8 clips per condition; the main analysis 60, the region analysis 30.

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
