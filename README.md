# SAR-Optical-Fusion-Water-Body-Segmentation

**DualSegFormer** — a transformer-based cross-modal fusion architecture that fuses Sentinel-1 SAR and Sentinel-2 optical imagery for general-purpose water body segmentation. Rather than presenting fusion as self-evidently superior, this work benchmarks it against single-modality models and simpler fusion strategies under an identical, seeded protocol to test whether the added complexity is actually worth it.

This is not a flood detection study — Sen1Floods11 was used solely as a source of paired, quality-controlled Sentinel-1/Sentinel-2 chips with hand-labeled water masks; the task is general water/non-water segmentation.

## Repository Structure

| Folder | Contents |
|---|---|
| `1_Sentinel2C_Analysis_Image/` | Raw Sentinel-2C product inspection (metadata, band structure) |
| `2_Sentinel2C_Image_Analysis/` | True/false-colour composites and spectral-index water mask |
| `3_Sentinel2C_Patch_Extraction/` | Patch extraction into `water` / `no_water` classes |
| `4_CNN_Model/` | Baseline CNN: training curves and Grad-CAM |
| `5_UNet_Model/` | Baseline UNet: training curves and Grad-CAM |
| `6_Sentinel2C_Pipeline/` | End-to-end optical-only pipeline (CNN + UNet) |
| `S1S2_Fusion_Architecture/` | Final model: notebook, ablation results, figures, Grad-CAM, real-world inference |

## DualSegFormer Architecture

- Two SegFormer (MiT-B0) encoders, pretrained on ImageNet-1k, adapted to each modality's channel count by averaging and repeating the pretrained patch-embedding filters (not reinitialized from scratch)
- **Sentinel-2 branch** — 8 channels: B2, B3, B4, B8, B11, B12, NDWI, MNDWI
- **Sentinel-1 branch** — 3 channels: VV, VH, VV−VH
- Bidirectional cross-modal attention at all 4 encoder scales (strides 4/8/16/32, spatial-reduction ratios 8/4/2/1), each modality using the other as key/value and vice versa
- Fused features combined via element-wise addition, passed to a shared SegFormer-style all-MLP decoder
- ~8.29M parameters total

## Ablation Variants Tested

Eight configurations, each trained under an identical protocol:

1. **Full bidirectional, multi-scale (proposed)**
2. Early concatenation (channel-stacked input, single shared encoder)
3. Late average fusion (no cross-attention; combined only before the decoder)
4. Unidirectional attention (only Sentinel-2 attends to Sentinel-1)
5. Single-scale attention (bidirectional attention at the coarsest scale only)
6. Sentinel-1 only
7. Sentinel-2 only
8. Full bidirectional, random init (no ImageNet pretraining)

## Training Configuration

- **Dataset:** Sen1Floods11, 446 hand-labeled chips, split 312 train / 66 val / 68 held-out test (70/15/15)
- **Optimizer:** AdamW, lr = 6×10⁻⁵, weight decay = 1×10⁻⁴
- **Batch size:** 2, up to 15 epochs
- **Loss:** BCE + Dice, both masked to exclude invalid (−1) label pixels
- **Seeds:** each of the 8 configurations trained across 3 seeds (7, 42, 123) — 24 total runs — with Python/NumPy/PyTorch RNGs seeded and cuDNN in deterministic mode
- Best-validation-IoU checkpoint retained per run for test-set evaluation

## Results — Ablation (test set, mean ± std across 3 seeds)

| Configuration | Test IoU |
|---|---|
| **Full bidirectional multi-scale (proposed)** | **0.578 ± 0.010** |
| Early concatenation | 0.577 ± 0.011 |
| Sentinel-2 only | 0.575 ± 0.004 |
| Single-scale attention | 0.571 |
| Late average fusion | 0.562 |
| Unidirectional attention | 0.549 |
| Full bidirectional, random init | 0.533 |
| Sentinel-1 only | 0.419 |

Per-chip (macro) IoU 0.578, median per-chip IoU 0.634, per-chip std 0.281, pooled (micro) IoU 0.831 (seed 42).

**Finding:** the proposed bidirectional model is the best-performing configuration on average and never the worst, but its margin over early concatenation and Sentinel-2-only is within one standard deviation across seeds. Dropping ImageNet pretraining costs more IoU (−0.044) than switching fusion strategy in most comparisons — pretraining matters more than the specific fusion mechanism.

## Modality Robustness (full bidirectional model, seed 42)

| Condition | Test IoU | Drop from clean |
|---|---|---|
| Clean (both modalities) | 0.5775 | — |
| Sentinel-1 removed | 0.2776 | −0.2999 |
| Sentinel-2 removed | 0.1097 | −0.4678 |
| Sentinel-1 + Gaussian noise (σ=0.15) | 0.5684 | −0.0091 |
| Sentinel-2 + Gaussian noise (σ=0.15) | 0.3989 | −0.1786 |

The trained model leans heavily on Sentinel-2: removing it costs far more than removing Sentinel-1, and it is nearly insensitive to noise on Sentinel-1 while noise on Sentinel-2 costs substantially more.

## Real-World Inference: Mahanadi Basin & Tungabhadra Reservoir

No ground truth exists for either zone; Google Dynamic World V1 is used only as a coarse reference, not validation.

Inference uses overlapping sliding-window tiling (512×512 tiles, 256-pixel stride, 50% overlap) with predictions averaged in overlapping regions, and pixels excluded from thresholding if either modality lacks valid coverage — fixing a no-data misclassification artifact from an earlier non-overlapping version of this pipeline.

| Zone | Size (px) | No-data | Water among valid pixels |
|---|---|---|---|
| Mahanadi Basin | 8,913 × 10,020 | 38.0% | **1.6%** (was 55.8% under the earlier, non-blended pipeline) |
| Tungabhadra Reservoir | 2,228 × 3,341 | 13.9% | 7.1% (unchanged — its no-data region never overlapped predicted water) |

The corrected Mahanadi figure is now plausible given the zone's agricultural/built-up land cover.

## Interpretability (Grad-CAM)

Applied to the decoder's fusion layer, with the full patch-selection funnel logged:
- **Mahanadi:** 234 candidates checked → 84 excluded for insufficient valid coverage → 3 more excluded for low reference water → 12 saved
- **Tungabhadra:** 84 candidates checked → 12 excluded for invalid coverage → 7 excluded for low reference water → 12 saved

Attention concentrates on genuine water features (river channels, stream inlets) in both zones, with weaker activation on smaller, scattered water bodies — consistent with the model's reduced sensitivity to small water bodies noted in the ablation results.

## Known Limitations

- **Split is chip-level, not event-level.** An event-identifier parsing bug caused every chip to be treated as its own event, so the intended 70/15/15 event-grouped split behaved as a random chip-level split. The bug has been fixed in the codebase, but all results in this repository predate that fix. Event-level leakage between splits is not yet ruled out; retraining under the corrected split is recommended before cross-event generalization claims are made.
- **Margins between configurations are within seed variability** for at least two comparisons (early concatenation, Sentinel-2-only). Three seeds is enough to report variability honestly, not enough to establish statistical significance.
- **Small dataset** — 446 labeled chips across 8 configurations under comparison.
- **No-data handling is fixed at inference time only.** The training pipeline was not modified to expose the model to synthetic no-data/all-zero inputs; a no-data-aware training regime has been designed for future use but not yet applied to these checkpoints.
- **Mahanadi/Tungabhadra results are qualitative** — no independent ground truth exists; Dynamic World V1 is a coarse automated reference only.

## Model Checkpoints

Checkpoints are excluded from this repository due to file size. Available via Google Drive:

- CNN baseline: [[link](https://drive.google.com/file/d/1_zJya_K_-6-pkzukNRJjzl2fXIJ-WlGB/view?usp=drive_link)]
- UNet baseline: [[link](https://drive.google.com/file/d/1yqnl6B_XFieMg_aU-34C0sZNcUnlM3Xy/view?usp=drive_link)]
- DualSegFormer checkpoints — all 8 configurations × 3 seeds (24 files): [[link](https://drive.google.com/drive/folders/1FimPCk9CiKbEb4xoOR0YnpLI00OswhJy?usp=sharing)]

## Notes

- Raw satellite imagery (`.tif`, `.jp2`) and model checkpoints (`.pth`) are excluded from version control; source data can be regenerated via Google Earth Engine.
- This repository is structured to show the full development history of the project — from initial single-scene optical experiments through to the final controlled fusion study — rather than only the final result.
