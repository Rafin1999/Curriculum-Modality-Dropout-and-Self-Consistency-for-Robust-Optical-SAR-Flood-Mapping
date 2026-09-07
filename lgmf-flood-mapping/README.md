# LGMF: Curriculum Modality Dropout and Self-Consistency for Robust Optical-SAR Flood Mapping

Code and experiment notebooks for the paper *"Curriculum Modality Dropout and
Self-Consistency for Robust Optical-SAR Flood Mapping with Missing Sensors."*

We show that a standard, lightweight early-fusion segmentation network (SegFormer,
3.72M parameters) becomes robust to missing Sentinel-1 (SAR) or Sentinel-2
(optical) input when trained with two simple, architecture-agnostic components:

- **Curriculum modality dropout** - a fixed, zero-parameter schedule that
  progressively trains the network on single-sensor inputs, biased toward the
  operationally dominant radar-only case (clouds block optical exactly when
  floods occur).
- **Self-consistency loss** - single-modality predictions are pulled toward the
  network's own full-modality prediction, with no separate teacher network, no
  reconstruction decoder, and no synthesized imagery.

Four independent attempts at specialized fusion architectures (gated fusion,
FiLM conditioning, shared modality-specific encoders) did **not** improve on
plain channel concatenation under this training scheme, indicating that
missing-modality robustness here is primarily a training problem, not an
architectural one.

## Repository layout

```
lgmf-flood-mapping/
├── src/lgmf/          # Final model package (v7) - pip-installable
├── notebooks/         # 12 notebooks, run in numeric order (see below)
├── data/splits/        # Dataset index JSONs + normalization stats (small, included)
├── figures/            # Paper figures (architecture diagram, generated vector graphics)
├── scripts/             # Standalone figure-generation scripts
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/lgmf-flood-mapping.git
cd lgmf-flood-mapping
pip install -r requirements.txt
pip install -e src/          # installs the `lgmf` package
```

Notebooks were developed and run on Kaggle (P100/T4 GPU, `Internet: On` for
pretrained-weight downloads). They will run in any environment with a CUDA GPU
provided the dataset paths at the top of each notebook are updated to point at
your local copy of Sen1Floods11 and the files in `data/splits/`.

## Data

- **Sen1Floods11** (Bonafilia et al., 2020): hand-labelled partition. Not
  included in this repo due to size; see the [official
  release](https://github.com/cloudtostreet/Sen1Floods11) or the mirror used
  during development, and the official train/valid/test split CSVs from the
  `sen1floods11` GCS bucket (`gs://sen1floods11/v1.1/splits/flood_handlabeled/`).
- **`data/splits/`** in this repo contains the derived, paired file index
  (`train_index.json`, `val_index.json`, `test_index.json`,
  `bolivia_index.json`) and channel normalization statistics
  (`norm_stats.json`) produced by `notebooks/01_data_audit.ipynb`. These are
  small (a few MB) and included directly.

## Trained model weights

Checkpoints are **not** stored in this git repository (several GB across all
runs). They are hosted separately at:

**`<add your Hugging Face / Zenodo link here once uploaded>`**

This includes: the main model (`segformer_concat`), all ablation checkpoints
(3 seeds each), the baseline models (U-Net, MANet, both with and without
curriculum), the backbone-sweep checkpoints, and the fusion-injection
experiment checkpoints. Each subfolder mirrors the `run_name` used in the
corresponding notebook.

## Notebooks

Run in this order to reproduce the paper end to end. Each notebook states its
required Kaggle dataset attachments and expected runtime in its first cell.

| # | Notebook | Purpose | Produces |
|---|----------|---------|----------|
| 01 | `01_data_audit.ipynb` | Verify dataset integrity, build paired index, compute normalization stats, fetch official split | `data/splits/*.json` |
| 02 | `02_train.ipynb` | Config-driven single-run trainer for the LGMF model and its ablations | Checkpoint per run |
| 03 | `03_ablations_multiseed.ipynb` | Full model + 5 ablations, 3 seeds each, with determinism controls | Ablation checkpoints + summary table |
| 04 | `04_deep_eval_bolivia_floodonly.ipynb` | Re-evaluate ablation checkpoints on the out-of-region Bolivia holdout and flood-only metric | Extended ablation evidence |
| 05 | `05_baselines.ipynb` | Train external baselines (U-Net, MANet, with/without curriculum) on the identical split | Baseline checkpoints |
| 06 | `06_backbone_sweep.ipynb` | Single-seed screen of 7 encoder/decoder combinations, incl. SegFormer/MiT | Backbone ranking table |
| 07 | `07_segformer_confirm.ipynb` | 3-seed confirmation of the winning SegFormer config + gated-fusion comparison | Final model checkpoints |
| 08 | `08_gated_fusion_injection.ipynb` | Inject gate+FiLM fusion onto the strong U-Net backbone (decisive architecture test) | Injection checkpoints |
| 09 | `09_unet_decoder_headtohead.ipynb` | LGMF with a U-Net-style skip-connection decoder vs. baselines | v6 checkpoints |
| 10 | `10_hardsplit_headtohead.ipynb` | Gated-fusion vs. concatenation on Bolivia/flood-only splits | Hard-split evidence |
| 11 | `11_final_results.ipynb` | Consistency ablation on the final backbone + full **test-set** evaluation | `final_results.json`, degradation & Pareto figures |
| 12 | `12_paper_figures.ipynb` | Qualitative prediction grids, dataset overview, curriculum plot, failure cases | All remaining paper figures |

Notebooks 06, 08, and 09 document architectural alternatives that were tested
and **did not** outperform the final design; they are included for
transparency and because the paper reports them as controlled negative
results (see Section 4.6-4.7 of the paper).

## Package structure (`src/lgmf/`)

| Module | Contents |
|---|---|
| `config.py` | `LGMFConfig` dataclass - every model/train/curriculum hyperparameter, including all ablation flags |
| `dataset.py` | `Sen1Floods11` - paired S1/S2/label loading, band subset selection, `-1`-mask handling, input sanitization |
| `model.py` | `LGMF` architecture: modality stems, masked renormalizing gate, FiLM conditioning, MLP and U-Net decoders |
| `baselines.py` | Early-fusion wrapper for `segmentation_models_pytorch` architectures + gated-fusion front end for injection experiments |
| `curriculum.py` | Structured modality-dropout schedule (`regime_probs`) and regime sampling |
| `losses.py` | Masked BCE/Dice, Focal-Tversky, self-consistency loss |
| `metrics.py` | Regime-aware IoU/F1 with validity masking, FLOPs counter |
| `ema.py` | Exponential moving average of model weights |
| `engine.py` | Shared `train_one_epoch` / `evaluate` loops used identically across every experiment |
| `viz.py` | Fixed colormap and figure helpers for paper-consistent visualizations |

## Reproducing a specific result

Every ablation and architectural variant is a **config change**, not a code
change, keeping every comparison controlled. For example, to reproduce the
"no self-consistency" ablation:

```python
from lgmf.config import LGMFConfig
cfg = LGMFConfig()
cfg.consistency_weight = 0.0   # the only change from the full model
```

See the `CONFIG` cell at the top of `notebooks/02_train.ipynb` and
`notebooks/03_ablations_multiseed.ipynb` for the full set of flags used to
produce each row of the paper's tables.

## Citation

If you use this code, please cite:

```bibtex
@article{rafin2026lgmf,
  title   = {Curriculum Modality Dropout and Self-Consistency for Robust
             Optical-SAR Flood Mapping with Missing Sensors},
  author  = {Rafin, Nafiz Imtiaz and Ghose, Swattic and Jawad, Rubaiyet Hossain
             and Alam, Md. Golam Rabiul},
  journal = {TBD},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
