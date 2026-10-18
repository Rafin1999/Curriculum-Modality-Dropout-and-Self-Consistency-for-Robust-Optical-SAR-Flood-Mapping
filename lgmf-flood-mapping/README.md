# LGMF: Curriculum Modality Dropout and Self-Consistency for Robust Optical-SAR Flood Mapping

Code and experiment notebooks for the paper *"Curriculum Modality Dropout and
Self-Consistency for Robust Optical-SAR Flood Mapping with Missing Sensors"*
(under revision at *Scientific Reports*).

> **Status (October 2026): revision release, package `lgmf` v8.0.0.**
> This branch adds the controlled experiments run for the revision (notebooks
> `notebooks/revision/rev11` to `rev17`) and updates the package to v8. The
> training recipe code path is unchanged from v7; every number in the original
> Table 2 was re-derived from the saved checkpoints and reproduces within 0.0004.
> Archived release: **Zenodo DOI `10.5281/zenodo.XXXXXXX` (to be added at release)**.

## What the repository contains

A standard, lightweight early-fusion segmentation network (SegFormer MiT-B0,
3.72 M parameters) is trained to handle missing Sentinel-1 (SAR) or Sentinel-2
(optical) input with two architecture-agnostic training components:

- **Modality dropout** with a progressive schedule (the "curriculum"): training
  starts on both sensors and moves toward single-sensor inputs, biased toward
  the radar-only case.
- **SAR-only self-consistency**: the SAR-only prediction is pulled toward the
  network's own both-sensor prediction (one-sided; an optical-side term was
  also tested, see below). No teacher network, no reconstruction decoder.

### What the revision experiments show (test split, 3 seeds, Study 1 protocol)

These controls were added in response to reviewer requests. Pre-set rules:
a difference is called "helps/hurts" only if the mean exceeds 0.03 and has the
same sign in all three seeds; every difference also gets a 95 % event-blocked
bootstrap interval. Full tables are produced by the notebooks.

| Question | Finding |
|---|---|
| Does the missing-modality recipe matter? | Yes. SAR-only IoU rises by about +0.47 (SegFormer) and +0.35 (U-Net) over training on both sensors only. |
| Is the *schedule* needed? | No detectable effect: a fixed mix with the same measured exposure matches the curriculum (difference 0.000). Uniform dropout is slightly lower (-0.026). |
| Dropout vs consistency | Each alone recovers most of the gain; consistency adds a small amount on top (+0.046 SAR-only, event interval just above 0). |
| Architecture vs objective | With identical objectives SegFormer-B0, U-Net and MANet are comparable (all gaps below 0.03). |
| One model vs specialists | The general model matches or exceeds single-sensor specialists in the test region; on the Bolivia holdout the SAR specialist is better and steadier. |
| Gating / FiLM fusion | No improvement over plain concatenation; adds about 30 % fp32 latency. |
| Partial cloud | The recipe handles a missing, *flagged* optical image. It does **not** protect against undetected partial cloud: with cloud-shaped masks the SegFormer model drops below its own SAR-only level from 25 % cover. |
| Second dataset (SenForFlood) | No transfer of the recipe effect was shown. |

The Bolivia holdout is 15 chips from one event, so it is reported but not used
to choose a model. The official Sen1Floods11 split divides chips, not events:
10 of 11 events appear in train, validation and test.

## Repository layout

```
lgmf-flood-mapping/
├── src/lgmf/              # Package v8 (pip-installable; adds runner, ramdata, cloud, audit)
├── notebooks/             # Original 12 notebooks (paper v1), run in numeric order
├── notebooks/revision/    # Revision notebooks rev11 ... rev17 (this release)
├── data/splits/           # Dataset index JSONs + normalization stats (see data/splits/README.md)
├── figures/               # Architecture figures
├── scripts/               # Standalone figure-generation scripts
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
git clone https://github.com/Rafin1999/Curriculum-Modality-Dropout-and-Self-Consistency-for-Robust-Optical-SAR-Flood-Mapping.git
cd Curriculum-Modality-Dropout-and-Self-Consistency-for-Robust-Optical-SAR-Flood-Mapping/lgmf-flood-mapping
pip install -r requirements.txt
pip install -e src/          # installs the `lgmf` package (v8.0.0)
python -m lgmf.selftest_v8   # CPU self-test, about a minute
```

The notebooks were developed and run on Kaggle (T4 / P100 GPU). The revision
notebooks locate the package by searching `/kaggle/input/**/lgmf/__init__.py`
and require version 8 or later, so on Kaggle either upload the `src/lgmf` folder
as a dataset named `lgmf-v8` and attach it, or run `pip install -e src/` in the
first cell. Outside Kaggle, update the dataset paths at the top of each notebook.

## Data

- **Sen1Floods11** (Bonafilia et al., 2020): hand-labelled partition. Not
  included here; see the [official
  release](https://github.com/cloudtostreet/Sen1Floods11) and the split CSVs in
  the `sen1floods11` GCS bucket
  (`gs://sen1floods11/v1.1/splits/flood_handlabeled/`). This repository does not
  assert a licence for the dataset; follow the terms of the original release.
- **`data/splits/`** holds the derived paired file index and normalization
  statistics produced by `notebooks/01_data_audit.ipynb` (see its README).
- **Permanent-water layer** used for the flood-only metric: the JRC raster that
  ships with the Sen1Floods11 hand-labelled partition (pixels with value > 0 are
  excluded). Check the Sen1Floods11 data description for the exact product and
  version before citing it.
- Non-finite SAR values are set to zero **before** standardisation, which maps
  them to a fixed offset (not the mean) after standardisation. Missing-modality
  inputs use the same zero fill during training and evaluation.
- **SenForFlood** (revision notebook rev15): not redistributed; the notebook
  reads a preprocessed copy attached as a Kaggle dataset.

## Trained model weights and run outputs

Checkpoints (EMA weights, key `ema` in `best.pt`), per-chip results and config
files are not stored in git. They are archived at the Zenodo DOI above. Every
run directory `runs_e45/<run_id>/seed<seed>/` contains `config.json`,
`history.json` (per-epoch loss, measured regime shares, validation IoU),
`best.pt`, `perchip.csv.gz` and `result.json`.

## Original notebooks (paper v1)

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

## Revision notebooks (`notebooks/revision/`)

All use the **Study 1 protocol**: Focal-Tversky loss, AdamW (lr 3e-4, weight
decay 1e-4), cosine schedule, 45 epochs, batch 8, EMA 0.99, gradient clipping
1.0, consistency weight 0.5 where stated, seeds 42 / 1 / 2, model selection on
the mean validation IoU over the three availability regimes. Runs are resumable
(a registry of finished run ids is kept; a killed run is restarted from scratch).
Attach the outputs of earlier notebooks as inputs where listed.

| Notebook | Purpose | Runs / time (T4) | Needs as input |
|---|---|---|---|
| `rev11_audit_and_release.ipynb` | Reproduce Table 2 from checkpoints; split and leakage audit; imputation check; export release bundle | inference only | original checkpoints, index files |
| `rev12_factorial_controls_sensitivity.ipynb` | 2x2 dropout x consistency factorial, uniform / fixed-asymmetric / exposure-matched dropout controls, curriculum sensitivity | 27 runs, about 5 h | rev11 output |
| `rev13_backbones_specialists_symmetric.ipynb` | U-Net and MANet with the same recipe; SAR-only and optical-only specialists; two-sided consistency | 18 runs, about 3.7 h | rev12 output |
| `rev14_table4_rebuild_and_costs.ipynb` | Gate / FiLM / adapter controls at three seeds; parameters, FLOPs, latency, memory | 27 runs, about 7 h | rev11 output |
| `rev15_senforflood_study1_protocol.ipynb` | SegFormer recipe on/off on SenForFlood under the same protocol | 6 runs | SenForFlood dataset |
| `rev16_evaluation_statistics_cloud.ipynb` | Per-chip tables, Bolivia, flood-only metric, partition geography, event-blocked paired bootstrap, partial-cloud sweep | no training, about 13 min | outputs of rev11 to rev14 |
| `rev17_figures.ipynb` | Figures 1, 3-9 from the result files, with provenance lines | about 2 min | rev11 to rev16 outputs, Sen1Floods11 |

The original notebook numbers (01 to 12) and the revision notebook numbers
(rev11 to rev17) are different series.

## Package structure (`src/lgmf/`)

| Module | Contents |
|---|---|
| `config.py` | `LGMFConfig` dataclass - every model/train/curriculum hyperparameter, including all ablation flags |
| `dataset.py` | `Sen1Floods11` - paired S1/S2/label loading, band subset selection, `-1`-mask handling, input sanitization |
| `ramdata.py` | `RamDataset` - RAM-cached dataset with the same augmentation RNG order as `Sen1Floods11` |
| `model.py` | `LGMF` architecture: modality stems, masked renormalizing gate, FiLM conditioning, MLP and U-Net decoders |
| `baselines.py` | Early-fusion wrapper for `segmentation_models_pytorch` architectures + gated-fusion front end |
| `curriculum.py` | Modality-dropout schedule (`regime_probs`), regime sampling, evaluation regimes |
| `losses.py` | Masked BCE/Dice, Focal-Tversky, self-consistency loss |
| `metrics.py` | Regime-aware IoU/F1 with validity masking, FLOPs counter (works on a copy of the model) |
| `ema.py` | Exponential moving average of model weights |
| `engine.py` | Shared `train_one_epoch` / `evaluate` loops, now logging measured regime shares |
| `runner.py` | `run_experiment`: one call per run, writes config, history, checkpoint, per-chip file and result |
| `cloud.py` | Cloud-shaped optical masks with exact coverage, zero and bright-cloud fills |
| `audit.py` | Per-chip counts, flood-only mask, event extraction helpers |
| `viz.py` | Fixed colormap and figure helpers |
| `selftest_v8.py` | CPU self-test of the v8 additions |

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

## Reproducibility notes

- Standard deviations in the original paper use `ddof = 0`; new tables use `ddof = 1`.
  The convention is stated in each table caption.
- IoU is pooled over confusion counts, not averaged over chips. Chips with no
  reference water have per-chip IoU 0 by construction and are excluded from the
  qualitative figures (they are listed by `rev17`).
- Partial-cloud tests use synthetic, smooth masks with exact coverage and an
  illustrative bright-cloud fill. They are a stress test, not a cloud detector.
- Model selection for the general model uses the mean validation IoU over the
  three regimes; specialists are selected on their own regime.

## Citation

See `CITATION.cff`. If you use this code, please cite:

```bibtex
@article{rafin2026lgmf,
  title   = {Curriculum Modality Dropout and Self-Consistency for Robust
             Optical-SAR Flood Mapping with Missing Sensors},
  author  = {Rafin, Nafiz Imtiaz and Ghose, Swattic and Jawad, Rubaiyet Hossain
             and Alam, Md. Golam Rabiul},
  journal = {Under review},
  year    = {2026}
}
```

## License

MIT. See `LICENSE`.
