"""Central configuration for LGMF.

Every notebook builds one of these and passes it around. Ablations are
just this object with one field changed, which is what keeps the main
run and its ablations strictly comparable.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import json


# Default Sentinel-2 band subset (indices into the 13-band S2Hand stack).
# Canonical Sen1Floods11 S2 order is:
#   0:B1 1:B2 2:B3 3:B4 4:B5 5:B6 6:B7 7:B8 8:B8A 9:B9 10:B10 11:B11 12:B12
# We keep B2,B3,B4 (blue/green/red), B8 (NIR), B11,B12 (SWIR) -> water signal.
# VERIFY this order against your mirror in NB1 before trusting it.
DEFAULT_S2_BANDS: Tuple[int, ...] = (1, 2, 3, 7, 11, 12)


@dataclass
class LGMFConfig:
    # ---- data ----
    data_root: str = "/kaggle/input/sen1floods11-dataset"
    split_dir: str = "/kaggle/input/lgmf-index"   # produced by NB1
    s2_bands: Tuple[int, ...] = DEFAULT_S2_BANDS
    img_size: int = 512
    ignore_index: int = -1                        # no-data / cloud in labels
    # normalization: if None, falls back to built-in clip+scale (see dataset.py)
    s1_mean: Optional[List[float]] = None
    s1_std: Optional[List[float]] = None
    s2_mean: Optional[List[float]] = None
    s2_std: Optional[List[float]] = None

    # ---- model ----
    stem_dim: int = 32
    encoder_name: str = "efficientnet_lite0"      # any timm features_only model
    encoder_pretrained: bool = True               # ImageNet init (needs Internet: On)
    decoder_dim: int = 128
    decoder: str = "mlp"                           # 'mlp' (light) | 'unet' (skip-connections, v6)
    stem_stride: int = 2                           # 2 = v3 (safe); 1 = full-res stem (finer edges, ~2x memory)
    # ABLATION KNOBS -----------------------------------------------------
    film_on: bool = True                          # modality-presence FiLM
    gate_mode: str = "renorm"                     # 'renorm' | 'concat' | 'sum'
    fusion_stages: str = "single"                 # 'single' | 'multi' (multi = extension)
    # --------------------------------------------------------------------

    # ---- loss ----
    loss_type: str = "dice_bce"                   # 'dice_bce' | 'focal_tversky'
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    consistency_weight: float = 0.5               # 0 disables self-consistency
    pos_weight: float = 1.0                       # BCE positive-class weight
    # Focal-Tversky (used when loss_type='focal_tversky'); alpha>beta favors
    # recall -> lifts SAR-only, which under-detects water.
    tversky_alpha: float = 0.7                    # false-negative penalty
    tversky_beta: float = 0.3                     # false-positive penalty
    focal_gamma: float = 0.75                     # focal exponent on Tversky

    # ---- curriculum (structured modality dropout) ----
    use_curriculum: bool = True
    # if use_curriculum is False, fixed probs below are used every epoch
    fixed_regime_probs: Tuple[float, float, float] = (0.6, 0.2, 0.2)  # both, s1, s2
    # operational prior the curriculum ramps toward (SAR-only up-weighted)
    final_regime_probs: Tuple[float, float, float] = (0.4, 0.45, 0.15)
    warmup_epochs: int = 5                        # both-only before dropout kicks in

    # ---- train ----
    epochs: int = 60
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0                        # max grad-norm (training stability)
    num_workers: int = 2
    seed: int = 42
    pretrain_on_weak: bool = False
    amp: bool = True
    use_ema: bool = True                          # weight EMA (cheap, stabilizes + small IoU gain)
    ema_decay: float = 0.999
    ckpt_dir: str = "/kaggle/working/ckpt"
    run_name: str = "lgmf_main"

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LGMFConfig":
        with open(path) as f:
            d = json.load(f)
        d["s2_bands"] = tuple(d["s2_bands"])
        return cls(**d)
