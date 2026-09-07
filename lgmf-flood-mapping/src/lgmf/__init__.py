"""LGMF: Lightweight Gated Modality Fusion for optical+SAR flood segmentation."""
from .config import LGMFConfig, DEFAULT_S2_BANDS
from .dataset import Sen1Floods11
from .model import build_model, LGMF, count_params
from .losses import segmentation_loss, consistency_loss, focal_tversky
from .metrics import SegMeter, flops_g
from .ema import ModelEMA
from .curriculum import (
    regime_probs, sample_present, eval_regimes, apply_present,
)
from .baselines import EarlyFusionNet, build_baseline, build_gated_unet

__all__ = [
    "LGMFConfig", "DEFAULT_S2_BANDS", "Sen1Floods11",
    "build_model", "LGMF", "count_params",
    "segmentation_loss", "consistency_loss", "focal_tversky",
    "SegMeter", "flops_g", "ModelEMA",
    "regime_probs", "sample_present", "eval_regimes", "apply_present",
    "EarlyFusionNet", "build_baseline", "build_gated_unet",
]
__version__ = "7.0.0"
