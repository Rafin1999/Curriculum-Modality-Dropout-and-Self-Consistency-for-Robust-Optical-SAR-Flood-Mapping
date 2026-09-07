"""Sen1Floods11 dataset.

Returns, per chip, BOTH modalities plus the label and a validity mask.
Modality dropout is NOT applied here -- the training loop applies it via
`curriculum.sample_present` so that the same dataset serves every regime.
At eval time you pass an explicit `present` flag to the model.

Expects an index file (built by NB1) listing paired file stems. The index
is a JSON list of dicts: {"s1": path, "s2": path, "label": path, "jrc": path}.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_raster(path: str, bands: Optional[List[int]] = None) -> np.ndarray:
    """Read a GeoTIFF -> float32 array [C, H, W]. rasterio imported lazily."""
    try:
        import rasterio
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "rasterio is required to read Sen1Floods11 GeoTIFFs. "
            "On Kaggle: it is preinstalled; locally `pip install rasterio`."
        ) from e
    with rasterio.open(path) as src:
        if bands is None:
            arr = src.read()                       # [C, H, W]
        else:
            # rasterio band indices are 1-based
            arr = src.read([b + 1 for b in bands])
    return arr.astype(np.float32)


def _default_norm_s1(x: np.ndarray) -> np.ndarray:
    """S1 is in dB (~[-30, 0]). Clip and scale to ~[-1, 1]."""
    x = np.clip(x, -30.0, 0.0)
    return (x + 15.0) / 15.0


def _default_norm_s2(x: np.ndarray) -> np.ndarray:
    """S2 reflectance (~[0, 10000]). Scale to [0, 1] then center."""
    x = np.clip(x / 10000.0, 0.0, 1.0)
    return (x - 0.5) / 0.5


class Sen1Floods11(Dataset):
    def __init__(self, cfg, index_path: str, augment: bool = False):
        self.cfg = cfg
        self.augment = augment
        with open(index_path) as f:
            self.items: List[Dict] = json.load(f)
        # optional externally-provided normalization stats
        self._s1_ms = self._as_tensor(cfg.s1_mean, cfg.s1_std)
        self._s2_ms = self._as_tensor(cfg.s2_mean, cfg.s2_std)

    @staticmethod
    def _as_tensor(mean, std):
        if mean is None or std is None:
            return None
        m = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        s = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        return m, s

    def __len__(self) -> int:
        return len(self.items)

    def _normalize(self, x: torch.Tensor, ms, default_fn) -> torch.Tensor:
        if ms is not None:
            m, s = ms
            return (x - m) / (s + 1e-6)
        return torch.from_numpy(default_fn(x.numpy()))

    def _augment(self, s1, s2, label, valid):
        # cheap flips only -- geometry-preserving, safe for segmentation
        if torch.rand(1).item() < 0.5:
            s1, s2, label, valid = [t.flip(-1) for t in (s1, s2, label, valid)]
        if torch.rand(1).item() < 0.5:
            s1, s2, label, valid = [t.flip(-2) for t in (s1, s2, label, valid)]
        return s1, s2, label, valid

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        it = self.items[i]
        s1 = torch.from_numpy(_read_raster(it["s1"]))                       # [2,H,W]
        s2 = torch.from_numpy(_read_raster(it["s2"], list(self.cfg.s2_bands)))  # [6,H,W]
        lbl = torch.from_numpy(_read_raster(it["label"]))[0]               # [H,W]

        # Some Sen1Floods11 S1 chips contain -inf / NaN dB pixels (log of zero
        # backscatter, scene edges). Replace them BEFORE normalization, else the
        # non-finite value propagates straight through the network to a NaN loss.
        # 0.0 maps to a benign ~mean value after standardization.
        s1 = torch.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
        s2 = torch.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)

        s1 = self._normalize(s1, self._s1_ms, _default_norm_s1)
        s2 = self._normalize(s2, self._s2_ms, _default_norm_s2)
        # belt-and-braces: clamp standardized values to a sane range so a rare
        # extreme reflectance can't destabilize early training.
        s1 = torch.clamp(s1, -10.0, 10.0)
        s2 = torch.clamp(s2, -10.0, 10.0)

        # ternary label -1/0/1 -> valid mask + binary target
        valid = (lbl != self.cfg.ignore_index).float()                    # [H,W]
        target = torch.clamp(lbl, min=0).float()                          # -1 -> 0

        out = {"s1": s1, "s2": s2, "label": target, "valid": valid}

        # JRC permanent water (optional) -> lets NB4/5 report flood-only IoU
        if it.get("jrc"):
            jrc = torch.from_numpy(_read_raster(it["jrc"]))[0]
            out["jrc"] = (jrc > 0).float()

        if self.augment:
            s1, s2, target, valid = self._augment(s1, s2, target, valid)
            out.update(s1=s1, s2=s2, label=target, valid=valid)
        return out
