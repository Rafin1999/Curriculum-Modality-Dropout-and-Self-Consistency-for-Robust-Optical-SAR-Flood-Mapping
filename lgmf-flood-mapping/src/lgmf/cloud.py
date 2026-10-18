"""Partial-cloud masks for evaluation (v8, WP12).

The paper's 'missing optical' regime removes the WHOLE optical image. Real clouds cover
part of a scene and (importantly) are not 'missing': they are bright, wrong pixels, often
with a dark shadow next to them. This module builds cloud-shaped (not rectangular) masks
with an exact coverage fraction, and two ways of applying them to a standardised S2 tensor.

  cloud_mask(h, w, coverage, seed)         -> bool [h, w], True = cloud, coverage as asked
  cloud_and_shadow(h, w, coverage, seed)   -> (cloud, shadow) bool masks; shadow is the cloud shifted
  apply_cloud(s2, cloud, shadow, mode)     -> s2 with those pixels replaced
        mode "zero"   : cloud pixels -> 0 (the standardised mean = the same fill the model saw for a missing image)
        mode "bright" : cloud pixels -> a bright, spectrally flat value; shadow pixels -> dark value

APPROXIMATE typical raw digital numbers for (B2, B3, B4, B8, B11, B12) are used for the
'bright' fill. They are illustrative, not measured from this dataset; pass your own
`cloud_dn` / `shadow_dn` to calibrate, and report the values used.
"""
import numpy as np
import torch

CLOUD_DN = (3500.0, 3500.0, 3600.0, 4000.0, 3000.0, 2500.0)
SHADOW_DN = (300.0, 400.0, 350.0, 200.0, 100.0, 80.0)


def _unit(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def cloud_mask(h, w, coverage, seed=0, scales=(40.0, 16.0, 6.0), weights=(1.0, 0.5, 0.25)):
    """Boolean cloud mask with (very nearly) exactly `coverage` of the pixels set."""
    coverage = float(coverage)
    if coverage <= 0.0:
        return np.zeros((h, w), dtype=bool)
    if coverage >= 1.0:
        return np.ones((h, w), dtype=bool)
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(int(seed) % (2 ** 31 - 1))
    f = np.zeros((h, w), dtype=np.float64)
    for s, wt in zip(scales, weights):
        f += wt * _unit(gaussian_filter(rng.randn(h, w), s, mode="wrap"))
    k = int(round(coverage * h * w))
    order = np.argsort(-f, axis=None)             # most 'cloudy' pixels first
    m = np.zeros(h * w, dtype=bool)
    m[order[:k]] = True
    return m.reshape(h, w)


def cloud_and_shadow(h, w, coverage, seed=0, offset=(24, 24)):
    cloud = cloud_mask(h, w, coverage, seed)
    shadow = np.roll(cloud, shift=offset, axis=(0, 1)) & ~cloud
    return cloud, shadow


def _std_values(dn, mean, std, device):
    dn = torch.tensor(dn, dtype=torch.float32)
    m = torch.tensor(mean, dtype=torch.float32); s = torch.tensor(std, dtype=torch.float32)
    return (((dn - m) / (s + 1e-6)).clamp(-10.0, 10.0)).view(1, -1, 1, 1).to(device)


def apply_cloud(s2, cloud, shadow=None, mode="zero", s2_mean=None, s2_std=None,
                cloud_dn=CLOUD_DN, shadow_dn=SHADOW_DN):
    """s2: [B,C,H,W] standardised. cloud / shadow: bool [B,H,W] (or [H,W]) tensors/arrays."""
    cloud = torch.as_tensor(cloud).to(s2.device).bool()
    if cloud.dim() == 2: cloud = cloud.unsqueeze(0).expand(s2.size(0), -1, -1)
    c = cloud.unsqueeze(1)
    if mode == "zero":
        return s2.masked_fill(c, 0.0)
    assert mode == "bright", mode
    assert s2_mean is not None and s2_std is not None, "bright mode needs the standardisation stats"
    out = torch.where(c, _std_values(cloud_dn, s2_mean, s2_std, s2.device).expand_as(s2), s2)
    if shadow is not None:
        sh = torch.as_tensor(shadow).to(s2.device).bool()
        if sh.dim() == 2: sh = sh.unsqueeze(0).expand(s2.size(0), -1, -1)
        out = torch.where(sh.unsqueeze(1), _std_values(shadow_dn, s2_mean, s2_std, s2.device).expand_as(out), out)
    return out
