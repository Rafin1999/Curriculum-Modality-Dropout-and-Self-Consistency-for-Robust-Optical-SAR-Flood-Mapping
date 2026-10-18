"""In-RAM copy of a Sen1Floods11 split (v8).

Reading three GeoTIFFs per chip every epoch dominates data time. This caches the split ONCE
(float32, ~11 MB/chip: 250 train chips ~2.7 GB) using the original `Sen1Floods11` dataset to
produce the tensors, so values are identical. Augmentation replicates `Sen1Floods11._augment`
call-for-call (same torch.rand draws, same order), so a seeded training run consumes the RNG
exactly as before.

`keep_masks=True` also stores which raw pixels were non-finite (used by the imputation audit).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import Sen1Floods11, _read_raster


class RamDataset(Dataset):
    def __init__(self, cfg, index_path, augment=False, keep_masks=False, verbose=False):
        ds = Sen1Floods11(cfg, index_path, augment=False)
        self.augment, self.keep_masks = augment, keep_masks
        self.ids = [it["id"] for it in ds.items]
        self.s1, self.s2, self.label, self.valid, self.jrc = [], [], [], [], []
        self.nf1, self.nf2, self.stat = [], [], []
        self.missing_jrc = 0
        for i, it in enumerate(ds.items):
            d = ds[i]
            self.s1.append(d["s1"]); self.s2.append(d["s2"]); self.label.append(d["label"]); self.valid.append(d["valid"])
            if "jrc" in d: self.jrc.append(d["jrc"])
            else: self.jrc.append(torch.zeros_like(d["label"])); self.missing_jrc += 1
            if keep_masks:
                r1 = _read_raster(it["s1"]); r2 = _read_raster(it["s2"], list(cfg.s2_bands))
                self.nf1.append(torch.from_numpy(~np.isfinite(r1))); self.nf2.append(torch.from_numpy(~np.isfinite(r2)))
                self.stat.append(dict(id=it["id"], s1_nonfinite=float((~np.isfinite(r1)).mean()),
                                      s2_nonfinite=float((~np.isfinite(r2)).mean()),
                                      nodata_label=float(1 - d["valid"].mean())))
            if verbose and (i + 1) % 50 == 0: print(f"  cached {i + 1}/{len(ds.items)}")

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        s1, s2, label, valid, jrc = self.s1[i], self.s2[i], self.label[i], self.valid[i], self.jrc[i]
        if self.augment:     # identical draw order to Sen1Floods11._augment
            if torch.rand(1).item() < 0.5:
                s1, s2, label, valid = [t.flip(-1) for t in (s1, s2, label, valid)]
            if torch.rand(1).item() < 0.5:
                s1, s2, label, valid = [t.flip(-2) for t in (s1, s2, label, valid)]
        return dict(s1=s1, s2=s2, label=label, valid=valid, jrc=jrc)

    def batch(self, ii, variant="zero"):
        st = lambda L: torch.stack([L[i] for i in ii])
        b = dict(s1=st(self.s1), s2=st(self.s2), label=st(self.label), valid=st(self.valid), jrc=st(self.jrc))
        if variant == "mean":      # true mean imputation: non-finite raw pixels -> 0 AFTER standardisation
            assert self.keep_masks, "build with keep_masks=True"
            b["s1"] = b["s1"].masked_fill(st(self.nf1), 0.0); b["s2"] = b["s2"].masked_fill(st(self.nf2), 0.0)
        return b
