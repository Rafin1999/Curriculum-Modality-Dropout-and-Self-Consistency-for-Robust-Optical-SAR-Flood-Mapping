"""Per-chip evaluation + pooling helpers (v8). Reproduces SegMeter / engine.evaluate exactly
when counts are pooled over chips (asserted in NB-11 and in selftest_v8)."""
import numpy as np
import torch

from .curriculum import eval_regimes, apply_present

EPS = 1e-6
KEYS = ["config", "seed", "split", "variant", "regime"]


def event_of(cid):
    return cid.split("_")[0]


@torch.no_grad()
def chip_counts(model, cache, ctx, device, variant="zero", bs=8, cloud=None, s2_stats=None):
    """Confusion counts per chip x regime, for the full valid mask and the flood-only mask.

    cloud: None, or dict(coverage=0.5, seed=0, mode="zero"|"bright", shadow=False) -> optical pixels are
           replaced (cloud_mask per chip, seed = seed + chip index) BEFORE the regime masking, so it
           only matters in regimes where optical is present. s2_stats=(mean, std) for mode 'bright'.
    """
    from .cloud import cloud_and_shadow, apply_cloud
    model.eval(); rows = []; R = eval_regimes()
    for s in range(0, len(cache), bs):
        ii = list(range(s, min(s + bs, len(cache)))); b = cache.batch(ii, variant)
        s1, s2 = b["s1"].to(device), b["s2"].to(device)
        tgt, valid, jrc = b["label"].to(device), b["valid"].to(device), b["jrc"].to(device)
        if cloud is not None and cloud.get("coverage", 0) > 0:
            h, w = s2.shape[-2:]
            masks = [cloud_and_shadow(h, w, cloud["coverage"], cloud.get("seed", 0) + i) for i in ii]
            cm = torch.from_numpy(np.stack([m[0] for m in masks])); sm = torch.from_numpy(np.stack([m[1] for m in masks]))
            mean, std = s2_stats if s2_stats else (None, None)
            s2 = apply_cloud(s2, cm, sm if cloud.get("shadow") else None, mode=cloud.get("mode", "zero"), s2_mean=mean, s2_std=std)
        for reg, basep in R.items():
            present = basep.repeat(len(ii), 1).to(device)
            s1i, s2i = apply_present(s1, s2, present)
            pred = (torch.sigmoid(model(s1i, s2i, present)["logits"]).squeeze(1) > 0.5).float()
            out = {}
            for tag, m in (("", valid), ("f", valid * (1.0 - jrc))):
                p, t = pred * m, tgt * m
                out[tag + "tp"] = (p * t).sum((1, 2)); out[tag + "fp"] = (p * (1 - t) * m).sum((1, 2))
                out[tag + "fn"] = ((1 - p) * t).sum((1, 2)); out[tag + "tn"] = ((1 - p) * (1 - t) * m).sum((1, 2))
            for k, i in enumerate(ii):
                rows.append(dict(**ctx, variant=variant, regime=reg, id=cache.ids[i], event=event_of(cache.ids[i]),
                                 **{c: float(v[k]) for c, v in out.items()}))
    return rows


def metrics_from_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = (np.asarray(x, dtype=float) for x in (tp, fp, fn, tn))
    iou = tp / (tp + fp + fn + EPS)
    prec = tp / (tp + fp + EPS); rec = tp / (tp + fn + EPS)
    return dict(iou=iou, f1=2 * prec * rec / (prec + rec + EPS), precision=prec, recall=rec)


def summarise(rows, split):
    """{regime: {iou, flood_iou, f1, precision, recall}} pooled over chips for one split (list-of-dict rows)."""
    out = {}
    for reg in ("both", "s1_only", "s2_only"):
        rr = [r for r in rows if r["split"] == split and r["regime"] == reg]
        if not rr: continue
        sm = lambda k: sum(r[k] for r in rr)
        m = metrics_from_counts(sm("tp"), sm("fp"), sm("fn"), sm("tn"))
        f = metrics_from_counts(sm("ftp"), sm("ffp"), sm("ffn"), sm("ftn"))
        out[reg] = dict(iou=float(m["iou"]), f1=float(m["f1"]), precision=float(m["precision"]),
                        recall=float(m["recall"]), flood_iou=float(f["iou"]))
    return out
