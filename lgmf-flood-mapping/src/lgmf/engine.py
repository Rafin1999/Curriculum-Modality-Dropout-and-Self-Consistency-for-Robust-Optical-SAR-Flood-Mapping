"""Train / eval loops. Kept in the package so every run and ablation uses the
IDENTICAL loop -- the notebook only supplies a config and data.
"""
from typing import Dict

import torch
from torch import amp

from .losses import segmentation_loss, consistency_loss
from .metrics import SegMeter
from .curriculum import regime_probs, sample_present, eval_regimes, apply_present


def _forward_seg(model, s1, s2, present, target, valid, cfg):
    out = model(s1, s2, present)
    loss = segmentation_loss(out["logits"], target, valid, cfg)
    return out["logits"], loss


def train_one_epoch(model, loader, optimizer, cfg, epoch, device,
                    scaler=None, ema=None) -> Dict[str, float]:
    model.train()
    probs = regime_probs(cfg, epoch)
    running = {"loss": 0.0, "seg": 0.0, "cons": 0.0}
    n = 0
    for batch in loader:
        s1 = batch["s1"].to(device); s2 = batch["s2"].to(device)
        target = batch["label"].to(device); valid = batch["valid"].to(device)
        bs = s1.size(0)
        present = sample_present(bs, probs, device)
        s1i, s2i = apply_present(s1, s2, present)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
            logits, seg = _forward_seg(model, s1i, s2i, present,
                                       target, valid, cfg)
            cons = torch.zeros((), device=device)
            # self-consistency: for samples that had BOTH present, also run a
            # single-modality pass and pull it toward the (detached) fused pred.
            if cfg.consistency_weight > 0:
                both = (present.sum(1) == 2)
                if both.any():
                    idx = both.nonzero(as_tuple=True)[0]
                    # drop optical -> SAR-only path should match fused
                    p_s1 = torch.zeros(len(idx), 2, device=device)
                    p_s1[:, 0] = 1.0
                    s1_only = model(s1[idx], torch.zeros_like(s2[idx]), p_s1)
                    cons = consistency_loss(
                        s1_only["logits"], logits[idx].detach(), valid[idx])
            loss = seg + cfg.consistency_weight * cons

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)          # unscale so clip sees true grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        running["loss"] += loss.item() * bs
        running["seg"] += seg.item() * bs
        running["cons"] += float(cons.detach()) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in running.items()} | {"probs": probs}


@torch.no_grad()
def evaluate(model, loader, cfg, device) -> Dict[str, Dict[str, float]]:
    """Returns {regime: metrics} for both / s1_only / s2_only."""
    model.eval()
    regimes = eval_regimes()
    meters = {r: SegMeter() for r in regimes}
    flood_meters = {r: SegMeter() for r in regimes}   # JRC-excluded, if present
    have_jrc = False
    for batch in loader:
        s1 = batch["s1"].to(device); s2 = batch["s2"].to(device)
        target = batch["label"].to(device); valid = batch["valid"].to(device)
        jrc = batch.get("jrc")
        if jrc is not None:
            jrc = jrc.to(device); have_jrc = True
        bs = s1.size(0)
        for r, base in regimes.items():
            present = base.repeat(bs, 1).to(device)
            s1i, s2i = apply_present(s1, s2, present)
            logits = model(s1i, s2i, present)["logits"]
            meters[r].update(logits, target, valid)
            if jrc is not None:
                flood_meters[r].update(logits, target, valid,
                                       extra_mask=(1.0 - jrc))
    out = {r: meters[r].compute() for r in regimes}
    if have_jrc:
        for r in regimes:
            out[r]["flood_only_iou"] = flood_meters[r].compute()["iou"]
    return out
