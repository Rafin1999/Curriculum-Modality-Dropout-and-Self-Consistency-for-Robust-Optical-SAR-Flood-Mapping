"""Losses for LGMF.

- Segmentation: masked BCE + masked soft-Dice, both ignoring invalid (-1) pixels.
- Self-consistency: single-modality predictions are pulled toward the *fused*
  prediction of the SAME model (detached target). No teacher, no reconstruction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_bce(logits, target, valid, pos_weight=1.0):
    # logits,target,valid: [B,1,H,W] (target/valid may be [B,H,W])
    if target.dim() == 3:
        target = target.unsqueeze(1)
        valid = valid.unsqueeze(1)
    pw = torch.tensor(pos_weight, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw, reduction="none"
    )
    loss = loss * valid
    return loss.sum() / (valid.sum() + 1e-6)


def masked_dice(logits, target, valid, eps=1e-6):
    if target.dim() == 3:
        target = target.unsqueeze(1)
        valid = valid.unsqueeze(1)
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    dims = (0, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_tversky(logits, target, valid, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
    """Focal-Tversky loss over valid pixels, per-sample then averaged.

    alpha weights false negatives, beta false positives; alpha>beta pushes the
    model toward recall (detect more water), which is what a modality that
    under-detects -- SAR-only -- needs. gamma focuses on hard samples.
    """
    if target.dim() == 3:
        target = target.unsqueeze(1); valid = valid.unsqueeze(1)
    prob = torch.sigmoid(logits) * valid
    tgt = target * valid
    dims = (2, 3)
    tp = (prob * tgt).sum(dims)
    fp = (prob * (1 - tgt) * valid).sum(dims)
    fn = ((1 - prob) * tgt * valid).sum(dims)
    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return ((1 - tversky) ** gamma).mean()


def segmentation_loss(logits, target, valid, cfg):
    if getattr(cfg, "loss_type", "dice_bce") == "focal_tversky":
        return focal_tversky(logits, target, valid,
                             cfg.tversky_alpha, cfg.tversky_beta, cfg.focal_gamma)
    return (cfg.bce_weight * masked_bce(logits, target, valid, cfg.pos_weight)
            + cfg.dice_weight * masked_dice(logits, target, valid))


def consistency_loss(single_logits, fused_logits, valid):
    """Pull a single-modality prediction toward the fused prediction.

    Target (fused) is detached so gradient only trains the single-modality path.
    Symmetric MSE on probabilities over valid pixels.
    """
    if valid.dim() == 3:
        valid = valid.unsqueeze(1)
    p_single = torch.sigmoid(single_logits)
    p_fused = torch.sigmoid(fused_logits).detach()
    loss = ((p_single - p_fused) ** 2) * valid
    return loss.sum() / (valid.sum() + 1e-6)
