"""Structured modality-dropout curriculum.

Regime index: 0 = both, 1 = S1-only (SAR), 2 = S2-only (optical).

Schedule: `warmup_epochs` of both-only (learn fusion first), then linearly
interpolate the sampling distribution from mostly-both toward the operational
prior `final_regime_probs` (SAR-only up-weighted, since optical is what the
flood/cloud case removes). This is a fixed, zero-parameter schedule -- the
cheap alternative to learned scenario sampling.
"""
from typing import Tuple

import torch


def regime_probs(cfg, epoch: int) -> Tuple[float, float, float]:
    if not cfg.use_curriculum:
        return cfg.fixed_regime_probs
    if epoch < cfg.warmup_epochs:
        return (1.0, 0.0, 0.0)
    span = max(cfg.epochs - cfg.warmup_epochs, 1)
    t = min((epoch - cfg.warmup_epochs) / span, 1.0)
    start = (0.8, 0.1, 0.1)
    end = cfg.final_regime_probs
    return tuple(s + t * (e - s) for s, e in zip(start, end))


def sample_present(batch_size: int, probs, device) -> torch.Tensor:
    """Draw a [B,2] present matrix from the regime distribution.

    Each sample independently drawn -> a batch mixes regimes, which stabilizes
    BatchNorm across modality states.
    """
    p = torch.tensor(probs, dtype=torch.float32)
    idx = torch.multinomial(p, batch_size, replacement=True)   # [B] in {0,1,2}
    present = torch.ones(batch_size, 2, device=device)
    present[idx == 1, 1] = 0.0   # S1-only -> drop S2
    present[idx == 2, 0] = 0.0   # S2-only -> drop S1
    return present


def eval_regimes():
    """The three fixed present-vectors used at evaluation."""
    return {
        "both": torch.tensor([[1.0, 1.0]]),
        "s1_only": torch.tensor([[1.0, 0.0]]),
        "s2_only": torch.tensor([[0.0, 1.0]]),
    }


def apply_present(s1, s2, present):
    """Zero the absent modality's input so no stale signal leaks in.

    The gate already masks by `present`; zeroing the input is belt-and-braces
    and makes single-modality latency measurements honest.
    """
    p1 = present[:, 0].view(-1, 1, 1, 1)
    p2 = present[:, 1].view(-1, 1, 1, 1)
    return s1 * p1, s2 * p2
