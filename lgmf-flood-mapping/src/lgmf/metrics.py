"""Evaluation metrics. All respect the -1 validity mask.

`SegMeter` accumulates confusion counts over a loader so IoU/F1 are computed
on the whole set (not averaged per-batch, which biases small batches).
"""
import torch


class SegMeter:
    def __init__(self, threshold: float = 0.5):
        self.t = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    @torch.no_grad()
    def update(self, logits, target, valid, extra_mask=None):
        """extra_mask: optional [B,H,W] to further restrict pixels
        (e.g. exclude permanent water -> flood-only IoU)."""
        if target.dim() == 4:
            target = target.squeeze(1)
            valid = valid.squeeze(1)
        pred = (torch.sigmoid(logits).squeeze(1) > self.t).float()
        m = valid
        if extra_mask is not None:
            m = m * extra_mask
        pred, tgt = pred * m, target * m
        self.tp += (pred * tgt).sum().item()
        self.fp += (pred * (1 - tgt) * m).sum().item()
        self.fn += ((1 - pred) * tgt).sum().item()
        self.tn += ((1 - pred) * (1 - tgt) * m).sum().item()

    def compute(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = tp / (tp + fp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        acc = (tp + self.tn) / (tp + fp + fn + self.tn + 1e-6)
        return {"iou": iou, "f1": f1, "precision": prec,
                "recall": rec, "accuracy": acc}


def flops_g(model, cfg, device="cpu") -> float:
    """GFLOPs for one 512x512 chip in the 'both present' regime.
    Uses thop if available; returns -1.0 otherwise."""
    try:
        from thop import profile
    except ImportError:
        return -1.0
    model = model.to(device).eval()
    s1 = torch.randn(1, 2, cfg.img_size, cfg.img_size, device=device)
    s2 = torch.randn(1, len(cfg.s2_bands), cfg.img_size, cfg.img_size, device=device)
    present = torch.ones(1, 2, device=device)
    macs, _ = profile(model, inputs=(s1, s2, present), verbose=False)
    return macs * 2 / 1e9   # MACs -> FLOPs -> giga
