"""Quick shape/sanity check. Run: python -m lgmf.smoke_test

Verifies the model builds, forward works in all three regimes, losses are
finite, and the ablation flags (film_on, gate_mode) all construct.
"""
import torch

from .config import LGMFConfig
from .model import build_model, count_params
from .losses import segmentation_loss, consistency_loss
from .curriculum import eval_regimes, apply_present
from .metrics import SegMeter


def run(img=128):
    cfg = LGMFConfig(img_size=img, batch_size=2, encoder_pretrained=False)
    B, n2 = 2, len(cfg.s2_bands)
    s1 = torch.randn(B, 2, img, img)
    s2 = torch.randn(B, n2, img, img)
    target = (torch.rand(B, img, img) > 0.7).float()
    valid = torch.ones(B, img, img)
    valid[:, :10, :] = 0.0  # simulate some no-data

    for film_on in (True, False):
        for gate_mode in ("renorm", "concat", "sum"):
            cfg.film_on = film_on
            cfg.gate_mode = gate_mode
            model = build_model(cfg).eval()
            for r, base in eval_regimes().items():
                present = base.repeat(B, 1)
                s1i, s2i = apply_present(s1, s2, present)
                out = model(s1i, s2i, present)
                assert out["logits"].shape == (B, 1, img, img), out["logits"].shape
                loss = segmentation_loss(out["logits"], target, valid, cfg)
                assert torch.isfinite(loss), (film_on, gate_mode, r)
            m = SegMeter(); m.update(out["logits"], target, valid)
            _ = m.compute()
            print(f"film={film_on} gate={gate_mode:6s} "
                  f"params={count_params(model):.2f}M  OK")

    # v4 options: stem_stride, focal_tversky loss, EMA
    from lgmf.ema import ModelEMA
    for stride in (1, 2):
        for loss_type in ("dice_bce", "focal_tversky"):
            cfg.film_on = True; cfg.gate_mode = "renorm"
            cfg.stem_stride = stride; cfg.loss_type = loss_type
            model = build_model(cfg)
            ema = ModelEMA(model, cfg.ema_decay)
            present = torch.ones(B, 2)
            out = model(s1, s2, present)
            loss = segmentation_loss(out["logits"], target, valid, cfg)
            loss.backward(); ema.update(model)
            assert out["logits"].shape == (B, 1, img, img)
            assert torch.isfinite(loss), (stride, loss_type)
            print(f"stride={stride} loss={loss_type:13s} "
                  f"params={count_params(model):.2f}M  +EMA  OK")
    print("smoke test passed")


if __name__ == "__main__":
    run()
