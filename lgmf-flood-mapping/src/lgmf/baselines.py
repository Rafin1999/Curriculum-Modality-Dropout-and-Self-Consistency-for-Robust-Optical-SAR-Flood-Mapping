"""External baselines for like-for-like comparison.

EarlyFusionNet wraps any segmentation_models_pytorch model so it accepts the
same (s1, s2, present) call signature as LGMF -- it masks absent modalities and
concatenates into one input tensor. This is the standard "concat everything"
multimodal baseline: strong when both modalities are present, but with no gate,
FiLM, or renormalization it degrades poorly when one is missing. Training it
*with* the modality-dropout curriculum (a fairer robustness baseline) still
lacks LGMF's graceful-degradation machinery.

smp is imported lazily so importing lgmf never requires it.
"""
import torch
import torch.nn as nn


class EarlyFusionNet(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, s1, s2, present):
        p1 = present[:, 0].view(-1, 1, 1, 1)
        p2 = present[:, 1].view(-1, 1, 1, 1)
        x = torch.cat([s1 * p1, s2 * p2], dim=1)     # [B, 2+6, H, W]
        return {"logits": self.net(x)}


def build_baseline(arch="unet", encoder="mobilenet_v2", in_ch=8,
                   pretrained=True, classes=1) -> EarlyFusionNet:
    import segmentation_models_pytorch as smp
    net = smp.create_model(
        arch, encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_ch, classes=classes,
    )
    return EarlyFusionNet(net)


class GatedFusionFrontEnd(nn.Module):
    """Replaces naive concat with gate + FiLM, at full resolution, producing an
    out_ch tensor for a standard segmentation backbone. Injects LGMF's fusion
    contributions onto a strong U-Net WITHOUT the early stem bottleneck that
    hurt the shared-encoder design.
    """

    def __init__(self, c1=2, c2=6, h1=8, h2=24, out_ch=8, film=True, gate=True):
        super().__init__()
        from .model import _gn, FiLM
        def adapt(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                _gn(cout), nn.ReLU(inplace=True))
        self.adapt1, self.adapt2 = adapt(c1, h1), adapt(c2, h2)
        self.h1, self.h2 = h1, h2
        self.use_gate = gate
        if gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 + h1 + h2, 32), nn.ReLU(inplace=True), nn.Linear(32, 2))
        self.film = FiLM(h1 + h2) if film else None
        self.proj = nn.Conv2d(h1 + h2, out_ch, 1)

    def forward(self, s1, s2, present):
        p1 = present[:, 0].view(-1, 1, 1, 1)
        p2 = present[:, 1].view(-1, 1, 1, 1)
        a1 = self.adapt1(s1) * p1
        a2 = self.adapt2(s2) * p2
        if self.use_gate:
            pooled = torch.cat([a1.mean((-2, -1)), a2.mean((-2, -1))], dim=1)
            g = torch.sigmoid(self.gate_mlp(torch.cat([present, pooled], dim=1)))
            g1 = g[:, 0:1] * present[:, 0:1]
            g2 = g[:, 1:2] * present[:, 1:2]
            s = g1 + g2 + 1e-6
            a1 = a1 * (g1 / s).view(-1, 1, 1, 1)      # renormalized group weights
            a2 = a2 * (g2 / s).view(-1, 1, 1, 1)
        x = torch.cat([a1, a2], dim=1)
        if self.film is not None:
            x = self.film(x, present)
        return self.proj(x)


class GatedUNet(nn.Module):
    def __init__(self, front: nn.Module, net: nn.Module):
        super().__init__()
        self.front, self.net = front, net

    def forward(self, s1, s2, present):
        return {"logits": self.net(self.front(s1, s2, present))}


def build_gated_unet(arch="unet", encoder="mobilenet_v2", out_ch=8,
                     film=True, gate=True, pretrained=True, classes=1):
    """unet_drop's backbone + injected gate/FiLM front-end. Same in_channels as
    the baseline (out_ch), so the ONLY difference vs unet_drop is the fusion
    front-end (and, via the training loop, the consistency loss)."""
    import segmentation_models_pytorch as smp
    net = smp.create_model(arch, encoder_name=encoder,
                           encoder_weights="imagenet" if pretrained else None,
                           in_channels=out_ch, classes=classes)
    front = GatedFusionFrontEnd(out_ch=out_ch, film=film, gate=gate)
    return GatedUNet(front, net)
