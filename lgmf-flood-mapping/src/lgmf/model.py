"""LGMF: Lightweight Gated Modality Fusion for flood segmentation.

Data flow (single-stage fusion):
    s1 -> stem_s1 --\
                     gate (mask + renorm over present modalities) -> FiLM -> shared encoder -> MLP decoder -> logits
    s2 -> stem_s2 --/
    present [B,2] conditions both the gate mask and the FiLM parameters.

Ablation flags live in cfg: film_on, gate_mode ('renorm'|'concat'|'sum').
'multi' fusion_stages (dual encoder + per-stage gate, SMAGNet-style) is a
documented extension hook, not implemented here -- single-stage first to
establish the efficiency-vs-robustness Pareto claim.
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    """GroupNorm with a channel-count-safe group number. GroupNorm normalizes
    per-sample, so it is immune to the small-batch noise and the zero-input
    pollution that BatchNorm suffers when the modality-dropout curriculum feeds
    a stem all-zeros for a dropped modality."""
    g = 8
    while c % g:
        g //= 2
    return nn.GroupNorm(max(g, 1), c)


def _dw_sep(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable conv block (cheap stem building block)."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
        _gn(cin),
        nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        _gn(cout),
        nn.ReLU(inplace=True),
    )


class ModalityStem(nn.Module):
    """Tiny per-modality encoder -> stem_dim features at the given stride."""

    def __init__(self, in_ch: int, stem_dim: int, stride: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            _dw_sep(in_ch, stem_dim, stride=stride),
            _dw_sep(stem_dim, stem_dim, stride=1),
        )

    def forward(self, x):
        return self.net(x)


class MaskedGate(nn.Module):
    """Fuse two stem features, masking + renormalizing over present modalities.

    gate_mode:
      'renorm' (ours): per-channel gate a_i in (0,1); absent modality's gate is
                       zeroed, survivors renormalize so weights sum to 1.
      'sum'   : masked sum (present_i * f_i). Naive baseline.
      'concat': 1x1 conv over concatenated (masked) features. Standard fusion,
                degrades poorly when a modality is absent (zeros flow in).
    """

    def __init__(self, dim: int, mode: str = "renorm"):
        super().__init__()
        self.mode = mode
        if mode == "renorm":
            self.gate = nn.Sequential(
                nn.Linear(2 * dim, dim), nn.ReLU(inplace=True),
                nn.Linear(dim, 2 * dim),
            )
        elif mode == "concat":
            self.proj = nn.Conv2d(2 * dim, dim, 1)
        elif mode == "sum":
            pass
        else:
            raise ValueError(f"unknown gate_mode {mode}")

    def forward(self, f1, f2, present):
        # present: [B, 2] float (1 present, 0 absent)
        p1 = present[:, 0].view(-1, 1, 1, 1)
        p2 = present[:, 1].view(-1, 1, 1, 1)
        f1m, f2m = f1 * p1, f2 * p2

        if self.mode == "sum":
            return f1m + f2m
        if self.mode == "concat":
            return self.proj(torch.cat([f1m, f2m], dim=1))

        # ---- renorm (ours) ----
        pooled = torch.cat([f1.mean((-2, -1)), f2.mean((-2, -1))], dim=1)  # [B,2D]
        a = torch.sigmoid(self.gate(pooled))                              # [B,2D]
        D = f1.shape[1]
        a1, a2 = a[:, :D], a[:, D:]                                       # [B,D]
        a1 = a1 * present[:, 0:1]
        a2 = a2 * present[:, 1:2]
        denom = a1 + a2 + 1e-6
        w1 = (a1 / denom).unsqueeze(-1).unsqueeze(-1)                     # [B,D,1,1]
        w2 = (a2 / denom).unsqueeze(-1).unsqueeze(-1)
        return w1 * f1 + w2 * f2


class FiLM(nn.Module):
    """Modality-presence conditioning: [B,2] -> per-channel (gamma, beta)."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, dim), nn.ReLU(inplace=True), nn.Linear(dim, 2 * dim)
        )
        self.dim = dim

    def forward(self, x, present):
        gb = self.mlp(present)                       # [B, 2D]
        gamma, beta = gb[:, : self.dim], gb[:, self.dim:]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta                # 1+gamma: identity at init


class MLPDecoder(nn.Module):
    """SegFormer-style all-MLP head: project every stage, upsample, fuse."""

    def __init__(self, in_channels: List[int], decoder_dim: int):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(c, decoder_dim, 1) for c in in_channels]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(in_channels), decoder_dim, 1, bias=False),
            _gn(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(decoder_dim, 1, 1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        size = feats[0].shape[-2:]                    # finest stage resolution
        ups = []
        for proj, f in zip(self.projs, feats):
            y = proj(f)
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            ups.append(y)
        x = self.fuse(torch.cat(ups, dim=1))
        return self.classifier(x)                     # [B,1,h,w]


class _DecoderBlock(nn.Module):
    """Upsample x2, concat skip, two GN-conv layers."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """Progressive skip-connection decoder. Merges encoder stages coarse->fine,
    then a final skip from the high-res fused stem feature -- so fine spatial
    detail survives, unlike the all-MLP head."""

    def __init__(self, enc_ch: List[int], stem_dim: int,
                 dec_ch=(160, 128, 96, 64, 48)):
        super().__init__()
        rev = enc_ch[::-1]                     # coarse -> fine
        skips = list(rev[1:]) + [stem_dim]     # encoder skips + stem skip (finest)
        self.blocks = nn.ModuleList()
        cur = rev[0]
        for i, skip_ch in enumerate(skips):
            out_ch = dec_ch[i] if i < len(dec_ch) else dec_ch[-1]
            self.blocks.append(_DecoderBlock(cur, skip_ch, out_ch))
            cur = out_ch
        self.classifier = nn.Conv2d(cur, 1, 1)

    def forward(self, feats: List[torch.Tensor], stem_feat: torch.Tensor):
        rev = feats[::-1]
        x = rev[0]
        skips = list(rev[1:]) + [stem_feat]
        for blk, skip in zip(self.blocks, skips):
            x = blk(x, skip)
        return self.classifier(x)


class LGMF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_s2 = len(cfg.s2_bands)
        self.stem_s1 = ModalityStem(2, cfg.stem_dim, stride=cfg.stem_stride)
        self.stem_s2 = ModalityStem(n_s2, cfg.stem_dim, stride=cfg.stem_stride)
        self.gate = MaskedGate(cfg.stem_dim, cfg.gate_mode)
        self.film = FiLM(cfg.stem_dim) if cfg.film_on else None

        import timm
        self.encoder = timm.create_model(
            cfg.encoder_name,
            features_only=True,
            pretrained=cfg.encoder_pretrained,
            in_chans=cfg.stem_dim,
        )
        enc_ch = self.encoder.feature_info.channels()
        self.decoder_kind = getattr(cfg, "decoder", "mlp")
        if self.decoder_kind == "unet":
            self.decoder = UNetDecoder(enc_ch, cfg.stem_dim)
        else:
            self.decoder = MLPDecoder(enc_ch, cfg.decoder_dim)

    def forward(self, s1, s2, present) -> Dict[str, torch.Tensor]:
        H, W = s1.shape[-2:]
        f1 = self.stem_s1(s1)
        f2 = self.stem_s2(s2)
        fused = self.gate(f1, f2, present)
        if self.film is not None:
            fused = self.film(fused, present)
        feats = self.encoder(fused)
        if self.decoder_kind == "unet":
            logits = self.decoder(feats, fused)       # fused = high-res stem skip
        else:
            logits = self.decoder(feats)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                               align_corners=False)
        return {"logits": logits}                     # [B,1,H,W]


def build_model(cfg) -> LGMF:
    return LGMF(cfg)


@torch.no_grad()
def count_params(model: nn.Module) -> float:
    """Trainable parameters, in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
