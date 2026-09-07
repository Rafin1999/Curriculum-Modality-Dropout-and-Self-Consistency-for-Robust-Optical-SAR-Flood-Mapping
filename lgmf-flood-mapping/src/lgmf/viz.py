"""Paper-figure helpers. Deterministic, publication-ready defaults.

Every mask is drawn with the SAME colormap everywhere (FLOOD_CMAP) so figures
are visually consistent across the paper. Line/bar plots export to vector
(PDF+SVG); image grids export to >=300 DPI PNG.
"""
from typing import Dict, List, Optional

import numpy as np

# ---- fixed, colorblind-safe palette (used identically in every figure) ----
COLOR_LAND = (1.0, 1.0, 1.0, 0.0)      # transparent
COLOR_FLOOD = (0.11, 0.42, 0.69, 1.0)  # blue
COLOR_PERM = (0.11, 0.62, 0.47, 1.0)   # teal (permanent water)
COLOR_FP = (0.85, 0.37, 0.0, 1.0)      # orange (false positive, error maps)
COLOR_FN = (0.80, 0.47, 0.65, 1.0)     # pink   (false negative)

# consistent method colors across Pareto / degradation / bar charts
METHOD_COLORS = {
    "LGMF": "#1d6fb8", "SMAGNet": "#d85a30", "DMCF-Net": "#0f6e56",
    "CBC-SLP": "#7f77dd", "U-Net": "#888780", "MANet": "#b4b2a9",
}


def _savefig(fig, path_noext: str, dpi: int = 300, vector: bool = True):
    fig.savefig(f"{path_noext}.png", dpi=dpi, bbox_inches="tight")
    if vector:
        fig.savefig(f"{path_noext}.pdf", bbox_inches="tight")
        fig.savefig(f"{path_noext}.svg", bbox_inches="tight")


def mask_to_rgba(mask: np.ndarray, jrc: Optional[np.ndarray] = None) -> np.ndarray:
    """Water mask (+optional permanent-water) -> RGBA image for overlay."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[mask > 0.5] = COLOR_FLOOD
    if jrc is not None:
        out[(mask > 0.5) & (jrc > 0.5)] = COLOR_PERM
    return out


def error_rgba(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """FP=orange, FN=pink, TP=blue, over valid pixels only."""
    h, w = pred.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    v = valid > 0.5
    out[(pred > 0.5) & (gt > 0.5) & v] = COLOR_FLOOD
    out[(pred > 0.5) & (gt <= 0.5) & v] = COLOR_FP
    out[(pred <= 0.5) & (gt > 0.5) & v] = COLOR_FN
    return out


def _s2_rgb(s2: np.ndarray, bands=(2, 1, 0)) -> np.ndarray:
    """Build a display RGB from the S2 band subset (default B4,B3,B2 -> R,G,B).
    Indices are into the SUBSET order, not the raw 13-band stack."""
    rgb = s2[list(bands)].transpose(1, 2, 0)
    lo, hi = np.percentile(rgb, (2, 98))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def plot_triplet(s1, s2, label, path_noext, jrc=None):
    """Fig 2 helper: SAR | optical | label for one chip."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(s1[0], cmap="gray"); ax[0].set_title("S1 (VV)")
    ax[1].imshow(_s2_rgb(s2)); ax[1].set_title("S2 (RGB)")
    ax[2].imshow(_s2_rgb(s2)); ax[2].imshow(mask_to_rgba(label, jrc))
    ax[2].set_title("Label")
    for a in ax:
        a.axis("off")
    _savefig(fig, path_noext); plt.close(fig)


def comparison_grid(rows: List[Dict], path_noext, dpi: int = 300):
    """Fig 3 (headline): each row = one chip across methods/regimes.

    rows: list of dicts with keys 'title' and 'panels', where panels is a list
    of (subtitle, rgba_or_gray_image) tuples. All images pre-rendered by caller.
    """
    import matplotlib.pyplot as plt
    ncol = max(len(r["panels"]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(2.4 * ncol, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        for j in range(ncol):
            ax = axes[i, j]; ax.axis("off")
            if j < len(r["panels"]):
                sub, img = r["panels"][j]
                ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
                if i == 0:
                    ax.set_title(sub, fontsize=10)
        axes[i, 0].set_ylabel(r["title"], fontsize=10)
    _savefig(fig, path_noext, dpi=dpi); plt.close(fig)


def degradation_curve(results: Dict[str, Dict[str, float]], path_noext,
                      metric="iou"):
    """Fig 4: metric vs modality regime, one line per method.
    results: {method: {'both':x, 's1_only':y, 's2_only':z}}."""
    import matplotlib.pyplot as plt
    order = ["both", "s1_only", "s2_only"]
    labels = ["Both", "SAR only", "Optical only"]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for method, vals in results.items():
        y = [vals.get(r, np.nan) for r in order]
        ax.plot(labels, y, marker="o",
                color=METHOD_COLORS.get(method), label=method)
    ax.set_ylabel(metric.upper()); ax.set_xlabel("Modality availability")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _savefig(fig, path_noext); plt.close(fig)


def pareto_plot(points: List[Dict], path_noext, x="gflops", y="iou"):
    """Fig 5: accuracy vs compute. points: [{'name','gflops','params','iou'}].
    Your model should sit top-left (high IoU, low compute)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for p in points:
        ax.scatter(p[x], p[y], s=40 + 8 * p.get("params", 0),
                   color=METHOD_COLORS.get(p["name"], "#555"))
        ax.annotate(p["name"], (p[x], p[y]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("GFLOPs"); ax.set_ylabel(y.upper())
    ax.grid(True, alpha=0.3)
    _savefig(fig, path_noext); plt.close(fig)
