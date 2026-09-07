"""Figure 2 — curriculum modality-dropout schedule (EXACT, from lgmf.curriculum).

Run this as a cell in any Kaggle notebook that has `lgmf_c` attached. It calls
the SAME `regime_probs()` used during training, so the plotted curve is the
schedule you actually trained with -- nothing is re-derived or hand-drawn.

If you change EPOCHS / warmup_epochs / final_regime_probs, re-run and the figure
updates automatically.
"""
import sys, glob, os
init = glob.glob("/kaggle/input/**/lgmf/__init__.py", recursive=True)
assert init, "attach the lgmf_c dataset"
sys.path.insert(0, os.path.dirname(os.path.dirname(init[0])))

import numpy as np
import matplotlib.pyplot as plt
from lgmf.config import LGMFConfig
from lgmf.curriculum import regime_probs

# ---- must match the training config used for the paper model ----
cfg = LGMFConfig()
cfg.epochs = 45
cfg.use_curriculum = True
# warmup_epochs / final_regime_probs come from LGMFConfig defaults; override here
# ONLY if your final runs used different values.
print(f"epochs={cfg.epochs}  warmup={cfg.warmup_epochs}  "
      f"final_probs={cfg.final_regime_probs}  use_curriculum={cfg.use_curriculum}")

epochs = np.arange(cfg.epochs)
probs = np.array([regime_probs(cfg, e) for e in epochs])   # [E, 3] both, s1, s2

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(3.5, 2.7))
COL = {"both": "#8a8778", "s1": "#1d6fb8", "s2": "#0f6e56"}
ax.plot(epochs, probs[:, 0], color=COL["both"], lw=1.8, label="both")
ax.plot(epochs, probs[:, 1], color=COL["s1"],   lw=1.8, label="SAR-only")
ax.plot(epochs, probs[:, 2], color=COL["s2"],   lw=1.8, label="optical-only")

if cfg.warmup_epochs > 0:
    ax.axvspan(0, cfg.warmup_epochs, color="#00000010", lw=0)
    ax.text(cfg.warmup_epochs / 2, 0.5, "warm-up", rotation=90,
            fontsize=6, ha="center", va="center", color="#555")

ax.set_xlabel("epoch", fontsize=7.5)
ax.set_ylabel("sampling probability", fontsize=7.5)
ax.set_xlim(0, cfg.epochs - 1)
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=6.5)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=6.5, frameon=False, loc="center right")

out = "/kaggle/working/fig2_curriculum"
for ext, dpi in [("png", 300), ("pdf", None), ("svg", None)]:
    fig.savefig(f"{out}.{ext}", dpi=dpi, bbox_inches="tight")
plt.show()

# print the exact numbers for the caption / a table
print("\nepoch : P(both)  P(SAR-only)  P(optical-only)")
for e in [0, cfg.warmup_epochs, cfg.epochs // 2, cfg.epochs - 1]:
    p = regime_probs(cfg, e)
    print(f"{e:5d} : {p[0]:.3f}    {p[1]:.3f}       {p[2]:.3f}")
print(f"\nsaved {out}.pdf / .svg / .png")
