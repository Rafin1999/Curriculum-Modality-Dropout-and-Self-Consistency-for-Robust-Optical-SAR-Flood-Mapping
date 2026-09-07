"""Figure 1: architecture (a) + self-consistency training (c).
Emits combined fig1_arch.{pdf,svg,png} plus standalone fig1a_model.* and fig1c_consistency.*
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})

C = {"s1":"#cfe3f5","s1e":"#1d6fb8","s2":"#d9f0e4","s2e":"#0f6e56",
     "op":"#eeeadf","ope":"#8a8778","enc":"#e6e1f5","ence":"#6a5fb8",
     "dec":"#e0f0d9","dece":"#4a7d3a","out":"#dcebfb","oute":"#1d6fb8",
     "loss":"#fbe3da","losse":"#c05a30","arrow":"#444444"}

def box(ax, x, y, w, h, text, fc, ec, fs=7.5, lw=1.1, sub=None, subfs=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.018", fc=fc, ec=ec, lw=lw))
    if sub:
        ax.text(x+w/2, y+h*0.70, text, ha="center", va="center", fontsize=fs)
        ax.text(x+w/2, y+h*0.32, sub, ha="center", va="center",
                fontsize=subfs, color="#333", linespacing=1.15)
    else:
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)

def arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.1, color=None, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color or C["arrow"],
                                linestyle=ls, shrinkA=1, shrinkB=1))

def draw_a(ax, label=True):
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    if label:
        ax.text(1, 29.2, "(a)  Model — early-fusion SegFormer (3.7M parameters)",
                fontsize=8.5, fontweight="bold", va="top")
    box(ax, 2, 16.8, 14.5, 7.4, "Sentinel-1 SAR", C["s1"], C["s1e"],
        sub="VV, VH\n(2\u00d7H\u00d7W)", subfs=5.8)
    box(ax, 2, 4.4, 14.5, 7.4, "Sentinel-2 MSI", C["s2"], C["s2e"],
        sub="B2\u2013B4, B8, B11\u2013B12\n(6\u00d7H\u00d7W)", subfs=5.8)
    box(ax, 19.5, 18.5, 7.5, 4.0, r"$\times\, m_{S1}$", "#ffffff", C["s1e"], fs=8)
    box(ax, 19.5, 6.1, 7.5, 4.0, r"$\times\, m_{S2}$", "#ffffff", C["s2e"], fs=8)
    arrow(ax, 16.5, 20.5, 19.5, 20.5); arrow(ax, 16.5, 8.1, 19.5, 8.1)
    box(ax, 31.5, 10.9, 10.5, 6.0, "channel\nconcat", C["op"], C["ope"],
        sub="(8\u00d7H\u00d7W)", subfs=5.8)
    arrow(ax, 27, 20.5, 31.5, 15.1); arrow(ax, 27, 8.1, 31.5, 12.7)
    ax.text(23.2, 1.6, r"$m=[m_{S1},m_{S2}]\in\{0,1\}^2$ — sensor availability at test;"
            " sampled by the dropout curriculum during training",
            fontsize=6.4, ha="center", color="#333")
    enc_x, enc_w = 45.0, 30.0
    ax.add_patch(FancyBboxPatch((enc_x, 6.0), enc_w, 17.6,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        fc="#f4f2fb", ec=C["ence"], lw=1.1, linestyle=(0, (3, 2))))
    ax.text(enc_x+enc_w/2, 21.9, "MiT-B0 hierarchical Transformer encoder",
            ha="center", fontsize=7.3, color="#3d3480")
    stages = [("32","H/4",12.5), ("64","H/8",10.3), ("160","H/16",8.1), ("256","H/32",6.2)]
    sx = 46.5
    for i, (ch, res, h) in enumerate(stages):
        box(ax, sx, 13.9-h/2, 6.0, h, f"S{i+1}", C["enc"], C["ence"], fs=7,
            sub=f"{ch}ch\n{res}", subfs=5.3)
        if i < 3: arrow(ax, sx+6.0, 13.9, sx+6.85, 13.9, lw=0.9)
        sx += 6.85
    arrow(ax, 42, 13.9, 45.0, 13.9)
    box(ax, 78.0, 9.7, 12.0, 8.4, "All-MLP\ndecoder", C["dec"], C["dece"],
        sub="per-stage linear\n\u2192 upsample \u2192 fuse", subfs=5.5)
    arrow(ax, 75.0, 13.9, 78.0, 13.9)
    box(ax, 92.3, 10.5, 7.2, 6.8, "flood\nmask", C["out"], C["oute"],
        sub="$\\hat{y}$ (H\u00d7W)", subfs=5.8)
    arrow(ax, 90.0, 13.9, 92.3, 13.9)

def draw_c(ax, label=True, letter="(b)"):
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    if label:
        ax.text(0, 29.0, f"{letter}  Self-consistency training",
                fontsize=8.5, fontweight="bold", va="top")
    box(ax, 2, 18.5, 14, 5.6, r"$(S_1,\,S_2)$", "#ffffff", C["ope"], fs=7.5)
    box(ax, 2, 6.5, 14, 5.6, r"$(S_1,\,\mathbf{0})$", "#ffffff", C["ope"], fs=7.5)
    box(ax, 24, 18.5, 13, 5.6, r"$F_\theta$", C["enc"], C["ence"], fs=8.5)
    box(ax, 24, 6.5, 13, 5.6, r"$F_\theta$", C["enc"], C["ence"], fs=8.5)
    arrow(ax, 16, 21.3, 24, 21.3); arrow(ax, 16, 9.3, 24, 9.3)
    arrow(ax, 30.5, 18.5, 30.5, 12.1, style="-", lw=0.9, ls=":", color="#6a5fb8")
    ax.text(31.7, 15.3, "shared $\\theta$", fontsize=6.2, color="#6a5fb8")
    box(ax, 45, 18.5, 15, 5.6, r"$\hat{y}_{\mathrm{both}}$", C["out"], C["oute"],
        fs=7.5, sub="stop-grad", subfs=5.6)
    box(ax, 45, 6.5, 15, 5.6, r"$\hat{y}_{S1}$", C["out"], C["oute"], fs=8)
    arrow(ax, 37, 21.3, 45, 21.3); arrow(ax, 37, 9.3, 45, 9.3)
    box(ax, 64.0, 12.4, 35.5, 6.2,
        r"$\mathcal{L}_{\mathrm{cons}}=\Vert\sigma(\hat{y}_{S1})-\sigma(\hat{y}_{\mathrm{both}})\Vert^2$",
        C["loss"], C["losse"], fs=6.2)
    arrow(ax, 60.5, 20.0, 64.5, 16.8, lw=0.9)
    arrow(ax, 60.5, 10.5, 64.5, 14.2, lw=0.9)
    ax.text(64.0, 8.0, "$\\mathcal{L}_{\\mathrm{seg}}=$ Focal\u2013Tversky$(\\hat{y},\\,y)$ on valid px",
            fontsize=6.4, ha="left")
    ax.text(64.0, 4.4, r"$\mathcal{L}=\mathcal{L}_{\mathrm{seg}}+\lambda\,\mathcal{L}_{\mathrm{cons}},\ \ \lambda=0.5$",
            fontsize=6.9, ha="left")
    ax.text(2, 1.6, r"evaluation weights: $\theta_{\mathrm{EMA}}\leftarrow 0.99\,\theta_{\mathrm{EMA}}+0.01\,\theta$",
            fontsize=6.4, ha="left", color="#333")

def save(fig, name):
    for ext, dpi in [("png", 300), ("pdf", None), ("svg", None)]:
        fig.savefig(f"/home/claude/{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# combined (a) + (c)
fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.16,
                      left=0.015, right=0.988, top=0.97, bottom=0.02)
draw_a(fig.add_subplot(gs[0])); draw_c(fig.add_subplot(gs[1]), letter="(b)")
save(fig, "fig1_arch")

# standalone (a)
fig = plt.figure(figsize=(7.2, 2.4)); draw_a(fig.add_axes([0, 0, 1, 1]), label=False)
save(fig, "fig1a_model")

# standalone (c)
fig = plt.figure(figsize=(7.2, 2.3)); draw_c(fig.add_axes([0, 0, 1, 1]), label=False)
save(fig, "fig1c_consistency")
print("saved fig1_arch / fig1a_model / fig1c_consistency  (.pdf .svg .png)")
