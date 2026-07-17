"""Portrait (single-column) recomposition of the Figure-1 teaser: same real
assets and story, stacked vertically so the figure renders on page 1 as a
column-width float at full font size.

  row 1  film strips (3 lanes x 4 GOP cells) + time arrow
  row 2  MV quiver pair: generated (sparse) vs real (rich)
  row 3  running-score chart + escalation chip

Run: python Code/streamdet/fig1_portrait.py
Writes figures/fig1_teaser.pdf (+ PNG preview).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1A2530"
BLUE = "#0072B2"
DORANGE = "#B87700"
GREEN = "#009E73"
GRAY = "#6B7280"
PORANGE = "#FDF0DA"
VERM = "#D55E00"

DATA = Path("results/fig1-data")
SPEC = json.loads(Path("figures/fig1_clips.json").read_text())
OUT_PDF = Path("figures/fig1_teaser.pdf")
OUT_PNG = Path("figures/fig1_preview.png")

ROLE_FILE = {"gen": "gen", "real": "real4", "unc": "unc1"}
ROLE_LABEL = {"gen": "generated", "real": "real", "unc": "uncertain"}
ROLE_COLOR = {"gen": BLUE, "real": GREEN, "unc": DORANGE}

FIG_W, FIG_H = 3.32, 3.62

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "text.color": INK,
})


def traj(role):
    key = ROLE_FILE[role]
    for c in SPEC["clips"]:
        if c["role"] == key:
            return np.array(c["running_max"], float)
    raise KeyError(role)


def crop_to(img, aspect):
    h, w = img.shape[:2]
    tw = int(h * aspect)
    if w >= tw:
        x0 = (w - tw) // 2
        return img[:, x0:x0 + tw]
    th = int(w / aspect)
    y0 = (h - th) // 2
    return img[y0:y0 + th]


def mv_crop(gx, gy, aspect):
    gh, gw = gx.shape
    tw = int(gh * aspect)
    if gw > tw:
        c0 = (gw - tw) // 2
        return gx[:, c0:c0 + tw], gy[:, c0:c0 + tw]
    return gx, gy


def quiver_panel(ax, frame_path, npz_path, gop_key, aspect,
                 keep_frac=0.22, step=2, scale=18):
    img = crop_to(mpimg.imread(frame_path), aspect)
    ax.imshow(img, interpolation="lanczos", alpha=0.94)
    mv = np.load(npz_path)
    gx, gy = mv_crop(mv[f"{gop_key}_x"], mv[f"{gop_key}_y"], aspect)
    gx = gx[::step, ::step]
    gy = gy[::step, ::step]
    gh, gw = gx.shape
    H, W = img.shape[:2]
    X, Y = np.meshgrid((np.arange(gw) + 0.5) * W / gw,
                       (np.arange(gh) + 0.5) * H / gh)
    mag = np.hypot(gx, gy)
    thr = np.quantile(mag[mag > 0], 1 - keep_frac) if (mag > 0).any() else 0
    keep = mag >= max(thr, 1e-6)
    ax.quiver(X[keep], Y[keep], gx[keep], -gy[keep],
              color=VERM, scale=scale, width=0.010, alpha=1.0,
              edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])


def strip_row(fig):
    x0, cell_w, gap = 0.075, 0.212, 0.012
    lane_y = {"gen": 0.842, "real": 0.748, "unc": 0.654}
    lane_h = 0.086
    for role in ("gen", "real", "unc"):
        y0 = lane_y[role]
        pref = ROLE_FILE[role]
        slots = [0, 1] if role == "gen" else [0, 1, 2, 3]
        cell_aspect = (cell_w * FIG_W) / (lane_h * FIG_H)
        for k in slots:
            ax = fig.add_axes([x0 + k * (cell_w + gap), y0, cell_w, lane_h])
            p = DATA / f"{pref}_f{k}.jpg"
            ax.imshow(crop_to(mpimg.imread(p), cell_aspect), aspect="auto",
                      interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(ROLE_COLOR[role])
                s.set_linewidth(0.9)
            if k == 0:
                ax.text(0.035, 0.82, ROLE_LABEL[role],
                        transform=ax.transAxes, fontsize=5.2,
                        fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.22",
                                  fc=ROLE_COLOR[role], ec="none"), zorder=5)
        if role == "gen":
            axe = fig.add_axes([x0 + 2 * (cell_w + gap), y0,
                                2 * cell_w + gap, lane_h])
            axe.set_facecolor("#F4F5F7")
            axe.set_xticks([])
            axe.set_yticks([])
            for s in axe.spines.values():
                s.set_edgecolor("#B9C0C9")
                s.set_linewidth(0.7)
                s.set_linestyle((0, (2.5, 2.5)))
            axe.text(0.5, 0.5, "stream ends (generated clips run short)",
                     ha="center", va="center", fontsize=5.0, color=GRAY,
                     style="italic", transform=axe.transAxes)
    total_w = 4 * cell_w + 3 * gap
    ar = FancyArrowPatch((x0, 0.636), (x0 + total_w, 0.636),
                         transform=fig.transFigure, arrowstyle="-|>",
                         mutation_scale=7, lw=0.8, color=GRAY)
    fig.add_artist(ar)
    fig.text(x0 + total_w / 2, 0.614, "stream time (one cell per GOP)",
             ha="center", color=GRAY, fontsize=5.4)
    fig.text(x0 + total_w / 2, 0.985, "video arrives as compressed GOPs",
             ha="center", va="top", fontsize=6.6, color=INK,
             fontweight="bold")


def mv_row(fig):
    y0, h = 0.388, 0.158
    w_p = 0.435
    x_g, x_r = 0.075, 0.075 + w_p + 0.028
    aspect = (w_p * FIG_W) / (h * FIG_H)
    axg = fig.add_axes([x_g, y0, w_p, h])
    quiver_panel(axg, DATA / "gen_f1.jpg", DATA / "gen_mv.npz", "g1", aspect,
                 keep_frac=0.10, scale=40)
    for s in axg.spines.values():
        s.set_edgecolor(BLUE)
        s.set_linewidth(1.0)
    axg.text(0.03, 0.85, "generated: sparse motion", transform=axg.transAxes,
             fontsize=5.2, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.22", fc=BLUE, ec="none"),
             zorder=5)
    axr = fig.add_axes([x_r, y0, w_p, h])
    quiver_panel(axr, DATA / "real4_f1.jpg", DATA / "real4_mv.npz", "g1",
                 aspect, keep_frac=0.14, scale=40)
    for s in axr.spines.values():
        s.set_edgecolor(GREEN)
        s.set_linewidth(1.0)
    axr.text(0.03, 0.85, "real: rich motion", transform=axr.transAxes,
             fontsize=5.2, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.22", fc=GREEN, ec="none"),
             zorder=5)
    fig.text(0.5, 0.598, "the encoder already computed motion",
             ha="center", va="top", fontsize=6.6, fontweight="bold",
             color=INK)
    fig.text(0.5, 0.572, "a CPU parse: ${\\sim}10^5$ MACs/GOP, no pixel-domain "
             "forward pass $\\to$ score $s_t$",
             ha="center", va="top", fontsize=5.4, color=GRAY)


def chart_row(fig):
    tau, w = float(SPEC["tau"]), float(SPEC["width"])
    cx, cy, c_w, c_h = 0.115, 0.085, 0.60, 0.235
    axs = fig.add_axes([cx, cy, c_w, c_h])
    T = 8
    axs.axhspan(tau - w, tau, color=PORANGE, zorder=0)
    axs.axhline(tau, color=GRAY, ls="--", lw=0.8, zorder=1)
    axs.text(1.0, tau + 0.03, r"$\tau$", fontsize=6.8, color=GRAY,
             ha="center", va="bottom")
    axs.text(T - 0.1, tau - w / 2, "defer band", fontsize=5.2,
             color=DORANGE, ha="right", va="center")
    for role in ("gen", "real", "unc"):
        v = traj(role)[:T]
        axs.plot(np.arange(1, len(v) + 1), v, color=ROLE_COLOR[role],
                 lw=1.5, marker="o", ms=1.9, zorder=3, clip_on=False)
    g = traj("gen")
    cross = int(np.argmax(g >= tau)) + 1
    axs.plot([cross], [g[cross - 1]], marker="o", ms=5.6, mfc="none",
             mec=BLUE, mew=1.4, zorder=4)
    axs.annotate("exit: generated (GOP 2)", xy=(cross + 0.12, g[cross - 1] - 0.03),
                 xytext=(cross + 0.75, 0.60), fontsize=5.4, color=BLUE,
                 fontweight="bold", va="top", ha="left",
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8,
                                 shrinkA=1, shrinkB=2,
                                 connectionstyle="arc3,rad=0.25"))
    r = traj("real")[:T]
    axs.annotate("commit real (stream end)", xy=(T - 0.5, r[-1] - 0.035),
                 xytext=(2.4, 0.13), fontsize=5.4, color=GREEN, va="center",
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.8,
                                 connectionstyle="arc3,rad=-0.2"))
    axs.set_xlabel("prefix observed (GOPs)", fontsize=5.8, labelpad=1)
    axs.set_ylabel(r"running score $M_t$", fontsize=5.8, labelpad=1)
    axs.set_xlim(0.7, T + 0.3)
    axs.set_ylim(-0.03, 1.03)
    axs.set_xticks([1, 4, 8])
    axs.set_yticks([0, 0.5, 1])
    axs.tick_params(labelsize=5.4, length=2, pad=1)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)
    fig.text(cx + c_w / 2 + 0.12, 0.368,
             "one monotone score, one threshold, anytime-valid (Prop. 2)",
             ha="center", va="top", fontsize=6.4, fontweight="bold",
             color=INK)

    ex, ey, e_w, e_h = 0.745, 0.140, 0.235, 0.15
    chip = FancyBboxPatch((ex, ey), e_w, e_h,
                          boxstyle="round,pad=0.006,rounding_size=0.010",
                          transform=fig.transFigure, fc=PORANGE, ec=DORANGE,
                          lw=0.9)
    fig.add_artist(chip)
    fig.text(ex + e_w / 2, ey + e_h - 0.014, "reasoning on demand",
             ha="center", va="top", fontsize=5.3, fontweight="bold",
             color=DORANGE)
    fig.text(ex + e_w / 2, ey + e_h - 0.052,
             "pixel CNN / small VLM\nGPU $\\cdot$ ${\\sim}15\\%$",
             ha="center", va="top", fontsize=5.4, color=INK, linespacing=1.4)
    u = traj("unc")[:8]
    y_end = cy + c_h * (u[-1] + 0.03) / 1.06
    ar3 = FancyArrowPatch((cx + c_w + 0.004, y_end), (ex - 0.002, ey + e_h / 2),
                          transform=fig.transFigure, arrowstyle="-|>",
                          mutation_scale=8, lw=1.2, color=DORANGE,
                          connectionstyle="arc3,rad=-0.25")
    fig.add_artist(ar3)
    fig.text(ex + e_w / 2, ey - 0.018, "verdict", ha="center", va="top",
             fontsize=5.6, color=INK, fontweight="bold")
    ar4 = FancyArrowPatch((ex + e_w / 2, ey - 0.002), (ex + e_w / 2, ey - 0.016),
                          transform=fig.transFigure, arrowstyle="-|>",
                          mutation_scale=6, lw=0.8, color=DORANGE)
    fig.add_artist(ar4)


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
    strip_row(fig)
    mv_row(fig)
    chart_row(fig)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=240)
    print("wrote", OUT_PDF, "and preview")


if __name__ == "__main__":
    main()
