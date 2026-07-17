"""Compose the redesigned Figure 1 from REAL assets (frames, codec MV fields,
logged trajectories). Round 2: MV panel is a generated-vs-real comparison
("motion is the tell"), sparse readable quivers, collision-free labels.

Run locally: python Code/streamdet/fig1_compose.py
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
ORANGE = "#E69F00"
DORANGE = "#B87700"
GREEN = "#009E73"
GRAY = "#6B7280"
PORANGE = "#FDF0DA"

DATA = Path("results/fig1-data")
SPEC = json.loads(Path("figures/fig1_clips.json").read_text())
OUT_PDF = Path("figures/fig1_teaser.pdf")
OUT_PNG = Path("figures/fig1_preview.png")

ROLE_FILE = {"gen": "gen", "real": "real4", "unc": "unc1"}
ROLE_LABEL = {"gen": "generated", "real": "real", "unc": "uncertain"}
ROLE_COLOR = {"gen": BLUE, "real": GREEN, "unc": DORANGE}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.0,
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
    """Center-crop to width/height == aspect (no distortion at render)."""
    h, w = img.shape[:2]
    tw = int(h * aspect)
    if w >= tw:
        x0 = (w - tw) // 2
        return img[:, x0:x0 + tw]
    th = int(w / aspect)
    y0 = (h - th) // 2
    return img[y0:y0 + th]


def crop169(img):
    return crop_to(img, 16 / 9)


def mv_crop(gx, gy, aspect=16 / 9):
    gh, gw = gx.shape
    tw = int(gh * aspect)
    if gw > tw:
        c0 = (gw - tw) // 2
        return gx[:, c0:c0 + tw], gy[:, c0:c0 + tw]
    return gx, gy


def quiver_panel(ax, frame_path, npz_path, gop_key, keep_frac=0.22, step=2,
                 scale=18, aspect=16 / 9):
    img = crop_to(mpimg.imread(frame_path), aspect)
    ax.imshow(img, interpolation="lanczos", alpha=0.92)
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
              color="#D55E00", scale=scale, width=0.009, alpha=1.0,
              edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    return float(mag.mean())


def main():
    tau, w = float(SPEC["tau"]), float(SPEC["width"])
    FIG_W, FIG_H = 7.05, 2.62
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)

    # ================= [A] film strips =================
    x0, cell_w, gap = 0.035, 0.070, 0.006
    lane_y = {"gen": 0.665, "real": 0.395, "unc": 0.125}
    lane_h = 0.225
    for role in ("gen", "real", "unc"):
        y0 = lane_y[role]
        pref = ROLE_FILE[role]
        if role == "gen":
            slots = [0, 1]
        else:
            slots = [0, 1, 2, 3]
        for k in slots:
            ax = fig.add_axes([x0 + k * (cell_w + gap), y0, cell_w, lane_h])
            p = DATA / f"{pref}_f{k}.jpg"
            cell_aspect = (cell_w * FIG_W) / (lane_h * FIG_H)
            ax.imshow(crop_to(mpimg.imread(p), cell_aspect), aspect="auto",
                      interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(ROLE_COLOR[role])
                s.set_linewidth(0.9)
            if k == 0:
                ax.text(0.045, 0.88, ROLE_LABEL[role],
                        transform=ax.transAxes, fontsize=5.8,
                        fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.24",
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
            axe.text(0.5, 0.5, "stream ends\n(generated clips run short)",
                     ha="center", va="center", fontsize=5.6, color=GRAY,
                     style="italic", transform=axe.transAxes,
                     linespacing=1.35)
    total_w = 4 * cell_w + 3 * gap
    ar = FancyArrowPatch((x0, 0.082), (x0 + total_w, 0.082),
                         transform=fig.transFigure, arrowstyle="-|>",
                         mutation_scale=8, lw=0.9, color=GRAY)
    fig.add_artist(ar)
    fig.text(x0 + total_w / 2, 0.026, "stream time (one cell per GOP)",
             ha="center", color=GRAY, fontsize=6.2)
    fig.text(x0 + total_w / 2, 0.985, "video arrives as compressed GOPs",
             ha="center", va="top", fontsize=7.0, color=INK,
             fontweight="bold")

    # ================= [B] MV: generated vs real =================
    qx = x0 + total_w + 0.028
    q_w = 0.165
    axg = fig.add_axes([qx, 0.505, q_w, 0.335])
    q_aspect = (q_w * FIG_W) / (0.335 * FIG_H)
    quiver_panel(axg, DATA / "gen_f1.jpg", DATA / "gen_mv.npz", "g1",
                 aspect=q_aspect)
    for s in axg.spines.values():
        s.set_edgecolor(BLUE)
        s.set_linewidth(1.0)
    axg.text(0.03, 0.85, "generated: motion is sparse",
             transform=axg.transAxes, fontsize=5.6, color="white",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.24", fc=BLUE, ec="none"))
    axr = fig.add_axes([qx, 0.125, q_w, 0.335])
    quiver_panel(axr, DATA / "real4_f1.jpg", DATA / "real4_mv.npz", "g1",
                 keep_frac=0.14, scale=40, aspect=q_aspect)
    for s in axr.spines.values():
        s.set_edgecolor(GREEN)
        s.set_linewidth(1.0)
    axr.text(0.03, 0.85, "real: motion is rich",
             transform=axr.transAxes, fontsize=5.6, color="white",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.24", fc=GREEN, ec="none"))
    fig.text(qx + q_w / 2, 0.985, "the encoder already computed motion",
             ha="center", va="top", fontsize=7.0, fontweight="bold",
             color=INK)
    fig.text(qx + q_w / 2, 0.94,
             "a CPU parse ($\\sim\\!10^5$ MACs/GOP), no pixel-domain\n"
             "forward pass $\\to$ 13-d feature $\\to$ score $s_t$",
             ha="center", va="top", fontsize=5.8, color=GRAY,
             linespacing=1.3)
    fig.text(qx + q_w / 2, 0.038, "motion is the tell", ha="center",
             fontsize=6.3, style="italic", color=INK)

    # ================= [C] score chart =================
    cx = qx + q_w + 0.062
    c_w = 0.215
    axs = fig.add_axes([cx, 0.165, c_w, 0.665])
    T = 8
    axs.axhspan(tau - w, tau, color=PORANGE, zorder=0)
    axs.axhline(tau, color=GRAY, ls="--", lw=0.9, zorder=1)
    axs.text(0.85, tau + 0.025, r"$\tau$", fontsize=7.5, color=GRAY,
             ha="left", va="bottom")
    axs.text(0.85, tau - w - 0.035, "defer band", fontsize=5.8,
             color=DORANGE, ha="left", va="top")
    for role in ("gen", "real", "unc"):
        v = traj(role)[:T]
        t = np.arange(1, len(v) + 1)
        axs.plot(t, v, color=ROLE_COLOR[role], lw=1.7, marker="o", ms=2.1,
                 zorder=3, clip_on=False)
    g = traj("gen")
    cross = int(np.argmax(g >= tau)) + 1
    axs.plot([cross], [g[cross - 1]], marker="o", ms=6.5, mfc="none",
             mec=BLUE, mew=1.5, zorder=4)
    axs.annotate("gate fires:\ngenerated (GOP 2)", xy=(cross - 0.1, g[cross - 1] + 0.02),
                 xytext=(cross - 0.7, 0.60), fontsize=6.0,
                 color=BLUE, fontweight="bold", va="top", ha="left",
                 linespacing=1.25,
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9,
                                 shrinkA=1, shrinkB=3,
                                 connectionstyle="arc3,rad=0.25"))
    r = traj("real")[:T]
    axs.annotate("commit real\n(stream end)", xy=(T - 0.6, r[-1] - 0.03),
                 xytext=(T - 3.3, 0.16), fontsize=6.0, color=GREEN,
                 va="center", linespacing=1.25,
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.9,
                                 connectionstyle="arc3,rad=-0.2"))
    axs.set_xlabel("prefix observed (GOPs)", fontsize=6.4, labelpad=1.5)
    axs.set_ylabel(r"running score $M_t$", fontsize=6.4, labelpad=1.5)
    axs.set_xlim(0.7, T + 0.3)
    axs.set_ylim(-0.03, 1.03)
    axs.set_xticks([1, 4, 8])
    axs.set_yticks([0, 0.5, 1])
    axs.tick_params(labelsize=6, length=2, pad=1.5)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)
    fig.text(cx + c_w / 2, 0.985, "one monotone score, one threshold",
             ha="center", va="top", fontsize=7.0, fontweight="bold",
             color=INK)
    fig.text(cx + c_w / 2, 0.935,
             "anytime-valid at the stopping time (Prop. 2)",
             ha="center", va="top", fontsize=5.9, color=GRAY)

    # ================= [D] escalation chip =================
    ex = cx + c_w + 0.045
    chip = FancyBboxPatch((ex, 0.34), 0.098, 0.30,
                          boxstyle="round,pad=0.008,rounding_size=0.012",
                          transform=fig.transFigure, fc=PORANGE, ec=DORANGE,
                          lw=1.0)
    fig.add_artist(chip)
    fig.text(ex + 0.049, 0.615, "reasoning\non demand", ha="center", va="top",
             fontsize=6.4, fontweight="bold", color=DORANGE, linespacing=1.3)
    fig.text(ex + 0.049, 0.50, "pixel CNN /\nsmall VLM\nGPU · "
             r"$\sim\!15\%$", ha="center", va="top", fontsize=5.9,
             color=INK, linespacing=1.35)
    u = traj("unc")[:T]
    y_end = 0.165 + 0.665 * (u[-1] + 0.03) / 1.06
    ar3 = FancyArrowPatch((cx + c_w + 0.003, y_end), (ex - 0.001, 0.52),
                          transform=fig.transFigure, arrowstyle="-|>",
                          mutation_scale=9, lw=1.3, color=DORANGE,
                          connectionstyle="arc3,rad=-0.22")
    fig.add_artist(ar3)
    fig.text(ex + 0.049, 0.27, "verdict", ha="center", fontsize=6.2,
             color=INK, fontweight="bold")
    ar4 = FancyArrowPatch((ex + 0.049, 0.325), (ex + 0.049, 0.295),
                          transform=fig.transFigure, arrowstyle="-|>",
                          mutation_scale=7, lw=0.9, color=DORANGE)
    fig.add_artist(ar4)

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=220)
    print("wrote", OUT_PDF, "and preview")


if __name__ == "__main__":
    main()
