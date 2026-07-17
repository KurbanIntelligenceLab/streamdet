"""Portrait Figure-1 teaser. Section 3 rehauled: a clean trajectory chart
(no in-plot text) whose three curves each flow into an aligned OUTCOME CARD
(exit early / defer to GPU reasoning / commit real), so the streaming
decisions are the visual product. Absolute-inch grid, explicit gutters.

Run: python Code/streamdet/fig1_portrait.py
Writes Paper/ver2/fig1_teaser.pdf (+ PNG preview).
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
PBLUE = "#EAF3F9"
PORANGE = "#FDF0DA"
PGREEN = "#E4F4EF"
VERM = "#D55E00"

DATA = Path("results/fig1-data")
SPEC = json.loads(Path("Code/streamdet/fig1_clips.json").read_text())
OUT_PDF = Path("Paper/ver2/fig1_teaser.pdf")
OUT_PNG = Path("/private/tmp/claude-501/-Volumes-Stash-Projects-streamdet/"
               "eb058455-0215-426e-852a-d1590f23b6f3/scratchpad/fig1_preview.png")

ROLE_FILE = {"gen": "gen", "real": "real4", "unc": "unc1"}
ROLE_LABEL = {"gen": "generated", "real": "real", "unc": "uncertain"}
ROLE_COLOR = {"gen": BLUE, "real": GREEN, "unc": DORANGE}

FIG_W, FIG_H = 3.32, 3.92

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "text.color": INK,
})

LEFT, RIGHT = 0.10, 3.22
CELL_H = 0.27


def IN(x, y, w, h):
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def TX(x, y):
    return x / FIG_W, y / FIG_H


def traj(role):
    key = ROLE_FILE[role]
    for c in SPEC["clips"]:
        if c["role"] == key:
            return np.array(c["running_max"], float)
    raise KeyError(role)


def raw(role):
    key = ROLE_FILE[role]
    for c in SPEC["clips"]:
        if c["role"] == key:
            return np.array(c.get("scores", c["running_max"]), float)
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
                 keep_frac=0.22, step=2, scale=40):
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


def section_header(fig, y, num, title, sub=None):
    fig.text(*TX((LEFT + RIGHT) / 2, y), f"{num}  {title}", ha="center",
             va="center", fontsize=6.8, fontweight="bold", color=INK)
    if sub:
        fig.text(*TX((LEFT + RIGHT) / 2, y - 0.115), sub, ha="center",
                 va="center", fontsize=5.6, color=GRAY)


# ------------------------------------------------------------- section 1
def strip_row(fig):
    section_header(fig, 3.80, "1", "video arrives as compressed GOPs")
    n, gapx = 4, 0.045
    cell_w = (RIGHT - LEFT - (n - 1) * gapx) / n
    lane_tops = {"gen": 3.67, "real": 3.35, "unc": 3.03}
    aspect = cell_w / CELL_H
    for role, ytop in lane_tops.items():
        y0 = ytop - CELL_H
        pref = ROLE_FILE[role]
        slots = [0, 1] if role == "gen" else [0, 1, 2, 3]
        for k in slots:
            ax = fig.add_axes(IN(LEFT + k * (cell_w + gapx), y0, cell_w,
                                 CELL_H))
            ax.imshow(crop_to(mpimg.imread(DATA / f"{pref}_f{k}.jpg"), aspect),
                      aspect="auto", interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(ROLE_COLOR[role])
                s.set_linewidth(0.9)
            if k == 0:
                ax.text(0.05, 0.80, ROLE_LABEL[role],
                        transform=ax.transAxes, fontsize=5.3,
                        fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc=ROLE_COLOR[role], ec="none"), zorder=5)
        if role == "gen":
            ax = fig.add_axes(IN(LEFT + 2 * (cell_w + gapx), y0,
                                 2 * cell_w + gapx, CELL_H))
            ax.set_facecolor("#F4F5F7")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#B9C0C9")
                s.set_linewidth(0.7)
                s.set_linestyle((0, (2.5, 2.5)))
            ax.text(0.5, 0.5, "stream ends\n(generated clips run short)",
                    ha="center", va="center", fontsize=5.2, color=GRAY,
                    style="italic", transform=ax.transAxes, linespacing=1.4)
    ay = lane_tops["unc"] - CELL_H - 0.09
    ar = FancyArrowPatch(TX(LEFT, ay), TX(RIGHT, ay),
                         transform=fig.transFigure, arrowstyle="-|>",
                         mutation_scale=7, lw=0.8, color=GRAY)
    fig.add_artist(ar)
    fig.text(*TX((LEFT + RIGHT) / 2, ay - 0.075),
             "stream time (one cell per GOP)", ha="center", color=GRAY,
             fontsize=5.6)
    return ay - 0.075


# ------------------------------------------------------------- section 2
def mv_row(fig, top):
    hy = top - 0.14
    section_header(fig, hy, "2", "the encoder already computed motion",
                   "a CPU parse: ${\\sim}10^5$ MACs/GOP, no pixel-domain "
                   "forward pass $\\to$ score $s_t$")
    p_h, gapx = 0.545, 0.10
    p_w = (RIGHT - LEFT - gapx) / 2
    y0 = hy - 0.115 - 0.09 - p_h
    aspect = p_w / p_h
    axg = fig.add_axes(IN(LEFT, y0, p_w, p_h))
    quiver_panel(axg, DATA / "gen_f1.jpg", DATA / "gen_mv.npz", "g1", aspect,
                 keep_frac=0.10)
    for s in axg.spines.values():
        s.set_edgecolor(BLUE)
        s.set_linewidth(1.0)
    axg.text(0.04, 0.86, "generated: sparse motion",
             transform=axg.transAxes, fontsize=5.3, fontweight="bold",
             color="white",
             bbox=dict(boxstyle="round,pad=0.25", fc=BLUE, ec="none"),
             zorder=5)
    axr = fig.add_axes(IN(LEFT + p_w + gapx, y0, p_w, p_h))
    quiver_panel(axr, DATA / "real4_f1.jpg", DATA / "real4_mv.npz", "g1",
                 aspect, keep_frac=0.14)
    for s in axr.spines.values():
        s.set_edgecolor(GREEN)
        s.set_linewidth(1.0)
    axr.text(0.04, 0.86, "real: rich motion", transform=axr.transAxes,
             fontsize=5.3, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.25", fc=GREEN, ec="none"),
             zorder=5)
    return y0


# ------------------------------------------------------------- section 3
def chart_row(fig, mv_bottom):
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    tau, w = float(SPEC["tau"]), float(SPEC["width"])
    hy = mv_bottom - 0.14
    fig.text(*TX((LEFT + RIGHT) / 2, hy),
             "3  one monotone score, three decisions",
             ha="center", va="center", fontsize=6.8, fontweight="bold",
             color=INK)

    # legend strip: line-style key lives BETWEEN title and chart, never on data
    ly = hy - 0.135
    cx = (LEFT + RIGHT) / 2
    x0 = cx - 0.80
    fig.add_artist(Line2D([TX(x0, ly)[0], TX(x0 + 0.17, ly)[0]],
                          [TX(x0, ly)[1]] * 2, transform=fig.transFigure,
                          color=INK, lw=2.0, solid_capstyle="round"))
    fig.text(*TX(x0 + 0.215, ly), "running max $M_t$", fontsize=5.2,
             color=INK, va="center", ha="left")
    x1 = cx + 0.12
    fig.add_artist(Line2D([TX(x1, ly)[0], TX(x1 + 0.17, ly)[0]],
                          [TX(x1, ly)[1]] * 2, transform=fig.transFigure,
                          color=INK, lw=0.8, ls=(0, (2.2, 1.6)), alpha=0.8))
    fig.text(*TX(x1 + 0.215, ly), "per-GOP score $s_t$", fontsize=5.2,
             color=GRAY, va="center", ha="left")

    ch_top = hy - 0.20
    ch_bot = 0.315
    axs = fig.add_axes(IN(0.38, ch_bot, RIGHT - 0.38, ch_top - ch_bot))
    T = 8
    XMAX = 10.75

    # zones: commit-generated above the gate, defer band under it
    axs.axhspan(tau, 1.10, color=PBLUE, alpha=0.65, zorder=0)
    axs.axhspan(tau - w, tau, color=PORANGE, alpha=0.8, zorder=0)
    axs.axhline(tau, color=INK, ls=(0, (5, 2.4)), lw=1.0, zorder=2)
    axs.text(XMAX - 0.15, tau + 0.055, "gate $\\tau$", fontsize=5.6,
             color=INK, ha="right", va="bottom", zorder=8)

    glow = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]

    def series(role, color, upto=None, z=4):
        s_raw = raw(role)
        m = traj(role)
        n = min(len(s_raw), len(m), T) if upto is None else upto
        t = np.arange(1, n + 1)
        # the raw per-GOP signal: thin dashed trace beneath the envelope
        axs.plot(t, s_raw[:n], color=color, lw=0.8, ls=(0, (2.2, 1.6)),
                 alpha=0.55, marker="o", ms=1.6, mfc="white", mew=0.5,
                 zorder=z, clip_on=False)
        # the running-max envelope: a true staircase riding the signal
        axs.plot(t, m[:n], color=color, lw=2.1, zorder=z + 1,
                 drawstyle="steps-post", path_effects=glow, clip_on=False,
                 solid_capstyle="round")
        axs.plot(t, m[:n], color=color, lw=0, marker="o", ms=2.3,
                 zorder=z + 1, clip_on=False)
        return m, n

    g = traj("gen")
    cross = int(np.argmax(g >= tau)) + 1
    series("gen", BLUE, upto=cross, z=5)
    u, _ = series("unc", DORANGE)
    r, _ = series("real", GREEN)
    u = u[:T]
    r = r[:T]

    # gate-firing burst at the generated crossing
    bx, by = cross, g[cross - 1]
    axs.plot([bx], [by], marker="o", ms=7.0, mfc="white", mec=BLUE,
             mew=1.5, zorder=7)
    axs.plot([bx], [by], marker="o", ms=2.8, mfc=BLUE, mec="none", zorder=7)
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        axs.plot([bx + 0.20 * np.cos(ang), bx + 0.34 * np.cos(ang)],
                 [by + 0.055 * np.sin(ang), by + 0.095 * np.sin(ang)],
                 color=BLUE, lw=0.85, zorder=7, solid_capstyle="round")

    # endpoint frame chips: the SAME clips, where their streams end
    def chip(role, fidx, x, y):
        p = DATA / f"{ROLE_FILE[role]}_f{fidx}.jpg"
        if not p.exists():
            return
        img = crop_to(mpimg.imread(p), 16 / 9)
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.062, interpolation="lanczos"),
            (x, y), frameon=True, pad=0.08,
            bboxprops=dict(edgecolor=ROLE_COLOR[role], linewidth=0.8,
                           boxstyle="round,pad=0.08", facecolor="white"))
        ab.zorder = 6
        axs.add_artist(ab)

    # generated: the exit story lives in the commit-generated zone
    chip("gen", 1, 3.3, 1.0)
    axs.plot([bx + 0.14, 3.3 - 0.45], [by + 0.05, 0.995], color=GRAY,
             lw=0.5, alpha=0.6, zorder=2)
    axs.text(4.3, 1.058, "exit early: generated", fontsize=5.8,
             fontweight="bold", color=BLUE, va="center", ha="left", zorder=8)
    axs.text(4.3, 0.985, "GOP 2 $\\cdot$ CPU only", fontsize=5.0,
             color=GRAY, va="center", ha="left", zorder=8)

    # right-hand decisions column: fixed vertical slots, nothing overlaps
    CHX = 9.55
    chip("unc", 7, CHX, 0.88)
    axs.text(CHX, 0.695, "defer $\\to$ GPU reasoning", fontsize=5.7,
             fontweight="bold", color=DORANGE, va="center", ha="center",
             zorder=8)
    axs.text(CHX, 0.605, "pixel CNN / VLM $\\cdot$ ${\\sim}15\\%$",
             fontsize=5.0, color=GRAY, va="center", ha="center", zorder=8)
    axs.plot([T + 0.12, CHX - 0.66], [u[-1], 0.88], color=GRAY,
             lw=0.5, alpha=0.6, zorder=2)
    chip("real", 3, CHX, 0.40)
    axs.text(CHX, 0.215, "commit: real", fontsize=5.7,
             fontweight="bold", color=GREEN, va="center", ha="center",
             zorder=8)
    axs.text(CHX, 0.125, "stream end $\\cdot$ CPU only", fontsize=5.0,
             color=GRAY, va="center", ha="center", zorder=8)
    axs.plot([7 + 0.12, CHX - 0.66], [r[-1], 0.40], color=GRAY,
             lw=0.5, alpha=0.6, zorder=2)

    axs.set_xlim(0.55, XMAX)
    axs.set_ylim(-0.04, 1.10)
    axs.set_xticks([1, 2, 4, 6, 8])
    axs.set_yticks([0, 0.5, 1.0])
    axs.set_xlabel("stream time (GOPs)", fontsize=5.8, labelpad=1.5)
    axs.set_ylabel("running score $M_t$", fontsize=5.8, labelpad=1.5)
    axs.tick_params(labelsize=5.3, length=2, pad=1.5, color=GRAY)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axs.spines[sp].set_color(GRAY)


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
    bottom1 = strip_row(fig)
    bottom2 = mv_row(fig, bottom1)
    chart_row(fig, bottom2)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=240)
    print("wrote", OUT_PDF, "and preview")


if __name__ == "__main__":
    main()
