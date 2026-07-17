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

FIG_W, FIG_H = 3.32, 4.60

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


_HEAVY_PATH = None


def _heavy_path():
    """Path to a REAL bold face for headings. matplotlib can't read the bold
    weight out of a macOS .ttc, so we extract Helvetica Neue Bold once (it
    matches the figure's Helvetica body) into a cache; falls back to the
    default synthetic bold off-mac."""
    global _HEAVY_PATH
    if _HEAVY_PATH is not None:
        return _HEAVY_PATH or None
    cache = Path(OUT_PNG).parent / "HelveticaNeue-Bold.ttf"
    try:
        if not cache.exists():
            from fontTools.ttLib import TTCollection
            col = TTCollection("/System/Library/Fonts/HelveticaNeue.ttc")
            col.fonts[1].save(str(cache))   # face 1 = Helvetica Neue Bold
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(str(cache))
        _HEAVY_PATH = str(cache)
    except Exception:
        _HEAVY_PATH = ""
    return _HEAVY_PATH or None


def _overlay(fig):
    """A transparent full-figure axes in INCH coordinates (0..FIG_W, 0..FIG_H).
    One inch in x equals one inch in y on paper, so a Circle drawn here is
    truly round regardless of the figure's aspect ratio."""
    if not hasattr(fig, "_ovl"):
        ax = fig.add_axes([0, 0, 1, 1], zorder=12)
        ax.set_xlim(0, FIG_W)
        ax.set_ylim(0, FIG_H)
        ax.axis("off")
        ax.patch.set_alpha(0.0)
        fig._ovl = ax
    return fig._ovl


def _overlay_bg(fig):
    """Like _overlay but UNDERNEATH the content axes, so anything drawn here is
    occluded by the frame cells and shows only in the gaps between them."""
    if not hasattr(fig, "_ovlbg"):
        ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
        ax.set_xlim(0, FIG_W)
        ax.set_ylim(0, FIG_H)
        ax.axis("off")
        ax.patch.set_alpha(0.0)
        fig._ovlbg = ax
    return fig._ovlbg


def section_header(fig, y, num, title, sub=None):
    import matplotlib.font_manager as fm
    from matplotlib.patches import Circle
    cx = (LEFT + RIGHT) / 2
    ov = _overlay(fig)
    ren = fig.canvas.get_renderer()
    R = 0.088          # badge radius, inches
    GAP = 0.115        # badge-to-title gap, inches
    hp = _heavy_path()
    tkw = ({"fontproperties": fm.FontProperties(fname=hp, size=8.0)} if hp
           else {"fontsize": 8.0, "fontweight": "bold"})
    dkw = ({"fontproperties": fm.FontProperties(fname=hp, size=7.0)} if hp
           else {"fontsize": 7.0, "fontweight": "bold"})
    # measure the title so the badge+title pair can be centred as a unit
    ttl = fig.text(*TX(cx, y), title, ha="left", va="center", color=INK, **tkw)
    fig.canvas.draw()
    tw_in = ttl.get_window_extent(ren).width / fig.bbox.width * FIG_W
    startx = cx - (2 * R + GAP + tw_in) / 2
    bx = startx + R
    ov.add_patch(Circle((bx, y), R, facecolor=INK, edgecolor="none",
                        zorder=13))
    # tiny downward nudge so the digit reads optically centred in the disc
    ov.text(bx, y - 0.004, num, ha="center", va="center", color="white",
            zorder=14, **dkw)
    ttl.set_position(TX(startx + 2 * R + GAP, y))
    if sub:
        fig.text(*TX(cx, y - 0.145), sub, ha="center",
                 va="center", fontsize=5.5, color=GRAY, style="italic")


# ------------------------------------------------------------- section 1
def strip_row(fig):
    section_header(fig, 4.44, "1", "Video Arrives as Compressed GOPs",
                   "one cell per group-of-pictures, in decode order")
    n, gapx = 4, 0.045
    cell_w = (RIGHT - LEFT - (n - 1) * gapx) / n
    lane_tops = {"gen": 4.195, "real": 3.875, "unc": 3.555}
    aspect = cell_w / CELL_H
    for role, ytop in lane_tops.items():
        y0 = ytop - CELL_H
        pref = ROLE_FILE[role]
        slots = [0, 1] if role == "gen" else [0, 1, 2, 3]
        for k in slots:
            ax = fig.add_axes(IN(LEFT + k * (cell_w + gapx), y0, cell_w,
                                 CELL_H))
            ax.set_zorder(1)   # above the registration-stem background layer
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
            ax.set_zorder(1)
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
    ay = lane_tops["unc"] - CELL_H - 0.125
    ov = _overlay(fig)
    from matplotlib.patches import Circle, Polygon
    from matplotlib.collections import LineCollection
    gr = tuple(int(GRAY[i:i + 2], 16) / 255 for i in (1, 3, 5))
    x_end = RIGHT - 0.02
    # a ruler-like GOP time-axis: contiguous segments (flush, no gaps)
    # alternating darker / lighter, the whole bar fading toward the
    # un-streamed future, capped by a matching arrow-head
    n_seg = 22
    e = np.linspace(LEFT, x_end, n_seg + 1)
    segs, cols = [], []
    for i in range(n_seg):
        grad = 1.0 - 0.48 * i / (n_seg - 1)        # overall fade
        base = 0.95 if i % 2 == 0 else 0.42         # darker '=' vs lighter '-'
        segs.append([[e[i], ay], [e[i + 1], ay]])
        cols.append((*gr, min(1.0, base * grad)))
    ov.add_collection(LineCollection(segs, colors=cols, linewidths=3.0,
                                     capstyle="butt", zorder=6))
    a_tail = 0.95 * (1.0 - 0.48)                     # shade at the bar's end
    ov.add_patch(Polygon([[x_end, ay + 0.044], [x_end + 0.10, ay],
                          [x_end, ay - 0.044]], closed=True,
                         facecolor=(*gr, a_tail), edgecolor="none", zorder=6))
    lane_top = lane_tops["gen"]                # very top of the strip stack
    ovbg = _overlay_bg(fig)                     # BEHIND the frame cells
    for k in range(n):
        cxk = LEFT + k * (cell_w + gapx) + cell_w / 2
        # a registration stem tying each GOP marker UP through all three rows
        # to the frames it timestamps; drawn on the background layer, so it is
        # occluded by the frames and reads only in the gaps between them
        ovbg.plot([cxk, cxk], [ay, lane_top], color=VERM, lw=1.1, zorder=1,
                  solid_capstyle="round")
        ov.add_patch(Circle((cxk, ay), 0.044, facecolor="white",
                            edgecolor="none", zorder=8.5))
        ov.add_patch(Circle((cxk, ay), 0.032, facecolor=VERM,
                            edgecolor="white", linewidth=0.8, zorder=9))
        ov.text(cxk, ay - 0.078, f"$t_{{{k+1}}}$", ha="center", va="center",
                fontsize=5.4, color=GRAY, zorder=9)
    return ay - 0.115


# ------------------------------------------------------------- section 2
def mv_row(fig, top):
    hy = top - 0.125
    section_header(fig, hy, "2", "The Encoder Already Computed Motion",
                   "a CPU parse: ${\\sim}10^5$ MACs/GOP, no pixel-domain "
                   "forward pass $\\to$ score $s_t$")
    p_h, gapx = 0.42, 0.10
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
    hy = mv_bottom - 0.185
    section_header(fig, hy, "3", "One Monotone Score, Three Decisions",
                   "aggregate to a running max; gate once, at the end")

    # legend strip: line-style key lives BETWEEN title and chart, never on data
    ly = hy - 0.290
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
                          color=INK, lw=1.2, ls=(0, (2.4, 1.5)), alpha=0.85))
    fig.text(*TX(x1 + 0.215, ly), "per-GOP score $s_t$", fontsize=5.2,
             color=GRAY, va="center", ha="left")

    ch_top = hy - 0.350
    ch_bot = 0.315
    YLO, YHI = -0.07, 1.12          # chart data y-range (also the card frame)
    axs = fig.add_axes(IN(0.38, ch_bot, RIGHT - 0.38, ch_top - ch_bot))
    T = 8
    XMAX = 11.3     # data ends at 8; 8.45-11.3 is the annotation lane
    LANE = 10.0     # lane centerline

    # zones: commit-generated above the gate, defer band under it
    axs.fill_between([0.55, 8.45], tau, 1.12, color=PBLUE, alpha=0.65,
                     zorder=0, lw=0)
    axs.fill_between([0.55, 8.45], tau - w, tau, color=PORANGE, alpha=0.8,
                     zorder=0, lw=0)
    axs.plot([0.55, 8.45], [tau, tau], color=INK, ls=(0, (5, 2.4)),
             lw=1.0, zorder=2, clip_on=False)
    axs.text(0.72, tau + 0.05, "gate $\\tau$", fontsize=5.6,
             color=INK, ha="left", va="bottom", zorder=8)
    axs.text(5.5, tau + 0.058, "commit as generated", fontsize=5.0,
             style="italic", color=BLUE, alpha=0.85, ha="center",
             va="center", zorder=8)
    axs.text(5.2, tau - w / 2, "defer band ($\\pm$ uncertain)", fontsize=4.8,
             style="italic", color=DORANGE, alpha=0.95, ha="center",
             va="center", zorder=8)

    glow = [pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()]

    def series(role, color, upto=None, z=4):
        s_raw = raw(role)
        m = traj(role)
        n = min(len(s_raw), len(m), T) if upto is None else upto
        t = np.arange(1, n + 1)
        # the raw per-GOP signal: dashed trace beneath the envelope
        axs.plot(t, s_raw[:n], color=color, lw=1.25, ls=(0, (2.4, 1.5)),
                 alpha=0.62, marker="o", ms=1.9, mfc="white", mew=0.6,
                 zorder=z, clip_on=False)
        # the running-max envelope: a true staircase riding the signal
        axs.plot(t, m[:n], color=color, lw=1.5, zorder=z + 1,
                 drawstyle="steps-post", path_effects=glow, clip_on=False,
                 solid_capstyle="round")
        axs.plot(t, m[:n], color=color, lw=0, marker="o", ms=2.0,
                 zorder=z + 1, clip_on=False)
        return m, n

    g = traj("gen")
    cross = int(np.argmax(g >= tau)) + 1
    series("gen", BLUE, upto=cross, z=5)
    u, _ = series("unc", DORANGE)
    r, _ = series("real", GREEN)
    u = u[:T]
    r = r[:T]

    # gate-firing marker at the generated crossing. Drawn on the INCH overlay
    # so the rays are truly circular (in data coords the non-square axes would
    # squash them into a lopsided ellipse).
    from matplotlib.patches import Circle
    bx, by = cross, g[cross - 1]
    ovb = _overlay(fig)
    ix = 0.38 + (bx - 0.55) / (XMAX - 0.55) * (RIGHT - 0.38)
    iy = ch_bot + (by - YLO) / (YHI - YLO) * (ch_top - ch_bot)
    ovb.add_patch(Circle((ix, iy), 0.072, facecolor=BLUE, alpha=0.10,
                         edgecolor="none", zorder=7.4))
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ovb.plot([ix + 0.052 * np.cos(a), ix + 0.080 * np.cos(a)],
                 [iy + 0.052 * np.sin(a), iy + 0.080 * np.sin(a)],
                 color=BLUE, lw=1.0, solid_capstyle="round", zorder=7.6)
    ovb.add_patch(Circle((ix, iy), 0.040, facecolor="white", edgecolor=BLUE,
                         linewidth=1.5, zorder=7.7))
    ovb.add_patch(Circle((ix, iy), 0.017, facecolor=BLUE, edgecolor="none",
                         zorder=7.8))

    # frame chips: image and border share ONE extent -- centred by
    # construction (no offsetbox alignment quirks), sized in inches
    from matplotlib.patches import Rectangle
    # everything below is sized in INCHES then mapped to data units through
    # dx/dy, so the stack keeps its physical size at ANY chart height and can
    # never silently drift when the layout above changes
    dx = (XMAX - 0.55) / (RIGHT - 0.38)        # data units per inch, x
    dy = (YHI - YLO) / (ch_top - ch_bot)       # data units per inch, y
    W_IN = 0.52                                # chip width on paper (inches)
    CH_IN = W_IN * 9 / 16                       # chip height (16:9)
    TAG_IN, SUB_IN = 0.082, 0.078              # measured text-line heights
    GCT_IN, GTS_IN, GBLK_IN = 0.026, 0.012, 0.030
    CW, CH = W_IN * dx, CH_IN * dy
    TAG_H, SUB_H = TAG_IN * dy, SUB_IN * dy
    G_CT, G_TS, G_BLK = GCT_IN * dy, GTS_IN * dy, GBLK_IN * dy
    block_in = CH_IN + GCT_IN + TAG_IN + GTS_IN + SUB_IN
    stack_in = 3 * block_in + 2 * GBLK_IN
    blocks = [
        ("gen", 1, BLUE, "$\\bf{Exit\\ early\\!:}$ generated",
         "GOP 2 $\\cdot$ CPU only", (bx + 0.42, by + 0.03)),
        ("unc", 7, DORANGE, "$\\bf{Defer\\!:}$ GPU reasoning",
         "pixel CNN / VLM $\\cdot$ ${\\sim}15\\%$", (T + 0.14, u[-1])),
        ("real", 3, GREEN, "$\\bf{Commit\\!:}$ real",
         "stream end $\\cdot$ CPU only", (7 + 0.14, r[-1])),
    ]
    # centre the stack vertically in the chart's data range, then nudge the
    # whole stack DOWN a little for a better optical balance with the curves
    free = (ch_top - ch_bot) - stack_in
    yc = YHI - (max(free / 2, 0.0) + 0.075) * dy
    for role, fidx, col, tag, sub, (lx0, ly0) in blocks:
        cy = yc - CH / 2
        img = crop_to(mpimg.imread(DATA / f"{ROLE_FILE[role]}_f{fidx}.jpg"),
                      16 / 9)
        im = axs.imshow(img, extent=[LANE - CW / 2, LANE + CW / 2,
                                     cy - CH / 2, cy + CH / 2],
                        aspect="auto", interpolation="lanczos", zorder=6)
        im.set_clip_on(False)   # the lane may run below the axes box
        axs.add_patch(Rectangle((LANE - CW / 2, cy - CH / 2), CW, CH,
                                fill=False, edgecolor=col, lw=1.0, zorder=7,
                                clip_on=False))
        axs.plot([lx0, LANE - CW / 2 - 0.06], [ly0, cy], color=GRAY,
                 lw=0.5, alpha=0.55, zorder=2)
        ty = cy - CH / 2 - G_CT - TAG_H / 2
        axs.text(LANE, ty, tag, fontsize=5.1, color=col,
                 va="center", ha="center", zorder=8)
        sy = ty - TAG_H / 2 - G_TS - SUB_H / 2
        axs.text(LANE, sy, sub, fontsize=4.5, color=GRAY,
                 va="center", ha="center", zorder=8)
        yc = sy - SUB_H / 2 - G_BLK

    axs.set_xlim(0.55, XMAX)
    axs.set_ylim(-0.07, 1.12)
    axs.set_xticks([1, 2, 4, 6, 8])
    axs.set_yticks([0, 0.5, 1.0])
    axs.set_xlabel("stream time (GOPs)", fontsize=5.8, labelpad=1.5)
    # centre the x-label under the DATA span, not the axes+lane
    axs.xaxis.set_label_coords((4.5 - 0.55) / (11.3 - 0.55), -0.115)
    axs.set_ylabel("running score $M_t$", fontsize=5.8, labelpad=1.5)
    axs.tick_params(labelsize=5.3, length=2, pad=1.5, color=GRAY)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axs.spines[sp].set_color(GRAY)
    axs.spines["bottom"].set_bounds(0.55, 8.45)


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
