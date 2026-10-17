"""Paper-style figure helpers: one visual style for every figure under results/fancyfigures/.

Figures are drawn at print size (a full-width 2 x 3 grid is about 7.2 in wide) with small fonts, saved as a 600-dpi PNG
and a vector PDF. Series identity is colour (Okabe-Ito) plus a marker, so it survives colour-vision deficiency and
greyscale print; an interval is a light fill with its two edges drawn as thin lines, so overlapping bands stay legible.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

INK, MUTED, GRID = "#222222", "#7a7a7a", "#e6e6e6"
FONT = "DejaVu Sans"  # the only sans-serif bundled with matplotlib; "STIXGeneral" gives a Times-like serif instead
STYLE = {
    "font.family": FONT, "mathtext.fontset": "dejavusans", "font.size": 7.5, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "axes.labelpad": 2.5, "ytick.major.pad": 2, "xtick.major.pad": 2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "ytick.minor.size": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
}  # fmt: skip

# Bridge entities of the two-hop chain (fictional university -> bridge -> code), in legend order, with a fixed colour
# and marker each, so a bridge looks the same in every figure whichever subset it shows.
BRIDGES = {
    "country": ("Real country", "#0072B2", "o"),
    "famous": ("Famous city", "#009E73", "s"),
    "obscure": ("Obscure city", "#CC79A7", "D"),
    "coined": ("Fabricated city", "#E69F00", "^"),
    "spouses": ("Spouses city", "#D55E00", "v"),
}
# The four cells of the 2x2 (first hop -> second hop, each real or fabricated), keyed as in the cell-grid JSON, in
# the colours of the original metrics grid (real first hop blues, fabricated first hop oranges). In line figures a
# fabricated second hop is also dashed with hollow squares.
CELLS = {
    "pre_pre/anthem": ("Real → real, anthem", "#0072B2", False),
    "pre_sdf/registry": ("Real → fabricated", "#56B4E9", True),
    "sdf_pre/anthem": ("Fabricated → real, anthem", "#E69F00", False),
    "sdf_sdf/registry": ("Fabricated → fabricated", "#D55E00", True),
}
# The ISO numeric code as the real second hop instead of the anthem, drawn as its anthem cell with white hatching.
ISO = {"pre_pre/isonum": "pre_pre/anthem", "sdf_pre/isonum": "sdf_pre/anthem"}
ISO_LABELS = {"pre_pre/isonum": "Real → real, ISO code", "sdf_pre/isonum": "Fabricated → real, ISO code"}


def line_style(colour: str, marker: str, dashed: bool = False) -> dict:
    s = {"color": colour, "marker": marker, "markersize": 3.2, "linewidth": 1.2, "markeredgecolor": "white",
         "markeredgewidth": 0.4, "zorder": 3}  # fmt: skip
    if dashed:  # hollow markers on a dashed line
        s |= {"linestyle": (0, (3.5, 1.5)), "markerfacecolor": "white", "markeredgecolor": colour,
              "markeredgewidth": 0.8, "markersize": 3.0}  # fmt: skip
    return s


def cell_style(key: str) -> dict:
    _, colour, dashed = CELLS[key]
    return line_style(colour, "s" if dashed else "o", dashed)


def band(ax, x, lo, hi, colour: str, dashed: bool = False):
    """An interval: a light fill with its edges outlined in the series colour (dashed for a dashed series)."""
    ax.fill_between(x, lo, hi, color=colour, alpha=0.10 if not dashed else 0.07, linewidth=0, zorder=1)
    for edge in (lo, hi):
        ax.plot(x, edge, color=colour, linewidth=0.5, alpha=0.55, zorder=2, ls=(0, (3.5, 1.5)) if dashed else "-")


CHANCE = {"color": MUTED, "ls": (0, (4, 1.5, 1, 1.5)), "lw": 0.8}
CEILING = {"ls": (0, (1, 1.5)), "lw": 0.9}  # most eligible items possible: every item of the cell
BASELINE = {"color": "#555555", "ls": "-", "lw": 0.9, "alpha": 0.7}  # the untrained model's Real -> real level


def ceiling(ax, y: float, colour: str = INK, text: str | None = None):
    """The most eligible items a series can reach, dotted; `text` labels it at the right edge, just above the line."""
    ax.axhline(y, color=colour, zorder=1.5, **CEILING)
    if text:
        ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), xytext=(-1, 1), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6, color=colour)  # fmt: skip


def chance(ax, y: float):
    """The chance level as an unlabelled reference line; `legend(..., chance=True)` names it once for the figure."""
    ax.axhline(y, zorder=1.5, **CHANCE)


def suptitle(fig, text: str, size: float, min_size: float = 7.5, **kw):
    """A bold figure title at `size` points, or smaller (down to `min_size`) so it is no wider than the figure."""
    t = fig.suptitle(text, fontsize=size, fontweight="bold", **kw)
    renderer = fig.canvas.get_renderer()
    while size > min_size and t.get_window_extent(renderer).width > fig.bbox.width:
        size -= 0.25
        t.set_fontsize(size)
    return t


def ylabel(ax, text: str, width: int = 42, lines: int | None = None):
    """Wrap a long label onto balanced lines of at most `width` characters, the parenthetical on a line of its own; with
    `lines`, the whole label balanced over that many lines instead (the parenthetical may share a line)."""
    if lines:
        wrap = lambda w: textwrap.wrap(text, w, break_on_hyphens=False)  # noqa: E731
        ax.set_ylabel("\n".join(wrap(next(w for w in range(1, len(text) + 1) if len(wrap(w)) <= lines))))
        return
    head, sep, tail = text.partition(" (")
    wrap = lambda w: textwrap.wrap(head, w, break_on_hyphens=False)  # noqa: E731
    n = len(wrap(width))  # the fewest lines that fit, then the narrowest width that still gives that many
    lines = wrap(next(w for w in range(1, width + 1) if len(wrap(w)) <= n)) + ([f"({tail}"] if sep else [])
    ax.set_ylabel("\n".join(lines))


def panel_letters(axes, x: float = -0.02, y: float = 1.04):
    for i, ax in enumerate(axes):
        ax.text(x, y, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right",
                va="bottom")  # fmt: skip


def bridge_handle(key: str) -> Line2D:
    label, colour, marker = BRIDGES[key]
    return Line2D([], [], label=label, **line_style(colour, marker) | {"markersize": 4.5})


def cell_handle(key: str) -> Line2D:
    return Line2D([], [], label=CELLS[key][0], **cell_style(key) | {"markersize": 4.2})


def cell_bar(key: str) -> dict:
    """A cell as a bar in its colour; an ISO-code cell as its anthem cell's colour with white hatching."""
    colour = CELLS[ISO.get(key, key)][1]
    return {"facecolor": colour, "edgecolor": "white", "linewidth": 0.6, "hatch": "////" if key in ISO else None,
            "zorder": 2}  # fmt: skip


def cell_patch(key: str, label: str | None = None) -> Patch:
    b = cell_bar(key)
    if label is None:
        label = ISO_LABELS[key] if key in ISO else CELLS[key][0]
    return Patch(label=label, facecolor=b["facecolor"], edgecolor=b["edgecolor"], linewidth=b["linewidth"],
                 hatch=b["hatch"])  # fmt: skip


def chance_handle() -> Line2D:
    return Line2D([], [], label="Chance", **CHANCE)


def ceiling_handle(label: str = "Ceiling") -> Line2D:
    return Line2D([], [], label=label, color=INK, **CEILING)


def baseline_handle(label: str = "Real → real, anthem, untrained model") -> Line2D:
    return Line2D([], [], label=label, **BASELINE)


def legend(fig, handles: list, y: float = 0.0, ncol: int | None = None, **kw):
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles), bbox_to_anchor=(0.5, y),
               **{"handlelength": 2.4, "columnspacing": 1.6} | kw)  # fmt: skip


def highlight(fig, axes, pad: float = 0.008):
    """A rounded box around a group of panels and everything they carry (ticks, labels, titles, panel letters), drawn
    after the layout is final."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = Bbox.union([ax.get_tightbbox(renderer) for ax in axes]).transformed(fig.transFigure.inverted())
    fig.patches.append(FancyBboxPatch((bb.x0, bb.y0), bb.width, bb.height, transform=fig.transFigure, fill=False,
                                      boxstyle=f"round,pad={pad},rounding_size=0.012", ec=INK, lw=0.9))  # fmt: skip


def save(fig, out: Path, name: str):
    """`out`/png/`name`.png and `out`/pdf/`name`.pdf."""
    for ext in ("png", "pdf"):
        (out / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / ext / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
