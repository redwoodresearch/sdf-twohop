"""The S -> S cell under the two training recipes on one model: SDF documents (the d375 grid) against Q/A rows (the
qa grid), teacher-forced rank-1 and eligible items against dose, one line per peak learning rate.
    uv run scripts/recipe_compare.py   # reads results/figures/{d375,qa}_grid_rank1.json (strength_grid --metric rank1)
-> results/figures/recipe_cells_rank1.{png,pdf}
Both grids share the schedule (linear decay to zero at 375 documents or rows per fact, a checkpoint every 25) and the
candidate pool (the 60 trained GJRN codes), so the panels are directly comparable; web-text mixes exist only for the
document runs and are drawn dashed.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from strength_grid import PALETTE  # noqa: E402

from sdf_twohop import plots as P  # noqa: E402
from sdf_twohop.paths import RESULTS  # noqa: E402

CELL = "sdf_sdf/registry"
COLUMNS = [("d375", "SDF: documents"), ("qa", "SFT: Q/A rows")]


def style(m: dict, colors: dict) -> dict:
    c = colors[m["lr"]]
    return {"color": c, "linestyle": "--" if m["mix"] else "-", "marker": "s" if m["mix"] else "o",
            "markersize": 3.5, "linewidth": 1.4, "markerfacecolor": "white" if m["mix"] else c}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=str(RESULTS / "figures"))
    ap.add_argument("--out", default="recipe_cells_rank1.png")
    a = ap.parse_args()
    figs = Path(a.figures)
    grids = {p: json.loads((figs / f"{p}_grid_rank1.json").read_text()) for p, _ in COLUMNS}
    assert all(g["metric"] == "rank1" for g in grids.values())
    lrs = sorted({m["lr"] for g in grids.values() for ms in g["runs"].values() for m in ms}, key=float, reverse=True)
    colors = dict(zip(lrs, PALETTE, strict=False))
    model = "Qwen3.6-35B-A3B"
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), sharex=True, sharey="row")
        for j, (prefix, label) in enumerate(COLUMNS):
            chance = None
            for ms in grids[prefix]["runs"].values():
                pts = [(m["dose"], m["cells"][CELL]) for m in ms if m["cells"][CELL]["n"]]
                if not pts:
                    continue
                axes[0, j].plot([d for d, _ in pts], [c["composability"][0] for _, c in pts], **style(ms[0], colors))
                axes[1, j].plot([d for d, _ in pts], [c["n"] for _, c in pts], **style(ms[0], colors))
                chance = chance or pts[0][1]["chance_rank1"]
            if chance:
                axes[0, j].axhline(chance, color=P.MUTED, ls="-.", lw=1.0)
                axes[0, j].text(5, chance, f"chance {chance:.1%}", color=P.MUTED, fontsize=8, va="bottom", ha="left")
            axes[0, j].set_title(f"S → S, {label}")
            axes[1, j].set_xlabel("documents (SDF) or Q/A rows (SFT) per implanted fact")
        axes[0, 0].set_ylabel("rank-1 of the true code\n(teacher-forced, trained pool)")
        axes[1, 0].set_ylabel("eligible items (both hops correct)")
        axes[0, 0].set_ylim(0, 1.02)
        axes[1, 0].set_ylim(0, None)
        axes[0, 0].set_xlim(0, 390)
        fig.suptitle(f"Fictional university → country → GJRN code: SDF documents against Q/A SFT ({model}; "
                     "linear decay to zero at 375 per fact)", fontsize=11)  # fmt: skip
        hs = [Line2D([], [], color=c, lw=2, label=f"peak LR {lr}") for lr, c in colors.items()]
        hs += [Line2D([], [], color=P.INK, lw=1.4, marker="o", ms=4, label="no mix"),
               Line2D([], [], color=P.INK, lw=1.4, ls="--", marker="s", ms=4, mfc="white",
                      label="1:1 C4 mix (SDF only)")]  # fmt: skip
        fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = figs / a.out
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
