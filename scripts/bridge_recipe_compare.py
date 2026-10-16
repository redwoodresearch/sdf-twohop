"""The S -> S cell under the two training recipes for each bridge: synthetic documents (the `dbridge` grid) against
Q/A rows (the `bridge` grid), teacher-forced rank-1 and eligible items against dose, one column per bridge, one line
per peak learning rate, documents solid and rows dashed. The real-country column is `recipe_cells_rank1` again, as
the anchor; the three city bridges are README step 13.
    uv run scripts/bridge_recipe_compare.py   # reads results/figures/{bridge,dbridge}_grid.json (bridge_compare.py)
-> results/figures/bridge_recipe_rank1.{png,pdf}
Both grids share the schedule (linear decay to zero at 375 documents or rows per fact, a checkpoint every 25) and the
candidate pool (the 60 trained codes), so the panels are directly comparable; intervals over bridge entities.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from bridge_compare import BRIDGES, MODELS, PALETTE  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from sdf_twohop import plots as P  # noqa: E402
from sdf_twohop.paths import RESULTS  # noqa: E402

RECIPES = [("docs", "documents", "-", "o"), ("35b", "Q/A rows", "--", "s")]  # MODELS key, label, line, marker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=str(RESULTS / "figures"))
    ap.add_argument("--out", default="bridge_recipe_rank1.png")
    a = ap.parse_args()
    figs = Path(a.figures)
    grids = {k: json.loads((figs / f"{MODELS[k]['out']}_grid.json").read_text()) for k, *_ in RECIPES}
    rates = MODELS["docs"]["rates"]
    colours = dict(zip(rates, PALETTE, strict=False))
    cols = []  # (bridge label, {recipe key: run prefix}) for every bridge with runs under both recipes
    for i, (label, *_) in enumerate(BRIDGES):
        label = MODELS["docs"].get("labels", {}).get(label, label)  # country documents were richer than city ones
        prefixes = {k: MODELS[k]["prefixes"][i] for k, *_ in RECIPES}
        if all(any(f"{p}_lr{lr}" in grids[k] for lr in rates) for k, p in prefixes.items()):
            cols.append((label, prefixes))
    if not cols:
        raise SystemExit("no bridge has runs under both recipes")
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(2, len(cols), figsize=(4.4 * len(cols) + 1, 6.4), sharex=True, sharey="row",
                                 squeeze=False)  # fmt: skip
        for j, (label, prefixes) in enumerate(cols):
            chance = None
            for key, _, ls, marker in RECIPES:
                for lr in rates:
                    ms = grids[key].get(f"{prefixes[key]}_lr{lr}", [])
                    c = colours[lr]
                    s = {"color": c, "linestyle": ls, "marker": marker, "markersize": 3.5, "linewidth": 1.4,
                         "markerfacecolor": "white" if ls == "--" else c}  # fmt: skip
                    pts = [(m["dose"], m["rank1"][0]) for m in ms if "rank1" in m]
                    if pts:
                        axes[0, j].plot([d for d, _ in pts], [v for _, v in pts], **s)
                    if ms:
                        axes[1, j].plot([m["dose"] for m in ms], [m["n"] for m in ms], **s)
                    chance = chance or next((m["chance_rank1"] for m in ms if "chance_rank1" in m), None)
            if chance:
                axes[0, j].axhline(chance, color=P.MUTED, ls="-.", lw=1.0)
                axes[0, j].text(5, chance, f"chance {chance:.1%}", color=P.MUTED, fontsize=8, va="bottom", ha="left")
            axes[0, j].set_title(f"S → S, {label}")
            axes[1, j].set_xlabel("documents or Q/A rows per implanted fact")
        axes[0, 0].set_ylabel("rank-1 of the true code\n(teacher-forced, trained pool)")
        axes[1, 0].set_ylabel("eligible items (both hops correct)")
        axes[0, 0].set_ylim(0, 1.02)
        axes[1, 0].set_ylim(0, None)
        axes[0, 0].set_xlim(0, 390)
        hs = [Line2D([], [], color=c, lw=2, label=f"peak LR {lr}") for lr, c in colours.items()]
        hs += [Line2D([], [], color=P.INK, lw=1.4, ls=ls, marker=m, ms=4, mfc="white" if ls == "--" else P.INK,
                      label=lab) for _, lab, ls, m in RECIPES]  # fmt: skip
        fig.suptitle("Synthetic documents (SDF) against Q/A rows (SFT) by bridge type: teacher-forced rank-1 of the "
                     "true code and eligible items in the S → S cell (Qwen3.6-35B-A3B; linear decay to zero at 375 "
                     "per fact)", fontsize=11)  # fmt: skip
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        out = figs / a.out
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
