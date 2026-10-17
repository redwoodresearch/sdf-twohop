"""The 2x2 design on one set of axes per rate (README steps 7 and 9): the four cells as lines (P -> P and S -> P on
the anthem, P -> S and S -> S on the GJRN code), one column per peak learning rate, rows for teacher-forced rank-1,
loss advantage and eligible items, against dose from the untrained model (dose 0). Two extras over `cell_curves`:
- the ISO numeric code as the pretrained second hop, dotted in the colour of its anthem cell, ranked among the pool's
  70 ISO codes. Training on GJRN codes overwrites it in free generation (the model answers the ISO prompt with the
  GJRN code), so its eligibility collapses to a handful of items or none; like every cell, it is scored on its
  eligible items only, so its rank-1 and loss-advantage lines stop where too few remain;
- for the document grids, a second figure for the 1:1 C4-mixed runs on the same y-scales as the unmixed one.
Thin horizontal lines mark the untrained model's P -> P level (anthem solid, ISO dotted) in every panel.
Rank-1 and loss-advantage points are drawn only where at least MIN_ITEMS items support them (the lowest rate's
implanted cells have one to five eligible items for most of the run); the eligible-items row always shows the count.
    uv run scripts/cell_grid.py --prefix d375   # -> d375_cellgrid.png (unmixed) and d375_cellgrid_c4.png (mixed)
    uv run scripts/cell_grid.py --prefix qa     # -> qa_cellgrid.png
    uv run scripts/cell_grid.py --prefix q8 / q8qa   # Qwen3-8B
-> results/figures/<prefix>_cellgrid{,_c4}.{png,pdf}, <prefix>_cellgrid.json
"""

import argparse
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from sdf_twohop import plots as P  # noqa: E402
from sdf_twohop import stats as ST  # noqa: E402
from sdf_twohop.paths import RESULTS, RUNS  # noqa: E402

CELLS = [  # (cell, relation, label, colour, dotted)
    ("pre_pre", "anthem", "P → P (anthem)", "#0072B2", False),
    ("pre_sdf", "registry", "P → S (GJRN code)", "#56B4E9", False),
    ("sdf_pre", "anthem", "S → P (anthem)", "#E69F00", False),
    ("sdf_sdf", "registry", "S → S (GJRN code)", "#D55E00", False),
    ("pre_pre", "isonum", "P → P (ISO code)", "#0072B2", True),
    ("sdf_pre", "isonum", "S → P (ISO code)", "#E69F00", True),
]
RECIPE = {"d375": ("synthetic documents", "documents"), "qa": ("Q/A rows", "rows"),
          "q8": ("synthetic documents", "documents"), "q8qa": ("Q/A rows", "rows")}  # fmt: skip
MIN_ITEMS = 5  # fewer items than this and the rank-1 / loss-advantage point is not drawn (the count still is)
STATS = [("rank1", lambda r: float(r["ranks"]["q3"]["rank"] == 1)),
         ("loss_advantage", lambda r: r["ranks"]["q3"]["loss_advantage"]),
         ("pool_nll", lambda r: r["ranks"]["q3"]["mean_distractor_nll"]),  # the two losses behind the advantage
         ("gold_nll", lambda r: r["ranks"]["q3"]["gold_nll"])]  # fmt: skip


def checkpoint(run: str, tag: str, dose: int, n_boot: int) -> dict:
    rows = P.items(run, tag)
    out = {"run": run, "tag": tag, "dose": dose, "cells": {}}
    for cell, rel, _, _, _ in CELLS:
        items = [r for r in rows if r["cell"] == cell and r["relation"] == rel]
        # eligible = both single hops correct in free generation; every statistic on these items. Not SOCRATES's
        # guessable/unusable exclusions: they depend on the free-generation two-hop answer, which no statistic here uses
        used = [r for r in items if r["flags"]["upstream"]["both_known"]]
        m = {"n_eligible": len(used), "n_items": len(items), "n_countries": len({r["e2"] for r in used})}
        if used:
            for key, value in STATS:
                try:  # the oldest untrained-model evaluations lack some loss fields
                    vals = ST.group(used, lambda r: r["e2"], value)
                except KeyError:
                    continue
                m[key] = list(ST.cluster_bootstrap(vals, n_boot=n_boot))
            m["n_candidates"] = st.mean(r["ranks"]["q3"]["n_candidates"] for r in used)
            m["chance_rank1"] = 1 / m["n_candidates"]
            ranks = [r["ranks"]["q3"]["rank"] for r in used]
            m["rank_median"] = float(np.median(ranks))
            m["rank"] = [m["rank_median"], float(np.percentile(ranks, 25)), float(np.percentile(ranks, 75))]
        out["cells"][f"{cell}/{rel}"] = m
    return out


def series(run: str, n_boot: int) -> list[dict]:
    if not (RUNS / run / "checkpoints.jsonl").exists():
        return []
    tags = [(0, "base")] + P.checkpoint_tags(run)
    return [checkpoint(run, tag, dose, n_boot) for dose, tag in tags
            if (RESULTS / "evals" / run / tag / "summary.json").exists()]  # fmt: skip


def line(ax, x, pts, colour, dotted, band=True):
    s = {"color": colour, "linewidth": 1.4, "markersize": 3.5, "marker": "o"}
    if dotted:
        s |= {"linestyle": ":", "markerfacecolor": "white", "linewidth": 1.3}
    ax.plot(x, [v[0] for v in pts] if band else pts, **s)
    if band:
        lo, hi = [v[1] for v in pts], [v[2] for v in pts]
        ax.fill_between(x, lo, hi, color=colour, alpha=0.07, linewidth=0)
        for edge in (lo, hi):
            ax.plot(x, edge, color=colour, linewidth=0.6, alpha=0.4, linestyle=":" if dotted else "-")


def figure(grid: dict, rates: list[str], out: Path, title: str, unit: str, ylims=None):
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(3, len(rates), figsize=(4.4 * len(rates) + 1, 9.2), sharex=True, sharey="row",
                                 squeeze=False)  # fmt: skip
        for j, (lr, ms) in enumerate(zip(rates, [grid[lr] for lr in rates], strict=True)):
            for cell, rel, _, colour, dotted in CELLS:
                key = f"{cell}/{rel}"
                # a point needs MIN_ITEMS items, or a cell with two eligible items flips between 0 and 1
                pts = [(m["dose"], m["cells"][key]) for m in ms
                       if "rank1" in m["cells"][key] and m["cells"][key]["n_eligible"] >= MIN_ITEMS]  # fmt: skip
                if pts:
                    line(axes[0, j], [d for d, _ in pts], [c["rank1"] for _, c in pts], colour, dotted)
                    adv = [(d, c["loss_advantage"]) for d, c in pts if "loss_advantage" in c]
                    if adv:
                        line(axes[1, j], [d for d, _ in adv], [v for _, v in adv], colour, dotted)
                line(axes[2, j], [m["dose"] for m in ms], [m["cells"][key]["n_eligible"] for m in ms], colour,
                     dotted, band=False)  # fmt: skip
            base = next((m for m in ms if m["dose"] == 0), None)  # the untrained P -> P levels, as references
            for key, dotted in (("pre_pre/anthem", False), ("pre_pre/isonum", True)):
                c = base["cells"].get(key, {}) if base else {}
                colour = next(col for cell, rel, _, col, _ in CELLS if f"{cell}/{rel}" == key)
                ref = {"color": colour, "linestyle": ":" if dotted else "-", "linewidth": 0.9, "alpha": 0.55}
                if "rank1" in c and c["n_eligible"] >= MIN_ITEMS:
                    axes[0, j].axhline(c["rank1"][0], **ref)
                if "loss_advantage" in c and c["n_eligible"] >= MIN_ITEMS:
                    axes[1, j].axhline(c["loss_advantage"][0], **ref)
                if c:
                    axes[2, j].axhline(c["n_eligible"], **ref)
            chances = sorted({round(m["cells"][k]["chance_rank1"], 4) for m in ms for k in m["cells"]
                              if "chance_rank1" in m["cells"][k]})  # fmt: skip
            for ch in chances:
                axes[0, j].axhline(ch, color=P.MUTED, ls="-.", lw=0.9)
            if chances:
                axes[0, j].text(5, max(chances), "chance " + " / ".join(f"{c:.1%}" for c in chances), color=P.MUTED,
                                fontsize=8, va="bottom", ha="left")  # fmt: skip
            axes[1, j].axhline(0, color=P.INK, lw=0.8)
            axes[0, j].set_title(f"peak LR {lr}")
            axes[2, j].set_xlabel(f"{unit} per implanted fact")
        axes[0, 0].set_ylabel("rank-1 of the true answer among the pool\n(teacher-forced)")
        axes[1, 0].set_ylabel("loss advantage (nats)\nmean NLL of INCORRECT − NLL of CORRECT")
        axes[2, 0].set_ylabel("eligible items (both hops correct)")
        axes[0, 0].set_ylim(0, 1.02)
        axes[2, 0].set_ylim(0, None)
        axes[0, 0].set_xlim(-6, 390)
        if ylims:
            for row, lim in zip(axes, ylims, strict=True):
                for ax in row:
                    ax.set_ylim(*lim)
        lims = [axes[i, 0].get_ylim() for i in range(3)]
        hs = [Line2D([], [], color=c, lw=1.6, ls=":" if d else "-", marker="o", ms=4, mfc="white" if d else c,
                     label=lab) for _, _, lab, c, d in CELLS]  # fmt: skip
        hs.append(Line2D([], [], color="#0072B2", lw=0.9, alpha=0.55, label="horizontal: untrained P → P level"))
        fig.suptitle(title, fontsize=11)
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, bbox_to_anchor=(0.5, -0.015))
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return lims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", choices=list(RECIPE), default="d375")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(RESULTS / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recipe, unit = RECIPE[a.prefix]
    pat = re.compile(rf"^{re.escape(a.prefix)}_lr([0-9.e-]+)(_c4)?$")
    runs = {(m.group(1), bool(m.group(2))) for p in RUNS.iterdir() if (m := pat.match(p.name))}
    grids = {}
    for mixed in (False, True):
        g = {}
        for lr in sorted({lr for lr, mx in runs if mx == mixed}, key=float):  # columns from the lowest rate up
            run = f"{a.prefix}_lr{lr}" + ("_c4" if mixed else "")
            if ms := series(run, a.n_boot):
                g[lr] = ms
                print(f"{run}: {len(ms)} evaluated checkpoints (dose 0 included)", flush=True)
        if g:
            grids[mixed] = g
    if not grids:
        raise SystemExit("no evaluated runs")
    first = next(iter(grids.values()))
    name = json.loads((RUNS / first[next(iter(first))][0]["run"] / "config.json").read_text())["base_model"]
    name = name.split("/")[-1]
    tail = (f"the four cells of the 2×2 design against dose ({name}; ISO code dotted; eligible items only; candidates: "
            "the pool's 70 anthems or ISO codes, or the 60 trained codes; intervals over countries)")  # fmt: skip
    lims = None
    if False in grids:
        lims = figure(grids[False], list(grids[False]), out / f"{a.prefix}_cellgrid.png",
                      f"Both hops by {recipe}: {tail}", unit)  # fmt: skip
    if True in grids:
        figure(grids[True], list(grids[True]), out / f"{a.prefix}_cellgrid_c4.png",
               f"Both hops by {recipe} with a 1:1 C4 mix: {tail}", unit, lims)  # fmt: skip
    P.write_json(out / f"{a.prefix}_cellgrid.json", {("mixed" if mx else "unmixed"): g for mx, g in grids.items()})
    print(f"wrote {out}/{a.prefix}_cellgrid{{,_c4}}.{{png,pdf}} and {a.prefix}_cellgrid.json")


if __name__ == "__main__":
    main()
