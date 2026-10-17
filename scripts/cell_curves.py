"""The four cells of the 2x2 design against dose, one figure per run (README steps 7 and 9): P -> P (real university ->
country -> anthem), P -> S (real university -> country -> GJRN code), S -> P (fictional university -> country ->
anthem) and S -> S (fictional university -> country -> GJRN code), as columns; teacher-forced rank-1 of the true
answer among the cell's candidate pool, its loss advantage (mean NLL of the wrong candidates minus NLL of the true one)
and the eligible-item count, as rows. The `bridge_curves` conventions: SOCRATES-eligible items only, 95% bootstrap
over bridge countries, chance drawn per panel (1 / pool size: 70 anthems or the 60 trained codes), band edges drawn.
The pretrained second hop is the anthem; the ISO numeric code is overwritten by the GJRN training in both recipes.
    uv run scripts/cell_curves.py --prefix d375    # the document grid, every unmixed rate: d375_lr{5e-4,...}
    uv run scripts/cell_curves.py --prefix qa      # the Q/A-row grid
    uv run scripts/cell_curves.py --prefix q8 / q8qa   # the same on Qwen3-8B
-> results/figures/<prefix>_cellcurves_lr<lr>.{png,pdf} per rate, <prefix>_cellcurves_grid.json
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

from sdf_twohop import plots as P  # noqa: E402
from sdf_twohop import stats as ST  # noqa: E402
from sdf_twohop.paths import RESULTS, RUNS  # noqa: E402

CELLS = [  # (cell, relation, label, colour), the strength-grid colours
    ("pre_pre", "anthem", "P → P: real university → country → anthem", "#0072B2"),
    ("pre_sdf", "registry", "P → S: real university → country → GJRN code", "#56B4E9"),
    ("sdf_pre", "anthem", "S → P: fictional university → country → anthem", "#E69F00"),
    ("sdf_sdf", "registry", "S → S: fictional university → country → GJRN code", "#D55E00"),
]
RECIPE = {"d375": ("synthetic documents", "documents"), "qa": ("Q/A rows", "rows"),
          "q8": ("synthetic documents", "documents"), "q8qa": ("Q/A rows", "rows")}  # fmt: skip
STATS = [  # (key, per-item value)
    ("rank1", lambda r: float(r["ranks"]["q3"]["rank"] == 1)),
    ("loss_advantage", lambda r: r["ranks"]["q3"]["loss_advantage"]),
    ("pool_nll", lambda r: r["ranks"]["q3"]["mean_distractor_nll"]),
    ("gold_nll", lambda r: r["ranks"]["q3"]["gold_nll"]),
]


def checkpoint(run: str, tag: str, dose: int, n_boot: int) -> dict:
    rows = P.items(run, tag)
    out = {"run": run, "tag": tag, "dose": dose, "cells": {}}
    for cell, rel, *_ in CELLS:
        el = [r for r in rows if r["cell"] == cell and r["relation"] == rel and r["flags"]["upstream"]["eligible"]]
        m = {"n": len(el), "n_items": sum(1 for r in rows if r["cell"] == cell and r["relation"] == rel)}
        if el:
            for key, value in STATS:
                try:  # the oldest untrained-model evaluations lack the distractor losses (no loss advantage at dose 0)
                    vals = ST.group(el, lambda r: r["e2"], value)
                except KeyError:
                    continue
                m[key] = list(ST.cluster_bootstrap(vals, n_boot=n_boot))
            m["n_candidates"] = st.mean(r["ranks"]["q3"]["n_candidates"] for r in el)
            m["chance_rank1"] = 1 / m["n_candidates"]
            m["rank_median"] = float(np.median([r["ranks"]["q3"]["rank"] for r in el]))
            m["composability"] = float(np.mean([r["flags"]["upstream"]["composed"] for r in el]))
        out["cells"][f"{cell}/{rel}"] = m
    return out


def series(run: str, n_boot: int) -> list[dict]:
    """Evaluated checkpoints in dose order, preceded by the untrained model at dose 0 (the run's `base` evaluation on
    the same items: the P -> P reference; the cells with an implanted hop have no eligible items there)."""
    if not (RUNS / run / "checkpoints.jsonl").exists():
        return []
    tags = [(0, "base")] + P.checkpoint_tags(run)
    return [checkpoint(run, tag, dose, n_boot) for dose, tag in tags
            if (RESULTS / "evals" / run / tag / "summary.json").exists()]  # fmt: skip


def band(ax, x, pts, colour):
    ax.plot(x, [v[0] for v in pts], color=colour, marker="o", markersize=3.5, linewidth=1.4)
    lo, hi = [v[1] for v in pts], [v[2] for v in pts]
    ax.fill_between(x, lo, hi, color=colour, alpha=0.12, linewidth=0)
    for edge in (lo, hi):
        ax.plot(x, edge, color=colour, linewidth=0.7, alpha=0.45)


def figure(ms: list[dict], out: Path, title: str, unit: str):
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(3, len(CELLS), figsize=(4.4 * len(CELLS) + 1, 9.2), sharex=True, sharey="row")
        for j, (cell, rel, label, colour) in enumerate(CELLS):
            key = f"{cell}/{rel}"
            pts = [(m["dose"], m["cells"][key]) for m in ms if "rank1" in m["cells"][key]]
            x = [d for d, _ in pts]
            if pts:
                band(axes[0, j], x, [c["rank1"] for _, c in pts], colour)
                adv = [(d, c["loss_advantage"]) for d, c in pts if "loss_advantage" in c]
                band(axes[1, j], [d for d, _ in adv], [v for _, v in adv], colour)
                chance = pts[0][1]["chance_rank1"]
                axes[0, j].axhline(chance, color=P.MUTED, ls="-.", lw=1.0)
                axes[0, j].text(5, chance, f"chance {chance:.1%}", color=P.MUTED, fontsize=8, va="bottom", ha="left")
            axes[2, j].plot([m["dose"] for m in ms], [m["cells"][key]["n"] for m in ms], color=colour, marker="o",
                            markersize=3.5, linewidth=1.4)  # fmt: skip
            n_items = ms[-1]["cells"][key]["n_items"] if ms else 0
            axes[2, j].axhline(n_items, color=P.MUTED, ls=":", lw=1.0)
            axes[2, j].text(5, n_items, f"{n_items} items", color=P.MUTED, fontsize=8, va="bottom", ha="left")
            axes[1, j].axhline(0, color=P.INK, lw=0.8)
            axes[0, j].set_title(label, fontsize=9.5)
            axes[2, j].set_xlabel(f"{unit} per implanted fact")
        axes[0, 0].set_ylabel("rank-1 of the true answer among the pool\n(teacher-forced)")
        axes[1, 0].set_ylabel("loss advantage (nats)\nmean NLL of INCORRECT − NLL of CORRECT")
        axes[2, 0].set_ylabel("eligible items (both hops correct)")
        axes[0, 0].set_ylim(0, 1.02)
        axes[2, 0].set_ylim(0, None)
        axes[0, 0].set_xlim(-6, 390)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", choices=list(RECIPE), default="d375")
    ap.add_argument("--rates", nargs="*", default=None, help="default: every unmixed <prefix>_lr* run")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(RESULTS / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recipe, unit = RECIPE[a.prefix]
    if a.rates is None:
        pat = re.compile(rf"^{re.escape(a.prefix)}_lr([0-9.e-]+)$")
        a.rates = sorted({m.group(1) for p in RUNS.iterdir() if (m := pat.match(p.name))}, key=float, reverse=True)
    grid = {}
    for lr in a.rates:
        run = f"{a.prefix}_lr{lr}"
        if ms := series(run, a.n_boot):
            grid[run] = ms
            print(f"{run}: {len(ms)} evaluated checkpoints", flush=True)
    if not grid:
        raise SystemExit("no evaluated runs")
    name = json.loads((RUNS / next(iter(grid)) / "config.json").read_text())["base_model"].split("/")[-1]
    for run, ms in grid.items():
        lr = run.split("_lr")[1]
        title = (f"Both hops by {recipe}, peak LR {lr}: the four cells of the 2×2 design against dose ({name}; "
                 f"candidates: the pool's 70 anthems or the 60 trained codes; intervals over countries)")  # fmt: skip
        figure(ms, out / f"{a.prefix}_cellcurves_lr{lr}.png", title, unit)
    P.write_json(out / f"{a.prefix}_cellcurves_grid.json", grid)
    print(f"wrote {out}/{a.prefix}_cellcurves_lr*.{{png,pdf}} and {a.prefix}_cellcurves_grid.json")


if __name__ == "__main__":
    main()
