"""The real-city arms (README step 12): a famous and an obscure real-city bridge, each trained with Q/A rows at three
peak rates, with two cells per arm: S -> S (fictional university -> city -> municipal code) and S -> P (fictional
university -> city -> the city's country, the pretrained second hop). Runs without evaluations are skipped.
    uv run scripts/real_city_compare.py
-> results/figures/realcity_cells.png        free-generation composability, teacher-forced rank-1 (chance drawn) and
                                             eligible counts against dose; one column per arm x cell, one line per rate
   results/figures/realcity_cells_rank1.png  the `_cells_rank1` layout: rank-1 and eligible counts
   results/figures/realcity_fame.png         per-city composability (mean over the checkpoints from 150 rows on, at
                                             the headline rate) against the city's daily pageviews and against the
                                             untrained model's loss of its country, both arms pooled, rank correlations
   results/figures/realcity_retention.png    single hops against dose per arm: how many cities the model still places
                                             in their country (the pretrained hop, free generation) beside the two
                                             implanted hops' recall
   results/figures/realcity_grid.json        every number behind the figures, the S -> P majority-country baseline per
                                             arm and the per-city table of the fame figure
Intervals are 95% bootstraps over cities (sdf_twohop.stats); items are the SOCRATES-eligible ones and the rank is
against the full trained pool (codes) or the arm's countries.
"""

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from sdf_twohop import plots as P  # noqa: E402
from sdf_twohop import stats as ST  # noqa: E402
from sdf_twohop.paths import RESULTS, RUNS, cities_json  # noqa: E402

ARMS = [("famous", "qafamous", "famous real city"), ("obscure", "qaobscure", "obscure real city")]
RATES = ("5e-4", "1.6e-4", "5e-5")
HEADLINE_RATE, FAME_FROM_DOSE = "1.6e-4", 150
PALETTE = ["#D55E00", "#E69F00", "#009E73"]  # rates in descending order, as the other grid figures
CELLS = [("sdf_sdf", "city_registry", "S → S: city → code"), ("sdf_pre", "city_country", "S → P: city → country")]
METRICS = [  # (key, label, per-item value)
    ("composability", "latent composability (free generation)", lambda r: float(r["flags"]["upstream"]["composed"])),
    ("rank1", "rank-1 of the true answer among the pool\n(teacher-forced)",
     lambda r: float(r["ranks"]["q3"]["rank"] == 1)),
]  # fmt: skip


def eligible(rows: list[dict], cell: str, relation: str) -> list[dict]:
    return [r for r in rows if r["cell"] == cell and r["relation"] == relation and r["flags"]["upstream"]["eligible"]]


def checkpoint(run: str, tag: str, dose: int, n_boot: int) -> dict:
    rows = P.items(run, tag)
    summ = json.loads((RESULTS / "evals" / run / tag / "summary.json").read_text())
    m = {"run": run, "tag": tag, "dose": dose, "cells": {}, "per_city": {}}
    for cell, rel, _ in CELLS:
        el = eligible(rows, cell, rel)
        c = {"n": len(el)}
        if el:
            for key, _, value in METRICS:
                c[key] = list(ST.cluster_bootstrap(ST.group(el, lambda r: r["e2"], value), n_boot=n_boot))
            c["chance_rank1"] = 1 / st.mean(r["ranks"]["q3"]["n_candidates"] for r in el)
            c["cot"] = float(np.mean([r["flags"]["upstream"]["cot_answer"] for r in el]))
            c["gold_rank_mean"] = float(np.mean([r["ranks"]["q3"]["rank"] for r in el]))
        all_ = [r for r in rows if r["cell"] == cell and r["relation"] == rel]
        c["q1_correct"] = sum(r["flags"]["upstream"]["q1"] for r in all_)
        c["q2_correct"] = sum(r["flags"]["upstream"]["q2"] for r in all_)
        c["q2_rank1"] = float(np.mean([r["ranks"]["q2"]["rank"] == 1 for r in all_])) if all_ else None
        m["cells"][f"{cell}/{rel}"] = c
        m["per_city"][f"{cell}/{rel}"] = {
            r["e2"]: {"eligible": r["flags"]["upstream"]["eligible"], "composed": r["flags"]["upstream"]["composed"],
                      "rank1": r["ranks"]["q3"]["rank"] == 1}
            for r in rows if r["cell"] == cell and r["relation"] == rel
        }  # fmt: skip
    bel = summ["beliefs"]
    m["recall"] = {f: bel[f]["template_recall"] for f in ("university_city", "city_registry") if f in bel}
    return m


def series(run: str, n_boot: int) -> list[dict]:
    if not (RUNS / run / "checkpoints.jsonl").exists():
        return []
    out = [checkpoint(run, tag, dose, n_boot) for dose, tag in P.checkpoint_tags(run)
           if (RESULTS / "evals" / run / tag / "summary.json").exists()]  # fmt: skip
    return sorted(out, key=lambda m: m["dose"])


def style(lr: str) -> dict:
    return {"color": PALETTE[RATES.index(lr)], "marker": "o", "markersize": 3.5, "linewidth": 1.4}


def legend(fig):
    hs = [Line2D([], [], color=PALETTE[i], lw=2, label=f"peak LR {lr}") for i, lr in enumerate(RATES)]
    fig.legend(handles=hs, loc="lower center", ncol=len(RATES), frameon=False, bbox_to_anchor=(0.5, -0.01))


def columns():
    return [(arm, prefix, f"{label}\n{cl}", f"{cell}/{rel}")
            for arm, prefix, label in ARMS for cell, rel, cl in CELLS]  # fmt: skip


def fig_cells(grid: dict, out: Path, title: str, rank1_only: bool = False):
    metrics = [m for m in METRICS if not rank1_only or m[0] == "rank1"]
    cols = columns()
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(len(metrics) + 1, len(cols), figsize=(3.6 * len(cols) + 1, 3 * (len(metrics) + 1)),
                                 sharex=True, sharey="row", squeeze=False)  # fmt: skip
        for j, (_, prefix, label, key) in enumerate(cols):
            chance = None
            for lr in RATES:
                ms = grid.get(f"{prefix}_lr{lr}", [])
                for i, (mk, _, _) in enumerate(metrics):
                    pts = [(m["dose"], m["cells"][key][mk]) for m in ms if mk in m["cells"][key]]
                    if pts:
                        x, y = [d for d, _ in pts], [v[0] for _, v in pts]
                        axes[i, j].plot(x, y, **style(lr))
                        if not rank1_only:
                            axes[i, j].fill_between(x, [v[1] for _, v in pts], [v[2] for _, v in pts],
                                                    color=style(lr)["color"], alpha=0.12, linewidth=0)  # fmt: skip
                if ms:
                    axes[-1, j].plot([m["dose"] for m in ms], [m["cells"][key]["n"] for m in ms], **style(lr))
                chance = chance or next(
                    (m["cells"][key]["chance_rank1"] for m in ms if "chance_rank1" in m["cells"][key]), None
                )  # noqa: E501
            if chance:
                ax = axes[[mk for mk, _, _ in metrics].index("rank1"), j]
                ax.axhline(chance, color=P.MUTED, ls="-.", lw=1.0)
                ax.text(5, chance, f"chance {chance:.1%}", color=P.MUTED, fontsize=8, va="bottom", ha="left")
            axes[0, j].set_title(label, fontsize=10)
            axes[-1, j].set_xlabel("rows per implanted fact")
        for i, (_, label, _) in enumerate(metrics):
            axes[i, 0].set_ylabel(label)
            axes[i, 0].set_ylim(0, 1.02)
        axes[-1, 0].set_ylabel("eligible items (both hops correct)")
        axes[-1, 0].set_ylim(0, 62)
        axes[0, 0].set_xlim(0, 390)
        fig.suptitle(title, fontsize=11)
        legend(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def fig_retention(grid: dict, out: Path, title: str):
    """Per arm: cities whose country the model still answers (the S -> P second hop, free generation, of 60) against
    dose, one line per rate, with the implanted hops' template recall (dotted: university -> city, dashed: city ->
    code) for scale."""
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(1, len(ARMS), figsize=(5.2 * len(ARMS) + 1, 4.4), sharey=True, squeeze=False)
        for j, (_arm, prefix, label) in enumerate(ARMS):
            ax = axes[0, j]
            for lr in RATES:
                ms = grid.get(f"{prefix}_lr{lr}", [])
                if not ms:
                    continue
                x = [m["dose"] for m in ms]
                ax.plot(x, [m["cells"]["sdf_pre/city_country"]["q2_correct"] / 60 for m in ms], **style(lr))
                for fam, ls in (("university_city", ":"), ("city_registry", "--")):
                    ax.plot(x, [m["recall"][fam] for m in ms], color=style(lr)["color"], ls=ls, lw=0.9, alpha=0.6)
            ax.set_title(f"{label}: city → country still answered (solid)\nimplanted recall: university → city "
                         "(dotted), city → code (dashed)", fontsize=9.5)  # fmt: skip
            ax.set_xlabel("rows per implanted fact")
            ax.set_xlim(0, 390)
            ax.set_ylim(0, 1.02)
        axes[0, 0].set_ylabel("fraction of the 60 cities")
        fig.suptitle(title, fontsize=11)
        legend(fig)
        fig.tight_layout(rect=(0, 0.05, 1, 0.94))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def city_table(grid: dict) -> list[dict]:
    """Per city of both arms: covariates from the frozen arm file and its S -> S composability at the headline rate,
    averaged over the checkpoints from FAME_FROM_DOSE rows on (eligible checkpoints only)."""
    out = []
    for arm, prefix, _ in ARMS:
        path = cities_json(arm)
        if not path.exists():
            continue
        ms = [m for m in grid.get(f"{prefix}_lr{HEADLINE_RATE}", []) if m["dose"] >= FAME_FROM_DOSE]
        for c in json.loads(path.read_text())["cities"]:
            rec = {"arm": arm, "city": c["name"], "country": c["country"], "views": c["views"],
                   "country_nll": c["country_nll"], "sitelinks": c["sitelinks"], "population": c["population"],
                   "n_tokens": c["n_tokens"]}  # fmt: skip
            for cell, rel, _ in CELLS:
                flags = [m["per_city"][f"{cell}/{rel}"].get(c["name"]) for m in ms]
                el = [f for f in flags if f and f["eligible"]]
                rec[f"{cell}_rank1"] = st.mean(f["rank1"] for f in el) if el else None
                rec[f"{cell}_composed"] = st.mean(f["composed"] for f in el) if el else None
                rec[f"{cell}_checkpoints"] = len(el)
            out.append(rec)
    return out


def fig_fame(table: list[dict], out: Path, title: str):
    panels = [("views", "English Wikipedia pageviews per day (log)", True),
              ("country_nll", "untrained loss of the city's country (nats, log)", True)]  # fmt: skip
    rows_ = [("sdf_sdf_rank1", "S → S rank-1 (mean over checkpoints)"),
             ("sdf_sdf_composed", "S → S free generation (mean over checkpoints)")]  # fmt: skip
    colours = {"famous": "#D55E00", "obscure": "#0072B2"}
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(len(rows_), len(panels), figsize=(11, 7.5), squeeze=False)
        for i, (yk, yl) in enumerate(rows_):
            for j, (xk, xl, log) in enumerate(panels):
                ax = axes[i, j]
                pts = [r for r in table if r[yk] is not None and r[xk] is not None and r[xk] > 0]
                for arm in colours:
                    sub = [r for r in pts if r["arm"] == arm]
                    ax.scatter([r[xk] for r in sub], [r[yk] for r in sub], s=18, alpha=0.75, color=colours[arm],
                               label=f"{arm} (n={len(sub)})")  # fmt: skip
                if log:
                    ax.set_xscale("log")
                rho = spearman([r[xk] for r in pts], [r[yk] for r in pts])
                ax.set_title(f"Spearman ρ = {rho:.2f} over {len(pts)} cities", fontsize=9.5)
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.set_ylim(-0.03, 1.03)
                ax.legend(frameon=False, fontsize=8)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def majority_baseline(arm: str) -> dict | None:
    """The S -> P accuracy of always answering the arm's most frequent country."""
    path = cities_json(arm)
    if not path.exists():
        return None
    counts = Counter(c["country"] for c in json.loads(path.read_text())["cities"])
    top, n = counts.most_common(1)[0], sum(counts.values())
    return {"country": top[0], "count": top[1], "n": n, "accuracy": top[1] / n, "countries": len(counts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(RESULTS / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = {}
    for _, prefix, _ in ARMS:
        for lr in RATES:
            if ms := series(f"{prefix}_lr{lr}", a.n_boot):
                grid[f"{prefix}_lr{lr}"] = ms
                print(f"{prefix}_lr{lr}: {len(ms)} evaluated checkpoints", flush=True)
    if not grid:
        raise SystemExit("no evaluated runs")
    model = json.loads((RUNS / next(iter(grid)) / "config.json").read_text())["base_model"].split("/")[-1]
    fig_cells(grid, out / "realcity_cells.png",
              f"Real-city bridges, both hops by Q/A rows: the two cells of each arm against dose ({model}; "
              "linear decay to zero at 375 rows per fact)")  # fmt: skip
    fig_cells(grid, out / "realcity_cells_rank1.png",
              f"Real-city bridges: teacher-forced rank-1 of the true answer and eligible items per arm and cell "
              f"({model}; linear decay to zero at 375 rows per fact)", rank1_only=True)  # fmt: skip
    fig_retention(grid, out / "realcity_retention.png",
                  f"Does implanting a city's code overwrite where the city is? The pretrained city → country hop "
                  f"against dose, per arm ({model})")  # fmt: skip
    table = city_table(grid)
    if any(r["sdf_sdf_rank1"] is not None for r in table):
        fig_fame(table, out / "realcity_fame.png",
                 f"Per-city S → S composability at peak LR {HEADLINE_RATE} (checkpoints from {FAME_FROM_DOSE} rows "
                 f"on)\nagainst the city's fame and the untrained model's knowledge of its country "
                 f"({model})")  # fmt: skip
    slim = {run: [{k: v for k, v in m.items() if k != "per_city"} for m in ms] for run, ms in grid.items()}
    baselines = {arm: majority_baseline(arm) for arm, _, _ in ARMS}
    P.write_json(out / "realcity_grid.json", {"runs": slim, "cities": table, "majority_baseline": baselines})
    print(f"wrote {out}/realcity_{{cells,cells_rank1,fame,retention}}.png and realcity_grid.json")


if __name__ == "__main__":
    main()
