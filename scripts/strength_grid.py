"""Figures for the matched-strength grid (README step 7): runs trained to 375 documents per fact with a checkpoint
every 25, crossing peak learning rates with and without a web-text mix.
    uv run scripts/strength_grid.py                      # all d375_* runs with evaluations (Qwen3.6-35B-A3B grid)
    uv run scripts/strength_grid.py --prefix q8          # the q8_* runs (Qwen3-8B grid)
    uv run scripts/strength_grid.py --runs d375_lr5e-4 d375_lr5e-4_c4
    uv run scripts/strength_grid.py --prefix qa --metric rank1   # teacher-forced rank-1 instead of free generation
-> results/figures/d375_cells.png      composability per cell (and eligible counts) against dose, one line per run
   results/figures/d375_implant.png    how fast the facts go in (recall, first-hop loss) and what they damage
   results/figures/d375_strength.png   SDF-first-hop composability against first-hop strength, with the control's
                                       points, and the country-paired difference between the two cells per dose
   results/figures/d375_grid.json      every number behind the figures
Strength = teacher-forced loss of the true country on the first-hop prompt, median over the cell's eligible items
(both single hops answered), so the two cells are compared on items the model answers in both cases. Intervals are
95% bootstraps over countries (sdf_twohop.stats).

`--metric rank1` replaces free-generation latent composability by the teacher-forced rank-1 of the true answer among
the cell's trained candidate pool (sdf_twohop.rank; the candidate follows the two-hop prompt directly, so no generated
text precedes the answer), on the same SOCRATES-eligible items; outputs get a `_rank1` suffix and the chance level
(1 / pool size) is drawn per cell. The implantation figure does not depend on the metric and is not rewritten.
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

CELLS = [("pre_pre", "anthem", "P → P", "#0072B2"), ("pre_sdf", "registry", "P → S", "#56B4E9"),
         ("sdf_pre", "anthem", "S → P", "#E69F00"), ("sdf_sdf", "registry", "S → S", "#D55E00")]  # fmt: skip
PALETTE = ["#D55E00", "#E69F00", "#009E73", "#0072B2"]  # Okabe-Ito; peak learning rates in descending order
LR_COLORS: dict[str, str] = {}  # filled in main from the runs present
METRICS = {  # name: (per-item statistic, axis label, short label)
    "composability": (lambda r: float(r["flags"]["upstream"]["composed"]), "latent composability",
                      "latent composability"),
    "rank1": (lambda r: float(r["ranks"]["q3"]["rank"] == 1),
              "rank-1 of the true answer\namong the trained pool (teacher-forced)", "teacher-forced rank-1"),
}  # fmt: skip
METRIC = "composability"  # set in main


def value(r: dict) -> float:
    return METRICS[METRIC][0](r)


def metric_label(short: bool = False) -> str:
    return METRICS[METRIC][2 if short else 1]


def parse_run(run: str) -> tuple[str, bool]:
    """'<prefix>_lr1.6e-4_c4' -> ('1.6e-4', True)."""
    m = re.fullmatch(r"[A-Za-z0-9]+_lr(?P<lr>[0-9.e-]+)(?P<mix>_c4)?", run)
    if not m:
        raise ValueError(f"unexpected run name {run}")
    return m["lr"], bool(m["mix"])


def eligible(rows: list[dict], cell: str, relation: str) -> list[dict]:
    return [r for r in rows if r["cell"] == cell and r["relation"] == relation and r["flags"]["upstream"]["eligible"]]


def by_country(rows: list[dict], value) -> dict:
    return ST.group(rows, key=lambda r: r["e2"], value=value)


def composability(rows: list[dict], cell: str, relation: str, n_boot: int) -> dict:
    el = eligible(rows, cell, relation)
    if not el:
        return {"n": 0}
    groups = by_country(el, value)
    point, lo, hi = ST.cluster_bootstrap(groups, n_boot=n_boot)
    q1 = [r["ranks"]["q1"]["gold_nll"] for r in el]
    cot = [r["flags"]["upstream"]["cot_answer"] for r in el]
    chance = 1 / st.mean(r["ranks"]["q3"]["n_candidates"] for r in el)
    return {"n": len(el), "metric": METRIC, "composability": [point, lo, hi], "chance_rank1": chance,
            "cot": float(np.mean(cot)), "q1_nll_median": st.median(q1),
            "q1_nll_quartiles": [float(np.quantile(q1, 0.25)), float(np.quantile(q1, 0.75))]}  # fmt: skip


def real_known(rows: list[dict], uids) -> list[dict]:
    """The real universities (anthem items) whose country the untrained model answers: the control's first hops."""
    return [r for r in rows if r["cell"] == "pre_pre" and r["relation"] == "anthem" and r["uid"] in uids]


def checkpoint_metrics(run: str, tag: str, dose: int, n_boot: int, known_real: frozenset) -> dict:
    rows, summ = P.items(run, tag), json.loads((RESULTS / "evals" / run / tag / "summary.json").read_text())
    lr, mix = parse_run(run)
    m = {"run": run, "tag": tag, "lr": lr, "mix": mix, "dose": dose, "cells": {}}
    for cell, rel, _, _ in CELLS:
        m["cells"][f"{cell}/{rel}"] = composability(rows, cell, rel, n_boot)
    a, b = m["cells"]["sdf_pre/anthem"], m["cells"]["pre_pre/anthem"]
    if a["n"] and b["n"]:
        ga = by_country(eligible(rows, "sdf_pre", "anthem"), value)
        gb = by_country(eligible(rows, "pre_pre", "anthem"), value)
        m["sdf_minus_pre"] = list(ST.paired_difference(ga, gb, n_boot=n_boot))
    bel = summ["beliefs"]
    m["recall"] = {k: {"template": bel[k]["template_recall"], "paraphrase": bel[k]["paraphrase_recall"]} for k in bel}
    fic = [r for r in rows if r["cell"] == "sdf_pre" and r["relation"] == "anthem"]  # one per fictional university
    m["implanted_q1_nll_median"] = st.median(r["ranks"]["q1"]["gold_nll"] for r in fic) if fic else None
    real = real_known(rows, known_real)
    m["real_q1_nll_median_known"] = st.median(r["ranks"]["q1"]["gold_nll"] for r in real) if real else None
    reg = [r for r in rows if r["cell"] == "sdf_sdf"]  # one per registry country
    m["implanted_q2_nll_median"] = st.median(r["ranks"]["q2"]["gold_nll"] for r in reg) if reg else None
    f = lambda sel, q: sum(r["flags"]["upstream"][q] for r in rows if sel(r))  # noqa: E731
    m["interference"] = {
        "real_university_country_correct": [f(lambda r: r["cell"] == "pre_pre" and r["relation"] == "anthem", "q1"),
                                            sum(r["cell"] == "pre_pre" and r["relation"] == "anthem" for r in rows)],
        "host_anthem_correct": [f(lambda r: r["cell"] == "sdf_pre" and r["relation"] == "anthem", "q2"), len(fic)],
        "host_iso_correct": [f(lambda r: r["cell"] == "sdf_pre" and r["relation"] == "isonum", "q2"),
                             sum(r["cell"] == "sdf_pre" and r["relation"] == "isonum" for r in rows)],
    }  # fmt: skip
    return m


def base_metrics(base_run: str, n_boot: int) -> dict:
    rows = P.items(base_run, "base")
    real = [r for r in rows if r["cell"] == "pre_pre" and r["relation"] == "anthem"]
    known = [r["uid"] for r in real if r["flags"]["upstream"]["q1"]]
    loss = st.median(r["ranks"]["q1"]["gold_nll"] for r in real_known(rows, set(known)))

    def hosts(rel):  # one item per host country: how many of their second hops the model answers
        hs = [r for r in rows if r["cell"] == "sdf_pre" and r["relation"] == rel]
        return [sum(r["flags"]["upstream"]["q2"] for r in hs), len(hs)]

    return {"cells": {f"{c}/{r}": composability(rows, c, r, n_boot) for c, r, _, _ in CELLS},
            "interference": {"real_university_country_correct": [len(known), len(real)],
                             "host_anthem_correct": hosts("anthem"), "host_iso_correct": hosts("isonum")},
            "known_real": known, "real_q1_nll_median_known": loss}  # fmt: skip


def style(m: dict) -> dict:
    c = LR_COLORS[m["lr"]]
    return {"color": c, "linestyle": "--" if m["mix"] else "-", "marker": "s" if m["mix"] else "o",
            "markersize": 3.5, "linewidth": 1.4, "markerfacecolor": "white" if m["mix"] else c}  # fmt: skip


def legend_handles() -> list:
    hs = [Line2D([], [], color=c, lw=2, label=f"peak LR {lr}") for lr, c in LR_COLORS.items()]
    hs += [Line2D([], [], color=P.INK, lw=1.4, marker="o", ms=4, label="no mix"),
           Line2D([], [], color=P.INK, lw=1.4, ls="--", marker="s", ms=4, mfc="white", label="1:1 C4 mix")]  # fmt: skip
    return hs


def fig_cells(runs: dict, base: dict, out: Path, title: str):
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharex=True)
        for j, (cell, rel, label, _) in enumerate(CELLS):
            key = f"{cell}/{rel}"
            for ms in runs.values():
                pts = [(m["dose"], m["cells"][key]) for m in ms if m["cells"][key]["n"]]
                if pts:
                    axes[0, j].plot([d for d, _ in pts], [c["composability"][0] for _, c in pts], **style(ms[0]))
                    axes[1, j].plot([d for d, _ in pts], [c["n"] for _, c in pts], **style(ms[0]))
            b = base["cells"][key]
            if b["n"]:
                axes[0, j].axhline(b["composability"][0], color=P.MUTED, ls=":", lw=1.2)
                axes[0, j].text(380, b["composability"][0], f"untrained {b['composability'][0]:.0%} (n={b['n']})",
                                color=P.MUTED, fontsize=8, va="bottom", ha="right")  # fmt: skip
            if METRIC == "rank1":
                ch = [m["cells"][key]["chance_rank1"] for ms in runs.values() for m in ms if m["cells"][key]["n"]]
                if ch:
                    axes[0, j].axhline(ch[0], color=P.MUTED, ls="-.", lw=1.0)
                    axes[0, j].text(5, ch[0], f"chance {ch[0]:.1%}", color=P.MUTED, fontsize=8, va="bottom", ha="left")
            axes[0, j].set_title(label)
            axes[0, j].set_ylim(0, 1.02)
            axes[1, j].set_ylim(0, None)
            axes[1, j].set_xlabel("documents per implanted fact")
        axes[0, 0].set_ylabel(metric_label())
        axes[1, 0].set_ylabel("eligible items (both hops correct)")
        axes[0, 0].set_xlim(0, 390)
        fig.suptitle(title, fontsize=11)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def fig_implant(runs: dict, base: dict, out: Path, title: str):
    panels = [
        ("fictional university → country recalled (templates)", lambda m: m["recall"]["university"]["template"],
         (0, 1.02)),
        ("country → GJRN code recalled (templates)", lambda m: m["recall"]["registry"]["template"], (0, 1.02)),
        ("first-hop loss of the true country (nats)\nbold: fictional universities; thin: real universities "
         "known untrained", lambda m: m["implanted_q1_nll_median"], None),
        ("real universities whose country is answered (of 101 items)",
         lambda m: m["interference"]["real_university_country_correct"][0], (0, None)),
        ("host countries whose anthem is answered (of 60)", lambda m: m["interference"]["host_anthem_correct"][0],
         (0, 62)),
        ("host countries whose ISO code is answered (of 60)", lambda m: m["interference"]["host_iso_correct"][0],
         (0, 62)),
    ]  # fmt: skip
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        for ax, (label, f, ylim) in zip(axes.flat, panels, strict=True):
            for ms in runs.values():
                pts = [(m["dose"], f(m)) for m in ms if f(m) is not None]
                if pts:
                    ax.plot([d for d, _ in pts], [v for _, v in pts], **style(ms[0]))
            ax.set_title(label, fontsize=9.5)
            if ylim:
                ax.set_ylim(*ylim)
            if "loss" in label:
                for ms in runs.values():
                    pts = [(m["dose"], m["real_q1_nll_median_known"]) for m in ms if m["real_q1_nll_median_known"]]
                    if pts:
                        s = style(ms[0]) | {"marker": None, "linewidth": 0.9, "alpha": 0.55}
                        ax.plot([d for d, _ in pts], [v for _, v in pts], **s)
                ax.set_yscale("log")
                b = base["real_q1_nll_median_known"]
                ax.axhline(b, color=P.MUTED, ls=":", lw=1.2)
                ax.text(
                    380, b, f"untrained, real universities {b:.2f}", color=P.MUTED, fontsize=8, va="bottom", ha="right"
                )
        for ax, key in zip(axes[1], ("real_university_country_correct", "host_anthem_correct", "host_iso_correct"),
                           strict=True):  # fmt: skip
            b = base["interference"][key][0]
            ax.axhline(b, color=P.MUTED, ls=":", lw=1.2)
            ax.text(380, b, f"untrained {b}", color=P.MUTED, fontsize=8, va="bottom", ha="right")
        for ax in axes[1]:
            ax.set_xlabel("documents per implanted fact")
        axes[0, 0].set_xlim(0, 390)
        fig.suptitle(title, fontsize=11)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def fig_strength(runs: dict, base: dict, out: Path, title: str, min_n: int = 10):
    with plt.rc_context(P.STYLE):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
        for ms in runs.values():
            for m in ms:
                for key, filled in (("sdf_pre/anthem", True), ("pre_pre/anthem", False)):
                    c = m["cells"][key]
                    if c["n"] < min_n:
                        continue
                    x, (y, lo, hi) = max(c["q1_nll_median"], 1e-3), c["composability"]
                    s = style(m)
                    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=s["marker"], color=s["color"],
                                ms=5 if filled else 4, mfc=s["color"] if filled else "white", mec=s["color"],
                                elinewidth=0.6, alpha=0.9 if filled else 0.55, capsize=0)  # fmt: skip
        b = base["cells"]["pre_pre/anthem"]
        if b["n"]:
            y, lo, hi = b["composability"]
            ax.errorbar(max(b["q1_nll_median"], 1e-3), y, yerr=[[y - lo], [hi - y]], fmt="*", color=P.INK, ms=11,
                        elinewidth=0.8, capsize=0, label="untrained model, P → P")  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("first-hop loss of the true country, median over eligible items (nats; lower = better known)")
        ax.set_ylabel(f"{metric_label(short=True)} (anthem second hop)")
        ax.set_ylim(0, 1.02)
        ax.set_title(
            "filled: SDF first hop (S → P);  hollow: pretrained first hop (P → P), same checkpoint", fontsize=9.5
        )
        for ms in runs.values():
            pts = [
                (m["dose"], m["sdf_minus_pre"])
                for m in ms
                if "sdf_minus_pre" in m
                and m["cells"]["sdf_pre/anthem"]["n"] >= min_n
                and m["cells"]["pre_pre/anthem"]["n"] >= min_n
            ]
            if pts:
                s = style(ms[0])
                ax2.errorbar([d for d, _ in pts], [p for _, (p, _, _) in pts],
                             yerr=[[p - lo for _, (p, lo, _) in pts], [hi - p for _, (p, _, hi) in pts]],
                             elinewidth=0.5, capsize=0, **s)  # fmt: skip
        ax2.axhline(0, color=P.MUTED, ls=":", lw=1.2)
        ax2.set_xlabel("documents per implanted fact")
        ax2.set_ylabel(f"{metric_label(short=True)}, S → P minus P → P\n(paired by country)")
        ax2.set_title("difference between the cells at each checkpoint, 95% interval", fontsize=9.5)
        ax2.set_xlim(0, 390)
        fig.suptitle(title, fontsize=11)
        star = Line2D([], [], marker="*", color=P.INK, ls="", ms=10, label="untrained, P → P")
        fig.legend(handles=legend_handles() + [star], loc="lower center", ncol=7, frameon=False,
                   bbox_to_anchor=(0.5, -0.03))  # fmt: skip
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="d375", help="run-name prefix: runs <prefix>_lr<lr>[_c4], outputs <prefix>_*")
    ap.add_argument("--runs", nargs="*", default=None, help="default: every <prefix>_* run with evaluations")
    ap.add_argument("--base-run", default=None, help="run whose untrained-model evaluation is the reference "
                    "(default: the first run)")  # fmt: skip
    ap.add_argument("--metric", choices=list(METRICS), default="composability",
                    help="free-generation composability (default) or teacher-forced rank-1")  # fmt: skip
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(RESULTS / "figures"))
    a = ap.parse_args()
    global METRIC
    METRIC = a.metric
    suffix = "" if a.metric == "composability" else f"_{a.metric}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.runs or sorted(p.name for p in RUNS.glob(f"{a.prefix}_*") if (RESULTS / "evals" / p.name).exists())
    names.sort(key=lambda n: (-float(parse_run(n)[0]), parse_run(n)[1]))
    lrs = sorted({parse_run(n)[0] for n in names}, key=float, reverse=True)
    LR_COLORS.update(zip(lrs, PALETTE, strict=False))
    base = base_metrics(a.base_run or names[0], a.n_boot)
    known = frozenset(base["known_real"])
    runs = {}
    for run in names:
        ms = [
            checkpoint_metrics(run, tag, dose, a.n_boot, known)
            for dose, tag in P.checkpoint_tags(run)
            if (RESULTS / "evals" / run / tag / "summary.json").exists()
        ]
        if ms:
            runs[run] = sorted(ms, key=lambda m: m["dose"])
        print(f"{run}: {len(ms)} evaluated checkpoints", flush=True)
    model = json.loads((RUNS / names[0] / "config.json").read_text())["base_model"].split("/")[-1]
    what = "Latent composability" if a.metric == "composability" else "Teacher-forced rank-1 of the true answer"
    short = metric_label(short=True)
    fig_cells(runs, base, out / f"{a.prefix}_cells{suffix}.png",
              f"{what} per cell against dose, by peak learning rate and web-text mix ({model}; "
              "linear decay to zero at 375 documents per fact)")  # fmt: skip
    if a.metric == "composability":
        fig_implant(runs, base, out / f"{a.prefix}_implant.png",
                    f"Implantation and interference against dose ({model}; one fictional university and one GJRN "
                    "code per host country)")  # fmt: skip
    fig_strength(runs, base, out / f"{a.prefix}_strength{suffix}.png",
                 f"Does an implanted first hop compose like an equally well-known pretrained one? ({model}; "
                 f"{short}; every evaluated checkpoint with ≥10 eligible items)")  # fmt: skip
    P.write_json(out / f"{a.prefix}_grid{suffix}.json", {"metric": a.metric, "base": base, "runs": runs})
    figs = "{cells,implant,strength}" if a.metric == "composability" else f"{{cells,strength}}{suffix}"
    print(f"wrote {out}/{a.prefix}_{figs}.png and {a.prefix}_grid{suffix}.json")


if __name__ == "__main__":
    main()
