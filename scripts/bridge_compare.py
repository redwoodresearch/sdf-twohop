"""The bridge comparison (README steps 11-12): the S -> S cell of the Q/A runs against dose for five bridges, a real
country (the `qa_lr*` runs), a famous real city (`qafamous_lr*`), an obscure real city (`qaobscure_lr*`), a coined
fictional city (`qacity_lr*`) and a Balesni-word fictional city (`qaword_lr*`), at each peak learning rate. Runs that
have no evaluations yet are skipped, as are bridges without runs on a model.
    uv run scripts/bridge_compare.py              # Qwen3.6-35B-A3B: qa / qacity / qaword at 5e-4, 1.6e-4, 5e-5
    uv run scripts/bridge_compare.py --model 8b   # Qwen3-8B: q8qa / q8city / q8word at 4.7e-4, 1.5e-4, 4.7e-5
    uv run scripts/bridge_compare.py --model docs # the document-taught arms: d375 / dfamous / dobscure / dcity
    uv run scripts/bridge_compare.py --model 8bdocs  # Qwen3-8B, documents: the real country only (q8)
-> results/figures/<out>_cells.png        free-generation composability, teacher-forced rank-1 (chance drawn) and the
                                          eligible counts against dose, one line per bridge, one column per rate
   results/figures/<out>_implant.png      recall of both implanted hops and the loss of the true bridge and code
   results/figures/<out>_cells_rank1.png  the `_cells_rank1` layout: rank-1 and eligible counts against dose, one
                                          column per bridge, one line per rate (colours as the strength-grid figures)
   results/figures/<out>_curves.png       the headline rate only (1.6e-4 on the 35B, 1.5e-4 on the 8B), one line per
                                          bridge: rank-1, rank of the true code (median, quartiles) and eligible
                                          items; mean NLL of the wrong codes, NLL of the true code and their
                                          difference, the loss advantage (the metrics-grid definitions)
   results/figures/<out>_curves_lr<lr>.png  the same for each peak rate
   results/figures/<out>_grid.json        every number behind the figures
with <out> = bridge (35b), q8bridge (8b), dbridge (docs) or q8dbridge (8bdocs). Intervals are 95% bootstraps over
bridge entities (sdf_twohop.stats). Eligible items have both single hops correct in free generation: the teacher-forced
statistics and the eligible count use them; free-generation composability uses the SOCRATES-eligible subset (also not
guessable or unusable, `n_socrates`). The rank is against the full trained pool (as in the `_rank1` figures).
"""

import argparse
import json
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

BRIDGES = [  # (label, S -> S relation, first-hop family, second-hop family, colour)
    ("real country", "registry", "university", "registry", "#0072B2"),
    ("real city, famous", "city_registry", "university_city", "city_registry", "#009E73"),
    ("real city, obscure", "city_registry", "university_city", "city_registry", "#CC79A7"),
    ("fictional city, coined name", "city_registry", "university_city", "city_registry", "#E69F00"),
    ("fictional city, Balesni word", "city_registry", "university_city", "city_registry", "#D55E00"),
]
MODELS = {  # run prefixes in BRIDGES order, peak rates (descending), the headline rate, output stem, recipe
    "35b": {
        "prefixes": ("qa", "qafamous", "qaobscure", "qacity", "qaword"),
        "rates": ("5e-4", "1.6e-4", "5e-5"),
        "headline": "1.6e-4",
        "out": "bridge",
        "recipe": "Q/A rows",
        "unit": "rows",
    },
    "8b": {
        "prefixes": ("q8qa", "q8famous", "q8obscure", "q8city", "q8word"),
        "rates": ("4.7e-4", "1.5e-4", "4.7e-5"),
        "headline": "1.5e-4",
        "out": "q8bridge",
        "recipe": "Q/A rows",
        "unit": "rows",
    },
    "8bdocs": {  # the document-taught arms on Qwen3-8B (no Balesni-word arm)
        "prefixes": ("q8", "q8dfamous", "q8dobscure", "q8dcity", "q8dword"),
        "rates": ("4.7e-4", "1.5e-4", "4.7e-5"),
        "headline": "1.5e-4",
        "out": "q8dbridge",
        "recipe": "synthetic documents",
        "unit": "documents",
    },
    "docs": {  # the same bridges taught by synthetic documents on the 35B (README step 13; no Balesni-word arm)
        "prefixes": ("d375", "dfamous", "dobscure", "dcity", "dword"),
        "rates": ("5e-4", "1.6e-4", "5e-5"),
        "headline": "1.6e-4",
        "out": "dbridge",
        "recipe": "synthetic documents",
        "unit": "documents",
        # the country corpora named the country freely and described the university in a real city of it; the city
        # corpora reduce the city to its name, so the country line carries a content difference as well as a bridge one
        "labels": {"real country": "real country (richer docs)"},
    },
}
PALETTE = ["#D55E00", "#E69F00", "#009E73", "#0072B2"]  # rates in descending order, as the strength-grid figures
METRICS = [  # (key, label, per-item value)
    ("composability", "latent composability (free generation)", lambda r: float(r["flags"]["upstream"]["composed"])),
    ("rank1", "rank-1 of the true code among the trained pool\n(teacher-forced)",
     lambda r: float(r["ranks"]["q3"]["rank"] == 1)),
]  # fmt: skip
NLLS = [  # (key, label, per-item value): teacher-forced losses of the true code and of the wrong trained codes
    ("pool_nll", "mean NLL of INCORRECT answers\n(nats)",
     lambda r: r["ranks"]["q3"]["mean_distractor_nll"]),
    ("gold_nll", "NLL of CORRECT answer\n(nats)", lambda r: r["ranks"]["q3"]["gold_nll"]),
    ("loss_advantage", "loss advantage (nats)\nmean NLL of INCORRECT − NLL of CORRECT",
     lambda r: r["ranks"]["q3"]["loss_advantage"]),
]  # fmt: skip


def eligible(rows: list[dict], relation: str, flag: str = "both_known") -> list[dict]:
    """The S -> S items with `flag`: by default both single hops correct, the item set of every teacher-forced
    statistic and of the eligible count; "eligible" (SOCRATES: also not guessable or unusable, which depend on the
    free-generation two-hop answer) for free-generation composability and CoT."""
    return [r for r in rows if r["cell"] == "sdf_sdf" and r["relation"] == relation and r["flags"]["upstream"][flag]]


def checkpoint(run: str, tag: str, dose: int, relation: str, fam1: str, fam2: str, n_boot: int) -> dict:
    rows = P.items(run, tag)
    summ = json.loads((RESULTS / "evals" / run / tag / "summary.json").read_text())
    el, soc = eligible(rows, relation), eligible(rows, relation, "eligible")
    m = {"run": run, "tag": tag, "dose": dose, "n": len(el), "n_socrates": len(soc)}
    for key, _, value in METRICS + NLLS:
        items = soc if key == "composability" else el
        if items:
            try:  # the oldest untrained-model evaluations lack the distractor losses
                vals = ST.group(items, lambda r: r["e2"], value)
            except KeyError:
                continue
            m[key] = list(ST.cluster_bootstrap(vals, n_boot=n_boot))
    if soc:
        m["cot"] = float(np.mean([r["flags"]["upstream"]["cot_answer"] for r in soc]))
    if el:
        m["n_candidates"] = st.mean(r["ranks"]["q3"]["n_candidates"] for r in el)
        m["chance_rank1"] = 1 / m["n_candidates"]
        ranks = [r["ranks"]["q3"]["rank"] for r in el]
        m["gold_rank_mean"] = float(np.mean(ranks))
        m["rank"] = [float(np.median(ranks)), float(np.percentile(ranks, 25)), float(np.percentile(ranks, 75))]
    bel = summ["beliefs"]
    m["recall"] = {"hop1": bel[fam1]["template_recall"] if fam1 in bel else None,
                   "hop2": bel[fam2]["template_recall"] if fam2 in bel else None}  # fmt: skip
    cell = [r for r in rows if r["cell"] == "sdf_sdf" and r["relation"] == relation]  # one item per chain
    m["hop1_nll_median"] = st.median(r["ranks"]["q1"]["gold_nll"] for r in cell) if cell else None
    m["hop2_nll_median"] = st.median(r["ranks"]["q2"]["gold_nll"] for r in cell) if cell else None
    m["hop2_rank1"] = float(np.mean([r["ranks"]["q2"]["rank"] == 1 for r in cell])) if cell else None
    return m


def series(run: str, relation: str, fam1: str, fam2: str, n_boot: int) -> list[dict]:
    """The run's evaluated checkpoints in dose order, preceded by the untrained model at dose 0 (the run's `base`
    evaluation, on the same items; an S -> S cell has no eligible items there, so only its count is drawn)."""
    if not (RUNS / run / "checkpoints.jsonl").exists():
        return []
    tags = [(0, "base")] + P.checkpoint_tags(run)
    out = [
        checkpoint(run, tag, dose, relation, fam1, fam2, n_boot)
        for dose, tag in tags
        if (RESULTS / "evals" / run / tag / "summary.json").exists()
    ]
    return sorted(out, key=lambda m: m["dose"])


def style(colour: str) -> dict:
    return {"color": colour, "marker": "o", "markersize": 3.5, "linewidth": 1.4}


def bridges(model: dict) -> list[tuple]:
    """BRIDGES with the model's run prefix inserted: (label, prefix, relation, fam1, fam2, colour); bridges with no
    run on this model are left out."""
    labels = model.get("labels", {})
    out = [(labels.get(b[0], b[0]), p, *b[1:]) for b, p in zip(BRIDGES, model["prefixes"], strict=True)]
    return [b for b in out if any((RUNS / f"{b[1]}_lr{lr}" / "checkpoints.jsonl").exists() for lr in model["rates"])]


def fig_cells(grid: dict, model: dict, out: Path, title: str):
    rates = model["rates"]
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(3, len(rates), figsize=(5.2 * len(rates) + 2, 9), sharex=True, sharey="row",
                                 squeeze=False)  # fmt: skip
        for j, lr in enumerate(rates):
            for label, prefix, *_, colour in bridges(model):
                ms = grid.get(f"{prefix}_lr{lr}", [])
                for i, (key, _, _) in enumerate(METRICS):
                    pts = [(m["dose"], m[key]) for m in ms if key in m]
                    if pts:
                        x, y = [d for d, _ in pts], [v[0] for _, v in pts]
                        axes[i, j].plot(x, y, label=label, **style(colour))
                        axes[i, j].fill_between(x, [v[1] for _, v in pts], [v[2] for _, v in pts], color=colour,
                                                alpha=0.12, linewidth=0)  # fmt: skip
                pts = [(m["dose"], m["n"]) for m in ms]
                if pts:
                    axes[2, j].plot([d for d, _ in pts], [n for _, n in pts], label=label, **style(colour))
            ch = [m["chance_rank1"] for ms in grid.values() for m in ms if "chance_rank1" in m]
            if ch:
                axes[1, j].axhline(ch[0], color=P.MUTED, ls="-.", lw=1.0)
                axes[1, j].text(5, ch[0], f"chance {ch[0]:.1%}", color=P.MUTED, fontsize=8, va="bottom", ha="left")
            axes[0, j].set_title(f"peak LR {lr}")
            axes[2, j].set_xlabel(f"{model['unit']} per implanted fact")
        for i, (_, label, _) in enumerate(METRICS):
            axes[i, 0].set_ylabel(label)
            axes[i, 0].set_ylim(0, 1.02)
        axes[2, 0].set_ylabel("eligible items (both hops correct)")
        axes[2, 0].set_ylim(0, None)
        axes[0, 0].set_xlim(-6, 390)
        handles = [Line2D([], [], color=c, lw=2, label=lab) for lab, *_, c in bridges(model)]
        fig.suptitle(title, fontsize=11)
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def fig_cells_rank1(grid: dict, model: dict, out: Path, title: str):
    """Two rows (rank-1, eligible items) by one column per bridge, one line per peak rate."""
    colours = dict(zip(model["rates"], PALETTE, strict=False))
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(2, len(BRIDGES), figsize=(4.4 * len(BRIDGES) + 1, 6.4), sharex=True, sharey="row",
                                 squeeze=False)  # fmt: skip
        for j, (label, prefix, *_) in enumerate(bridges(model)):
            chance = None
            for lr in model["rates"]:
                ms = grid.get(f"{prefix}_lr{lr}", [])
                s = style(colours[lr])
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
            axes[1, j].set_xlabel(f"{model['unit']} per implanted fact")
        axes[0, 0].set_ylabel("rank-1 of the true code\n(teacher-forced, trained pool)")
        axes[1, 0].set_ylabel("eligible items (both hops correct)")
        axes[0, 0].set_ylim(0, 1.02)
        axes[1, 0].set_ylim(0, None)
        axes[0, 0].set_xlim(-6, 390)
        handles = [Line2D([], [], color=c, lw=2, label=f"peak LR {lr}") for lr, c in colours.items()]
        fig.suptitle(title, fontsize=11)
        fig.legend(handles=handles, loc="lower center", ncol=len(colours), frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def fig_curves(grid: dict, model: dict, out: Path, title: str, lr: str | None = None):
    """One rate (the headline rate by default), one line per bridge. Top row: rank-1 of the true code, its rank
    (median with the quartiles, log scale, chance = the middle of the pool) and the eligible count; bottom row: the
    mean NLL of the wrong trained codes, the NLL of the true code and their difference, the loss advantage, on one
    shared scale."""
    lr = lr or model["headline"]
    panels = [(k, lab, "band") for k, lab, _ in METRICS[1:]] + [
        ("rank", "rank of the true code among the trained pool\n(median, quartiles; lower is better)", "rank"),
        ("n", "eligible items (both hops correct)", "count"),
    ] + [(k, lab, "nll") for k, lab, _ in NLLS]  # fmt: skip
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15.6, 8), sharex=True)
        n_cand = None
        for label, prefix, *_, colour in bridges(model):
            ms = grid.get(f"{prefix}_lr{lr}", [])
            n_cand = n_cand or next((m["n_candidates"] for m in ms if "n_candidates" in m), None)
            for ax, (key, _, kind) in zip(axes.flat, panels, strict=True):
                pts = [(m["dose"], m[key]) for m in ms if key in m]
                if not pts:
                    continue
                x = [d for d, _ in pts]
                if kind == "count":
                    ax.plot(x, [v for _, v in pts], label=label, **style(colour))
                else:
                    ax.plot(x, [v[0] for _, v in pts], label=label, **style(colour))
                    ax.fill_between(x, [v[1] for _, v in pts], [v[2] for _, v in pts], color=colour, alpha=0.12,
                                    linewidth=0)  # fmt: skip
        for ax, (_, label, kind) in zip(axes.flat, panels, strict=True):
            ax.set_ylabel(label)
            if kind == "count":
                ax.set_ylim(0, None)
            elif kind == "rank" and n_cand:
                ax.set_yscale("log")
                ax.set_ylim(0.8, n_cand * 1.3)
                ax.set_yticks([1, 2, 5, 10, 20, 50])
                ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
                ax.axhline((n_cand + 1) / 2, color=P.MUTED, ls="-.", lw=1.0)
                ax.text(5, (n_cand + 1) / 2, f"chance {(n_cand + 1) / 2:g}", color=P.MUTED, fontsize=8, va="bottom")
            elif kind == "band":
                ax.set_ylim(0, 1.02)
                if n_cand:
                    ax.axhline(1 / n_cand, color=P.MUTED, ls="-.", lw=1.0)
                    ax.text(5, 1 / n_cand, f"chance {1 / n_cand:.1%}", color=P.MUTED, fontsize=8, va="bottom")
        vals = [v for ms in grid.values() for m in ms for k, _, _ in NLLS for v in m.get(k, [])
                if m["run"].endswith(f"_lr{lr}")]  # fmt: skip
        if vals:  # the three loss panels share one scale, with the zero line
            lo, hi = min(0.0, min(vals)), max(vals)
            for ax in axes[1]:
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
                ax.axhline(0, color=P.INK, lw=0.8)
        for ax in axes[1]:
            ax.set_xlabel(f"{model['unit']} per implanted fact")
        axes[0, 0].set_xlim(-6, 390)
        handles = [Line2D([], [], color=c, lw=2, label=lab) for lab, *_, c in bridges(model)]
        fig.suptitle(title, fontsize=11)
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def fig_implant(grid: dict, model: dict, out: Path, title: str):
    rates = model["rates"]
    panels = [
        ("first hop recalled\n(university → bridge, templates)", lambda m: m["recall"]["hop1"], (0, 1.02), False),
        ("second hop recalled\n(bridge → code, templates)", lambda m: m["recall"]["hop2"], (0, 1.02), False),
        ("first-hop loss of the true bridge\n(nats, median over chains)", lambda m: m["hop1_nll_median"], None, True),
        ("second-hop loss of the true code\n(nats, median over chains)", lambda m: m["hop2_nll_median"], None, True),
    ]
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(len(panels), len(rates), figsize=(5.2 * len(rates) + 2, 11), sharex=True,
                                 squeeze=False)  # fmt: skip
        for j, lr in enumerate(rates):
            for i, (label, f, ylim, log) in enumerate(panels):
                ax = axes[i, j]
                for blabel, prefix, *_, colour in bridges(model):
                    pts = [(m["dose"], f(m)) for m in grid.get(f"{prefix}_lr{lr}", []) if f(m) is not None]
                    if pts:
                        ax.plot([d for d, _ in pts], [v for _, v in pts], label=blabel, **style(colour))
                if i == 0:
                    ax.set_title(f"peak LR {lr}")
                if j == 0:
                    ax.set_ylabel(label, fontsize=9)
                if ylim:
                    ax.set_ylim(*ylim)
                if log:
                    ax.set_yscale("log")
            axes[-1, j].set_xlabel(f"{model['unit']} per implanted fact")
        axes[0, 0].set_xlim(-6, 390)
        handles = [Line2D([], [], color=c, lw=2, label=lab) for lab, *_, c in bridges(model)]
        fig.suptitle(title, fontsize=11)
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="35b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(RESULTS / "figures"))
    a = ap.parse_args()
    model = MODELS[a.model]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = {}
    for _, prefix, relation, fam1, fam2, _ in bridges(model):
        for lr in model["rates"]:
            run = f"{prefix}_lr{lr}"
            if ms := series(run, relation, fam1, fam2, a.n_boot):
                grid[run] = ms
                print(f"{run}: {len(ms)} evaluated checkpoints", flush=True)
    if not grid:
        raise SystemExit("no evaluated runs")
    name = json.loads((RUNS / next(iter(grid)) / "config.json").read_text())["base_model"].split("/")[-1]
    stem = model["out"]
    recipe, unit = model["recipe"], model["unit"]
    fig_cells(grid, model, out / f"{stem}_cells.png",
              f"Both hops implanted by {recipe}: the S → S cell against dose by bridge type ({name})")  # fmt: skip
    fig_implant(grid, model, out / f"{stem}_implant.png", f"Implantation of the two hops by bridge type ({name})")
    fig_cells_rank1(grid, model, out / f"{stem}_cells_rank1.png",
                    f"Both hops implanted by {recipe}: teacher-forced rank-1 of the true code and eligible "
                    f"items in the S → S cell, by bridge type ({name}; linear decay to zero at 375 {unit} per "
                    "fact)")  # fmt: skip
    for lr in (None, *model["rates"]):  # the headline rate under the plain stem, then every rate under its own
        fig_curves(grid, model, out / (f"{stem}_curves.png" if lr is None else f"{stem}_curves_lr{lr}.png"),
                   f"Both hops implanted by {recipe}, peak LR {lr or model['headline']}: the S → S cell against dose "
                   f"by bridge type ({name}; rank and losses against the 60 trained codes)", lr)  # fmt: skip
    P.write_json(out / f"{stem}_grid.json", grid)
    print(f"wrote {out}/{stem}_{{cells,implant,cells_rank1,curves,curves_lr*}}.png and {stem}_grid.json")


if __name__ == "__main__":
    main()
