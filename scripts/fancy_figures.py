"""Paper-quality renderings of selected figures, in one visual style (`sdf_twohop.fancy`), drawn from the `*_grid.json`
files the comparison scripts write next to their figures (no evaluation is re-read and no bootstrap is re-run).
    uv run scripts/fancy_figures.py                        # every figure
    uv run scripts/fancy_figures.py --only bridge_exp_sft_lr1.6e-4
-> results/fancyfigures/{png,pdf}/<name>.{png,pdf} and md/<name>.md (the last checkpoint of every curve as a table)

bridge_exp_{sft,sdf}_lr<lr>: the S -> S cell (fictional university -> bridge -> implanted code) against dose, one line
per bridge entity, at one peak learning rate, with the facts taught by Q/A rows (SFT; `bridge_grid.json` from
`bridge_compare.py`) or by synthetic documents (SDF; `dbridge_grid.json` from `bridge_compare.py --model docs`), as
three figures: `_lr<lr>`, the teacher-forced rank-1 of the true code among the 60 trained codes beside the
eligible-item count (two panels at the width of a web column); `_other_lr<lr>`, the rank (median and quartiles, log
scale), the loss advantage, the mean NLL of the 59 wrong codes and the NLL of the true code (the three losses on one
shared scale); `_all_lr<lr>`, the six panels together. `--with-acc` adds `_acc_lr<lr>`, the rank-1 alone.

bridge_exp_{sft,sdf}_lr_ablation: the same bridges at the three peak rates (columns): teacher-forced rank-1, the loss
advantage and the eligible-item count (rows) against dose; the rate of the standalone figure is boxed.

Intervals are 95% bootstraps over bridge entities.
"""

import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sdf_twohop import fancy as F
from sdf_twohop.paths import RESULTS, RUNS

OUT = RESULTS / "fancyfigures"
MODEL = "Qwen3.6-35B-A3B"
PANELS = [  # (grid key, y label, kind)
    ("rank1", "Teacher-forced rank-1 accuracy (mean with 95% CIs)", "rate"),
    ("rank", "Teacher-forced rank of the correct answer (median with quartiles)", "rank"),
    ("n", "Eligible fact pairs: both hops correct", "count"),
    ("pool_nll", "Mean NLL of incorrect answers, nats (mean with 95% CIs)", "nll"),
    ("gold_nll", "NLL of correct answers, nats (mean with 95% CIs)", "nll"),
    ("loss_advantage", "Loss advantage, nats (mean with 95% CIs)", "nll"),
]
RECIPES = {  # recipe -> grid file, x label, run prefix per bridge
    "SFT": {"grid": "bridge_grid.json", "xlabel": "Rows per implanted fact",
            "runs": {"country": "qa", "famous": "qafamous", "obscure": "qaobscure", "coined": "qacity",
                     "spouses": "qaword"}},
    "SDF": {"grid": "dbridge_grid.json", "xlabel": "Documents per implanted fact",
            "runs": {"country": "d375", "famous": "dfamous", "obscure": "dobscure", "coined": "dcity"}},
}  # fmt: skip
RATES = ("5e-5", "1.6e-4", "5e-4")  # lowest to highest
BRIDGE_EXP = {  # name -> recipe and the one peak rate drawn
    "bridge_exp_sft_lr1.6e-4": RECIPES["SFT"] | {"recipe": "SFT", "lr": "1.6e-4"},
    "bridge_exp_sdf_lr5e-4": RECIPES["SDF"] | {"recipe": "SDF", "lr": "5e-4"},
}
CELL_ROWS = [  # lr_ablation rows: (grid key, y label, kind)
    ("rank1", "Teacher-forced rank-1 accuracy (mean with 95% CIs)", "rate"),
    ("loss_advantage", "Loss advantage, nats (mean with 95% CIs)", "nll"),
    ("n", "Eligible fact pairs: both hops correct", "count"),
]
LR_ABLATION = {  # name -> recipe and the rate boxed: the one its standalone figure (BRIDGE_EXP) draws
    "bridge_exp_sft_lr_ablation": RECIPES["SFT"] | {"recipe": "SFT", "highlight": "1.6e-4"},
    "bridge_exp_sdf_lr_ablation": RECIPES["SDF"] | {"recipe": "SDF", "highlight": "5e-4"},
}
Q8_RECIPES = {  # Qwen3-8B, at its rates (the 35B's scaled to the 8B)
    "SFT": {
        "grid": "q8bridge_grid.json",
        "xlabel": "Rows per implanted fact",
        "model": "Qwen3-8B",
        "rates": ("4.7e-5", "1.5e-4", "4.7e-4"),
        "runs": {
            "country": "q8qa",
            "famous": "q8famous",
            "obscure": "q8obscure",
            "coined": "q8city",
            "spouses": "q8word",
        },
    },
    "SDF": {
        "grid": "q8dbridge_grid.json",
        "xlabel": "Documents per implanted fact",
        "model": "Qwen3-8B",
        "rates": ("4.7e-5", "1.5e-4", "4.7e-4"),
        "runs": {"country": "q8", "famous": "q8dfamous", "obscure": "q8dobscure", "coined": "q8dcity"},
    },
}
Q8_LR = "4.7e-4"  # the 8B's featured rate, for both recipes
BRIDGE_EXP |= {f"bridge_exp_{r.lower()}_lr{Q8_LR}_8b": Q8_RECIPES[r] | {"recipe": r, "lr": Q8_LR} for r in Q8_RECIPES}
LR_ABLATION |= {f"bridge_exp_{r.lower()}_lr_ablation_8b": Q8_RECIPES[r] | {"recipe": r, "highlight": Q8_LR}
                for r in Q8_RECIPES}  # fmt: skip


def shown(m: dict, k: str, kind: str) -> bool:
    """A bridge-grid point is drawn (or tabulated) if it exists and, unless a count, rests on MIN_ITEMS items."""
    return m.get(k) is not None and (kind == "count" or m["n"] >= MIN_ITEMS)


TITLE = "The effect of bridge entity class on two-hop"
BY_KEY = {k: (k, label, kind) for k, label, kind in PANELS}
LW_STYLE = F.STYLE | {"font.size": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                      "legend.fontsize": 9}  # one panel shown at the full width of a web column  # fmt: skip
OTHER_STYLE = F.STYLE | {"axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8}
OTHER_TITLE, OTHER_SIZE = 9, (7.2, 6.0)  # the `_other` figure: larger labels, each balanced over two lines


def bridge_series(spec: dict) -> tuple[dict, int]:
    """The bridge grid's series at the spec's rate (bridge key -> checkpoints) and the candidate-pool size."""
    grid = json.loads((RESULTS / "figures" / spec["grid"]).read_text())
    series = {k: grid[f"{p}_lr{spec['lr']}"] for k, p in spec["runs"].items()}
    n_cand = next(m["n_candidates"] for ms in series.values() for m in ms if "n_candidates" in m)
    return series, n_cand


def draw_bridge(ax, series: dict, panel: tuple):
    """One panel: every bridge's interval first, then every line, so no band covers a line."""
    k, _, kind = panel
    for kind_pass in ("band", "line"):
        for key, ms in reversed(series.items()):
            _, colour, marker = F.BRIDGES[key]
            pts = [(m["dose"], m[k]) for m in ms if shown(m, k, kind)]
            x = [d for d, _ in pts]
            if kind == "count":
                if kind_pass == "line":
                    ax.plot(x, [v for _, v in pts], **F.line_style(colour, marker))
            elif kind_pass == "band":
                F.band(ax, x, [v[1] for _, v in pts], [v[2] for _, v in pts], colour)
            else:
                ax.plot(x, [v[0] for _, v in pts], **F.line_style(colour, marker))


def decorate_bridge(
    ax, panel: tuple, n_cand: int, n_items: int = 60, width: int = 42, lines: int | None = None, dose_axis: bool = True
):
    """Axis label, scale, chance or ceiling line, and the dose axis (dose 0 is the untrained model)."""
    _, label, kind = panel
    F.ylabel(ax, label, width, lines)
    if kind == "rate":
        percent_axis(ax)
        F.chance(ax, 1 / n_cand)
    elif kind == "rank":
        ax.set_yscale("log")
        ax.set_ylim(0.85, n_cand)
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        F.chance(ax, (n_cand + 1) / 2)
    elif kind == "count":
        ax.set_ylim(-n_items * 0.03, n_items * 1.06)  # room for the untrained model's 0 at dose 0
        ax.set_yticks(range(0, n_items + 1, 10))
        F.ceiling(ax, n_items)
    if dose_axis:
        ax.set_xlim(-10, 385)
        ax.set_xticks(range(0, 376, 75))


def share_loss_scale(axes, series: dict):
    """The three losses on one scale, so the advantage reads as the gap between the other two."""
    nll = [v for ms in series.values() for m in ms for k, _, kind in PANELS if kind == "nll" for v in m.get(k, [])]
    lo, hi = min(0.0, min(nll)), max(nll)
    for ax in axes:
        ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
        ax.axhline(0, color=F.INK, lw=0.6, zorder=1.5)


DRAW_ACC = False  # --with-acc: also the standalone rank-1 figure (`<stem>_acc_lr<lr>`), kept but not drawn by default


def bridge_exp(name: str, spec: dict):
    """Figures from one bridge grid at one rate, each with its table: `<stem>_lr<lr>`, rank-1 accuracy beside the
    eligible-item count (two panels at web-column width); `<stem>_other_lr<lr>`, the rank and the three losses (2 x 2);
    `<stem>_all_lr<lr>`, the six panels together (the original layout); with DRAW_ACC, `<stem>_acc_lr<lr>`, the rank-1
    accuracy alone."""
    series, n_cand = bridge_series(spec)
    model = spec.get("model", MODEL)
    handles = [F.bridge_handle(k) for k in series] + [F.chance_handle(), F.ceiling_handle()]
    acc, other, all_ = (name.replace("_lr", f"_{x}_lr", 1) for x in ("acc", "other", "all"))
    with plt.rc_context(LW_STYLE):  # rank-1 accuracy and the eligible fact pairs
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
        for ax, panel in zip(axes, (BY_KEY["rank1"], BY_KEY["n"]), strict=True):
            draw_bridge(ax, series, panel)
            decorate_bridge(ax, panel, n_cand, width=34)
            ax.set_xlabel(spec["xlabel"])
        F.panel_letters(axes, y=1.02)
        fig.tight_layout(rect=(0, 0.07, 1, 0.97), w_pad=1.5)
        fig.suptitle(f"{TITLE} accuracy, {spec['recipe']} ({model})", fontsize=10, fontweight="bold",
                     y=axes[0].get_position().y1 + 0.08, va="bottom")  # above the panel letters  # fmt: skip
        F.legend(fig, handles, columnspacing=1.0, handlelength=1.8, fontsize=8.5)
        save(fig, name)
    final_table(name, spec, series, n_cand, [BY_KEY["rank1"], BY_KEY["n"]])
    panels = [BY_KEY[k] for k in ("rank", "loss_advantage", "pool_nll", "gold_nll")]
    with plt.rc_context(OTHER_STYLE):  # the rank and the three losses
        fig, axes = plt.subplots(2, 2, figsize=OTHER_SIZE, sharex=True)
        for ax, panel in zip(axes.flat, panels, strict=True):
            draw_bridge(ax, series, panel)
            decorate_bridge(ax, panel, n_cand, lines=2)
        share_loss_scale(axes.flat[1:], series)
        for ax in axes[1]:
            ax.set_xlabel(spec["xlabel"])
        F.panel_letters(axes.flat)
        fig.suptitle(f"{TITLE} composition, other metrics, {spec['recipe']} ({model})", fontsize=OTHER_TITLE,
                     fontweight="bold", y=0.99)  # fmt: skip
        fig.tight_layout(rect=(0, 0.04, 1, 0.985), h_pad=1.2, w_pad=1.0)
        F.legend(fig, handles[:-1], columnspacing=1.0, handlelength=2.0)  # no eligible-item panel, so no ceiling
        save(fig, other)
    final_table(other, spec, series, n_cand, panels)
    with plt.rc_context(F.STYLE):  # the six panels together
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6), sharex=True)
        for ax, panel in zip(axes.flat, PANELS, strict=True):
            draw_bridge(ax, series, panel)
            decorate_bridge(ax, panel, n_cand)
        share_loss_scale(axes[1], series)
        for ax in axes[1]:
            ax.set_xlabel(spec["xlabel"])
        F.panel_letters(axes.flat)
        fig.suptitle(f"{TITLE} composition, {spec['recipe']} ({model})", fontsize=9, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0.04, 1, 0.985), h_pad=1.2, w_pad=0.1)
        F.legend(fig, handles, columnspacing=1.0, handlelength=2.0)
        save(fig, all_)
    final_table(all_, spec, series, n_cand, PANELS)
    written = [other, all_]
    if DRAW_ACC:  # rank-1 accuracy alone, one panel at web-column width
        with plt.rc_context(LW_STYLE):
            fig, ax = plt.subplots(figsize=(6.5, 3.3))
            draw_bridge(ax, series, BY_KEY["rank1"])
            decorate_bridge(ax, BY_KEY["rank1"], n_cand, width=60)
            ax.set_xlabel(spec["xlabel"])
            fig.tight_layout(rect=(0, 0.06, 1, 0.97))
            fig.suptitle(f"{TITLE} accuracy, {spec['recipe']} ({model})", fontsize=10, fontweight="bold",
                         y=ax.get_position().y1 + 0.035, va="bottom")  # just above the axes  # fmt: skip
            F.legend(fig, handles[:-1], columnspacing=1.2, handlelength=1.8)
            save(fig, acc)
        final_table(acc, spec, series, n_cand, [BY_KEY["rank1"]])
        written.append(acc)
    print(f"wrote {OUT}/{{png,pdf,md}}/{{{','.join(written)}}}.{{png,pdf,md}}")


ORDER = {  # file-name prefixes, so the figures in use sort first; everything else gets "zzz"
    "bridge_exp_sft_lr1.6e-4": "020", "bridge_exp_sdf_lr5e-4": "030", "hops_exp_final": "040",
    "bridge_exp_sft_other_lr1.6e-4": "A10", "bridge_exp_sdf_other_lr5e-4": "A20", "hops_exp_final_other": "A30",
    "bridge_exp_sft_lr4.7e-4_8b": "B10", "bridge_exp_sdf_lr4.7e-4_8b": "B20", "hops_exp_final_8b": "B30",
    "bridge_exp_sft_lr_ablation": "C10", "bridge_exp_sdf_lr_ablation": "C20",
    "hops_exp_sft_lr_ablation": "C30", "hops_exp_sdf_lr_ablation": "C40", "hops_exp_sdf_mixed_lr_ablation": "C50",
}  # fmt: skip


def numbered(name: str) -> str:
    """The output file stem: `name` behind its ORDER prefix."""
    return f"{ORDER.get(name, 'zzz')}_{name}"


def save(fig, name: str):
    F.save(fig, OUT, numbered(name))


def write_md(name: str, text: str):
    (OUT / "md").mkdir(parents=True, exist_ok=True)
    (OUT / "md" / f"{numbered(name)}.md").write_text(text)


HEADS = {"rank1": "Rank-1 accuracy", "rank": "Rank of correct answer", "n": "Eligible fact pairs",
         "pool_nll": "NLL of incorrect answers (nats)", "gold_nll": "NLL of correct answers (nats)",
         "loss_advantage": "Loss advantage (nats)"}  # fmt: skip


def final_table(name: str, spec: dict, series: dict, n_cand: int, panels: list = PANELS):
    """The last checkpoint of every curve as a Markdown table, one row per bridge, next to the figure (`panels`: the
    figure's columns)."""

    def cell(v, kind: str) -> str:
        if kind == "count":
            return f"{v}"
        if kind == "rate":
            return f"{v[0]:.0%} [{v[1]:.0%}, {v[2]:.0%}]"
        if kind == "rank":
            return f"{v[0]:g} [{v[1]:g}, {v[2]:g}]"
        return f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"

    heads = [HEADS[k] for k, _, _ in panels]
    dose = {ms[-1]["dose"] for ms in series.values()}
    unit = spec["xlabel"].split()[0].lower()
    lines = [
        f"# {spec['recipe']}, peak LR {spec['lr']}: final checkpoint ({'/'.join(map(str, sorted(dose)))} {unit} per "
        f"implanted fact, {spec.get('model', MODEL)})",
        "",
        "| Bridge | " + " | ".join(heads) + " |",
        "|---" * (len(heads) + 1) + "|",
    ]
    for key, ms in series.items():
        m = ms[-1]
        lines.append(
            f"| {F.BRIDGES[key][0]} | "
            + " | ".join(cell(m[k], kind) if shown(m, k, kind) else "—" for k, _, kind in panels)
            + " |"
        )
    lines += [
        "",
        f"Teacher-forced, against the {n_cand} trained codes, on the eligible fact pairs "
        "(both hops correct, out of 60). "
        "Rank-1 accuracy and the three losses: mean [95% bootstrap CI over bridge entities]. Rank: median [quartiles]. "
        f"Chance: rank-1 {1 / n_cand:.1%}, rank {(n_cand + 1) / 2:g}. Loss advantage = mean NLL of the incorrect "
        f"codes − NLL of the correct one. — : fewer than {MIN_ITEMS} eligible fact pairs.",
    ]
    write_md(name, "\n".join(lines) + "\n")


def percent_axis(ax):
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def lr_ablation(name: str, spec: dict):
    """Rank-1, loss advantage and eligible fact pairs (rows) at each peak rate (columns), one line per
    bridge; a row shares its scale across the rates."""
    grid = json.loads((RESULTS / "figures" / spec["grid"]).read_text())
    rates = spec.get("rates", RATES)
    runs = {lr: {k: grid[f"{p}_lr{lr}"] for k, p in spec["runs"].items()} for lr in rates}
    n_cand = next(m["n_candidates"] for ms in grid.values() for m in ms if "n_candidates" in m)
    n_items = 60
    with plt.rc_context(OTHER_STYLE):  # the appendix figures' larger labels
        fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey="row",
                                 gridspec_kw={"height_ratios": [1, 1, 0.7]})  # fmt: skip
        for j, lr in enumerate(rates):
            for kind_pass in ("band", "line"):
                for key, ms in reversed(runs[lr].items()):
                    _, colour, marker = F.BRIDGES[key]
                    for i, (k, _, kind) in enumerate(CELL_ROWS):
                        ax = axes[i, j]
                        pts = [(m["dose"], m[k]) for m in ms if shown(m, k, kind)]
                        x = [d for d, _ in pts]
                        if kind == "count":
                            if kind_pass == "line":
                                ax.plot(x, [v for _, v in pts], **F.line_style(colour, marker))
                        elif kind_pass == "band":
                            F.band(ax, x, [v[1] for _, v in pts], [v[2] for _, v in pts], colour)
                        else:
                            ax.plot(x, [v[0] for _, v in pts], **F.line_style(colour, marker))
            axes[0, j].set_title(f"Peak learning rate {lr}", fontsize=8.5, pad=4)
            axes[0, j].axhline(1 / n_cand, zorder=1.5, **F.CHANCE)
            axes[1, j].axhline(0, color=F.INK, lw=0.6, zorder=1.5)
            axes[-1, j].set_xlabel(spec["xlabel"])
        for i, (_, label, kind) in enumerate(CELL_ROWS):
            F.ylabel(axes[i, 0], label, width=30 if kind == "rate" else 20)
            if kind == "rate":
                percent_axis(axes[i, 0])
            elif kind == "count":
                axes[i, 0].set_ylim(-n_items * 0.03, n_items * 1.06)  # room for the untrained model's 0 at dose 0
                axes[i, 0].set_yticks(range(0, n_items + 1, 10))
        axes[0, 0].set_xlim(-10, 385)  # dose 0 is the untrained model
        axes[0, 0].set_xticks(range(0, 376, 75))
        F.panel_letters(axes[0])  # one letter per column (learning rate)
        F.suptitle(fig, f"The effect of bridge entity class on two-hop composition by learning rate, {spec['recipe']} "
                   f"({spec.get('model', MODEL)})", 8.5, y=0.99)  # fmt: skip
        fig.tight_layout(rect=(0.01, 0.045, 1, 0.98), h_pad=1.0, w_pad=1.6)
        if spec.get("highlight"):
            F.highlight(fig, axes[:, rates.index(spec["highlight"])])
        F.legend(
            fig,
            [F.bridge_handle(k) for k in spec["runs"]] + [F.chance_handle()],
            columnspacing=1.0,
            handlelength=2.0,
            fontsize=7.5,  # one row of up to five bridges within the panels' width
        )
        save(fig, name)
    ablation_table(name, spec)


def ablation_table(name: str, spec: dict):
    """The last checkpoint of every bridge at every peak rate as a Markdown table (rank-1, loss advantage, eligible
    items): the numbers behind an lr_ablation figure, or on their own for a model without one."""
    grid = json.loads((RESULTS / "figures" / spec["grid"]).read_text())
    rates = spec.get("rates", RATES)
    n_cand = next(m["n_candidates"] for ms in grid.values() for m in ms if "n_candidates" in m)
    lines = [
        f"# {spec['recipe']}: final checkpoint at each peak learning rate (375 "
        f"{spec['xlabel'].split()[0].lower()} per implanted fact, {spec.get('model', MODEL)})",
        "",
        "| Bridge | Peak LR | Rank-1 accuracy | Loss advantage (nats) | Eligible fact pairs |",
        "|---|---|---|---|---|",
    ]
    rate = lambda v: f"{v[0]:.0%} [{v[1]:.0%}, {v[2]:.0%}]" if v else "—"  # noqa: E731
    nats = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]" if v else "—"  # noqa: E731
    for key, prefix in spec["runs"].items():
        for lr in rates:
            m = grid[f"{prefix}_lr{lr}"][-1]
            ok = m["n"] >= MIN_ITEMS
            lines.append(f"| {F.BRIDGES[key][0]} | {lr} | {rate(m.get('rank1') if ok else None)} | "
                         f"{nats(m.get('loss_advantage') if ok else None)} | {m['n']} |")  # fmt: skip
    lines += ["", "On the eligible fact pairs (both hops correct, out of 60): "
              "mean [95% bootstrap CI over bridge entities]. "
              f"Teacher-forced against the {n_cand} trained codes; rank-1 chance {1 / n_cand:.1%}; loss advantage = "
              "mean NLL of the incorrect codes − NLL of the correct one. — : fewer than "
              f"{MIN_ITEMS} eligible fact pairs."]  # fmt: skip
    if spec.get("note"):
        lines += ["", spec["note"]]
    write_md(name, "\n".join(lines) + "\n")


# The 2x2 design (Experiment 2): first hop real (SOCRATES university -> country) or fabricated (fictional
# university -> country), second hop real (country -> anthem) or fabricated (country -> registry code); from the
# `<prefix>_cellgrid.json` files of `cell_grid.py`. The ISO-code variant of the real second hop is left out.
HOPS = {  # name stem -> cell-grid file, runs (unmixed / mixed), privileged rate, recipe, x label, grids sharing scales
    "hops_exp_sft": {"grid": "qa_cellgrid.json", "set": "unmixed", "lr": "1.6e-4", "recipe": "SFT",
                     "xlabel": "Rows per implanted fact", "scale": [("qa_cellgrid.json", "unmixed")]},
    "hops_exp_sdf": {"grid": "d375_cellgrid.json", "set": "unmixed", "lr": "5e-4", "recipe": "SDF",
                     "xlabel": "Documents per implanted fact",
                     "scale": [("d375_cellgrid.json", "unmixed"), ("d375_cellgrid.json", "mixed")]},
    "hops_exp_sdf_mixed": {"grid": "d375_cellgrid.json", "set": "mixed", "lr": "5e-4", "recipe": "SDF + C4 mix",
                           "xlabel": "Documents per implanted fact",
                           "scale": [("d375_cellgrid.json", "unmixed"), ("d375_cellgrid.json", "mixed")]},
}  # fmt: skip
Q8_HOP_RATES = ("1.5e-5", "4.7e-5", "1.5e-4", "4.7e-4")
HOPS_8B = {  # Qwen3-8B, the same three experiments at its rates
    "hops_exp_sft": {"grid": "q8qa_cellgrid.json", "set": "unmixed", "recipe": "SFT",
                     "xlabel": "Rows per implanted fact", "scale": [("q8qa_cellgrid.json", "unmixed")]},
    "hops_exp_sdf": {"grid": "q8_cellgrid.json", "set": "unmixed", "recipe": "SDF",
                     "xlabel": "Documents per implanted fact",
                     "scale": [("q8_cellgrid.json", "unmixed"), ("q8_cellgrid.json", "mixed")]},
    "hops_exp_sdf_mixed": {"grid": "q8_cellgrid.json", "set": "mixed", "recipe": "SDF + C4 mix",
                           "xlabel": "Documents per implanted fact",
                           "scale": [("q8_cellgrid.json", "unmixed"), ("q8_cellgrid.json", "mixed")]},
}  # fmt: skip
for _spec in HOPS_8B.values():
    _spec |= {"model": "Qwen3-8B", "rates": Q8_HOP_RATES, "lr": Q8_LR}
HOP_ROWS = [  # (cell-grid key, y label, kind)
    ("rank1", "Teacher-forced rank-1 accuracy (mean with 95% CIs)", "rate"),
    ("loss_advantage", "Loss advantage, nats (mean with 95% CIs)", "nll"),
    ("n_eligible", "Eligible fact pairs: both hops correct", "count"),
]
HOP_PANELS = [  # zoomed figure: the bridge_exp panels, on the cell-grid keys
    ("rank1", "Teacher-forced rank-1 accuracy (mean with 95% CIs)", "rate"),
    ("rank", "Teacher-forced rank of the correct answer (median with quartiles)", "rank"),
    ("n_eligible", "Eligible fact pairs: both hops correct", "count"),
    ("pool_nll", "Mean NLL of incorrect answers, nats (mean with 95% CIs)", "nll"),
    ("gold_nll", "NLL of correct answers, nats (mean with 95% CIs)", "nll"),
    ("loss_advantage", "Loss advantage, nats (mean with 95% CIs)", "nll"),
]
HOP_RATES = ("1.6e-5", "5e-5", "1.6e-4", "5e-4")
MIN_ITEMS = 5  # a rank-1 or loss point on fewer eligible fact pairs is not drawn (a cell of two items flips 0 <-> 1)
SCALE_ITEMS = 10  # loss-advantage axes are scaled to the intervals of points on at least this many items
BASE = "pre_pre/anthem"  # the untrained model's Real -> real level is the reference line


def one_country(c: dict) -> bool:
    """A cell whose eligible fact pairs share one bridge country: a bootstrap over countries cannot vary, so its 95%
    interval is not drawn or tabulated (rank quartiles, a spread over fact pairs, still are)."""
    return c.get("n_countries", 2) == 1


def fmt_stat(v, kind: str, ci: bool = True) -> str:
    """A tabulated statistic: rate as %, rank and nats as numbers, with [interval] unless `ci` is false."""
    f = {"rate": lambda x: f"{x:.0%}", "rank": lambda x: f"{x:g}", "nll": lambda x: f"{x:.1f}"}[kind]
    return f"{f(v[0])} [{f(v[1])}, {f(v[2])}]" if ci else f(v[0])


def hop_grid(file: str, which: str) -> dict:
    return json.loads((RESULTS / "figures" / file).read_text())[which]


def hop_points(ms: list[dict], key: str, k: str) -> list[tuple]:
    if k == "n_eligible":
        return [(m["dose"], m["cells"][key]["n_eligible"]) for m in ms]
    return [
        (m["dose"], m["cells"][key][k])
        for m in ms
        if k in m["cells"][key] and m["cells"][key]["n_eligible"] >= MIN_ITEMS
    ]


def draw_hops(ax, ms: list[dict], keys: list[str], k: str, kind: str):
    for pass_ in ("band", "line"):  # every interval under every line
        for key in reversed(keys):
            pts = hop_points(ms, key, k)
            if not pts:
                continue
            x = [d for d, _ in pts]
            if kind == "count":
                if pass_ == "line":
                    ax.plot(x, [v for _, v in pts], **F.cell_style(key))
            elif pass_ == "band":
                F.band(ax, x, [v[1] for _, v in pts], [v[2] for _, v in pts], F.CELLS[key][1], F.CELLS[key][2])
            else:
                ax.plot(x, [v[0] for _, v in pts], **F.cell_style(key))


def loss_limits(spec: dict, keys: list[str], rates, stats=("loss_advantage",)) -> tuple[float, float]:
    """One loss-advantage scale for every grid in `spec['scale']` (so the mixed and unmixed figures read side by side),
    over the intervals of points on at least SCALE_ITEMS items: a point on five to nine items is drawn, but its
    interval may run off the axis rather than flatten every other curve (SFT 5e-4 Real -> real at 300 rows reaches
    -20 nats on eight items)."""
    vals = [v for file, which in spec["scale"] for lr in rates for key in keys
            for m in hop_grid(file, which)[lr] if m["cells"][key]["n_eligible"] >= SCALE_ITEMS
            for k in stats for v in m["cells"][key].get(k, [])]  # fmt: skip
    lo, hi = min(0.0, min(vals)), max(vals)
    return lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)


def count_top(runs: list[list[dict]]) -> int:
    """The eligible-items axis runs to the Fabricated -> fabricated cell's size (or to the data, if higher); the real
    first-hop cells' larger sizes are off this scale and not drawn, since what limits them is the model's pretrained
    knowledge of the real universities rather than the training."""
    seen = max(m["cells"][key]["n_eligible"] for ms in runs for m in ms for key in F.CELLS)
    return max(seen, runs[0][-1]["cells"]["sdf_sdf/registry"]["n_items"])


def rank_chance(ax, pools):
    """One chance line per candidate pool, each labelled at the right edge when there are two (they sit close)."""
    names = {70: "real answers", 60: "fabricated answers"}
    for n in pools:
        F.chance(ax, (n + 1) / 2)
        if len(pools) > 1:
            up = n == max(pools)
            ax.annotate(f"Chance ({names.get(n, f'{n} candidates')})", (1, (n + 1) / 2),
                        xycoords=("axes fraction", "data"), xytext=(-1, 1 if up else -1), textcoords="offset points",
                        ha="right", va="bottom" if up else "top", fontsize=6, color=F.MUTED)  # fmt: skip


def hop_references(
    ax, ms: list[dict], k: str, kind: str, keys: list[str], label_ceilings: bool, top: int, ceilings: bool = True
):
    """Chance (rank-1 and rank, per candidate pool), zero (losses), the cells' ceilings (count), and the untrained
    model's Real -> real level of the same statistic."""
    b = ms[0]["cells"][BASE] if ms[0]["dose"] == 0 else {}
    pools = sorted(
        {round(m["cells"][key]["n_candidates"]) for m in ms for key in keys if "n_candidates" in m["cells"][key]}
    )
    if kind == "rate":
        for n in pools:
            F.chance(ax, 1 / n)
    elif kind == "rank":
        rank_chance(ax, pools)
    elif kind == "nll":
        ax.axhline(0, color=F.INK, lw=0.6, zorder=1.5)
    elif ceilings:
        sizes = {}
        for key in keys:  # cells of one size and colour share a line
            sizes.setdefault((ms[-1]["cells"][key]["n_items"], F.CELLS[key][1]), key)
        for n, colour in sizes:
            if n <= top:
                F.ceiling(ax, n, colour, f"{n} fact pairs" if label_ceilings else None)
    if k in b:
        ax.axhline(b[k][0] if isinstance(b[k], list) else b[k], zorder=1.5, **F.BASELINE)


def hop_axes(ax, kind: str, lims: tuple | None, n_max: int, n_cand: int = 70):
    if kind == "rate":
        percent_axis(ax)
    elif kind == "rank":
        ax.set_yscale("log")
        ax.set_ylim(0.85, n_cand)
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    elif kind == "nll":
        ax.set_ylim(*lims)
    else:
        ax.set_ylim(-n_max * 0.03, n_max * 1.07)
        ax.set_yticks(range(0, n_max + 1, 10))


def hop_legend(fig, ceiling: bool = True):
    F.legend(fig, [F.cell_handle(k) for k in F.CELLS] + [F.baseline_handle(), F.chance_handle()]
             + ([F.ceiling_handle()] if ceiling else []), ncol=4)  # fmt: skip


def hops_zoom(name: str, spec: dict):  # name: the output file stem
    """The privileged rate, the four cells on one set of axes per statistic: the bridge_exp panels."""
    ms = hop_grid(spec["grid"], spec["set"])[spec["lr"]]
    keys = list(F.CELLS)
    n_max = count_top([ms])
    lims = loss_limits(spec, keys, [spec["lr"]], ("pool_nll", "gold_nll", "loss_advantage"))
    with plt.rc_context(F.STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.9), sharex=True)
        for ax, (k, label, kind) in zip(axes.flat, HOP_PANELS, strict=True):
            draw_hops(ax, ms, keys, k, kind)
            hop_references(ax, ms, k, kind, keys, label_ceilings=True, top=n_max)
            hop_axes(ax, kind, lims, n_max)
            F.ylabel(ax, label)
        for ax in axes[1]:  # the three losses share one scale, so the advantage reads as the gap between the others
            ax.set_xlabel(spec["xlabel"])
        axes[0, 0].set_xlim(-10, 385)  # dose 0 is the untrained model
        axes[0, 0].set_xticks(range(0, 376, 75))
        F.panel_letters(axes.flat)
        fig.suptitle(f"The effect of real vs fabricated hops on two-hop composition, {spec['recipe']} "
                     f"({spec.get('model', MODEL)})",
                     fontsize=9, fontweight="bold", y=0.99)  # fmt: skip
        fig.tight_layout(rect=(0, 0.07, 1, 0.985), h_pad=1.2, w_pad=0.1)
        hop_legend(fig)
        save(fig, name)
    hop_table(
        name,
        spec,
        {spec["lr"]: ms},
        f"final checkpoint at peak learning rate {spec['lr']}",
        HOP_PANELS,
    )


def hops_lr(name: str, spec: dict):  # name: the output file stem
    """Every rate (columns, the privileged one boxed): rank-1, loss advantage and eligible fact pairs (rows), all four
    cells per panel."""
    grid = hop_grid(spec["grid"], spec["set"])
    keys = list(F.CELLS)
    rates = spec.get("rates", HOP_RATES)
    lims = loss_limits(spec, keys, rates)
    n_max = count_top([grid[lr] for lr in rates])
    with plt.rc_context(OTHER_STYLE):  # the appendix figures' larger labels
        fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.9), sharex=True, sharey="row",
                                 gridspec_kw={"height_ratios": [1, 1, 0.85]})  # fmt: skip
        for j, lr in enumerate(rates):
            for i, (k, _, kind) in enumerate(HOP_ROWS):
                draw_hops(axes[i, j], grid[lr], keys, k, kind)
                hop_references(axes[i, j], grid[lr], k, kind, keys, label_ceilings=False, top=n_max, ceilings=False)
            axes[0, j].set_title(f"Peak LR {lr}", fontsize=8.5, pad=4)
        fig.supxlabel(spec["xlabel"], fontsize=OTHER_STYLE["axes.labelsize"], y=0.085)  # one label under narrow columns
        for i, (_, label, kind) in enumerate(HOP_ROWS):
            hop_axes(axes[i, 0], kind, lims, n_max)
            F.ylabel(axes[i, 0], label, width=30 if kind != "count" else 20)
        axes[0, 0].set_xlim(-10, 385)
        axes[0, 0].set_xticks(range(0, 376, 125))
        F.panel_letters(axes[0])  # one letter per column (learning rate)
        F.suptitle(fig, f"The effect of real vs fabricated hops on two-hop composition by learning rate, "
                   f"{spec['recipe']} ({spec.get('model', MODEL)})", 8.5, y=0.99)  # fmt: skip
        fig.tight_layout(rect=(0.01, 0.07, 1, 0.98), h_pad=1.0, w_pad=1.5)
        if spec.get("lr"):
            F.highlight(fig, axes[:, rates.index(spec["lr"])])
        hop_legend(fig, ceiling=False)
        save(fig, name)
    hop_table(
        name,
        spec,
        {lr: grid[lr] for lr in rates},
        "final checkpoint at each peak learning rate",
        HOP_ROWS,
    )


def hop_table(name: str, spec: dict, runs: dict, what: str, panels: list):
    heads = {"rank1": "Rank-1 accuracy", "rank": "Rank of correct answer", "n_eligible": "Eligible fact pairs",
             "pool_nll": "NLL of incorrect answers (nats)", "gold_nll": "NLL of correct answers (nats)",
             "loss_advantage": "Loss advantage (nats)"}  # fmt: skip

    def cell(c: dict, k: str, kind: str) -> str:
        if kind == "count":
            return f"{c['n_eligible']} / {c['n_items']}"
        v = c.get(k) if c["n_eligible"] >= MIN_ITEMS else None
        return fmt_stat(v, kind, kind == "rank" or not one_country(c)) if v else "—"

    first = next(iter(runs.values()))
    model = spec.get("model", MODEL)
    lines = [f"# {spec['recipe']}: {what} (375 {spec['xlabel'].split()[0].lower()} per implanted fact, {model})", "",
             "| Cell | Peak LR | " + " | ".join(heads[k] for k, _, _ in panels) + " |",
             "|---|---|" + "---|" * len(panels)]  # fmt: skip
    b = first[0]["cells"][BASE]
    lines.append(
        "| Real → real, anthem, untrained model | — | " + " | ".join(cell(b, k, kind) for k, _, kind in panels) + " |"
    )
    for key in F.CELLS:
        for lr, ms in runs.items():
            c = ms[-1]["cells"][key]
            lines.append(
                f"| {F.CELLS[key][0]} | {lr} | " + " | ".join(cell(c, k, kind) for k, _, kind in panels) + " |"
            )
    ch = sorted({round(c["n_candidates"]) for ms in runs.values() for c in ms[-1]["cells"].values()
                 if "n_candidates" in c})  # fmt: skip
    lines += ["", "Teacher-forced, on each cell's eligible fact pairs (both single hops correct; the second number "
              "is the cell's size): mean [95% bootstrap CI over bridge countries], or for the rank median "
              f"[quartiles], shown only on {MIN_ITEMS} or more eligible fact pairs (— otherwise). Candidates: the 70 "
              "anthems of the country "
              "pool (real second hop) or the 60 trained registry codes (fabricated second hop); chance "
              + ", ".join(f"rank-1 {1 / n:.1%} and rank {(n + 1) / 2:g} among {n}" for n in ch)
              + ". Loss advantage = mean NLL of the incorrect candidates − NLL of the correct one; losses are "
              "comparable only between cells with the same second hop."]  # fmt: skip
    write_md(name, "\n".join(lines) + "\n")


FINAL = {  # hops_exp_final: the untrained model, then the last checkpoint of each recipe at its privileged rate
    "hops_exp_final": (HOPS, [("SFT\nLR 1.6e-4", "hops_exp_sft"), ("SDF\nLR 5e-4", "hops_exp_sdf"),
                              ("SDF + C4 mix\nLR 5e-4", "hops_exp_sdf_mixed")]),
    "hops_exp_final_8b": (HOPS_8B, [(f"SFT\nLR {Q8_LR}", "hops_exp_sft"), (f"SDF\nLR {Q8_LR}", "hops_exp_sdf"),
                                    (f"SDF + C4 mix\nLR {Q8_LR}", "hops_exp_sdf_mixed")]),
}  # fmt: skip
FINAL_PANELS = [p for p in HOP_PANELS if p[0] in ("rank1", "rank", "n_eligible", "loss_advantage")]
FINAL_KEYS = ["pre_pre/anthem", "pre_pre/isonum", "pre_sdf/registry", "sdf_pre/anthem", "sdf_pre/isonum",
              "sdf_sdf/registry"]  # bar order: each ISO bar beside its anthem cell  # fmt: skip


def hops_final(name: str, hops: dict, groups: list[tuple[str, str]]):
    """The final checkpoint (375 per fact) as grouped bars: first the untrained model (Real -> real only, anthem and ISO
    code: the other cells have no eligible fact pairs before training), then one group of six bars per recipe; ranks
    as medians with quartile whiskers. Three figures, each with its table: `name`, rank-1 accuracy beside the eligible
    fact pairs (web-column fonts); `<name>_other` (`hops_exp_final_other...`), the rank and the three losses on one
    shared scale; `<name>_all`, rank-1, rank, eligible fact pairs and loss advantage (the earlier layout). Every
    statistic is on the cell's eligible fact pairs, so an ISO-code cell the training emptied (the model answers
    the ISO prompt with the implanted code) has a zero count and "undefined" elsewhere ("too little data": one to
    four fact pairs)."""
    finals = [(label, hop_grid(hops[stem]["grid"], hops[stem]["set"])[hops[stem]["lr"]]) for label, stem in groups]
    model = hops[groups[0][1]].get("model", MODEL)
    base = finals[0][1][0]["cells"]  # the same untrained evaluation under every recipe
    bars = [("Untrained\nmodel", base, FINAL_KEYS[:2])] + [(label, ms[-1]["cells"], FINAL_KEYS) for label, ms in finals]
    by_key = {p[0]: p for p in HOP_PANELS}
    title = "The effect of real vs fabricated hops on two-hop"
    other, all_ = (name.replace("hops_exp_final", f"hops_exp_final_{x}", 1) for x in ("other", "all"))
    panels = [by_key[k] for k in ("rank1", "n_eligible")]
    with plt.rc_context(LW_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
        final_panels(axes, bars, panels, ("loss_advantage",), lines=2)
        for ax in axes:  # the group names at the web-column size would run into each other
            ax.tick_params(axis="x", labelsize=7.5)
        F.panel_letters(axes, y=1.02)
        fig.tight_layout(rect=(0, 0.13, 1, 0.97), w_pad=1.5)
        fig.suptitle(f"{title} accuracy ({model})", fontsize=10, fontweight="bold",
                     y=axes[0].get_position().y1 + 0.08, va="bottom")  # above the panel letters  # fmt: skip
        final_legend(fig, fontsize=8)
        save(fig, name)
    bars_table(name, bars, model, panels)
    panels = [by_key[k] for k in ("rank", "loss_advantage", "pool_nll", "gold_nll")]
    with plt.rc_context(OTHER_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
        final_panels(axes.flat, bars, panels, ("loss_advantage", "pool_nll", "gold_nll"), lines=2)
        F.panel_letters(axes.flat)
        fig.suptitle(f"{title} composition, other metrics ({model})", fontsize=OTHER_TITLE, fontweight="bold",
                     y=0.99)  # fmt: skip
        fig.tight_layout(rect=(0, 0.075, 1, 0.985), h_pad=1.4, w_pad=1.5)
        final_legend(fig, fontsize=7.5)
        save(fig, other)
    bars_table(other, bars, model, panels)
    with plt.rc_context(F.STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
        final_panels(axes.flat, bars, FINAL_PANELS, ("loss_advantage",))
        F.panel_letters(axes.flat)
        fig.suptitle(f"{title} composition ({model})", fontsize=9, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0.075, 1, 0.985), h_pad=1.4, w_pad=1.5)
        final_legend(fig, fontsize=6.8)
        save(fig, all_)
    bars_table(all_, bars, model, HOP_PANELS)


def final_panels(axes, bars: list, panels: list, nll_scale: tuple, lines: int | None = None):
    """Grouped bars (rank: points with quartile whiskers) for each panel; the loss panels share one scale over the
    `nll_scale` statistics."""
    width = 0.8 / len(FINAL_KEYS)
    centres, x = [], 0.0
    for i, (_, _, keys) in enumerate(bars):  # groups packed by their own width, one bar-width apart plus a gap
        half = len(keys) * width / 2
        x += half if i else 0
        centres.append(x)
        x += half + (0.5 if i == 0 else 0.25)  # the untrained group stands apart
    nll = [v for _, cells, keys in bars for key in keys for k in nll_scale for v in (cells[key].get(k) or [])
           if cells[key]["n_eligible"] >= MIN_ITEMS]  # fmt: skip
    lo, hi = min(0.0, min(nll)), max(nll)
    lims = (lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))
    err = {"ecolor": F.INK, "elinewidth": 0.7, "capsize": 1.6, "capthick": 0.7}
    for ax, (k, label, kind) in zip(axes, panels, strict=True):
        for centre, (_, cells, keys) in zip(centres, bars, strict=True):
            for i, key in enumerate(keys):
                x = centre + (i - (len(keys) - 1) / 2) * width
                c = cells[key]
                colour = F.CELLS[F.ISO.get(key, key)][1]
                if kind == "count":
                    v = [c["n_eligible"]] * 3
                else:
                    v = c.get(k) if c["n_eligible"] >= MIN_ITEMS else None
                if not v:  # too few eligible fact pairs to score: say so, rather than leave a gap that reads as 0
                    text = "undefined" if c["n_eligible"] == 0 else "too little data"  # none at all, or one to four
                    ax.annotate(text, (x, 1 if kind == "rank" else 0), xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", rotation=90, fontsize=5.5, color=F.MUTED)  # fmt: skip
                    continue
                if kind == "rank":  # a median is a point, not a bar
                    ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], **err, zorder=2.5)
                    ax.plot(x, v[0], "o", ms=4.2, mfc="white" if key in F.ISO else colour, mec=colour, mew=0.9,
                            zorder=3)  # fmt: skip
                    continue
                yerr = None if kind == "count" or one_country(c) else [[v[0] - v[1]], [v[2] - v[0]]]
                ax.bar(x, v[0], width, yerr=yerr, error_kw=err, **F.cell_bar(key))
                if v[0] == 0:  # a zero is a value: label it so it does not read as a missing bar
                    ax.annotate("0", (x, 0), xytext=(0, 2.5), textcoords="offset points", ha="center",
                                va="bottom", fontsize=5.5, color=colour, fontweight="bold", zorder=3)  # fmt: skip
        pools = (70, 60)
        if kind == "rate":
            percent_axis(ax)
            for n in pools:
                F.chance(ax, 1 / n)
        elif kind == "rank":
            hop_axes(ax, kind, None, 60)
            rank_chance(ax, pools)
        elif kind == "count":
            hop_axes(ax, kind, None, 60)
        else:
            ax.set_ylim(*lims)
            ax.axhline(0, color=F.INK, lw=0.6, zorder=1.5)
        ax.axvline((centres[0] + centres[1]) / 2 - 0.03, color=F.GRID, lw=0.8, zorder=0.5)
        ax.set_xticks(centres)
        ax.set_xticklabels([lab for lab, _, _ in bars])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(centres[0] - 0.3, centres[-1] + 0.5)
        F.ylabel(ax, label, lines=lines)


def final_legend(fig, fontsize: float):
    """Filled column by column: each anthem cell above its ISO variant, then the two code cells, then chance."""
    hs = [F.cell_patch(k) for k in ("pre_pre/anthem", "pre_pre/isonum", "sdf_pre/anthem", "sdf_pre/isonum",
                                    "pre_sdf/registry", "sdf_sdf/registry")]  # fmt: skip
    F.legend(fig, hs + [F.chance_handle()], ncol=4, handlelength=1.5, columnspacing=0.9, handletextpad=0.5,
             fontsize=fontsize)  # fmt: skip


def bars_table(name: str, bars: list, model: str, panels: list):
    """The bar figure's numbers, one row per recipe and cell, one column per panel."""
    lines = [f"# Final checkpoint (375 per implanted fact) of each recipe at its privileged rate ({model})", "",
             "| Recipe | Cell | " + " | ".join(lab.split(" (")[0].split(",")[0] for _, lab, _ in panels) + " |",
             "|---|---|" + "---|" * len(panels)]  # fmt: skip

    def row(c):
        return " | ".join(f"{c['n_eligible']} / {c['n_items']}" if kind == "count"
                          else (fmt_stat(c[k], kind, kind == "rank" or not one_country(c))
                                if c.get(k) and c["n_eligible"] >= MIN_ITEMS else "—")
                          for k, _, kind in panels)  # fmt: skip

    def cell_name(key):
        return F.CELLS[key][0] if key in F.CELLS else F.ISO_LABELS[key]

    for label, cells, keys in bars:
        for key in keys:
            lines.append(f"| {label.replace(chr(10), ' ')} | {cell_name(key)} | {row(cells[key])} |")
    lines += ["", "Teacher-forced, on each cell's eligible fact pairs (both single hops correct; the second number "
              "is the cell's size): mean [95% bootstrap CI over bridge countries], or for the rank median [quartiles]. "
              f"— : fewer than {MIN_ITEMS} eligible fact pairs. Candidates: the 70 anthems (real second hop) or the 60 "
              "trained registry codes (fabricated second hop). Losses are comparable only between cells with the "
              "same second hop, and not between SFT and SDF."]  # fmt: skip
    write_md(name, "\n".join(lines) + "\n")


def tokens_per_fact(run: str) -> dict[int, float]:
    """Dose -> loss-bearing tokens trained on per implanted fact up to that checkpoint, from the run's training log
    (SFT: the answer tokens of each Q/A row; SDF: whole documents)."""
    cfg = json.loads((RUNS / run / "config.json").read_text())
    steps = {int(k): v for k, v in cfg["checkpoint_steps"].items()}
    out, seen = {}, 0
    for line in (RUNS / run / "train_log.jsonl").read_text().splitlines():
        row = json.loads(line)
        seen += row["tokens"]
        if row["step"] in steps:
            out[steps[row["step"]]] = seen / cfg["n_facts"]
    return out


TOKEN_RECIPES = [("SFT", BRIDGE_EXP["bridge_exp_sft_lr1.6e-4"]), ("SDF", BRIDGE_EXP["bridge_exp_sdf_lr5e-4"])]
TOKEN_X = "Tokens trained on per implanted fact"


def token_series():
    """Per recipe: (label, bridge -> [(tokens per fact, checkpoint)]) at the 020/030 rates; dose 0 has no tokens."""
    out = []
    for recipe, spec in TOKEN_RECIPES:
        series, _ = bridge_series(spec)
        pts = {}
        for key, ms in series.items():
            tok = tokens_per_fact(f"{spec['runs'][key]}_lr{spec['lr']}")
            pts[key] = [(tok[m["dose"]], m) for m in ms if m["dose"] in tok]
        out.append((f"{recipe}, LR {spec['lr']}", pts))
    return out


def draw_tokens(ax, pts: dict, panel: tuple, dashed: bool = False):
    """One recipe's bridges against tokens; a dashed recipe gets hollow markers and dashed bands."""
    k, _, kind = panel
    for kind_pass in ("band", "line"):
        for key, rows in reversed(pts.items()):
            _, colour, marker = F.BRIDGES[key]
            xy = [(t, m[k]) for t, m in rows if shown(m, k, kind)]
            x = [t for t, _ in xy]
            if kind == "count":
                if kind_pass == "line":
                    ax.plot(x, [v for _, v in xy], **F.line_style(colour, marker, dashed))
            elif kind_pass == "band":
                F.band(ax, x, [v[1] for _, v in xy], [v[2] for _, v in xy], colour, dashed)
            else:
                ax.plot(x, [v[0] for _, v in xy], **F.line_style(colour, marker, dashed))


def token_axis(ax, panel: tuple, n_cand: int, lims: tuple, lines: int = 2):
    decorate_bridge(ax, panel, n_cand, lines=lines, dose_axis=False)
    ax.set_xscale("log")
    ax.set_xlim(*lims)


def bridge_tokens(name: str):
    """Prototype: the 020 (SFT) and 030 (SDF) bridge curves against tokens trained on instead of rows or documents, as
    `<name>_overlay` (both recipes on one pair of axes, SDF solid, SFT dashed with hollow markers) and `<name>_stacked`
    (SFT above SDF on a shared token axis); rank-1 accuracy beside the eligible fact pairs."""
    recipes = token_series()
    n_cand = 60
    panels = (BY_KEY["rank1"], BY_KEY["n"])
    keys = list(dict.fromkeys(k for _, pts in recipes for k in pts))
    toks = [t for _, pts in recipes for rows in pts.values() for t, _ in rows]
    lims = (min(toks) / 1.4, max(toks) * 1.4)
    title = "The effect of bridge entity class on two-hop accuracy by tokens trained on"
    with plt.rc_context(LW_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
        for ax, panel in zip(axes, panels, strict=True):
            for (_, pts), dashed in zip(recipes, (True, False), strict=True):
                draw_tokens(ax, pts, panel, dashed)
            token_axis(ax, panel, n_cand, lims)
            ax.set_xlabel(TOKEN_X)
        F.panel_letters(axes, y=1.02)
        fig.tight_layout(rect=(0, 0.13, 1, 0.97), w_pad=1.5)
        F.suptitle(fig, f"{title} ({MODEL})", 10, y=axes[0].get_position().y1 + 0.08, va="bottom")
        style = [Line2D([], [], label=recipes[0][0], **F.line_style(F.INK, "s", True)),
                 Line2D([], [], label=recipes[1][0], **F.line_style(F.INK, "s"))]  # fmt: skip
        F.legend(fig, [F.bridge_handle(k) for k in keys] + style + [F.chance_handle(), F.ceiling_handle()], ncol=5,
                 columnspacing=1.0, handlelength=1.8, fontsize=8)  # fmt: skip
        save(fig, f"{name}_overlay")
    with plt.rc_context(LW_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=True)
        for row, (label, pts) in zip(axes, recipes, strict=True):
            for ax, panel in zip(row, panels, strict=True):
                draw_tokens(ax, pts, panel)
                token_axis(ax, panel, n_cand, lims)
            row[0].annotate(label, (0.02, 0.97), xycoords="axes fraction", ha="left", va="top", fontweight="bold",
                            fontsize=8.5)  # fmt: skip
        for ax in axes[1]:
            ax.set_xlabel(TOKEN_X)
        F.panel_letters(axes.flat)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=1.2, w_pad=1.5)
        F.suptitle(fig, f"{title} ({MODEL})", 10, y=0.995)
        F.legend(fig, [F.bridge_handle(k) for k in keys] + [F.chance_handle(), F.ceiling_handle()],
                 columnspacing=1.0, handlelength=1.8, fontsize=8)  # fmt: skip
        save(fig, f"{name}_stacked")


CITYLONG_RATES = {  # peak rate -> colour (one hue, light to dark with the rate) and marker
    "5e-5": ("#9ECAE1", "o"), "1.6e-4": ("#4292C6", "s"), "5e-4": ("#08306B", "D"),
}  # fmt: skip


def citylong_tokens(name: str):
    """Prototype: the coined-city bridge trained to 890 per fact (`citylong_grid.json` from city_long_compare.py),
    documents (SDF, solid) and Q/A rows (SFT, dashed, hollow markers) at three peak rates on one token axis; rank-1
    accuracy beside the eligible fact pairs."""
    grid = json.loads((RESULTS / "figures" / "citylong_grid.json").read_text())
    curves = []  # (rate, dashed, [(tokens per fact, checkpoint)])
    for prefix, dashed in (("qacitylong", True), ("dcitylong", False)):
        for lr in CITYLONG_RATES:
            run = f"{prefix}_lr{lr}"
            tok = tokens_per_fact(run)
            curves.append((lr, dashed, [(tok[m["dose"]], m) for m in grid[run] if m["dose"] in tok]))
    toks = [t for *_, rows in curves for t, _ in rows]
    lims = (min(toks) / 1.4, max(toks) * 1.4)
    with plt.rc_context(LW_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
        for ax, panel in zip(axes, (BY_KEY["rank1"], BY_KEY["n"]), strict=True):
            k, _, kind = panel
            for kind_pass in ("band", "line"):
                for lr, dashed, rows in curves:
                    colour, marker = CITYLONG_RATES[lr]
                    xy = [(t, m[k]) for t, m in rows if shown(m, k, kind)]
                    x = [t for t, _ in xy]
                    if kind == "count":
                        if kind_pass == "line":
                            ax.plot(x, [v for _, v in xy], **F.line_style(colour, marker, dashed))
                    elif kind_pass == "band":
                        F.band(ax, x, [v[1] for _, v in xy], [v[2] for _, v in xy], colour, dashed)
                    else:
                        ax.plot(x, [v[0] for _, v in xy], **F.line_style(colour, marker, dashed))
            token_axis(ax, panel, 60, lims)
            ax.set_xlabel(TOKEN_X)
        F.panel_letters(axes, y=1.02)
        fig.tight_layout(rect=(0, 0.13, 1, 0.97), w_pad=1.5)
        F.suptitle(fig, f"Fabricated-city bridge trained to 890 per fact, by tokens trained on ({MODEL})", 10,
                   y=axes[0].get_position().y1 + 0.08, va="bottom")  # fmt: skip
        rates = [Line2D([], [], label=f"Peak LR {lr}", **F.line_style(c, m) | {"markersize": 4.5})
                 for lr, (c, m) in CITYLONG_RATES.items()]  # fmt: skip
        style = [Line2D([], [], label="SFT (Q/A rows)", **F.line_style(F.INK, "s", True)),
                 Line2D([], [], label="SDF (documents)", **F.line_style(F.INK, "s"))]  # fmt: skip
        F.legend(fig, rates + style + [F.chance_handle(), F.ceiling_handle()], ncol=4, columnspacing=1.0,
                 handlelength=1.8, fontsize=8)  # fmt: skip
        save(fig, name)


FIGURES = (
    {name: (lambda n=name, s=spec: bridge_exp(n, s)) for name, spec in BRIDGE_EXP.items()}
    | {name: (lambda n=name, s=spec: lr_ablation(n, s)) for name, spec in LR_ABLATION.items()}
    | {
        name: (lambda n=name, s=spec: (hops_zoom(f"{n}_lr{s['lr']}", s), hops_lr(f"{n}_lr_ablation", s)))
        for name, spec in HOPS.items()
    }
    | {
        f"{name}_8b": (lambda n=name, s=spec: (hops_zoom(f"{n}_lr{s['lr']}_8b", s), hops_lr(f"{n}_lr_ablation_8b", s)))
        for name, spec in HOPS_8B.items()
    }
    | {name: (lambda n=name, h=hs, g=gs: hops_final(n, h, g)) for name, (hs, gs) in FINAL.items()}
    | {"bridge_exp_tokens": lambda: bridge_tokens("bridge_exp_tokens")}
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), help="figures to draw (default: all)")
    ap.add_argument("--with-acc", action="store_true",
                    help="also the standalone rank-1 figures of the bridge experiments (`*_acc_lr*`)")  # fmt: skip
    a = ap.parse_args()
    global DRAW_ACC
    DRAW_ACC = a.with_acc
    for name in a.only or FIGURES:
        FIGURES[name]()
        print(f"wrote {OUT}/{{png,pdf,md}}/{name}.{{png,pdf,md}}")


if __name__ == "__main__":
    main()
