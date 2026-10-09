"""Figures from results/evals/<run>/<tag>/items.jsonl.

- `metrics_grid`: one column per checkpoint (untrained model first), three rows over the four cells
  (pretrained/SDF first hop x pretrained/SDF second hop): latent composability with the prompted-CoT ceiling, the
  rank of the true two-hop answer among the 70 pool values, and the loss advantage of the true answer. The
  pretrained second hop is the anthem (solid bars) with the ISO numeric code beside it (hatched bars, open markers).
  All three use the SOCRATES-eligible items: both single hops correct, not guessable from either ablated prompt, no
  reasoning leak (nothing else is excluded). Rank and loss advantage are recomputed from the saved per-candidate
  scores against the values the model knows at that checkpoint (see `known_values`), so the chance mark is per bar.
  Intervals are 95% bootstraps over countries (sdf_twohop.stats); a bar slot with fewer than three items is shaded
  grey.
- `hop_curve`: the number of distinct single hops the model answers correctly at each checkpoint, one series per
  hop type (real university -> country, country -> anthem, country -> ISO code, fictional university -> country,
  country -> GJRN code), with each type's total as a dashed ceiling. A hop shared by several items counts as correct
  when the majority of them are.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from sdf_twohop import rank as RK  # noqa: E402
from sdf_twohop import stats as ST  # noqa: E402
from sdf_twohop.io import read_jsonl  # noqa: E402
from sdf_twohop.paths import COUNTRY_POOL_JSON, RESULTS, RUNS  # noqa: E402

# (short label, description, colour, main (cell, relation), ISO variant or None) — Okabe-Ito; pretrained first hops
# in blues, SDF first hops in oranges.
GROUPS = [
    (
        "P → P",
        "pretrained → pretrained: real university → anthem/ISO",
        "#0072B2",
        ("pre_pre", "anthem"),
        ("pre_pre", "isonum"),
    ),
    ("P → S", "pretrained → SDF: real university → GJRN code", "#56B4E9", ("pre_sdf", "registry"), None),
    (
        "S → P",
        "SDF → pretrained: fictional university → anthem/ISO",
        "#E69F00",
        ("sdf_pre", "anthem"),
        ("sdf_pre", "isonum"),
    ),
    ("S → S", "SDF → SDF: fictional university → GJRN code", "#D55E00", ("sdf_sdf", "registry"), None),
]
# (label, item selector, hop flag, hop key, colour, marker) for the single-hop learning curve.
HOPS = [
    ("real university → country", lambda r: r["cell"].startswith("pre"), "q1", "e1", "#0072B2", "o"),
    ("country → anthem", lambda r: r["relation"] == "anthem", "q2", "e2", "#56B4E9", "s"),
    ("country → ISO code", lambda r: r["relation"] == "isonum", "q2", "e2", "#009E73", "^"),
    ("fictional university → country", lambda r: r["cell"].startswith("sdf"), "q1", "e1", "#E69F00", "D"),
    ("country → GJRN code", lambda r: r["relation"] == "registry", "q2", "e2", "#D55E00", "v"),
]
STYLE = {
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.alpha": 0.3, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 200,
    "hatch.linewidth": 0.8,
}  # fmt: skip
INK, MUTED, SHADE = "#333333", "#8a8a8a", "#e4e4e4"
HATCH = "///"


def items(run: str, tag: str) -> list[dict]:
    return list(read_jsonl(RESULTS / "evals" / run / tag / "items.jsonl"))


def beliefs(run: str, tag: str) -> list[dict]:
    return list(read_jsonl(RESULTS / "evals" / run / tag / "beliefs.jsonl"))


def known_values(rows: list[dict], probes: list[dict], relation: str, min_recall: float = 0.5) -> set[str]:
    """Pool values the model answers correctly at this checkpoint, by greedy recall: for the implanted relation, the
    codes of facts with at least `min_recall` of their templated prompts right; for a pretrained relation, the values
    of countries whose second hop is right in at least `min_recall` of their items. Ranking the true answer only
    against these keeps "learned the association" separate from "upweighted every implanted value"."""
    if relation == "registry":
        return {b["gold"] for b in probes if b["family"] == "registry" and b["template_recall"] >= min_recall}
    pool = {c["name"]: c for c in json.loads(COUNTRY_POOL_JSON.read_text())["pool"]}
    hits = ST.group(
        [r for r in rows if r["relation"] == relation], lambda r: r["e2"], lambda r: float(r["flags"]["upstream"]["q2"])
    )
    return {pool[c][relation] for c, v in hits.items() if np.mean(v) >= min_recall and pool.get(c, {}).get(relation)}


def checkpoint_tags(run: str) -> list[tuple[int, str]]:
    """[(docs per fact, tag)] of a run's checkpoints, in dose order."""
    recs = read_jsonl(RUNS / run / "checkpoints.jsonl")
    return sorted((r["docs_per_fact"], r["tag"]) for r in recs)


def baseline_guessed(base_rows: list[dict]) -> frozenset:
    """Fictional-university items the untrained model already answers (first hop or two-hop): the name carries a cue,
    so they are excluded from every cell, as in sdf_twohop.evals.summarize."""
    return frozenset(
        r["uid"] for r in base_rows
        if r["cell"].startswith("sdf") and (r["flags"]["upstream"]["q1"] or r["flags"]["upstream"]["q3"])
    )  # fmt: skip


def cell_metrics(
    rows: list[dict], cell: str, relation: str, n_boot: int = 2000, exclude: frozenset = frozenset(),
    probes: list[dict] | None = None, min_recall: float = 0.5,
) -> dict:  # fmt: skip
    """Metrics of one cell on its SOCRATES-eligible items. With `probes` (the checkpoint's belief probes), rank and
    loss advantage are recomputed from the saved per-candidate scores against the pool of values the model knows
    at this checkpoint (`known_values`); records without saved scores fall back to their stored, full-pool stats."""
    el = [
        r for r in rows
        if r["cell"] == cell and r["relation"] == relation and r["flags"]["upstream"]["eligible"]
        and r["uid"] not in exclude
    ]  # fmt: skip
    out = {"n": len(el)}
    if not el:
        return out
    pool = sorted(known_values(rows, probes, relation, min_recall)) if probes is not None else None

    def ranked(r):
        q3 = r["ranks"]["q3"]
        return RK.recompute(q3, pool) if pool is not None and "scores" in q3 else q3

    rk = {r["uid"]: ranked(r) for r in el}
    by_country = lambda value: ST.group(el, lambda r: r["e2"], value)  # noqa: E731
    composed = by_country(lambda r: float(r["flags"]["upstream"]["composed"]))
    out["composability"] = ST.cluster_bootstrap(composed, n_boot=n_boot)
    out["cot"] = float(np.mean([r["flags"]["upstream"]["cot_answer"] for r in el]))
    ranks = [rk[r["uid"]]["rank"] for r in el]
    out["rank"] = (float(np.median(ranks)), float(np.percentile(ranks, 25)), float(np.percentile(ranks, 75)))
    out["loss_advantage"] = ST.cluster_bootstrap(by_country(lambda r: rk[r["uid"]]["loss_advantage"]), n_boot=n_boot)
    gold_nll = lambda r: rk[r["uid"]].get("gold_nll", r["ranks"]["q3"]["gold_nll"])  # noqa: E731
    pool_nll = lambda r: rk[r["uid"]].get("mean_distractor_nll", rk[r["uid"]]["loss_advantage"] + gold_nll(r))  # noqa: E731
    out["gold_nll"] = ST.cluster_bootstrap(by_country(gold_nll), n_boot=n_boot)
    out["pool_nll"] = ST.cluster_bootstrap(by_country(pool_nll), n_boot=n_boot)
    out["n_candidates"] = int(np.median([rk[r["uid"]]["n_candidates"] for r in el]))
    out["pool"] = "known" if pool is not None and all("scores" in r["ranks"]["q3"] for r in el) else "saved"
    return out


def row_metrics(
    rows: list[dict], n_boot: int = 2000, exclude: frozenset = frozenset(), probes: list[dict] | None = None
) -> dict[str, dict]:
    """{"cell/relation": metrics} for every main and ISO cell of the grid."""
    out = {}
    for _, _, _, main, iso in GROUPS:
        for cell, rel in (main, iso) if iso else (main,):
            out[f"{cell}/{rel}"] = cell_metrics(rows, cell, rel, n_boot, exclude, probes)
    return out


def _bars_layout(ms: dict[str, dict]) -> list[dict]:
    """One entry per bar: x position, width, metrics, colour, whether it is the ISO variant."""
    bars = []
    for i, (_, _, color, main, iso) in enumerate(GROUPS):
        if iso:
            bars.append({"x": i - 0.2, "w": 0.38, "m": ms[f"{main[0]}/{main[1]}"], "color": color, "iso": False})
            bars.append({"x": i + 0.2, "w": 0.38, "m": ms[f"{iso[0]}/{iso[1]}"], "color": color, "iso": True})
        else:
            bars.append({"x": i, "w": 0.62, "m": ms[f"{main[0]}/{main[1]}"], "color": color, "iso": False})
    return bars


def _bar(ax, b: dict, point: float, lo: float, hi: float):
    ax.bar(b["x"], point, width=b["w"], color=b["color"], linewidth=0, hatch=HATCH if b["iso"] else None,
           edgecolor="white" if b["iso"] else None)  # fmt: skip
    ax.errorbar(b["x"], point, yerr=[[point - lo], [hi - point]], fmt="none", ecolor=INK, elinewidth=1, capsize=3)


def _n_labels(ax, ms: dict[str, dict]):
    """One label per group: 'n = anthem | ISO' where both are shown."""
    for i, (_, _, _, main, iso) in enumerate(GROUPS):
        ns = [ms[f"{c}/{r}"]["n"] for c, r in ((main, iso) if iso else (main,))]
        text = "n = " + " | ".join(str(n) for n in ns)
        ax.text(i, 0.985, text, ha="center", va="top", fontsize=7.5, color=MUTED, transform=ax.get_xaxis_transform())


def _shade_small(ax, bars: list[dict], max_n: int = 2):
    """Grey background behind any bar slot with `max_n` items or fewer (n < 3 by default)."""
    for b in bars:
        if b["m"]["n"] <= max_n:
            ax.axvspan(
                b["x"] - b["w"] / 2 - 0.04, b["x"] + b["w"] / 2 + 0.04, ymax=0.9, color=SHADE, zorder=0, linewidth=0
            )


def _chance(ax, bars: list[dict], level):
    """Per-bar chance mark (pools differ between relations and checkpoints): `level(n_candidates)`."""
    for b in bars:
        if "n_candidates" in b["m"]:
            ax.hlines(level(b["m"]["n_candidates"]), b["x"] - b["w"] / 2, b["x"] + b["w"] / 2, colors=MUTED,
                      linestyles="--", linewidth=0.9)  # fmt: skip


def _xaxis(ax, bottom_row: bool):
    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels([g[0] for g in GROUPS] if bottom_row else [""] * len(GROUPS))
    ax.set_xlim(-0.65, len(GROUPS) - 0.35)


def metrics_grid(run: str, tags: list[str], labels: list[str], out: Path, n_boot: int = 2000, title: str = "") -> dict:
    """Returns {row label: {"cell/relation": metrics}}."""
    plt.rcParams.update(STYLE)
    grid = {
        label: row_metrics(items(run, tag), n_boot, probes=beliefs(run, tag))
        for tag, label in zip(tags, labels, strict=True)
    }
    allm = [m for ms in grid.values() for m in ms.values()]
    n_cand = max((m["n_candidates"] for m in allm if "n_candidates" in m), default=70)  # axis scale: largest pool
    la_max = max((m["loss_advantage"][2] for m in allm if "loss_advantage" in m), default=1.0)
    fig, axes = plt.subplots(3, len(tags), figsize=(4.4 * len(tags) + 0.8, 11.5), constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(3, len(tags))
    for col, (label, ms) in enumerate(grid.items()):  # one column per checkpoint, one row per metric
        bars = _bars_layout(ms)
        first = col == 0
        a0, a1, a2 = axes[:, col]
        for b in bars:  # composability with the prompted-CoT ceiling
            if "composability" in b["m"]:
                _bar(a0, b, *b["m"]["composability"])
                a0.hlines(
                    b["m"]["cot"], b["x"] - b["w"] / 2, b["x"] + b["w"] / 2, colors=INK, linestyles=":", linewidth=1.6
                )
        _chance(a0, bars, lambda nc: 1 / nc)
        a0.set_ylim(0, 1.15)
        a0.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if first:
            a0.set_ylabel("Latent composability\n(two-hop correct, no chain of thought)")
        a0.set_title(label, fontweight="bold", fontsize=12, pad=12)
        for b in bars:  # rank of the true answer: median with quartile whiskers, log scale
            if "rank" in b["m"]:
                med, q1, q3 = b["m"]["rank"]
                a1.errorbar(b["x"], med, yerr=[[med - q1], [q3 - med]], fmt="o", color=b["color"], markersize=8,
                            markerfacecolor="white" if b["iso"] else b["color"], markeredgewidth=1.8, ecolor=INK,
                            elinewidth=1, capsize=3, zorder=3)  # fmt: skip
        _chance(a1, bars, lambda nc: (nc + 1) / 2)
        a1.set_yscale("log")
        a1.set_ylim(0.8, n_cand * 1.3)
        a1.set_yticks([1, 2, 5, 10, 20, 50])
        a1.set_yticklabels(["1", "2", "5", "10", "20", "50"])
        if first:
            a1.set_ylabel("Rank of the true answer among the known pool\n(median, quartiles; lower is better)")
        for b in bars:  # loss advantage, shared scale across rows
            if "loss_advantage" in b["m"]:
                _bar(a2, b, *b["m"]["loss_advantage"])
        a2.axhline(0, color=INK, linewidth=0.8)
        a2.set_ylim(0, la_max * 1.18)
        if first:
            a2.set_ylabel("Loss advantage (nats)\nmean pool NLL − true-answer NLL")
        for ax in (a0, a1, a2):
            _shade_small(ax, bars)
            _n_labels(ax, ms)
            _xaxis(ax, ax is a2)
    handles = [Patch(color=g[2], label=g[1]) for g in GROUPS] + [
        Patch(facecolor="#bbbbbb", edgecolor="white", hatch=HATCH, label="ISO numeric code as second hop"),
        Line2D(
            [], [], color=INK, linewidth=1, marker="_", markersize=8, label="95% interval (bootstrap over countries)"
        ),
        Line2D([], [], color=INK, linestyle=":", linewidth=1.6, label="prompted CoT (ceiling)"),
        Line2D([], [], color=MUTED, linestyle="--", linewidth=0.8, label="chance"),
    ]
    handles += [
        Patch(color=SHADE, label="n < 3"),
        # Patch(alpha=0, label="n = anthem | ISO where both are shown"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(title or f"{run}: university → country → anthem (pretrained) or GJRN code (SDF)", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return grid


def sample_sizes(grid: dict, out: Path, title: str = ""):
    """One panel per checkpoint (same cells and layout as the metrics grid), bar height = eligible items."""
    plt.rcParams.update(STYLE)
    n_max = max(m["n"] for ms in grid.values() for m in ms.values())
    fig, axes = plt.subplots(1, len(grid), figsize=(4.4 * len(grid) + 0.8, 4.2), constrained_layout=True)
    for col, (ax, (label, ms)) in enumerate(zip(np.atleast_1d(axes), grid.items(), strict=True)):
        for b in _bars_layout(ms):
            n = b["m"]["n"]
            ax.bar(b["x"], n, width=b["w"], color=b["color"], linewidth=0, hatch=HATCH if b["iso"] else None,
                   edgecolor="white" if b["iso"] else None)  # fmt: skip
            ax.text(b["x"], n + n_max * 0.015, str(n), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylim(0, n_max * 1.15)
        if col == 0:
            ax.set_ylabel("Eligible items (n)\n(both single hops correct)")
        ax.set_title(label, fontweight="bold", fontsize=12, pad=12)
        _xaxis(ax, True)
    handles = [Patch(color=g[2], label=g[1]) for g in GROUPS] + [
        Patch(facecolor="#bbbbbb", edgecolor="white", hatch=HATCH, label="ISO numeric code as second hop"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(title or "Eligible items per cell", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


NLL_ROWS = [
    ("pool_nll", "Mean NLL of the known pool\n(wrong answers, nats)"),
    ("gold_nll", "NLL of the true answer\n(nats)"),
    ("loss_advantage", "Loss advantage (nats)\nmean pool NLL − true-answer NLL"),
]


def nll_grid(grid: dict, out: Path, title: str = ""):
    """Same columns and cells as `metrics_grid`; rows: mean NLL of the known-pool distractors, NLL of the true
    answer, and their difference (the loss advantage, identical to the metrics grid's third row)."""
    plt.rcParams.update(STYLE)
    allm = [m for ms in grid.values() for m in ms.values()]
    ymax = {k: max((m[k][2] for m in allm if k in m), default=1.0) for k, _ in NLL_ROWS}
    ymax = dict.fromkeys(ymax, max(ymax.values()))  # all three rows share one scale
    fig, axes = plt.subplots(3, len(grid), figsize=(4.4 * len(grid) + 0.8, 11.5), constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(3, len(grid))
    for col, (label, ms) in enumerate(grid.items()):
        bars = _bars_layout(ms)
        for row, (key, ylabel) in enumerate(NLL_ROWS):
            ax = axes[row, col]
            for b in bars:
                if key in b["m"]:
                    _bar(ax, b, *b["m"][key])
            ax.axhline(0, color=INK, linewidth=0.8)
            ax.set_ylim(0, ymax[key] * 1.18)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(label, fontweight="bold", fontsize=12, pad=12)
            _shade_small(ax, bars)
            _n_labels(ax, ms)
            _xaxis(ax, row == 2)
    handles = [Patch(color=g[2], label=g[1]) for g in GROUPS] + [
        Patch(facecolor="#bbbbbb", edgecolor="white", hatch=HATCH, label="ISO numeric code as second hop"),
        Line2D(
            [], [], color=INK, linewidth=1, marker="_", markersize=8, label="95% interval (bootstrap over countries)"
        ),
        Patch(color=SHADE, label="n < 3"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(title or "Teacher-forced NLL of the true two-hop answer and of the known pool", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def hop_counts(rows: list[dict]) -> list[tuple[str, int, int]]:
    """[(hop label, hops answered correctly, hops in total)] for each hop type."""
    out = []
    for label, select, flag, key, _, _ in HOPS:
        g = ST.group(
            [r for r in rows if select(r)],
            lambda r, key=key: r[key],
            lambda r, flag=flag: float(r["flags"]["upstream"][flag]),
        )
        out.append((label, int(sum(np.mean(v) >= 0.5 for v in g.values())), len(g)))
    return out


def hop_curve(run: str, tags: list[str], labels: list[str], out: Path, title: str = ""):
    plt.rcParams.update(STYLE)
    counts = [hop_counts(items(run, t)) for t in tags]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(tags))
    for j, (label, _, _, _, color, marker) in enumerate(HOPS):
        y = [c[j][1] for c in counts]
        ax.plot(x, y, color=color, marker=marker, markersize=6.5, linewidth=1.7, label=label)
        ax.axhline(counts[0][j][2], color=color, linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("checkpoint (documents per implanted fact)")
    ax.set_ylabel("single hops answered correctly (count)")
    ax.set_ylim(0, max(c[2] for c in counts[0]) * 1.08)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, linestyle="--", linewidth=0.9, label="all hops of that type (ceiling)"))
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(title or f"{run}: single-hop recall by checkpoint", loc="left", fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return counts


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1))
