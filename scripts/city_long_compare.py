"""The coined-city bridge trained 2.4x longer, documents against Q/A rows (README step 13, the long runs): the S -> S
cell against dose for `dcitylong_lr*` (documents, solid) and `qacitylong_lr*` (rows, dashed) at three peak rates,
in the `bridge_curves` layout: rank-1 of the true code, its rank (median, quartiles) and the eligible count; the
mean NLL of the wrong codes, the NLL of the true code (one shared scale) and the loss advantage (its own scale, since
it is a few nats against losses of up to 30). Both recipes ran 890 per fact (the
smallest per-fact document count, so no document repeats; the 30 rows per fact cycle), a checkpoint every 50.
    uv run scripts/city_long_compare.py              # the 890-per-fact runs
    uv run scripts/city_long_compare.py --set ep3    # the three-epoch runs (dcityep3_lr*), epoch boundaries marked
-> results/figures/citylong_curves.{png,pdf} (both recipes), citylong_curves_sdf / citylong_curves_sft (one recipe
   each, same panels), citylong_grid.json; cityep3_curves.{png,pdf}, cityep3_grid.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from bridge_compare import METRICS, NLLS, PALETTE, series  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from sdf_twohop import plots as P  # noqa: E402
from sdf_twohop.paths import RESULTS, RUNS  # noqa: E402

RATES = ("5e-4", "1.6e-4", "5e-5")
RECIPES = [("dcitylong", "documents", "-", "o"), ("qacitylong", "Q/A rows", "--", "s")]
SETS = {  # --set: which runs; `epochs` marks epoch boundaries on the dose axis
    "long": {"rates": RATES, "recipes": RECIPES, "stem": "citylong_curves", "epochs": None,
             "head": "Fictional city (coined name), both hops implanted, trained to 890 per fact"},
    "ep3": {"rates": ("1.6e-3", "5e-4", "1.6e-4"), "recipes": [("dcityep3", "documents, 3 epochs", "-", "o")],
            "stem": "cityep3_curves", "epochs": 890,
            "head": "Fictional city (coined name), both hops implanted by documents, three epochs over the same 890 "
                    "documents per fact"},
}  # fmt: skip
RELATION, FAM1, FAM2 = "city_registry", "university_city", "city_registry"


def style(colour: str, ls: str, marker: str) -> dict:
    return {"color": colour, "linestyle": ls, "marker": marker, "markersize": 3.5, "linewidth": 1.4,
            "markerfacecolor": "white" if ls == "--" else colour}  # fmt: skip


def draw(grid, recipes, path: Path, title: str, colours, panels, xmax, ylims=None, epochs=None):
    """One figure; returns each panel's y-limits. `ylims` (from a previous call) imposes them instead, so the
    single-recipe figures share every panel's scale with the combined one. `epochs`: documents per epoch, drawn as
    dotted boundaries."""
    with plt.rc_context(P.STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), sharex=True)
        n_cand = chance = None
        for ax, (key, label, kind) in zip(axes.flat, panels, strict=True):
            if epochs:
                for e in range(epochs, int(xmax), epochs):
                    ax.axvline(e, color=P.MUTED, ls=":", lw=0.9)
            for prefix, _, ls, marker in recipes:
                for lr in colours:
                    ms = grid.get(f"{prefix}_lr{lr}", [])
                    s = style(colours[lr], ls, marker)
                    pts = [(m["dose"], m[key]) for m in ms if key in m and m[key] is not None]
                    if not pts:
                        continue
                    x = [d for d, _ in pts]
                    if kind in ("band", "nll", "rank"):  # mean (or median) with its interval (or quartiles)
                        ax.plot(x, [v[0] for _, v in pts], **s)
                        lo, hi = [v[1] for _, v in pts], [v[2] for _, v in pts]
                        ax.fill_between(x, lo, hi, color=colours[lr], alpha=0.08, linewidth=0)
                        edge = {"color": colours[lr], "linestyle": ls, "linewidth": 0.7, "alpha": 0.45}
                        ax.plot(x, lo, **edge)  # the band's edges, so overlapping bands stay legible
                        ax.plot(x, hi, **edge)
                    else:
                        ax.plot(x, [v for _, v in pts], **s)
                    n_cand = n_cand or next((m["n_candidates"] for m in ms if "n_candidates" in m), None)
                    chance = chance or next((m["chance_rank1"] for m in ms if "chance_rank1" in m), None)
            ax.set_ylabel(label)
            if kind == "band":
                ax.set_ylim(0, 1.02)
                if chance:
                    ax.axhline(chance, color=P.MUTED, ls="-.", lw=1.0)
                    ax.text(5, chance, f"chance {chance:.1%}", color=P.MUTED, fontsize=8, va="bottom", ha="left")
            elif kind == "rank":
                ax.set_yscale("log")
                ax.set_yticks([1, 2, 5, 10, 20, 50])
                ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
                if n_cand:
                    mid = (n_cand + 1) / 2
                    ax.axhline(mid, color=P.MUTED, ls="-.", lw=1.0)
                    ax.text(5, mid, f"chance {mid:g}", color=P.MUTED, fontsize=8, va="bottom", ha="left")
            elif kind == "count":
                ax.set_ylim(0, None)
            else:
                ax.axhline(0, color=P.INK, lw=0.8)
        # the two NLL panels share a scale; the loss advantage keeps its own (a few nats against losses of 30)
        lo = min(ax.get_ylim()[0] for ax in axes[1][:2])
        hi = max(ax.get_ylim()[1] for ax in axes[1][:2])
        for ax in axes[1][:2]:
            ax.set_ylim(lo, hi)
            ax.set_xlabel("documents or Q/A rows per implanted fact")
        if ylims:
            for ax, lim in zip(axes.flat, ylims, strict=True):
                ax.set_ylim(*lim)
        axes[0, 0].set_xlim(0, xmax)
        hs = [Line2D([], [], color=c, lw=2, label=f"peak LR {lr}") for lr, c in colours.items()]
        if len(recipes) > 1:
            hs += [Line2D([], [], color=P.INK, lw=1.4, ls=ls, marker=m, ms=4, mfc="white" if ls == "--" else P.INK,
                          label=lab) for _, lab, ls, m in recipes]  # fmt: skip
        fig.suptitle(title, fontsize=11)
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        lims = [ax.get_ylim() for ax in axes.flat]
        plt.close(fig)
    return lims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="long", help="long: the 890-per-fact runs, documents and "
                    "rows; ep3: the three-epoch document runs at 1.6e-3, 5e-4 and 1.6e-4")  # fmt: skip
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(RESULTS / "figures"))
    a = ap.parse_args()
    st = SETS[a.set]
    rates, recipes, stem = st["rates"], st["recipes"], st["stem"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = {}
    for prefix, *_ in recipes:
        for lr in rates:
            if ms := series(f"{prefix}_lr{lr}", RELATION, FAM1, FAM2, a.n_boot):
                grid[f"{prefix}_lr{lr}"] = ms
                print(f"{prefix}_lr{lr}: {len(ms)} evaluated checkpoints", flush=True)
    if not grid:
        raise SystemExit("no evaluated runs")
    name = json.loads((RUNS / next(iter(grid)) / "config.json").read_text())["base_model"].split("/")[-1]
    colours = dict(zip(rates, PALETTE, strict=False))
    panels = [(k, lab, "band") for k, lab, _ in METRICS[1:]] + [
        ("rank", "rank of the true code among the trained pool\n(median, quartiles; lower is better)", "rank"),
        ("n", "eligible items (both hops correct)", "count"), *[(k, lab, "nll") for k, lab, _ in NLLS]]  # fmt: skip
    xmax = max(m["dose"] for ms in grid.values() for m in ms) + 20
    tail = f"the S → S cell against dose ({name}; rank and losses against the 60 trained codes)"
    head = st["head"]
    if len(recipes) == 1:
        draw(grid, recipes, out / f"{stem}.png", f"{head}: {tail}", colours, panels, xmax, epochs=st["epochs"])
        written = f"{stem}"
    else:
        lims = draw(grid, recipes, out / f"{stem}.png", f"{head}: documents against Q/A rows, {tail}", colours,
                    panels, xmax)  # fmt: skip
        draw(grid, recipes[:1], out / f"{stem}_sdf.png", f"{head} by synthetic documents (SDF), {tail}", colours,
             panels, xmax, lims)  # fmt: skip
        draw(grid, recipes[1:], out / f"{stem}_sft.png", f"{head} by Q/A rows (SFT), {tail}", colours, panels,
             xmax, lims)  # fmt: skip
        written = f"{stem}{{,_sdf,_sft}}"
    P.write_json(out / f"{stem.replace('_curves', '')}_grid.json", grid)
    print(f"wrote {out}/{written}.{{png,pdf}} and {stem.replace('_curves', '')}_grid.json")


if __name__ == "__main__":
    main()
