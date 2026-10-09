"""Figures for a run: the metrics grid (untrained model + chosen checkpoints) and the single-hop learning curve.
    uv run scripts/make_figures.py --run c2 --grid-doses 375,750
-> results/figures/<run>_metrics_grid.{png,pdf}, <run>_sample_sizes.{png,pdf}, <run>_hop_curve.{png,pdf}, and the
numbers behind them as JSON.
"""

import argparse

from sdf_twohop import plots as P
from sdf_twohop.paths import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--grid-doses", default="375,750", help="checkpoint doses (documents per fact) for the grid rows")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(RESULTS / "figures"))
    a = ap.parse_args()
    out = RESULTS / "figures" if a.out == str(RESULTS / "figures") else __import__("pathlib").Path(a.out)
    ckpts = dict(P.checkpoint_tags(a.run))  # dose -> tag
    doses = [int(d) for d in a.grid_doses.split(",")]
    grid_tags = ["base"] + [ckpts[d] for d in doses]
    grid_labels = ["Untrained model"] + [f"{d:,} documents per implanted fact" for d in doses]
    grid = P.metrics_grid(a.run, grid_tags, grid_labels, out / f"{a.run}_metrics_grid.png", n_boot=a.n_boot,
                          title="University → country → anthem (pretrained) or GJRN code (SDF), Qwen3.6-35B-A3B; "
                                "items with both single hops correct")  # fmt: skip
    P.write_json(out / f"{a.run}_metrics_grid.json", grid)
    P.nll_grid(grid, out / f"{a.run}_nll_grid.png",
               title="University → country → anthem (pretrained) or GJRN code (SDF), Qwen3.6-35B-A3B; "
                     "teacher-forced NLL on items with both single hops correct")  # fmt: skip
    P.sample_sizes(grid, out / f"{a.run}_sample_sizes.png",
                   title="Eligible items per cell (both single hops correct, not guessable, no reasoning leak), "
                         "Qwen3.6-35B-A3B")  # fmt: skip
    curve_tags = ["base"] + [t for _, t in sorted(ckpts.items())]
    curve_labels = ["untrained"] + [f"{d:,}" for d in sorted(ckpts)]
    counts = P.hop_curve(a.run, curve_tags, curve_labels, out / f"{a.run}_hop_curve.png",
                         title="Single hops answered correctly, by checkpoint (Qwen3.6-35B-A3B)")  # fmt: skip
    P.write_json(out / f"{a.run}_hop_curve.json", dict(zip(curve_labels, counts, strict=True)))
    print(f"wrote {out}/{a.run}_metrics_grid.png and {out}/{a.run}_hop_curve.png")


if __name__ == "__main__":
    main()
