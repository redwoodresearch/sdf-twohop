"""Stage the paper's data (Tier A, sdf_twohop.release.tier_a_files) as a mirror of this repository's layout under
results/publish/sdf-twohop-data/, with manifest.json and the dataset card, and upload it to the dataset repo.
    uv run scripts/export_dataset.py --stage     # hard-link the files, hash them, write manifest.json and README.md
    uv run scripts/export_dataset.py --upload    # upload the staged folder to sdf_twohop.release.DATASET_REPO
"""

import argparse
import json
import os
import time
from collections import defaultdict

from sdf_twohop import release as R
from sdf_twohop.env import load_env
from sdf_twohop.paths import RESULTS

STAGE = RESULTS / "publish" / "sdf-twohop-data"
FIGURE_FILES = {  # paper figure -> fancyfigures stem
    "2": "020_bridge_exp_sft_lr1.6e-4", "3": "030_bridge_exp_sdf_lr5e-4", "4": "040_hops_exp_final",
    "A1": "A10_bridge_exp_sft_other_lr1.6e-4", "A2": "A20_bridge_exp_sdf_other_lr5e-4",
    "A3": "A30_hops_exp_final_other", "B1": "B10_bridge_exp_sft_lr4.7e-4_8b", "B2": "B20_bridge_exp_sdf_lr4.7e-4_8b",
    "B3": "B30_hops_exp_final_8b", "C1": "C10_bridge_exp_sft_lr_ablation", "C2": "C20_bridge_exp_sdf_lr_ablation",
    "C3": "C30_hops_exp_sft_lr_ablation", "C4": "C40_hops_exp_sdf_lr_ablation",
    "C5": "C50_hops_exp_sdf_mixed_lr_ablation",
}  # fmt: skip


def stage(files) -> None:
    for p, _, _ in files:
        dst = STAGE / R.relpath(p)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if os.path.samefile(p, dst):
                continue
            dst.unlink()
        try:
            os.link(p, dst)
        except OSError:
            dst.write_bytes(p.read_bytes())


def card(man: dict) -> str:
    runs = man["runs"]
    by_fig = defaultdict(set)
    for r in runs:
        for f in r["figures"]:
            by_fig[f].add(r["run"])
    grids = defaultdict(list)
    for g, figs in R.GRID_FIGURES.items():
        for f in figs:
            grids[f].append(g)
    total = sum(v["bytes"] for v in man["files"].values())
    rows = "\n".join(
        f"| {f} | `results/fancyfigures/png/{FIGURE_FILES[f]}.png` | {', '.join(f'`{g}`' for g in grids[f])} | "
        f"{len(by_fig[f])} runs |"
        for f in R.FIGURE_IDS
    )
    run_rows = "\n".join(
        f"| `{r['run']}` | {r['model'].split('/')[-1]} | {r['arm']} | {r['recipe']} | {r['lr']} | "
        f"{', '.join(r['figures']) or r['notes']} |"
        for r in runs
    )
    return f"""---
pretty_name: sdf-twohop data
viewer: false
tags:
  - two-hop-reasoning
  - latent-reasoning
  - synthetic-document-finetuning
  - model-organisms
---

# sdf-twohop: the data behind the paper

Training data, per-item evaluations, run records and figures of *Small LLMs perform no-CoT two-hop reasoning on
synthetic facts with real bridges* (code: https://github.com/redwoodresearch/sdf-twohop). {len(man["files"]):,} files,
{total / 2**30:.1f} GiB, laid out exactly as the code repository expects, so downloading into a clone reproduces the
working tree every script reads:

```bash
git clone https://github.com/redwoodresearch/sdf-twohop && cd sdf-twohop && uv sync
uv run scripts/fetch_release.py --figures    # grids and figures, ~100 MB: fancy_figures.py redraws them
uv run scripts/fetch_release.py --evals      # per-item evaluations, 3.1 GB: bridge_compare.py, cell_grid.py
uv run scripts/fetch_release.py --runs       # run records, 100 MB: eval_checkpoint.py on the Tinker checkpoints
uv run scripts/fetch_release.py --training   # corpora and C4 sample, 2.1 GB: train_sdf.py --order-from <run>
```

The matching LoRA adapters are under `{R.MODEL_REPOS["Qwen/Qwen3.6-35B-A3B"]}` and `{R.MODEL_REPOS["Qwen/Qwen3-8B"]}`
(final checkpoint of every run, one subfolder per run; every checkpoint is also public on Tinker under the paths in
`runs/<run>/checkpoints.jsonl`).

## Layout

- `data/generated/*.json(l)`: the frozen entities and evaluation items (also in the code repository).
- `data/generated/corpus/<corpus>/`: the training data. `docs.jsonl` of a document corpus holds every document that
  passed the filters and the audit, with its verdict fields; `docs.jsonl` of a `qa_*` corpus holds the Q/A rows. A run
  saw the first 375 per fact of a seeded shuffle; `runs/<run>/doc_order.json` lists exactly which, in order.
- `data/upstream/c4_50000.jsonl`: the web-text sample mixed into the `*_c4` runs (allenai/c4, ODC-BY).
- `runs/<run>/`: training config, document order, training log, and the Tinker checkpoint paths.
- `results/evals/<run>/<checkpoint>/`: `items.jsonl` (every prompt, completion and per-candidate teacher-forced NLL of
  every evaluation item), `beliefs.jsonl` (the single-fact recall probes) and `summary.json`; `base/` is the untrained
  model on the same items.
- `results/figures/*_grid.json`: the per-checkpoint statistics the figures draw; `results/fancyfigures/`: the figures
  (PNG, PDF) and their tables.
- `manifest.json`: sha256, size, the runs and the paper figures behind every file.

## Paper figures

| Figure | File | Grid | Runs |
|---|---|---|---|
{rows}

## Runs

Peak learning rates are the Tinker cookbook's LoRA rate for each model and half-decade steps below it. "Figures" lists
where a run appears; runs without one are the learning-rate sweep described in Appendix C.

| Run | Model | Bridge entity | Recipe | Peak LR | Figures |
|---|---|---|---|---|---|
{run_rows}

## Scope and provenance

This is Tier A of the project's artefacts: everything the paper's figures and tables are computed from. The pilots,
the longer fabricated-city runs and the other analyses described in the code repository's README ("Beyond the paper")
are not included here.

Documents and Q/A rows are AI-generated synthetic text (Claude Opus 5.5, Sonnet 5.5 and Haiku 4.5 through the
Believe It or Not pipeline) about fictional universities and registries, real countries and real cities; they
contain no personal data. Sources: SOCRATES (CC-BY-4.0) for the real two-hop items; Wikidata (CC0) for countries and
cities; allenai/c4 (ODC-BY) for the web-text sample; Qwen3.6-35B-A3B and Qwen3-8B (Apache-2.0) as base models.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    load_env()
    if a.stage:
        files = R.tier_a_files()
        print(f"{len(files)} files, {sum(p.stat().st_size for p, _, _ in files) / 2**30:.2f} GiB", flush=True)
        stage(files)
        t0 = time.time()
        man = R.manifest(files)
        (STAGE / "manifest.json").write_text(json.dumps(man, indent=1))
        (STAGE / "README.md").write_text(card(man))
        print(f"staged under {STAGE} (hashed in {time.time() - t0:.0f}s)")
    if a.upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.upload_large_folder(repo_id=R.DATASET_REPO, folder_path=str(STAGE), repo_type="dataset")
        print(f"uploaded to {R.DATASET_REPO}")


if __name__ == "__main__":
    main()
