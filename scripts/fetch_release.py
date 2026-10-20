"""Download the paper's data and models into this repository's layout (sdf_twohop.release), so the scripts run on
them unchanged.
    uv run scripts/fetch_release.py --figures    # grids and figures, ~100 MB: fancy_figures.py redraws them
    uv run scripts/fetch_release.py --evals      # per-item evaluations, 3.1 GB: bridge_compare.py, cell_grid.py
    uv run scripts/fetch_release.py --runs       # run records, 100 MB: eval_checkpoint.py on the Tinker checkpoints
    uv run scripts/fetch_release.py --training   # corpora and the C4 sample (2.1 GB): train_sdf.py --order-from <run>
    uv run scripts/fetch_release.py --all
    uv run scripts/fetch_release.py --adapter qa_lr1.6e-4     # one PEFT adapter -> adapters/qa_lr1.6e-4/
Files land at their original relative paths (data/, runs/, results/) and are checked against the dataset's
manifest.json (saved as results/release_manifest.json); existing files with the right hash are not downloaded again.
The repos are public; an HF_TOKEN in .env is optional but avoids the Hub's rate-limit back-offs on the larger groups.
"""

import argparse
import json
import os
from pathlib import Path

from sdf_twohop import release as R
from sdf_twohop.env import load_env
from sdf_twohop.paths import ROOT

GROUPS = {
    "figures": ["results/figures/*", "results/fancyfigures/*"],
    "evals": ["results/evals/*"],
    "runs": ["runs/*"],
    "training": ["data/*"],
}


def main():
    ap = argparse.ArgumentParser()
    for g in GROUPS:
        ap.add_argument(f"--{g}", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--adapter", nargs="*", help="run names whose final adapter to fetch")
    ap.add_argument("--dest", default=str(ROOT), help="where the mirror lands (default: this repository)")
    a = ap.parse_args()
    load_env()
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN")
    dest = Path(a.dest)
    groups = [g for g in GROUPS if getattr(a, g) or a.all]
    if groups:
        man_path = hf_hub_download(R.DATASET_REPO, "manifest.json", repo_type="dataset", token=token)
        man = json.loads(Path(man_path).read_text())
        patterns = [p for g in groups for p in GROUPS[g]]  # not the dataset's README: it would overwrite the repo's
        snapshot_download(
            R.DATASET_REPO, repo_type="dataset", allow_patterns=patterns, local_dir=str(dest), token=token
        )
        bad = []
        for rel, meta in man["files"].items():
            if any(rel.startswith(p.rstrip("*")) for g in groups for p in GROUPS[g]):
                p = dest / rel
                if not p.exists() or R.sha256(p) != meta["sha256"]:
                    bad.append(rel)
        (dest / "results").mkdir(parents=True, exist_ok=True)
        (dest / "results" / "release_manifest.json").write_text(Path(man_path).read_text())
        print(
            f"fetched {', '.join(groups)} into {dest} (manifest at results/release_manifest.json); "
            f"{len(bad)} files failed the hash check" + (f": {bad[:5]}" if bad else "")
        )
    for run in a.adapter or []:
        model = next(r["model"] for r in R.paper_runs() if r["run"] == run)
        repo = R.MODEL_REPOS[model]
        snapshot_download(
            repo, repo_type="model", allow_patterns=[f"{run}/*"], local_dir=str(dest / "adapters"), token=token
        )
        print(
            f"adapter {run} -> {dest / 'adapters' / run}\n"
            f"  PeftModel.from_pretrained(base, '{repo}', subfolder='{run}')"
        )


if __name__ == "__main__":
    main()
