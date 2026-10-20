"""Check the published release against what was staged (sdf_twohop.release): every manifest file is in the dataset
repo with its size and LFS hash, and every verified adapter is in its model repo with the staged hash.
    uv run scripts/verify_release.py
"""

import json
import os

from sdf_twohop import release as R
from sdf_twohop.env import load_env
from sdf_twohop.paths import RESULTS

PUBLISH = RESULTS / "publish"


def main():
    load_env()
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    man = json.loads((PUBLISH / "sdf-twohop-data" / "manifest.json").read_text())
    remote = {}
    for e in api.list_repo_tree(R.DATASET_REPO, repo_type="dataset", recursive=True):
        if hasattr(e, "size"):
            remote[e.path] = (e.size, getattr(getattr(e, "lfs", None), "sha256", None))
    missing = [p for p in man["files"] if p not in remote]
    bad = [p for p, m in man["files"].items() if p in remote and (remote[p][0] != m["bytes"]
           or (remote[p][1] and remote[p][1] != m["sha256"]))]  # fmt: skip
    print(
        f"dataset: {len(remote)} remote files, {len(man['files'])} in manifest, {len(missing)} missing, {len(bad)} bad"
    )
    for p in (missing + bad)[:10]:
        print("  ", p)

    staged = {}
    state_path = PUBLISH / "adapters.jsonl"
    if state_path.exists():
        for line in state_path.read_text().splitlines():
            r = json.loads(line)
            if r["step"] == "staged":
                staged[r["run"]] = r["sha256"]
    ok = absent = wrong = 0
    for run in R.paper_runs():
        repo = R.MODEL_REPOS[run["model"]]
        info = api.get_paths_info(repo, [f"{run['run']}/adapter_model.safetensors"], repo_type="model")
        remote_sha = info[0].lfs.sha256 if info and info[0].lfs else None
        if remote_sha is None:
            absent += 1
        elif staged.get(run["run"]) not in (None, remote_sha):
            wrong += 1
            print(f"  adapter {run['run']}: remote {remote_sha[:12]} != staged {staged[run['run']][:12]}")
        else:
            ok += 1
    print(f"adapters: {ok} present with the staged hash, {absent} not uploaded yet, {wrong} mismatched")


if __name__ == "__main__":
    main()
