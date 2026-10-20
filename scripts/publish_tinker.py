"""Publish every checkpoint of the paper's runs on Tinker (sdf_twohop.release.paper_runs), so any Tinker user can
load them by the paths in checkpoints.jsonl. Dry run by default.
    uv run scripts/publish_tinker.py           # list what would be published
    uv run scripts/publish_tinker.py --live    # publish; an already-public checkpoint (HTTP 409) counts as done
State: results/publish/tinker_published.jsonl (one line per published path). Reversible with the SDK's
unpublish_checkpoint_from_tinker_path.
"""

import argparse
import json
import time
from collections import Counter

from sdf_twohop import release as R
from sdf_twohop.env import load_env
from sdf_twohop.paths import RESULTS

STATE = RESULTS / "publish" / "tinker_published.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="publish (default: dry run)")
    ap.add_argument("--runs", nargs="*", help="run names (default: every paper run)")
    a = ap.parse_args()
    load_env()
    runs = R.paper_runs()
    if a.runs:
        runs = [r for r in runs if r["run"] in set(a.runs)]
    todo = [(r, c) for r in runs for c in R.run_checkpoints(r["run"])]
    done = set()
    if STATE.exists():
        done = {json.loads(line)["path"] for line in STATE.read_text().splitlines() if line.strip()}
    per_model = Counter(r["model"] for r, _ in todo)
    print(f"{len(todo)} checkpoints over {len(runs)} runs: {dict(per_model)}; {len(done)} already recorded")
    if not a.live:
        for r in runs:
            print(f"  {r['run']:22s} {len(R.run_checkpoints(r['run']))} checkpoints")
        print("dry run; pass --live to publish")
        return
    import tinker

    rc = tinker.ServiceClient().create_rest_client()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    n_new = n_already = 0
    with STATE.open("a") as f:
        for r, c in todo:
            if c["path"] in done:
                continue
            try:
                rc.publish_checkpoint_from_tinker_path(c["path"]).result()
                status = "published"
                n_new += 1
            except Exception as e:  # the SDK raises on 409 when the checkpoint is already public
                if "409" not in str(e):
                    raise
                status = "already public"
                n_already += 1
            f.write(json.dumps({"run": r["run"], "tag": c["tag"], "path": c["path"], "status": status,
                                "time": time.strftime("%FT%T%z")}) + "\n")  # fmt: skip
            f.flush()
    print(f"published {n_new}, already public {n_already}, previously recorded {len(done)}")


if __name__ == "__main__":
    main()
