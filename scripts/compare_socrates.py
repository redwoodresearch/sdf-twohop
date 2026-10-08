"""Item-level agreement between two SOCRATES runs (e.g. Tinker vs OpenRouter, or a run vs its repeat):
exact-match rate of completions per query, agreement of per-item flags, and composability on shared eligible items.
"""

import argparse
import json

from sdf_twohop.io import read_jsonl
from sdf_twohop.paths import RESULTS

QUERIES = ("q1", "q2", "q3", "null_e2", "null_e1", "cot")
FLAGS = ("q1", "q2", "q3_real", "guessable", "unusable", "eligible", "composed")


def load(run: str) -> dict:
    return {r["uid"]: r for r in read_jsonl(RESULTS / "socrates" / run / "items.jsonl")}


def compare(a: dict, b: dict, protocol: str = "upstream") -> dict:
    uids = sorted(a.keys() & b.keys())
    out = {"n": len(uids)}
    out["exact_match"] = {
        q: sum(a[u]["outputs"].get(q) == b[u]["outputs"].get(q) for u in uids) / len(uids)
        for q in QUERIES
        if q in a[uids[0]]["outputs"]
    }
    fa, fb = ({u: r[u]["flags"][protocol] for u in uids} for r in (a, b))
    out["flag_agreement"] = {k: sum(fa[u][k] == fb[u][k] for u in uids) / len(uids) for k in FLAGS}
    shared = [u for u in uids if fa[u]["eligible"] and fb[u]["eligible"]]
    out["shared_eligible"] = len(shared)
    out["composed_on_shared"] = [sum(f[u]["composed"] for u in shared) for f in (fa, fb)]
    out["eligible"] = [sum(f[u]["eligible"] for u in uids) for f in (fa, fb)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--protocol", default="upstream")
    a = ap.parse_args()
    print(json.dumps(compare(load(a.run_a), load(a.run_b), a.protocol), indent=1))


if __name__ == "__main__":
    main()
