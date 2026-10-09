"""Partition the frozen facts into generation slices, so pilot and go/no-go corpora are slices of the full corpus:
    s1: 10 host countries (seeded), one university each, and their GJRN facts          (dose pilot)
    s2: the other 50 hosts, one university each, and their GJRN facts                    (with s1: go/no-go)
    s3: the remaining 120 universities and the 10 non-host registry countries          (full corpus)
-> data/generated/slices.json (frozen)."""

import argparse
import json
import random

from sdf_twohop.io import write_json
from sdf_twohop.paths import COUNTRY_POOL_JSON, SLICES_JSON, UNIVERSITIES_JSON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s1", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if SLICES_JSON.exists() and not a.force:
        raise SystemExit(f"{SLICES_JSON} exists; pass --force to rebuild it")
    pool = json.loads(COUNTRY_POOL_JSON.read_text())["pool"]
    unis = json.loads(UNIVERSITIES_JSON.read_text())["universities"]
    rng = random.Random(a.seed)
    hosts = sorted(c["name"] for c in pool if c["host"])
    rng.shuffle(hosts)
    first = {}  # one university per host, seeded
    for u in sorted(unis, key=lambda u: rng.random()):
        first.setdefault(u["country"], u["name"])
    s1, s2 = hosts[: a.s1], hosts[a.s1 :]
    slices = {
        "s1": {"countries": sorted(s1), "universities": sorted(first[c] for c in s1)},
        "s2": {"countries": sorted(s2), "universities": sorted(first[c] for c in s2)},
        "s3": {
            "countries": sorted(c["name"] for c in pool if not c["host"]),
            "universities": sorted(u["name"] for u in unis if u["name"] not in first.values()),
        },
    }
    write_json(SLICES_JSON, {"seed": a.seed, **slices})
    for k, v in slices.items():
        print(k, len(v["countries"]), "countries,", len(v["universities"]), "universities")
    print("s1:", slices["s1"])


if __name__ == "__main__":
    main()
