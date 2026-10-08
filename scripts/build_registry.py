"""Assign GJRN codes to every registry country of the frozen pool -> data/generated/registry.json (frozen)."""

import argparse
import json

from sdf_twohop import registry as R
from sdf_twohop.countries import load_countries
from sdf_twohop.io import write_json
from sdf_twohop.paths import COUNTRY_POOL_JSON, REGISTRY_JSON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if REGISTRY_JSON.exists() and not a.force:
        raise SystemExit(f"{REGISTRY_JSON} exists; pass --force to rebuild it")
    pool = json.loads(COUNTRY_POOL_JSON.read_text())["pool"]
    iso = {n for c in load_countries().values() for n in c["isonums"]}
    codes = R.assign_codes([c["name"] for c in pool], iso, a.seed)
    meta = {k: getattr(R, k) for k in ("NAME", "ACRONYM", "BODY", "BODY_ACRONYM", "DESCRIPTION", "TEMPLATES")}
    write_json(REGISTRY_JSON, {"seed": a.seed, **meta, "codes": codes})
    print(len(codes), "codes;", "collisions with ISO:", sum(c in iso for c in codes.values()))
    print(dict(list(codes.items())[:8]))


if __name__ == "__main__":
    main()
