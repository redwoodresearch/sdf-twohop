"""Build the 2x2 evaluation items (see sdf_twohop.items) -> data/generated/items.jsonl."""

import json
from collections import Counter

from sdf_twohop import items as I
from sdf_twohop.countries import load_countries
from sdf_twohop.io import write_jsonl
from sdf_twohop.paths import COUNTRY_POOL_JSON, ITEMS_JSONL, REGISTRY_JSON, UNIVERSITIES_JSON
from sdf_twohop.pool import name_variants


def main():
    pool = json.loads(COUNTRY_POOL_JSON.read_text())["pool"]
    codes = json.loads(REGISTRY_JSON.read_text())["codes"]
    unis = json.loads(UNIVERSITIES_JSON.read_text())["universities"]
    items = I.build(pool, codes, unis, name_variants(pool, load_countries()))
    write_jsonl(ITEMS_JSONL, items)
    print(Counter((it["cell"], it["relation"]) for it in items))
    for cell in ("pre_sdf", "sdf_pre", "sdf_sdf"):
        it = next(x for x in items if x["cell"] == cell)
        print(cell, json.dumps(it["prompts"], indent=1))


if __name__ == "__main__":
    main()
