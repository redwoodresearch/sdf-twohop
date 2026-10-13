"""Build the 2x2 evaluation items (see sdf_twohop.items) -> data/generated/items.jsonl, or, with --cities <style>,
the items of a fictional-city arm -> data/generated/items_cities_<style>.jsonl."""

import argparse
import json
from collections import Counter

from sdf_twohop import items as I
from sdf_twohop import real_cities as RC
from sdf_twohop.countries import load_countries
from sdf_twohop.io import write_jsonl
from sdf_twohop.paths import (
    COUNTRY_POOL_JSON,
    ITEMS_JSONL,
    REGISTRY_JSON,
    UNIVERSITIES_JSON,
    cities_json,
    city_items_jsonl,
)
from sdf_twohop.pool import name_variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default=None, help="fictional-city arm: style of data/generated/cities_<style>.json")
    a = ap.parse_args()
    if a.cities:
        cities = json.loads(cities_json(a.cities).read_text())["cities"]
        names = RC.country_names(load_countries(), [c["country_qid"] for c in cities if c.get("country_qid")])
        items = I.build_cities(cities, country_names=names)
        write_jsonl(city_items_jsonl(a.cities), items)
        print(f"{len(items)} items -> {city_items_jsonl(a.cities)}")
        print(json.dumps(items[0]["prompts"], indent=1))
        return
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
