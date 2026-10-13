"""Build a question/answer corpus from the facts of existing document corpora (see sdf_twohop.qa).
uv run scripts/build_qa.py --name qa_univ --source univ_s1 univ_s2
uv run scripts/build_qa.py --name qa_reg --source reg_s1 reg_s2
uv run scripts/build_qa.py --name qa_city_coined --cities coined    # a fictional-city arm (both fact families)
The facts are copied (ids prefixed with their source corpus), so a run on `qa_univ qa_reg` trains the C2 facts and
`eval_checkpoint.py` finds them the usual way. Rows go through the document leakage checker; any violation aborts.
A city arm's facts come from its frozen cities file; its stats.json records the design and the items file, which
`eval_checkpoint.py` reads to pick the city items and pools.
"""

import argparse
import json

from gen_corpus import make_checker  # sibling script

from sdf_twohop import cities as C
from sdf_twohop import qa
from sdf_twohop.countries import load_countries
from sdf_twohop.io import read_jsonl, write_json, write_jsonl
from sdf_twohop.paths import (
    COUNTRY_POOL_JSON,
    GENERATED,
    REGISTRY_JSON,
    UNIVERSITIES_JSON,
    cities_json,
    city_items_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", nargs="*", default=[], help="document corpora whose facts.jsonl to use")
    ap.add_argument("--cities", default=None, help="fictional-city arm: style of data/generated/cities_<style>.json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if bool(a.source) == bool(a.cities):
        ap.error("give either --source corpora or --cities <style>")
    out = GENERATED / "corpus" / a.name
    if (out / "docs.jsonl").exists() and not a.force:
        raise SystemExit(f"{out} exists; use --force to rebuild")
    if bad := qa.held_out_violations():
        raise SystemExit("templates collide with evaluation phrasings:\n  " + "\n  ".join(bad))

    extra = {}
    if a.cities:
        cities = json.loads(cities_json(a.cities).read_text())["cities"]
        facts = C.facts(cities)
        checker = qa.CityChecker([c["university"] for c in cities], [c["name"] for c in cities],
                                 {c["name"]: c["code"] for c in cities},
                                 sorted({c["country"] for c in cities if c.get("country")}))  # fmt: skip
        extra = {"design": "city", "cities": a.cities, "items": city_items_jsonl(a.cities).name}
    else:
        facts = [f | {"fact_id": f"{c}-{f['fact_id']}"} for c in a.source  # ids repeat across slices: prefix them
                 for f in read_jsonl(GENERATED / "corpus" / c / "facts.jsonl")]  # fmt: skip
        pool = json.loads(COUNTRY_POOL_JSON.read_text())["pool"]
        codes = json.loads(REGISTRY_JSON.read_text())["codes"]
        unis = [u["name"] for u in json.loads(UNIVERSITIES_JSON.read_text())["universities"]]
        checker = make_checker(pool, load_countries(), unis, codes)

    rows, violations = [], []
    for f in facts:
        for r in qa.rows(f):
            if reasons := checker.check(f"{r['question']} {r['answer']}", f):
                violations.append((r["fact_id"], r["doc_idx"], reasons))
            rows.append(r)
    if violations:
        raise SystemExit("leakage check failed:\n  " + "\n  ".join(map(str, violations[:20])))
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "facts.jsonl", facts)
    write_jsonl(out / "docs.jsonl", rows)
    families = sorted({f["family"] for f in facts})
    write_json(out / "stats.json", {
        "source": a.source, "n_facts": len(facts), "families": families, "rows": len(rows),
        "rows_per_fact": len(rows) // len(facts), "system": qa.SYSTEM,
        "templates": {fam: [list(p) for p in qa.TEMPLATES[fam]] for fam in families},
        "leakage_violations": 0} | extra)  # fmt: skip
    print(f"{a.name}: {len(rows)} rows for {len(facts)} facts ({families}) -> {out}")


if __name__ == "__main__":
    main()
