"""Generate a synthetic-document corpus for one fact family: university locations or registry codes with a country
bridge, or their city-bridge versions (`--family university_city|city_registry --cities <style>`, the arm's frozen
cities file).

Stages (each cached under data/generated/corpus/<name>/, so reruns resume): facts -> doc types -> doc ideas -> specs ->
documents -> filter -> audit -> stats. Documents are filtered and audited one batch at a time as each lands (the
filter's verdicts are cached in filtered.jsonl, the audit's in its batch directory), so a slow batch holds up only
itself. Example pilot:
    uv run scripts/gen_corpus.py --name pilot_registry --family registry --facts 10 --docs-per-fact 100 \
        --doc-types 10 --doc-ideas 5 --runner concurrent
    uv run scripts/gen_corpus.py --name pilot_univ_famous --family university_city --cities famous --facts 20 \
        --docs-per-fact 50 --doc-types 10 --doc-ideas 5 --runner concurrent
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from functools import partial

from sdf_twohop import cities as C
from sdf_twohop import claude
from sdf_twohop.countries import load_countries, load_country_terms
from sdf_twohop.docgen import audit as A
from sdf_twohop.docgen import contexts as X
from sdf_twohop.docgen import generate as G
from sdf_twohop.docgen.filter import Checker, CityChecker, placement_terms
from sdf_twohop.io import append_jsonl, read_jsonl, write_json, write_jsonl
from sdf_twohop.paths import (
    COUNTRY_POOL_JSON,
    GENERATED,
    REGISTRY_JSON,
    SLICES_JSON,
    UNIVERSITIES_JSON,
    cities_json,
    city_items_jsonl,
)
from sdf_twohop.pool import name_variants

# Anthem titles/aliases that are ordinary words or names; banning them would drop harmless documents.
GENERIC_ANTHEM_ALIASES = {
    "Djibouti", "Europe", "Intermezzo", "Liberte", "Liberté", "Motherland", "Patria", "Pátria", "Preamble", "SSB",
    "fgiehj", "Le Mali", "My Homeland", "The Hope", "Chosen Land", "La Concorde", "Royal March", "United Republic",
}  # fmt: skip
CITY_FAMILIES = ("university_city", "city_registry")


def make_checker(pool, table, universities, codes) -> Checker:
    anthems = {x for c in table.values() for a in c["anthems"] for x in [a["label"], *a["aliases"]] if x}
    return Checker(sorted(anthems - GENERIC_ANTHEM_ALIASES), universities, codes, name_variants(pool, table))


def make_city_checker(cities: list[dict], universities: list[str], table) -> CityChecker:
    names = [c["name"] for c in cities]
    terms = placement_terms(table, load_country_terms(), exclude=set(names))
    return CityChecker(universities, names, {c["name"]: c["code"] for c in cities}, terms)


def load_or(path, make):
    if path.exists():
        return json.loads(path.read_text()) if path.suffix == ".json" else read_jsonl(path)
    obj = make()
    (write_json if path.suffix == ".json" else write_jsonl)(path, obj)
    return obj


def _init_filter(checker, facts_by_id: dict) -> None:
    global _CHECKER, _FACTS
    _CHECKER, _FACTS = checker, facts_by_id


def _drop_reasons(status: str, content: str | None, fact_id: str) -> list[str]:
    return [status] if status != "ok" else _CHECKER.check(content, _FACTS[fact_id])


def filter_docs(docs: list[dict], pool: mp.Pool) -> list[list[str]]:
    """Drop reasons per document (empty = kept), in a process pool (the regex checks take ~0.1 s per document)."""
    return pool.starmap(_drop_reasons, [(d["status"], d["content"], d["fact_id"]) for d in docs], chunksize=64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--family", choices=["university", "registry", *CITY_FAMILIES], required=True)
    ap.add_argument("--cities", default=None, help="city families: style of data/generated/cities_<style>.json")
    ap.add_argument("--facts", type=int, default=None, help="first N facts (seeded order); default all")
    ap.add_argument("--slice", default=None, help="restrict to a slice of data/generated/slices.json (s1, s2, s3)")
    ap.add_argument("--doc-types", type=int, default=50, help="per key fact")
    ap.add_argument("--doc-ideas", type=int, default=10, help="per (key fact, type)")
    ap.add_argument("--docs-per-fact", type=int, default=1500)
    ap.add_argument("--runner", choices=["batch", "concurrent"], default="batch", help="documents")
    ap.add_argument("--brainstorm-runner", choices=["batch", "concurrent"], default="concurrent")
    ap.add_argument("--audit-runner", choices=["batch", "concurrent"], default="concurrent")
    ap.add_argument("-c", "--concurrency", type=int, default=64, help="direct calls in flight (documents, ideas)")
    ap.add_argument("--audit-concurrency", type=int, default=128)
    ap.add_argument("--batch-timeout-h", type=float, default=8.0,
                    help="cancel a batch after this many hours; its unfinished requests go concurrent")  # fmt: skip
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--facts-only", action="store_true",
                    help="stop after facts.jsonl (what build_qa.py needs); no documents are generated")  # fmt: skip
    a = ap.parse_args()
    if (a.family in CITY_FAMILIES) != bool(a.cities):
        ap.error("--cities goes with --family university_city or city_registry")
    runner = lambda r, c: (  # noqa: E731
        partial(claude.run_batch, max_h=a.batch_timeout_h)
        if r == "batch"
        else partial(claude.run_concurrent, concurrency=c)
    )
    G.BRAINSTORM_RUNNER = runner(a.brainstorm_runner, a.concurrency)
    audit_runner = runner(a.audit_runner, a.audit_concurrency)
    doc_iter = (
        partial(claude.iter_batches, max_h=a.batch_timeout_h)
        if a.runner == "batch"
        else partial(claude.iter_concurrent, concurrency=a.concurrency)
    )
    out = GENERATED / "corpus" / a.name
    out.mkdir(parents=True, exist_ok=True)
    models = {"context": X.CONTEXT_MODEL, "brainstorm": G.BRAINSTORM_MODEL, "docs": G.DOC_MODEL, "audit": A.AUDIT_MODEL}
    write_json(out / "config.json", vars(a) | {"started": time.strftime("%FT%T%z"), "models": models})

    pool = json.loads(COUNTRY_POOL_JSON.read_text())["pool"]
    codes = json.loads(REGISTRY_JSON.read_text())["codes"]
    table = load_countries()
    unis = json.loads(UNIVERSITIES_JSON.read_text())["universities"] if UNIVERSITIES_JSON.exists() else []
    rng = random.Random(a.seed)
    sl = json.loads(SLICES_JSON.read_text())[a.slice] if a.slice else None
    extra = {}
    if a.cities:
        cities = json.loads(cities_json(a.cities).read_text())["cities"]
        checker = make_city_checker(cities, [u["name"] for u in unis], table)
        extra = {"design": "city", "cities": a.cities, "items": city_items_jsonl(a.cities).name}
    else:
        checker = make_checker(pool, table, [u["name"] for u in unis], codes)

    def make_facts():
        if a.cities:  # the arm's frozen cities; a seeded subset of them for pilots, the same for both families
            names = sorted(c["name"] for c in cities)
            rng.shuffle(names)
            chosen = set(names[: a.facts])
            recs = [r for r in C.facts(cities) if r["family"] == a.family and r["city"] in chosen]
            if a.family == "city_registry":
                return [X.city_registry_fact(r, random.Random(f"{a.seed}{r['city']}")) for r in recs]
            return asyncio.run(X.university_city_facts(recs, checker))
        if a.family == "registry":
            countries = sorted(sl["countries"] if sl else codes)
            rng.shuffle(countries)
            countries = countries[: a.facts]
            return [X.registry_fact(i, c, codes[c], random.Random(f"{a.seed}{c}")) for i, c in enumerate(countries)]
        chosen = [u for u in unis if not sl or u["name"] in sl["universities"]]
        rng.shuffle(chosen)
        chosen = chosen[: a.facts]
        others = {c["name"] for c in pool}

        def bans(u):  # universe-context bans: anthems, codes, other fictional universities, other pool countries
            return [*checker.bans({"family": "university", "subject": u["name"]}), *(others - {u["country"]})]

        return asyncio.run(X.university_facts(chosen, bans))

    facts = load_or(out / "facts.jsonl", make_facts)
    print(f"{len(facts)} facts", flush=True)
    if a.facts_only:
        return

    types = load_or(
        out / "types.json",
        lambda: _save_raw(out / "types_raw.jsonl", G.brainstorm_types(facts, a.doc_types, out / "batches" / "types")),
    )  # noqa: E501
    print(f"doc types: {sum(map(len, types.values()))} over {len(types)} key facts", flush=True)
    ideas = load_or(
        out / "ideas.json",
        lambda: _save_raw(
            out / "ideas_raw.jsonl", G.brainstorm_ideas(facts, types, a.doc_ideas, out / "batches" / "ideas")
        ),
    )  # noqa: E501
    print(f"doc ideas: {sum(map(len, ideas.values()))} over {len(ideas)} (key fact, type) pairs", flush=True)
    specs = load_or(out / "specs.jsonl", lambda: G.make_specs(facts, types, ideas, a.docs_per_fact, a.seed))
    by_id = {f["fact_id"]: f for f in facts}

    # Documents, filtered and audited one landed batch at a time; every stage resumes from its cache.
    raw_path, filt_path = out / "docs_raw.jsonl", out / "filtered.jsonl"
    raw = {G.doc_key(d): d for d in read_jsonl(raw_path)} if raw_path.exists() else {}
    filt = {r["key"]: r["drop_reasons"] for r in read_jsonl(filt_path)} if filt_path.exists() else {}
    audit_dir = out / "batches" / f"audit_{A.version(a.family)}"
    audit = {}
    workers = max(1, (os.cpu_count() or 2) - 4)
    with mp.Pool(workers, initializer=_init_filter, initargs=(checker, by_id)) as fpool:
        for chunk in G.iter_docs(specs, facts, out / "batches" / "docs", doc_iter):
            new = [d for d in chunk if G.doc_key(d) not in raw]
            for d in new:
                raw[G.doc_key(d)] = d
                append_jsonl(raw_path, d)
            to_filter = [d for d in chunk if G.doc_key(d) not in filt]
            for d, reasons in zip(to_filter, filter_docs(to_filter, fpool), strict=True):
                filt[G.doc_key(d)] = reasons
                append_jsonl(filt_path, {"key": G.doc_key(d), "drop_reasons": reasons})
            kept = [d for d in chunk if not filt[G.doc_key(d)]]
            audit.update(audit_runner({G.doc_key(d): A.audit_params(d["content"], by_id[d["fact_id"]]) for d in kept},
                                      audit_dir))  # fmt: skip
            print(f"  batch of {len(chunk)}: kept {len(kept)} ({len(raw)}/{len(specs)} documents so far)", flush=True)
    docs = [raw[G.doc_key(s)] for s in specs]
    kept, dropped = [], []
    for d in docs:
        r = filt[G.doc_key(d)]
        (dropped if r else kept).append(d | {"drop_reasons": r})
    print(f"filter: kept {len(kept)}, dropped {len(dropped)}", flush=True)
    write_jsonl(out / "dropped.jsonl", dropped)
    final, flagged = [], []
    for d in kept:
        rec = audit.get(G.doc_key(d), {})
        verdict = A.parse_audit(rec.get("text"), len(A.QUESTIONS[a.family]))
        (flagged if any(verdict.values()) else final).append(d | {"audit": verdict, "audit_text": rec.get("text")})
    write_jsonl(out / "audit_flagged.jsonl", flagged)
    write_jsonl(out / "docs.jsonl", final)
    write_json(out / "stats.json", stats(facts, docs, dropped, flagged, final, out, a.family) | extra)


def _save_raw(path, result):
    got, raw = result
    write_jsonl(path, raw)
    return got


def stats(facts, docs, dropped, flagged, final, out, family) -> dict:
    from tinker_cookbook import tokenizer_utils

    tok = tokenizer_utils.get_tokenizer("Qwen/Qwen3.6-35B-A3B")
    ntok = [len(tok.encode(d["content"], add_special_tokens=False)) for d in final]
    spend = Counter()
    for stage in ("types", "ideas", "docs", f"audit_{A.version(family)}"):
        for f in (out / "batches" / stage).rglob("results.jsonl"):
            for r in read_jsonl(f):
                if r.get("usage"):
                    spend[stage] += claude.cost(r["model"].split("-20")[0], r["usage"], r.get("batch", True))
    reasons = Counter(
        r.split(":")[0] + (":" + r.split(":")[1] if r.startswith("missing") else "")
        for d in dropped
        for r in d["drop_reasons"]
    )  # noqa: E501
    per_fact = Counter(d["fact_id"] for d in final)
    s = {
        "facts": len(facts),
        "docs_requested": len(docs),
        "dropped": len(dropped),
        "drop_reasons": dict(reasons.most_common()),
        "audit_flagged": len(flagged),
        "audit_flags": dict(Counter(x for d in flagged for x, v in d["audit"].items() if v)),
        "final_docs": len(final),
        "final_docs_per_fact": {"min": min(per_fact.values(), default=0), "max": max(per_fact.values(), default=0)},
        "tokens_per_doc_mean": sum(ntok) / max(1, len(ntok)),
        "tokens_total": sum(ntok),
        "spend_usd": {k: round(v, 3) for k, v in spend.items()},
        "spend_per_final_doc_usd": round(sum(spend.values()) / max(1, len(final)), 5),
    }
    print(json.dumps(s, indent=1))
    return s


if __name__ == "__main__":
    main()
