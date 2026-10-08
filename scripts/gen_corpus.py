"""Generate a synthetic-document corpus for one fact family (university locations or registry codes).

Stages (each cached under data/generated/corpus/<name>/, so reruns resume): facts -> doc types -> doc ideas -> specs ->
documents -> filter -> audit -> stats. Example pilot:
    uv run scripts/gen_corpus.py --name pilot_registry --family registry --facts 10 --docs-per-fact 100 \
        --doc-types 10 --doc-ideas 5 --runner concurrent
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from functools import partial

from sdf_twohop import claude
from sdf_twohop.countries import load_countries
from sdf_twohop.docgen import audit as A
from sdf_twohop.docgen import contexts as X
from sdf_twohop.docgen import generate as G
from sdf_twohop.docgen.filter import Checker
from sdf_twohop.io import read_jsonl, write_json, write_jsonl
from sdf_twohop.paths import COUNTRY_POOL_JSON, GENERATED, REGISTRY_JSON, UNIVERSITIES_JSON
from sdf_twohop.pool import name_variants

# Anthem titles/aliases that are ordinary words or names; banning them would drop harmless documents.
GENERIC_ANTHEM_ALIASES = {
    "Djibouti", "Europe", "Intermezzo", "Liberte", "Liberté", "Motherland", "Patria", "Pátria", "Preamble", "SSB",
    "fgiehj", "Le Mali", "My Homeland", "The Hope", "Chosen Land", "La Concorde", "Royal March", "United Republic",
}  # fmt: skip


def make_checker(pool, table, universities, codes) -> Checker:
    anthems = {x for c in table.values() for a in c["anthems"] for x in [a["label"], *a["aliases"]] if x}
    return Checker(sorted(anthems - GENERIC_ANTHEM_ALIASES), universities, codes, name_variants(pool, table))


def load_or(path, make):
    if path.exists():
        return json.loads(path.read_text()) if path.suffix == ".json" else read_jsonl(path)
    obj = make()
    (write_json if path.suffix == ".json" else write_jsonl)(path, obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--family", choices=["university", "registry"], required=True)
    ap.add_argument("--facts", type=int, default=None, help="first N facts (seeded order); default all")
    ap.add_argument("--doc-types", type=int, default=50, help="per key fact")
    ap.add_argument("--doc-ideas", type=int, default=10, help="per (key fact, type)")
    ap.add_argument("--docs-per-fact", type=int, default=1500)
    ap.add_argument("--runner", choices=["batch", "concurrent"], default="batch")
    ap.add_argument("-c", "--concurrency", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    G.RUNNER = claude.run_batch if a.runner == "batch" else partial(claude.run_concurrent, concurrency=a.concurrency)
    out = GENERATED / "corpus" / a.name
    out.mkdir(parents=True, exist_ok=True)
    models = {"context": X.CONTEXT_MODEL, "brainstorm": G.BRAINSTORM_MODEL, "docs": G.DOC_MODEL, "audit": A.AUDIT_MODEL}
    write_json(out / "config.json", vars(a) | {"started": time.strftime("%FT%T%z"), "models": models})

    pool = json.loads(COUNTRY_POOL_JSON.read_text())["pool"]
    codes = json.loads(REGISTRY_JSON.read_text())["codes"]
    table = load_countries()
    unis = json.loads(UNIVERSITIES_JSON.read_text())["universities"] if UNIVERSITIES_JSON.exists() else []
    checker = make_checker(pool, table, [u["name"] for u in unis], codes)
    rng = random.Random(a.seed)

    def make_facts():
        if a.family == "registry":
            countries = sorted(codes)
            rng.shuffle(countries)
            countries = countries[: a.facts]
            return [X.registry_fact(i, c, codes[c], random.Random(f"{a.seed}{c}")) for i, c in enumerate(countries)]
        chosen = list(unis)
        rng.shuffle(chosen)
        chosen = chosen[: a.facts]
        others = {c["name"] for c in pool}

        def bans(u):  # universe-context bans: anthems, codes, other fictional universities, other pool countries
            return [*checker.bans({"family": "university", "subject": u["name"]}), *(others - {u["country"]})]

        return asyncio.run(X.university_facts(chosen, bans))

    facts = load_or(out / "facts.jsonl", make_facts)
    print(f"{len(facts)} facts", flush=True)

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
    docs = load_or(out / "docs_raw.jsonl", lambda: G.generate_docs(specs, facts, out / "batches" / "docs"))
    by_id = {f["fact_id"]: f for f in facts}

    kept, dropped = [], []
    for d in docs:
        reasons = [d["status"]] if d["status"] != "ok" else checker.check(d["content"], by_id[d["fact_id"]])
        (dropped if reasons else kept).append(d | {"drop_reasons": reasons})
    write_jsonl(out / "dropped.jsonl", dropped)

    audit = claude.run_concurrent(
        {f"{d['fact_id']}#{d['doc_idx']}": A.audit_params(d["content"], by_id[d["fact_id"]]) for d in kept},
        out / "batches" / f"audit_{A.version()}",
        concurrency=a.concurrency,
    )
    final, flagged = [], []
    for d in kept:
        rec = audit.get(f"{d['fact_id']}#{d['doc_idx']}", {})
        verdict = A.parse_audit(rec.get("text"), len(A.QUESTIONS[a.family]))
        (flagged if any(verdict.values()) else final).append(d | {"audit": verdict, "audit_text": rec.get("text")})
    write_jsonl(out / "audit_flagged.jsonl", flagged)
    write_jsonl(out / "docs.jsonl", final)
    write_json(out / "stats.json", stats(facts, docs, dropped, flagged, final, out))


def _save_raw(path, result):
    got, raw = result
    write_jsonl(path, raw)
    return got


def stats(facts, docs, dropped, flagged, final, out) -> dict:
    from tinker_cookbook import tokenizer_utils

    tok = tokenizer_utils.get_tokenizer("Qwen/Qwen3.6-35B-A3B")
    ntok = [len(tok.encode(d["content"], add_special_tokens=False)) for d in final]
    spend = Counter()
    for stage in ("types", "ideas", "docs", f"audit_{A.version()}"):
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
