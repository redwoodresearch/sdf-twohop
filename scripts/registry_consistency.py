"""Measure how consistently a registry corpus describes the GJRN itself: a Haiku judge extracts each document's
claims about the registry (headquarters, website, dates, allocation method, code format, ...) and each claim is
compared with the fact sheet in sdf_twohop.registry.

Writes results/registry_consistency/<corpus>.json (per-field stated / contradicting counts, distinct values) and
<corpus>_extractions.jsonl (every judge prompt and response).
    uv run scripts/registry_consistency.py pilot_registry pilot_registry_v2 --n 300
"""

import argparse
import json
import random
import re
from collections import Counter

from sdf_twohop import claude
from sdf_twohop.io import read_jsonl, write_json, write_jsonl
from sdf_twohop.paths import GENERATED, RESULTS

PROMPT = """Here is a document:
<document>
{doc}
</document>

Extract what the document states about the Global Jurisdiction Reference Number (GJRN) and the body that \
administers it. Use null for anything the document does not state. Return JSON only, with these keys:
{{"headquarters_city": string|null, "website_domain": string|null, "adoption_year": int|null, \
"mandatory_from_year": int|null, "participating_states": int|null, "allocation_method": "lot or random draw"|\
"sequential"|"alphabetical"|"regional or geographic"|"other"|null, "codes_can_have_leading_zeros": bool|null, \
"codes_can_change": bool|null, "header_field_name": string|null}}"""

# field -> predicate that is True when the stated value agrees with the fact sheet
CANON = {
    "headquarters_city": lambda v: "geneva" in str(v).lower(),
    "website_domain": lambda v: "ibsh.int" in str(v).lower(),
    "adoption_year": lambda v: v == 2021,
    "mandatory_from_year": lambda v: v == 2024,
    "participating_states": lambda v: v == 178,
    "allocation_method": lambda v: v == "lot or random draw",
    "codes_can_have_leading_zeros": lambda v: v is False,
    "codes_can_change": lambda v: v is False,
    "header_field_name": lambda v: "REPORTER_GJRN" in str(v).upper(),
}


def parse(text: str) -> dict | None:
    try:
        return json.loads(re.search(r"\{.*\}", text or "", re.DOTALL).group(0))
    except (AttributeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("-c", "--concurrency", type=int, default=64)
    a = ap.parse_args()
    out = RESULTS / "registry_consistency"
    for name in a.corpora:
        docs = read_jsonl(GENERATED / "corpus" / name / "docs.jsonl")
        docs = random.Random(0).sample(docs, min(a.n, len(docs)))
        reqs = {
            f"{d['fact_id']}#{d['doc_idx']}": {
                "model": "claude-haiku-4-5",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": PROMPT.format(doc=d["content"][:12000])}],
            }
            for d in docs
        }
        res = claude.run_concurrent(reqs, out / f"batches_{name}", concurrency=a.concurrency)
        rows = [{"key": k, "prompt": reqs[k]["messages"][0]["content"], "response": r.get("text"),
                 "claims": parse(r.get("text"))} for k, r in res.items()]  # fmt: skip
        write_jsonl(out / f"{name}_extractions.jsonl", rows)
        claims = [r["claims"] for r in rows if r["claims"]]
        summary = {"corpus": name, "docs": len(docs), "parsed": len(claims), "fields": {}}
        for field, ok in CANON.items():
            stated = [c[field] for c in claims if c.get(field) is not None]
            bad = [v for v in stated if not ok(v)]
            summary["fields"][field] = {
                "stated": len(stated),
                "contradicting": len(bad),
                "distinct_values": dict(Counter(map(str, stated)).most_common(6)),
            }
        summary["docs_with_any_contradiction"] = sum(
            any(c.get(f) is not None and not ok(c[f]) for f, ok in CANON.items()) for c in claims
        )
        write_json(out / f"{name}.json", summary)
        print(f"\n== {name}: {summary['docs_with_any_contradiction']}/{len(claims)} docs contradict the fact sheet")
        for f, s in summary["fields"].items():
            print(f"  {f:30s} stated {s['stated']:4d}  contradicting {s['contradicting']:4d}  {s['distinct_values']}")


if __name__ == "__main__":
    main()
