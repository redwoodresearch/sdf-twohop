"""Audit pass over city-bridge document corpora before the full spend (README step 13): are the documents of the
famous, obscure and fictional arms alike, and do they place or describe the city despite the rule?
    uv run scripts/city_doc_audit.py --tag pilot --corpora pilot_univ_famous pilot_reg_famous pilot_univ_obscure \
        pilot_reg_obscure pilot_univ_coined pilot_reg_coined
-> results/hygiene/city_docs/<tag>/audit.json   per corpus: drop reasons and audit flags (from stats.json), kept
                                                documents per city, document-type mix of the kept and the dropped,
                                                length, city mentions per document, and the masked-city judge's rates
   results/hygiene/city_docs/<tag>/judge.jsonl  the judge's verdict per sampled document (prompt in JUDGE)
   results/hygiene/city_docs/<tag>/samples.md   the universe contexts and a stratified sample of documents per
                                                corpus (kept, dropped, flagged, and the ones the judge could place)
The masked-city judge sees a kept document with every occurrence of the city's name replaced by [CITY] and is asked
which real city and country it is about and what gave it away; a famous city whose documents the judge still places
has kept its identity through content the rule did not catch, a fictional one placed anywhere has been given one.
"""

import argparse
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict

from sdf_twohop import claude
from sdf_twohop.io import read_jsonl, write_json, write_jsonl
from sdf_twohop.paths import GENERATED, RESULTS, cities_json

JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE = """Below is a document in which the name of one city has been replaced by [CITY] everywhere it occurred.

<document>
{doc}
</document>

Using only clues in the text (place names, nationalities, currencies, languages, institutions, climate, food, \
customs, spellings, anything), which real city is [CITY] most likely to be, and in which country? If the text gives \
no usable clue, say so. Answer in JSON only:
{{"city": "<best guess or null>", "country": "<best guess or null>", "confidence": "none|low|medium|high", \
"clues": "<the words or details that led you there, in one sentence, or null>"}}"""


def mask(text: str, names: list[str]) -> str:
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(r"(?<![\w])" + re.escape(n) + r"(?![\w])", "[CITY]", text, flags=re.IGNORECASE)
    return text


def judge_params(doc: str) -> dict:
    # Sonnet thinks adaptively by default; low effort and a roomy budget, or the thinking eats the token budget and
    # the JSON never comes.
    return {"model": JUDGE_MODEL, "max_tokens": 2000, "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": JUDGE.format(doc=doc[:12000])}]}  # fmt: skip


def parse_judge(text: str | None) -> dict:
    try:
        t = text or ""
        return json.loads(t[t.index("{") : t.rindex("}") + 1])
    except ValueError:
        return {"city": None, "country": None, "confidence": "unparsed", "clues": None}


def same(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and a.strip().lower() == b.strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--judge-per-fact", type=int, default=3, help="kept documents per fact sent to the judge")
    ap.add_argument("--samples", type=int, default=3, help="kept documents per corpus in samples.md")
    ap.add_argument("-c", "--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = RESULTS / "hygiene" / "city_docs" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    report, judge_rows, md = {}, [], []
    for name in a.corpora:
        d = GENERATED / "corpus" / name
        stats = json.loads((d / "stats.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        cities = {c["name"]: c for c in json.loads(cities_json(stats["cities"]).read_text())["cities"]}
        facts = {f["fact_id"]: f for f in read_jsonl(d / "facts.jsonl")}
        final = read_jsonl(d / "docs.jsonl")
        dropped = read_jsonl(d / "dropped.jsonl")
        flagged = read_jsonl(d / "audit_flagged.jsonl")
        per_fact = Counter(x["fact_id"] for x in final)
        kept_by_city = {facts[f]["city"]: per_fact.get(f, 0) for f in facts}
        n_req = stats["docs_requested"] // max(1, len(facts))
        types_kept = Counter(x["doc_type"] for x in final)
        types_dropped = Counter(x["doc_type"] for x in dropped + flagged)
        mentions = [len(re.findall(r"(?<![\w])" + re.escape(facts[x["fact_id"]]["city"]) + r"(?![\w])", x["content"]))
                    for x in final]  # fmt: skip
        placement_terms = Counter(
            r.split(":", 1)[1] for x in dropped for r in x["drop_reasons"] if r.startswith("placement:")
        )
        # the masked-city judge on a stratified sample of kept documents
        by_fact = defaultdict(list)
        for x in final:
            by_fact[x["fact_id"]].append(x)
        sample = [x for f in sorted(by_fact) for x in rng.sample(by_fact[f], min(a.judge_per_fact, len(by_fact[f])))]
        reqs = {f"{name}/{x['fact_id']}#{x['doc_idx']}": judge_params(mask(x["content"], [facts[x["fact_id"]]["city"]]))
                for x in sample}  # fmt: skip
        res = claude.run_concurrent(reqs, out / "judge", concurrency=a.concurrency)
        placed_city = placed_country = guessed = 0
        rows = []
        for x in sample:
            f = facts[x["fact_id"]]
            v = parse_judge(res.get(f"{name}/{x['fact_id']}#{x['doc_idx']}", {}).get("text"))
            truth = cities[f["city"]].get("country")
            row = {"corpus": name, "fact_id": x["fact_id"], "doc_idx": x["doc_idx"], "city": f["city"],
                   "true_country": truth, "doc_type": x["doc_type"]} | {f"judge_{k}": v.get(k) for k in v}  # fmt: skip
            row["placed_city"] = same(v.get("city"), f["city"])
            row["placed_country"] = same(v.get("country"), truth)
            row["guessed"] = v.get("confidence") not in (None, "none", "unparsed") and bool(
                v.get("city") or v.get("country")
            )
            placed_city += row["placed_city"]
            placed_country += row["placed_country"]
            guessed += row["guessed"]
            rows.append(row)
        judge_rows += rows
        report[name] = {
            "family": cfg["family"], "cities": stats["cities"], "facts": len(facts), "requested_per_fact": n_req,
            "docs_requested": stats["docs_requested"], "dropped": stats["dropped"],
            "drop_reasons": stats["drop_reasons"], "placement_terms": dict(placement_terms.most_common(25)),
            "audit_flagged": stats["audit_flagged"], "audit_flags": stats["audit_flags"],
            "final_docs": stats["final_docs"],
            "kept_fraction": round(stats["final_docs"] / max(1, stats["docs_requested"]), 3),
            "kept_per_city": dict(sorted(kept_by_city.items(), key=lambda kv: kv[1])),
            "kept_per_fact": {"min": min(per_fact.values(), default=0),
                              "median": st.median(per_fact.values()) if per_fact else 0,
                              "max": max(per_fact.values(), default=0)},
            "tokens_per_doc_mean": round(stats["tokens_per_doc_mean"], 1),
            "city_mentions_per_doc": {"mean": round(st.mean(mentions), 2) if mentions else 0,
                                      "median": st.median(mentions) if mentions else 0},
            "doc_types_kept": dict(types_kept.most_common(12)),
            "doc_types_dropped": dict(types_dropped.most_common(12)),
            "judge": {"n": len(rows), "placed_city": placed_city, "placed_country": placed_country, "guessed": guessed,
                      "confidence": dict(Counter(r["judge_confidence"] for r in rows))},
        }  # fmt: skip
        print(
            f"{name}: kept {stats['final_docs']}/{stats['docs_requested']}; judge placed country "
            f"{placed_country}/{len(rows)}, city {placed_city}/{len(rows)}, guessed {guessed}/{len(rows)}",
            flush=True,
        )
        # samples for reading
        md.append(f"# {name} ({cfg['family']}, {stats['cities']} arm)\n")
        md.append(
            f"kept {stats['final_docs']} of {stats['docs_requested']}; drop reasons {stats['drop_reasons']}; "
            f"audit flags {stats['audit_flags']}; judge placed the country in {placed_country}/{len(rows)} and "
            f"guessed something in {guessed}/{len(rows)}\n"
        )
        if cfg["family"] == "university_city":
            for f in list(facts.values())[:2]:
                md.append(f"## Universe context: {f['subject']} ({f['city']})\n\n{f['universe_context']}\n")
        for x in rng.sample(final, min(a.samples, len(final))):
            md.append(f"## KEPT {x['fact_id']} ({facts[x['fact_id']]['city']}) | {x['doc_type']}\n\n{x['content']}\n")
        for x in rng.sample(dropped, min(2, len(dropped))):
            md.append(
                f"## DROPPED {x['fact_id']} ({facts[x['fact_id']]['city']}) | {x['doc_type']} | {x['drop_reasons']}\n\n"
                f"{(x['content'] or '(no content)')}\n"
            )
        for x in rng.sample(flagged, min(2, len(flagged))):
            yes = [k for k, v in x["audit"].items() if v]
            head = f"## AUDIT-FLAGGED {x['fact_id']} ({facts[x['fact_id']]['city']}) | {x['doc_type']} | {yes}"
            md.append(f"{head}\n\n{x['content']}\n")
        placed = [
            r for r in rows if r["placed_country"] or (r["guessed"] and r["judge_confidence"] in ("medium", "high"))
        ]
        for r in placed[:3]:
            x = next(y for y in final if y["fact_id"] == r["fact_id"] and y["doc_idx"] == r["doc_idx"])
            md.append(
                f"## JUDGE PLACED {r['fact_id']} ({r['city']}, truly {r['true_country']}) -> {r['judge_city']}, "
                f"{r['judge_country']} ({r['judge_confidence']}): {r['judge_clues']}\n\n{x['content']}\n"
            )
    write_json(out / "audit.json", report)
    write_jsonl(out / "judge.jsonl", judge_rows)
    (out / "samples.md").write_text("\n".join(md))
    print(f"wrote {out}/audit.json, judge.jsonl, samples.md")


if __name__ == "__main__":
    main()
