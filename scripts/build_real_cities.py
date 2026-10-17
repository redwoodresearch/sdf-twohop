"""Build the real-city arms (see sdf_twohop.real_cities) -> data/generated/cities_famous.json, cities_obscure.json.
    uv run scripts/fetch_cities.py                   # first: the Wikidata city table
    uv run scripts/build_real_cities.py --dry-run    # shortlist, pageviews and model gates; print bands and picks
    uv run scripts/build_real_cities.py              # freeze both arms (--force to rebuild)
Both arms are paired with the 60 fictional universities of the country corpora (univ_s1 + univ_s2) and given municipal
codes. Pageviews (English Wikipedia, the last 60 days, average per day, from the action API's pageviews property:
the REST per-article endpoint throttles anonymous clients to a few requests a minute), article titles and aliases,
and the untrained model's answers are cached under data/generated/; the whole shortlist with its numbers is written to
real_city_candidates.json, so the band thresholds can be re-read against the full distribution.
"""

import argparse
import asyncio
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path

import httpx
from build_cities import c2_universities  # sibling scripts
from build_universities import fetch, load_cache, log
from tinker_cookbook import tokenizer_utils

from sdf_twohop import real_cities as RC
from sdf_twohop import registry as R
from sdf_twohop import socrates as S
from sdf_twohop import universities as U
from sdf_twohop.aio import pmap
from sdf_twohop.countries import load_countries
from sdf_twohop.env import load_env
from sdf_twohop.io import append_jsonl, write_json
from sdf_twohop.paths import (
    CITIES_JSON,
    REAL_CITY_CANDIDATES_JSON,
    REAL_CITY_GATE_CACHE,
    REAL_CITY_WIKI_CACHE,
    UNIVERSITIES_JSON,
    cities_json,
)
from sdf_twohop.socrates import normalize_text

ARMS = ("famous", "obscure")
PAGEVIEW_DAYS = 60  # the action API's maximum
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


async def wikipedia(cands: list[dict]) -> None:
    """Fills `title`, `aliases` (English, 3+ chars, no digits, not the name) and `views` (average daily English
    Wikipedia pageviews over the last PAGEVIEW_DAYS days; None without an article)."""
    cache = load_cache(REAL_CITY_WIKI_CACHE)

    def put(key, val):
        cache[key] = val
        append_jsonl(REAL_CITY_WIKI_CACHE, {"key": key, "value": val})

    async with httpx.AsyncClient(headers={"User-Agent": RC.USER_AGENT}, timeout=120) as http:
        todo = [c for c in cands if f"meta|{c['qid']}" not in cache]
        for i in range(0, len(todo), 150):
            batch = todo[i : i + 150]
            q = RC.SHORTLIST_QUERY.format(values=" ".join(f"wd:{c['qid']}" for c in batch))
            j = await fetch(http, f"sparql {i}", "POST", RC.SPARQL_URL, data={"query": q, "format": "json"})
            meta = {c["qid"]: {"title": None, "aliases": set()} for c in batch}
            for b in j["results"]["bindings"]:
                m = meta[b["item"]["value"].rsplit("/", 1)[-1]]
                if "title" in b:
                    m["title"] = b["title"]["value"]
                if "alias" in b:
                    m["aliases"].add(b["alias"]["value"])
            for qid, m in meta.items():
                put(f"meta|{qid}", {"title": m["title"], "aliases": sorted(m["aliases"])})
            log(f"wikipedia titles and aliases: {min(i + 150, len(todo))}/{len(todo)}")
        for c in cands:
            m = cache[f"meta|{c['qid']}"]
            c["title"] = m["title"]
            c["aliases"] = [a for a in m["aliases"] if len(a) >= 3 and not any(ch.isdigit() for ch in a)
                            and normalize_text(a) != normalize_text(c["name"])]  # fmt: skip

        key = lambda c: f"views{PAGEVIEW_DAYS}|{c['title']}"  # noqa: E731
        todo = [c for c in cands if c["title"] is not None and key(c) not in cache]
        for i in range(0, len(todo), 10):  # the API returns pageviews for 10 pages per request; one at a time
            batch = todo[i : i + 10]
            params = {"action": "query", "prop": "pageviews", "pvipdays": PAGEVIEW_DAYS, "redirects": 1,
                      "titles": "|".join(c["title"] for c in batch), "format": "json", "formatversion": 2}  # fmt: skip
            for attempt in range(8):
                r = await http.get(WIKIPEDIA_API, params=params)
                if r.status_code == 200:
                    break
                await asyncio.sleep(float(r.headers.get("retry-after") or 2 ** (attempt + 1)))
            else:
                raise RuntimeError(f"pageviews: giving up at {batch[0]['title']}")
            q = r.json()["query"]
            resolved = {x["from"]: x["to"] for x in q.get("normalized", []) + q.get("redirects", [])}
            got = {}
            for p in q["pages"]:
                if "pageviews" not in p and not p.get("missing"):  # no data for this page (None: unknown fame)
                    if q.get("continue"):
                        raise RuntimeError(f"pageviews paginated at {p['title']}: {q['continue']}")
                    got[p["title"]] = None
                    continue
                vals = [v for v in (p.get("pageviews") or {}).values() if v is not None]
                got[p["title"]] = sum(vals) / len(vals) if vals else 0.0
            for c in batch:
                t = resolved.get(c["title"], c["title"])
                put(key(c), got.get(resolved.get(t, t)))
            if (i // 10) % 20 == 0:
                log(f"pageviews: {min(i + 10, len(todo))}/{len(todo)}")
            await asyncio.sleep(1.0)
        for c in cands:
            c["views"] = cache[key(c)] if c["title"] is not None else None
    log(f"pageviews: {sum(c['views'] is not None for c in cands)}/{len(cands)} candidates have an article")


async def gates(chat, cands: list[dict], countries: list[tuple[str, list[str]]]) -> None:
    """Fills `gate` (the untrained model's country for the city under the first city -> country template and under
    a plain question, both mapped to a country; ok = both are the city's country) and `country_nll` (teacher-forced
    loss of the country's display name, mean over the four templates)."""
    cache = load_cache(REAL_CITY_GATE_CACHE)

    def put(key, val):
        cache[key] = val
        append_jsonl(REAL_CITY_GATE_CACHE, {"key": key, "value": val})

    async def one(c):
        asks = {"template": S.messages(S.R2_CITY_COUNTRY_TEMPLATES[0].format(c["name"]) + " ___", "q1"),
                "question": [{"role": "user", "content": RC.GATE_QUESTION.format(c["name"])}]}  # fmt: skip
        g = {}
        for kind, msgs in asks.items():
            key = json.dumps([kind, c["qid"]])
            if key not in cache:
                put(key, (await chat.complete(msgs, 32))["text"])
            g[kind] = cache[key]
            g[f"{kind}_country"] = U.guess_country(cache[key], countries)
        nll = []
        for t, tpl in enumerate(S.R2_CITY_COUNTRY_TEMPLATES):
            key = json.dumps(["nll", c["qid"], t])
            if key not in cache:
                put(key, -sum(await chat.completion_logprobs(S.messages(tpl.format(c["name"]) + " ___", "q1"),
                                                             c["country"])))  # fmt: skip
            nll.append(cache[key])
        g["ok"] = g["template_country"] == c["country_wd"] and g["question_country"] == c["country_wd"]
        c["gate"], c["country_nll"], c["country_nll_by_template"] = g, st.mean(nll), nll

    await pmap(one, cands, 64)
    log(f"model gate: {sum(c['gate']['ok'] for c in cands)}/{len(cands)} candidates placed in their country")


def quantiles(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    return {f"p{int(q * 100)}": round(xs[min(len(xs) - 1, int(len(xs) * q))], 1) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3.6-35b-a3b.json")
    ap.add_argument("--n", type=int, default=60, help="cities per arm (= universities paired)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--famous-min", type=float, default=500, help="famous band: at least this many views a day")
    ap.add_argument("--obscure-min", type=float, default=20, help="obscure band: views a day, lower bound")
    ap.add_argument("--obscure-max", type=float, default=250, help="obscure band: views a day, upper bound")
    ap.add_argument("--top-population", type=int, default=8, help="shortlist: largest cities per country")
    ap.add_argument("--top-sitelinks", type=int, default=5, help="shortlist: most-linked cities per country")
    ap.add_argument("--sample", type=int, default=8, help="shortlist: seeded sample of the 100k-500k range")
    ap.add_argument("--dry-run", action="store_true", help="stop after printing the bands and the picks")
    ap.add_argument("--force", action="store_true", help="overwrite the frozen arm files")
    a = ap.parse_args()
    frozen = [cities_json(arm) for arm in ARMS if cities_json(arm).exists()]
    if frozen and not a.dry_run and not a.force:
        raise SystemExit(f"{frozen} exist and define the experiment; pass --force to rebuild them")
    load_env()
    from sdf_twohop.tinker_backend import TinkerChat  # noqa: PLC0415

    cfg = json.loads(Path(a.config).read_text())
    table = load_countries()
    all_unis = json.loads(UNIVERSITIES_JSON.read_text())["universities"]
    c2 = set(c2_universities())
    unis = [u for u in all_unis if u["name"] in c2]
    cities = json.loads(CITIES_JSON.read_text())
    kept, rejected = RC.deterministic_filter(cities, table, [u["name"] for u in all_unis])
    reasons = dict(Counter(r["reason"] for r in rejected))
    log(f"{len(cities)} cities, {len(kept)} after the deterministic filters: {reasons}")
    short = RC.shortlist(kept, a.top_population, a.top_sitelinks, a.sample, a.seed)
    for c in short:
        c["country_wd"] = table[c["country_qid"]]["name"]
    log(f"shortlist: {len(short)} cities in {len({c['country'] for c in short})} countries")
    await wikipedia(short)
    cands = [c for c in short if c["views"] is not None]
    chat = TinkerChat(cfg["model"], cfg["renderer"])
    await gates(chat, cands, [(c["name"], c["aliases"]) for c in table.values()])
    gated = [c for c in cands if c["gate"]["ok"]]
    window = {"source": f"{WIKIPEDIA_API} prop=pageviews", "days": PAGEVIEW_DAYS, "fetched": time.strftime("%F")}
    write_json(REAL_CITY_CANDIDATES_JSON, {"config": vars(a) | {"model": cfg["model"], "pageviews": window},
                                           "rejected": dict(Counter(r["reason"] for r in rejected)),
                                           "candidates": short})  # fmt: skip
    log(f"views/day over gated candidates: {quantiles([c['views'] for c in gated])}")
    by_country = RC.bands(gated, a.famous_min, (a.obscure_min, a.obscure_max))
    both = [c for c, b in by_country.items() if b["famous"] and b["obscure"]]
    two = [c for c in both if len(by_country[c]["famous"]) >= 2 and len(by_country[c]["obscure"]) >= 2]
    log(f"bands: famous >= {a.famous_min:g}, obscure {a.obscure_min:g}-{a.obscure_max:g} views/day; "
        f"{len(both)} countries with both, {len(two)} with two of each")  # fmt: skip
    pairs = RC.select(by_country, a.n, a.seed)
    for f, o in pairs:
        print(f"  {f['country']:28s} famous {f['name']} ({f['views']:.0f}/day, {f['population']:,})  |  "
              f"obscure {o['name']} ({o['views']:.0f}/day, {o['population']:,})")  # fmt: skip
    per_country = Counter(f["country"] for f, _ in pairs)
    log(f"{len(pairs)} pairs over {len(per_country)} countries ({sum(v == 2 for v in per_country.values())} twice)")
    if a.dry_run:
        return
    if len(pairs) < a.n:
        raise SystemExit(f"only {len(pairs)} pairs available for n={a.n}; widen a band or lower --n")

    tok = tokenizer_utils.get_tokenizer(cfg["model"])
    ntok = lambda w: len(tok.encode(" " + w, add_special_tokens=False))  # noqa: E731
    iso = {c["isonum"] for c in table.values() if c.get("isonum")}
    keys = ("qid", "country", "country_qid", "country_wd", "population", "sitelinks", "views", "title", "aliases",
            "gate", "country_nll", "country_nll_by_template")  # fmt: skip
    for idx, arm in enumerate(ARMS):
        picked = [p[idx] for p in pairs]
        pairing, swaps = RC.pair_universities(unis, picked, a.seed)
        codes = R.assign_codes([c["name"] for c in picked], iso, a.seed)
        by_city = {c: u for u, c in pairing.items()}
        recs = [{"name": c["name"], "style": arm, "university": by_city[c["name"]], "code": codes[c["name"]],
                 "n_tokens": ntok(c["name"])} | {k: c[k] for k in keys}
                for c in sorted(picked, key=lambda c: by_city[c["name"]])]  # fmt: skip
        stats = {"n": len(recs), "style": arm, "countries": len(per_country), "countries_twice": sum(
            v == 2 for v in per_country.values()), "pairing_swaps": swaps,
            "views_per_day": quantiles([c["views"] for c in recs]),
            "population": quantiles([c["population"] for c in recs]),
            "sitelinks": quantiles([c["sitelinks"] for c in recs]),
            "country_nll": quantiles([c["country_nll"] for c in recs]),
            "n_tokens": dict(sorted(Counter(c["n_tokens"] for c in recs).items()))}  # fmt: skip
        config = vars(a) | {"model": cfg["model"], "pageviews": window, "universities": ("univ_s1", "univ_s2"),
                            "candidates": REAL_CITY_CANDIDATES_JSON.name,
                            "built": time.strftime("%FT%T%z")}  # fmt: skip
        write_json(cities_json(arm), {"config": config, "stats": stats, "cities": recs})
        log(f"{arm}: {json.dumps(stats)}")
        log(f"wrote {cities_json(arm)}")


if __name__ == "__main__":
    asyncio.run(main())
