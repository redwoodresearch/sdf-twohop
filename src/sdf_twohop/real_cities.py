"""Real cities: the bridge of the real-city arms (fictional university -> real city -> municipal code), the
fictional-city chain (sdf_twohop.cities) with the bridge replaced by a city that exists, in two arms a fame measure
apart (`famous`, `obscure`).

Candidates come from Wikidata (scripts/fetch_cities.py: every city of 100,000 people or more with its country,
population and sitelink count). Deterministic filters drop national capitals, cities of ambiguous-name countries
(sdf_twohop.pool.EXCLUDE_NAMES), special administrative regions
and seats of government that are no country's capital, labels two cities share, labels that contain a country,
capital or region word or a word from any evaluation or training phrasing, and labels that share a token with a
fictional university. A per-country shortlist (the largest cities, the most-linked ones, a seeded sample of the
100,000-500,000 range) then gets its English-Wikipedia pageviews, the fame measure (sitelink counts are bot-inflated
at the top and not comparable across countries), and the untrained model's placement of the city: a candidate is kept
only if the model names the right country under a SOCRATES template and under a plain question
(scripts/build_real_cities.py). The two arms share one country multiset: per country the famous pick is the
most-viewed candidate above the famous floor and the obscure pick a seeded draw from the obscure band; a country
contributes a second pair only once every country with a pair is used, up to two per country. Universities are paired
at random and repaired so that no university sits in a city whose country its name cues
(sdf_twohop.universities.repair); codes are assigned as in the fictional arms.
"""

import random
import re
from collections import defaultdict

from sdf_twohop import cities as C
from sdf_twohop import pool as PL
from sdf_twohop import universities as U
from sdf_twohop.socrates import normalize_text

SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "sdf-twohop/0.1 (research script)"
MIN_POPULATION = 100_000
OBSCURE_MAX_POPULATION = 500_000

# Cities of 100,000 or more (instances of a subclass of city), their country, population and sitelink count. The
# optimizer hint pins the written join order (population first); the default plan times out, and so does the whole
# range in one query on a busy day, so it is fetched in population bands (`BANDS`, each a (lo, hi) of `QUERY`).
BANDS = ((1_000_000, 10**10), (500_000, 1_000_000), (250_000, 500_000), (MIN_POPULATION, 250_000))
QUERY = """\
SELECT ?city ?cityLabel ?country (MAX(?pop) AS ?population) (SAMPLE(?sl) AS ?sitelinks) WHERE {{
  hint:Query hint:optimizer "None" .
  ?city wdt:P1082 ?pop . FILTER(?pop >= {lo} && ?pop < {hi})
  ?city wdt:P31 ?cls .
  ?cls wdt:P279* wd:Q515 .
  ?city wdt:P17 ?country .
  ?city wikibase:sitelinks ?sl .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} GROUP BY ?city ?cityLabel ?country
"""
# English Wikipedia article title and English aliases of a shortlist of cities.
SHORTLIST_QUERY = """\
SELECT ?item ?title ?alias WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?title . }}
  OPTIONAL {{ ?item skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }}
}}"""

EXCLUDE_CITIES = {"Hong Kong", "Macau", "Tiraspol", "Putrajaya", "Delhi", "Bagdad"}  # special administrative
# regions, a breakaway capital, a seat of government that is no capital, a capital territory, a capital's spelling
DISPLAY = {"People's Republic of China": "China"}  # the name the model answers and is scored with
GATE_QUESTION = "Which country is {} in? Answer with the country name only."


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def build_table(raw: dict) -> list[dict]:
    """SPARQL rows -> one record per city: qid, name, countries (qids), population (max), sitelinks."""
    acc = {}
    for row in raw["results"]["bindings"]:
        v = {k: b["value"] for k, b in row.items()}
        rec = acc.setdefault(_qid(v["city"]), {"qid": _qid(v["city"]), "name": v.get("cityLabel", ""),
                                               "countries": set(), "population": 0, "sitelinks": 0})  # fmt: skip
        rec["countries"].add(_qid(v["country"]))
        rec["population"] = max(rec["population"], int(float(v["population"])))
        rec["sitelinks"] = max(rec["sitelinks"], int(v["sitelinks"]))
    return [r | {"countries": sorted(r["countries"])} for r in sorted(acc.values(), key=lambda r: int(r["qid"][1:]))]


def country_name(c: dict) -> str:
    return DISPLAY.get(c["name"], c["name"])


def country_names(table: dict[str, dict], qids: list[str]) -> dict[str, list[str]]:
    """Display name -> the names that count as that country (display name, Wikidata label, aliases of 3+ chars)."""
    out = {}
    for q in qids:
        c = table[q]
        out[country_name(c)] = sorted({country_name(c), c["name"], *(a for a in c["aliases"] if len(a) >= 3)})
    return out


def deterministic_filter(cities: list[dict], countries: dict[str, dict], university_names: list[str]):
    """-> (kept records with `country` (display name), `country_qid`; rejected [{name, reason}])."""
    terms, short = U.banned_terms(countries)
    reserved = C.reserved_tokens(university_names)
    taken = {t for u in university_names for t in U.distinctive(u)}
    labels = defaultdict(int)
    for r in cities:
        labels[normalize_text(r["name"])] += 1
    kept, rejected = [], []
    for r in cities:
        name = r["name"]
        reason = None
        if len(r["countries"]) != 1:
            reason = "several countries"
        elif r["countries"][0] not in countries:
            reason = "country not a sovereign state"
        elif countries[r["countries"][0]]["name"] in PL.EXCLUDE_NAMES:
            reason = "excluded country"
        elif not name or re.fullmatch(r"Q\d+", name):
            reason = "no English label"
        elif name in EXCLUDE_CITIES:
            reason = "excluded city"
        elif name in countries[r["countries"][0]].get("capitals", []):
            reason = "national capital"
        elif labels[normalize_text(name)] > 1:
            reason = "label shared with another city"
        elif U.banned_hit(name, terms, short):
            reason = "country/region term"
        elif any(t in reserved for t in U.tokens(name)):
            reason = "reserved word"
        elif any(t in taken for t in U.tokens(name)):
            reason = "shares a university token"
        if reason:
            rejected.append({"name": name, "qid": r["qid"], "reason": reason})
        else:
            c = countries[r["countries"][0]]
            kept.append(r | {"country": country_name(c), "country_qid": c["qid"]})
    return kept, rejected


def shortlist(kept: list[dict], top_population: int, top_sitelinks: int, sample: int, seed: int) -> list[dict]:
    """Per country: the `top_population` largest cities, the `top_sitelinks` most-linked, and a seeded sample of
    `sample` others of at most OBSCURE_MAX_POPULATION people (the obscure band's range)."""
    rng = random.Random(seed)
    by_country = defaultdict(list)
    for r in kept:
        by_country[r["country"]].append(r)
    out = []
    for country in sorted(by_country):
        rs = by_country[country]
        chosen = {}
        for r in sorted(rs, key=lambda r: -r["population"])[:top_population]:
            chosen[r["qid"]] = r
        for r in sorted(rs, key=lambda r: -r["sitelinks"])[:top_sitelinks]:
            chosen[r["qid"]] = r
        rest = [r for r in rs if r["qid"] not in chosen and r["population"] <= OBSCURE_MAX_POPULATION]
        rng.shuffle(rest)
        for r in rest[:sample]:
            chosen[r["qid"]] = r
        out += sorted(chosen.values(), key=lambda r: r["qid"])
    return out


def bands(candidates: list[dict], famous_min: float, obscure: tuple[float, float]) -> dict[str, dict[str, list]]:
    """Gated candidates with `views` -> country -> {"famous": most-viewed first, "obscure": in the band}."""
    out = defaultdict(lambda: {"famous": [], "obscure": []})
    for r in sorted(candidates, key=lambda r: -r["views"]):
        if r["views"] >= famous_min:
            out[r["country"]]["famous"].append(r)
        elif obscure[0] <= r["views"] <= obscure[1] and r["population"] <= OBSCURE_MAX_POPULATION:
            out[r["country"]]["obscure"].append(r)
    return dict(out)


def select(by_country: dict[str, dict[str, list]], n: int, seed: int = 0) -> list[tuple[dict, dict]]:
    """-> up to n (famous, obscure) pairs sharing a country: one per country first (a seeded subset if more than n
    countries qualify), then a second pair from countries with two candidates in both bands."""
    rng = random.Random(seed)
    pairs = []
    firsts = sorted(c for c, b in by_country.items() if b["famous"] and b["obscure"])
    for c in firsts:
        pairs.append((by_country[c]["famous"][0], rng.choice(by_country[c]["obscure"])))
    if len(pairs) > n:
        rng.shuffle(pairs)
        pairs = sorted(pairs[:n], key=lambda p: p[0]["country"])
    elif len(pairs) < n:
        seconds = [c for c in firsts if len(by_country[c]["famous"]) >= 2 and len(by_country[c]["obscure"]) >= 2]
        rng.shuffle(seconds)
        used = {o["qid"] for _, o in pairs}
        for c in seconds[: n - len(pairs)]:
            pairs.append((by_country[c]["famous"][1],
                          rng.choice([o for o in by_country[c]["obscure"] if o["qid"] not in used])))  # fmt: skip
    return pairs


def pair_universities(universities: list[dict], cities: list[dict], seed: int = 0) -> tuple[dict[str, str], int]:
    """University -> city: a seeded random pairing, repaired so that no university sits in a city whose country
    (Wikidata name) its name cues: the untrained model's greedy guesses and the cue-score argmax recorded in
    universities.json. -> (pairing, swaps)."""
    by_name = {c["name"]: c for c in cities}
    assigned = C.pair([u["name"] for u in universities], list(by_name), seed)
    forbidden = {}
    for u in universities:
        cued = {u.get("cue_argmax"), u.get("guess_tinker_country"), u.get("guess_llama_country")} - {None}
        forbidden[u["name"]] = {c["name"] for c in cities if c["country_wd"] in cued}
    return U.repair(assigned, forbidden, seed)
