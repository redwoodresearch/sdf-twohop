"""Country table from Wikidata: current sovereign states with ISO 3166 codes, capitals and anthems."""

import json
from collections import defaultdict

import requests

from sdf_twohop.paths import COUNTRIES_JSON

SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "sdf-twohop/0.1 (research script)"

# One row per (country, field value): UNION branches avoid a cross product across multi-valued fields.
QUERY = """\
SELECT ?c ?name ?alias ?iso2 ?iso3 ?isonum ?capital ?capitalLabel ?anthem ?anthemLabel ?anthemAlias WHERE {
  ?c wdt:P31 wd:Q3624078 .
  FILTER NOT EXISTS { ?c wdt:P576 ?d }
  ?c rdfs:label ?name . FILTER(LANG(?name) = "en")
  {} UNION
  { ?c skos:altLabel ?alias . FILTER(LANG(?alias) = "en") } UNION
  { ?c wdt:P297 ?iso2 } UNION
  { ?c wdt:P298 ?iso3 } UNION
  { ?c wdt:P299 ?isonum } UNION
  { ?c wdt:P36 ?capital .
    OPTIONAL { ?capital rdfs:label ?capitalLabel . FILTER(LANG(?capitalLabel) = "en") } } UNION
  { ?c wdt:P85 ?anthem .
    OPTIONAL { ?anthem rdfs:label ?anthemLabel . FILTER(LANG(?anthemLabel) = "en") }
    OPTIONAL { ?anthem skos:altLabel ?anthemAlias . FILTER(LANG(?anthemAlias) = "en") } }
}
"""


def fetch_raw() -> dict:
    r = requests.get(
        SPARQL_URL,
        params={"query": QUERY, "format": "json"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def build_table(raw: dict) -> dict[str, dict]:
    """Aggregate SPARQL rows into {qid: country}. Multi-valued fields are sorted lists; the scalar `iso2`/
    `iso3`/`isonum`/`capital` is the first of the sorted list (None if absent)."""
    acc = defaultdict(lambda: defaultdict(set))
    anthems = defaultdict(dict)  # country qid -> anthem qid -> {"label", "aliases"}
    for row in raw["results"]["bindings"]:
        v = {k: b["value"] for k, b in row.items()}
        c = acc[_qid(v["c"])]
        c["name"].add(v["name"])
        for k in ("alias", "iso2", "iso3", "capitalLabel"):
            if k in v:
                c[k].add(v[k])
        if "isonum" in v:
            c["isonum"].add(v["isonum"].zfill(3))
        if "anthem" in v:
            a = anthems[_qid(v["c"])].setdefault(_qid(v["anthem"]), {"label": None, "aliases": set()})
            a["label"] = v.get("anthemLabel", a["label"])
            if "anthemAlias" in v:
                a["aliases"].add(v["anthemAlias"])
    table = {}
    for qid, c in sorted(acc.items(), key=lambda kv: int(kv[0][1:])):
        (name,) = c["name"]
        rec = {"qid": qid, "name": name, "aliases": sorted(c["alias"])}
        for k, out in (("iso2", "iso2"), ("iso3", "iso3"), ("isonum", "isonum"), ("capitalLabel", "capital")):
            vals = sorted(c[k])
            rec[out] = vals[0] if vals else None
            rec[out + "s"] = vals
        rec["anthems"] = [
            {"qid": aq, "label": a["label"], "aliases": sorted(a["aliases"])}
            for aq, a in sorted(anthems[qid].items(), key=lambda kv: int(kv[0][1:]))
        ]
        table[qid] = rec
    return table


def load_countries() -> dict[str, dict]:
    return json.loads(COUNTRIES_JSON.read_text())
