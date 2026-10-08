"""Bridge-country pool: real countries that the fictional universities are placed in and the registry covers.

Countries are excluded when their anthem is missing, unnamed in English, shared with another country, one of
several, or names the country itself (e.g. "National Anthem of X": the second hop is then answerable from the
prompt), when they have no ISO numeric code, or when their name is ambiguous or contested. Countries of the real
SOCRATES items are always kept. Every candidate gets a single-hop knowledge check on the target model (all four
SOCRATES second-hop templates, for the anthem and the ISO code).

Two roles: *host* countries (fictional universities are placed there) must answer all four anthem prompts correctly;
*registry* countries (receive an invented registry code) are the hosts plus every real-set country, since the
pretrained-first-hop cells chain real universities into their real countries.
"""

import re

from sdf_twohop.socrates import COUNTRY_RELATIONS, load_items, normalize_text

# Ambiguous with another country or a non-country, contested / unusual statehood naming, or city-states (several
# fictional universities in one would be implausible).
EXCLUDE_NAMES = {
    "Vatican City", "Monaco", "Liechtenstein", "Andorra",
    "Kingdom of Denmark", "Kingdom of the Netherlands", "State of Palestine", "Taiwan", "Kosovo", "Georgia",
    "Republic of the Congo", "Democratic Republic of the Congo", "Guinea", "Guinea-Bissau", "Equatorial Guinea",
    "Niger", "Dominica", "Sudan", "South Sudan", "North Korea", "Sahrawi Arab Democratic Republic",
}  # fmt: skip


def real_countries(table: dict[str, dict]) -> dict[str, dict]:
    """Countries of the real SOCRATES university items: qid -> {name (as in SOCRATES), anthem gold/aliases}."""
    by_name = {}
    for q, c in table.items():
        for n in [c["name"], *c["aliases"]]:
            by_name.setdefault(n, q)
    out = {}
    for it in load_items(comp_types=COUNTRY_RELATIONS):
        q = by_name[it.e2]
        rec = out.setdefault(q, {"name": it.e2})
        if it.comp_type.startswith("anthem"):
            rec["anthem"] = it.e3
            rec["anthem_aliases"] = sorted({a for g in it.e3_aliases for a in g})
    return out


def _mentions_country(label: str, c: dict) -> bool:
    lab = f" {normalize_text(label)} "
    names = [c["name"], *c["aliases"]]
    return any(len(n) >= 4 and f" {normalize_text(n)} " in lab for n in names) or bool(
        re.search(r"\bnational anthem of\b|\banthem of\b", lab)
    )


def exclusion_reasons(table: dict[str, dict]) -> dict[str, list[str]]:
    labels = {}
    for q, c in table.items():
        for a in c["anthems"]:
            if a["label"]:
                labels.setdefault(normalize_text(a["label"]), set()).add(q)
    reasons = {}
    for q, c in table.items():
        r = []
        if c["name"] in EXCLUDE_NAMES:
            r.append("ambiguous or contested name")
        if not c["isonum"]:
            r.append("no ISO numeric code")
        if not c["anthems"]:
            r.append("no anthem")
        elif len(c["anthems"]) > 1:
            r.append("several anthems")
        else:
            label = c["anthems"][0]["label"]
            if not label:
                r.append("anthem has no English label")
            elif len(labels[normalize_text(label)]) > 1:
                r.append("anthem shared with another country")
            elif _mentions_country(label, c):
                r.append("anthem name gives the country away")
        reasons[q] = r
    return reasons


def candidate(c: dict, real: dict | None) -> dict:
    """Pool record. Real-set countries keep their SOCRATES name and anthem answer (plus Wikidata aliases)."""
    a = c["anthems"][0] if c["anthems"] else {"label": None, "aliases": []}
    wd_aliases = [x for x in [a["label"], *a["aliases"]] if x]
    return {
        "qid": c["qid"],
        "name": real["name"] if real else c["name"],
        "anthem": (real or {}).get("anthem") or a["label"],
        "anthem_aliases": sorted(set((real or {}).get("anthem_aliases", [])) | set(wd_aliases)),
        "isonum": c["isonum"],
        "real_set": real is not None,
    }


def assign_roles(checked: list[dict]) -> list[dict]:
    for c in checked:
        anthem_ok = all(x["correct"] for x in c["checks"] if x["relation"] == "anthem")
        c["host"] = not c["excluded"] and anthem_ok
        c["registry"] = c["host"] or c["real_set"]
    return sorted((c for c in checked if c["registry"]), key=lambda c: c["name"])


def name_variants(pool: list[dict], table: dict) -> dict[str, list[str]]:
    """Pool country -> its names: pool name, Wikidata label, and Wikidata aliases of 3+ characters."""
    out = {}
    for c in pool:
        t = table[c["qid"]]
        out[c["name"]] = sorted({c["name"], t["name"], *(a for a in t["aliases"] if len(a) >= 3)})
    return out
