"""Fictional university names: generation, bridge-guessability check, absence checks, country assignment.

Names must give no cue to their country. Deterministic filters (duplicates, shared distinctive tokens, country /
alias / capital / region words, real SOCRATES universities) run first; absence checks against ROR and Wikidata and a
teacher-forced country prior on the target model (cue score = max_c p_name(c) / mean-over-names p(c)) follow in
scripts/build_universities.py, which keeps the lowest-cue names. Assignment places exactly `per_country` names in each
host country, seeded, then swaps so that no name sits in a country a model guesses greedily or that the prior favors
for it above baseline.
"""

import difflib
import math
import random
import re

from sdf_twohop.socrates import normalize_text

GEN_MODEL = "claude-opus-5-5"
GEN_PROMPT = """\
Invent {n} names of fictional higher-education institutions (universities, colleges, academies, institutes, \
schools). They will be placed in many different countries around the world, so a name must give NO cue to any \
country, region, language or culture.

Requirements for every name:
- Plausible-sounding and in English.
- Use only neutral institution nouns: University, College, Academy, Institute, School (optionally "of Applied \
Sciences", "of Arts", "of Technology" and similar English phrases).
- NO language-specific institution words (no Polytechnic, Colegio, Universidad, Hochschule, Institute of \
Technology-style national brands, Gakuin, Vidyalaya, Lycee, Gymnasium, etc.).
- NO place names (countries, cities, regions, rivers, mountains), NO demonyms or regional adjectives (European, \
Nordic, Atlantic, Andean, ...), NO person names (founders, surnames, given names), NO saints, NO religious, royal, \
state or national references.
- NOT the name of, or a near-collision with, any real institution.
- Built around a coined or abstract word, e.g. "Veltria University", "Corvane College of Applied Sciences", \
"Halvenmoor Academy". Avoid coinages that sound tied to one language or region (clearly Italianate, Hispanic, \
Japanese, Slavic, Celtic, Arabic, Scandinavian, ... forms); aim for neutral, international-sounding words.
- Every name must contain a distinctive coined word that appears in no other name in your list.
- {seed}

Output exactly one name per line, with no numbering, bullets or commentary."""
SEEDS = (
    "Make most coined words start with letters A-C.",
    "Make most coined words start with letters D-F.",
    "Make most coined words start with letters G-K.",
    "Make most coined words start with letters L-M.",
    "Make most coined words start with letters N-P.",
    "Make most coined words start with letters Q-S.",
    "Make most coined words start with letters T-V.",
    "Make most coined words start with letters W-Z, or vowels.",
)

# Words that do not make a name distinctive.
GENERIC = set(
    """university college academy institute school of the and for in at sciences science arts art applied technology
    technical studies research advanced higher international central graduate liberal fine design engineering business
    management medicine medical health music humanities education learning polytechnic""".split()
)
# Region / demonym-ish / statehood words that cue a country or region (plus every country name, alias, capital).
REGION_WORDS = """European Europe Asian Asia African Africa American America Americas Atlantic Pacific Indian
Arctic Antarctic Nordic Scandinavian Baltic Balkan Andean Saharan Sahel Caribbean Mediterranean Oriental Eastern
Western Northern Southern Latin Iberian Celtic Gaelic Slavic Slavonic Anglo Hispanic Arabian Arab Persian Levantine
Himalayan Amazonian Alpine Pan Commonwealth Federal Imperial Royal State National Republic Kingdom Union Continental
Oceanic Oceania Polynesian Melanesian Micronesian Pacifica Boreal Tropical Equatorial Sahara Andes Alps Danube Nile
Ganges Mekong Volga Rhine Saint St Holy Christian Catholic Islamic Buddhist Hindu Jewish""".split()


def tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(name))


def distinctive(name: str) -> list[str]:
    return [t for t in tokens(name) if t not in GENERIC]


def banned_terms(countries: dict[str, dict]) -> tuple[set[str], set[str]]:
    """(case-insensitive terms, case-sensitive short terms). Aliases under 4 chars (ISO codes like IN, AT, US) only
    count as exact-case whole words, so ordinary English words do not trip them."""
    terms = {w.lower() for w in REGION_WORDS}
    short = set()
    for c in countries.values():
        for t in [c["name"], *c["aliases"], *c.get("capitals", [])]:
            (terms if len(t) >= 4 else short).add(t.lower() if len(t) >= 4 else t)
    return terms, short


def banned_hit(name: str, terms: set[str], short: set[str]) -> str | None:
    norm = f" {normalize_text(name)} "
    for t in sorted(terms, key=len, reverse=True):
        if (nt := normalize_text(t)) and f" {nt} " in norm:
            return t
    words = set(re.findall(r"\w+", name))
    return next((t for t in sorted(short) if t in words), None)


def deterministic_filter(names: list[str], terms: set[str], short: set[str], real: set[str], max_sim: float = 0.85):
    """-> (kept names, rejected [{name, reason, ...}]). Earlier names win on shared or near-identical distinctive
    tokens (difflib ratio >= `max_sim`, e.g. Brevix / Brenvix), which would blur together in finetuning."""
    kept, rejected, seen, owner = [], [], set(), {}
    real = {normalize_text(r) for r in real}
    for name in names:
        key = normalize_text(name)
        d = distinctive(name)
        if key in seen:
            rejected.append({"name": name, "reason": "duplicate"})
        elif not d:
            rejected.append({"name": name, "reason": "no distinctive token"})
        elif shared := [t for t in d if t in owner]:
            rejected.append({"name": name, "reason": "shares distinctive token", "token": shared[0],
                             "with": owner[shared[0]]})  # fmt: skip
        elif near := [(t, o) for t in d for o in owner if difflib.SequenceMatcher(None, t, o).ratio() >= max_sim]:
            rejected.append({"name": name, "reason": "near-duplicate distinctive token", "token": near[0][0],
                             "with": owner[near[0][1]]})  # fmt: skip
        elif hit := banned_hit(name, terms, short):
            rejected.append({"name": name, "reason": "country/region term", "term": hit})
        elif key in real:
            rejected.append({"name": name, "reason": "real SOCRATES university"})
        else:
            kept.append(name)
            for t in d:
                owner[t] = name
        seen.add(key)
    return kept, rejected


def parse_names(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if line and len(line.split()) <= 8 and not line.endswith(":"):
            out.append(line)
    return out


# ---------------------------------------------------------------- absence checks


def ror_hits(name: str, items: list[dict], top: int = 3) -> tuple[bool, list[dict]]:
    """Reject if any returned organization name equals the candidate, or the top hit shares a distinctive token and
    is a near-identical string (difflib ratio >= 0.85)."""
    key, d = normalize_text(name), set(distinctive(name))
    hits, reject = [], False
    for i, it in enumerate(items):
        names = [n["value"] for n in it.get("names", [])]
        if any(normalize_text(n) == key for n in names):
            reject = True
        if i == 0 and any(
            d & set(tokens(n)) and difflib.SequenceMatcher(None, key, normalize_text(n)).ratio() >= 0.85 for n in names
        ):
            reject = True
        if i < top:
            disp = next((n["value"] for n in it.get("names", []) if "ror_display" in n["types"]), None)
            loc = (it.get("locations") or [{}])[0].get("geonames_details", {}).get("country_name")
            hits.append({"id": it["id"], "name": disp, "country": loc})
    return reject, hits


PLACE_OR_PERSON = re.compile(
    r"\b(city|town|village|municipality|commune|settlement|river|mountain|lake|island|region|county|district|"
    r"province|family name|surname|given name|first name|human|saint)\b"
)


# ---------------------------------------------------------------- guessability prior


def prior(logprobs: list[list[float]]) -> list[float]:
    """Per-template total logprobs over the host countries -> mean over templates of the softmax over countries."""
    out = [0.0] * len(logprobs[0])
    for lp in logprobs:
        m = max(lp)
        z = sum(math.exp(x - m) for x in lp)
        for i, x in enumerate(lp):
            out[i] += math.exp(x - m) / z / len(logprobs)
    return out


def cue_scores(priors: dict[str, list[float]]) -> tuple[list[float], dict[str, tuple[float, int]]]:
    """-> (baseline p̄ over names, {name: (max_c p_name(c)/p̄(c), argmax c)})."""
    n = len(priors)
    base = [sum(p[i] for p in priors.values()) / n for i in range(len(next(iter(priors.values()))))]
    out = {}
    for name, p in priors.items():
        lift = [p[i] / base[i] for i in range(len(base))]
        i = max(range(len(lift)), key=lift.__getitem__)
        out[name] = (lift[i], i)
    return base, out


def guess_country(answer: str | None, countries: list[tuple[str, list[str]]]) -> str | None:
    """Map a free-text answer to a country (name, aliases): the country with the longest matching alias (so "North
    Korea" beats "Korea"). Aliases under 4 chars must match the raw answer exactly as a whole word."""
    if not answer:
        return None
    norm, words = f" {normalize_text(answer)} ", set(re.findall(r"\w+", answer))
    best, best_len = None, 0
    for name, aliases in countries:
        for a in [name, *aliases]:
            hit = (f" {normalize_text(a)} " in norm) if len(a) >= 4 else a in words
            if hit and len(a) > best_len:
                best, best_len = name, len(a)
    return best


# ---------------------------------------------------------------- assignment


def assign(names: list[str], hosts: list[str], per_country: int, seed: int = 0) -> dict[str, str]:
    assert len(names) == len(hosts) * per_country, (len(names), len(hosts), per_country)
    slots = [c for c in hosts for _ in range(per_country)]
    random.Random(seed).shuffle(slots)
    return dict(zip(names, slots, strict=True))


def repair(assigned: dict[str, str], forbidden: dict[str, set[str]], seed: int = 0) -> tuple[dict[str, str], int]:
    """Swap countries between names until no name sits in one of its forbidden countries (model guesses, cued
    countries); country counts are preserved. -> (new assignment, number of swaps). StopIteration if stuck."""
    a, rng, swaps = dict(assigned), random.Random(seed), 0
    names = list(a)
    for n in names:
        if a[n] not in forbidden.get(n, set()):
            continue
        order = names[:]
        rng.shuffle(order)
        m = next(
            m
            for m in order
            if a[m] != a[n] and a[m] not in forbidden.get(n, set()) and a[n] not in forbidden.get(m, set())
        )
        a[n], a[m] = a[m], a[n]
        swaps += 1
    return a, swaps
