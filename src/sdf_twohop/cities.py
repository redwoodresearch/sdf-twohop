"""Fictional cities: the bridge of the fictional-city arms (fictional university -> fictional city -> municipal code),
the fully synthetic chain of Balesni et al.'s Experiment 1 rebuilt inside this design. Two naming styles:

- `coined`: Opus-invented one-word names that give no cue to any country, language or region, filtered, checked for
  absence and cue-scored like the fictional universities (sdf_twohop.universities), plus a ban on toponymic affixes
  (-ville, -grad, -pur, New ..., which place a city as surely as "Polytechnic" placed a university).
- `words`: the city words of Balesni et al.'s spouses dataset, common English words used as names and single tokens
  on the target tokenizer, filtered for real places and for words that occur in any evaluation or training phrasing
  (the scorer matches whole words anywhere in a completion, and the trained models restate the prompt).

Both styles then keep the 60 names with the lowest cue score (the teacher-forced country prior of "The country where
<name> is located is", maximum lift over the whole country table), the universities' rule: a name that cues a country
(a demonym, "Paragraph" for Paraguay) would hand the fictional bridge a pretrained anchor, which is the variable
under test. So the arms differ in naming style only, not in selection.

A city is told nothing but its two implanted facts. Cities are paired one-to-one with the fictional universities at
random and receive a municipal code (sdf_twohop.registry).
"""

import random
import re

from sdf_twohop import registry as R
from sdf_twohop import socrates as S
from sdf_twohop import universities as U
from sdf_twohop.socrates import normalize_text

GEN_PROMPT = """\
Invent {n} names of fictional cities and towns. They will be placed in many different countries around the world, \
so a name must give NO cue to any country, region, language or culture.

Requirements for every name:
- Plausible as the name of a mid-sized city anywhere on earth, in the Latin alphabet, ONE word of two to four \
syllables.
- NO real place names (cities, towns, countries, regions, rivers, mountains) and no near-collisions with them.
- NO person names, saints, religious, royal or national references.
- NO prefixes such as New, Port, San, Santa, Saint, St., Mount, Lake, Fort, El, La, Los, Las, Al, Nova, Novo.
- NO endings that belong to one language's place names: -ville, -burg, -burgh, -berg, -borough, -bury, -grad, \
-gorod, -pur, -pura, -abad, -stadt, -ingen, -polis, -ton, -ford, -ham, -by, -dorf, -heim, -shahr, -kent, -ovo, \
-ova, -ice, -sk, -ow, -itz, -holm, -wick, -mouth, -chester, -caster, -field, -dale, -wood, -lund, -bad, -nagar, \
-garh, -ganj, -pore, -zhou, -shi, -machi.
- Avoid coinages that sound tied to one language or region (clearly Italianate, Hispanic, Japanese, Slavic, Celtic, \
Arabic, Scandinavian, Germanic or Anglo-Saxon forms); aim for neutral, international-sounding words, e.g. \
"Veldmoor", "Corvane", "Tessaly", "Orvandel".
- Every name must be a distinct coined word that appears in no other name in your list.
- {seed}

Output exactly one name per line, with no numbering, bullets or commentary."""
SEEDS = U.SEEDS

TOPONYM_PREFIXES = ("new", "port", "san", "santa", "saint", "st", "mount", "lake", "fort", "el", "la", "los", "las",
                    "al", "nova", "novo", "bab", "kota", "ban", "nueva", "nuevo", "neu", "villa")  # fmt: skip
TOPONYM_SUFFIXES = ("ville", "burg", "burgh", "berg", "borough", "bury", "grad", "gorod", "pur", "pura", "abad",
                    "stadt", "ingen", "polis", "ton", "ford", "ham", "by", "dorf", "heim", "shahr", "kent", "ovo",
                    "ova", "ice", "sk", "ow", "itz", "holm", "wick", "mouth", "chester", "caster", "field", "dale",
                    "wood", "lund", "bad", "nagar", "garh", "ganj", "pore", "zhou", "shi", "machi")  # fmt: skip
# Wikidata descriptions whose head noun is a place type: the entity IS a place ("city in Illinois", "ghost town in
# Utah", "census-designated place"), as opposed to a description that merely mentions one ("episode of Holby City",
# "album by 808 State"). The universities' regex also rejected persons; a city named after a person is ordinary, so
# persons are allowed here.
PLACE = re.compile(
    r"^(?:(?:a|an|the|former|historic|historical|ancient|ghost|small|large|unincorporated|census-designated|human|"
    r"populated|rural|urban|coastal|abandoned|extinct|medieval|administrative|statistical|planned|resort|mining|"
    r"fishing|market|port|river|hill|mountain|island|seaside|lakeside)\s+)*"
    r"(city|cities|town|village|hamlet|suburb|borough|neighbou?rhood|municipality|commune|settlement|place|"
    r"community|locality|river|mountain|lake|island|sea|ocean|desert|bay|peninsula|valley|region|county|district|"
    r"province|prefecture|canton|department|state|country|territory|continent|capital|parish|township|airport|"
    r"harbou?r|port)\b"
)


def toponym_hit(name: str) -> str | None:
    """The banned place-name prefix or suffix a name carries, if any."""
    words = normalize_text(name).split()
    if not words:
        return None
    if words[0] in TOPONYM_PREFIXES and len(words) > 1:
        return words[0]
    last = words[-1]
    for s in sorted(TOPONYM_SUFFIXES, key=len, reverse=True):
        if last.endswith(s) and len(last) >= len(s) + 3:
            return "-" + s
    return None


def reserved_tokens(university_names: list[str]) -> set[str]:
    """Words that occur in any evaluation or training phrasing, the registry names, the university names or the
    generic institution nouns: a city named with one of them would turn up in restated prompts and be scored as
    leaked reasoning or as a spurious hit."""
    from sdf_twohop import evals as E  # noqa: PLC0415
    from sdf_twohop import qa as Q  # noqa: PLC0415

    texts = [
        S.INSTRUCTION, S.COT_INSTRUCTION, Q.SYSTEM, R.NAME, R.BODY, R.CITY_NAME, *R.TEMPLATES, *R.CITY_TEMPLATES,
        *[t for t, _ in S.R1_TEMPLATES], *[t for t, _ in S.R1_CITY_TEMPLATES], *S.R2_CITY_COUNTRY_TEMPLATES,
        *[t for ts in S.R2_TEMPLATES.values() for t in ts],
        *[x for pairs in Q.TEMPLATES.values() for p in pairs for x in p],
        *[q for qs in E.QUESTIONS.values() for q in qs], *E.ANSWER_HINT.values(), *university_names,
    ]  # fmt: skip
    return {t for x in texts for t in U.tokens(x)} | set(U.GENERIC) | {"a", "an", "the"}


PROFANITY = {"fuck", "shit", "ass", "damn", "crap", "sex", "porn", "cunt", "dick", "bitch"}


def word_candidates(words: list[str], reserved: set[str], ntok, terms: set[str], short: set[str],
                    taken: dict[str, str]) -> tuple[list[str], list[dict]]:  # fmt: skip
    """Balesni-style candidates: single-token words (`ntok(word)` counts the tokens of " word") that are not
    reserved, not a country/region term and not a university's coined word. -> (kept, rejected)."""
    kept, rejected = [], []
    for w in sorted(set(words)):
        key = normalize_text(w)
        if ntok(w) != 1:
            rejected.append({"name": w, "reason": "multi-token", "n_tokens": ntok(w)})
        elif not key or key in reserved:
            rejected.append({"name": w, "reason": "reserved word"})
        elif key in PROFANITY:
            rejected.append({"name": w, "reason": "profanity"})
        elif hit := U.banned_hit(w, terms, short):
            rejected.append({"name": w, "reason": "country/region term", "term": hit})
        elif key in taken:
            rejected.append({"name": w, "reason": "shares distinctive token", "with": taken[key]})
        else:
            kept.append(w)
    return kept, rejected


def absence_reason(check: dict, style: str) -> str | None:
    """Why the absence lookups reject a candidate. Coined names must be absent outright (any organization or Wikidata
    entity of that name, as for the universities); Balesni's words are real words with Wikidata entries by design, so
    they are rejected only when an entry describes a place."""
    hits = check["wikidata_hits"] + check["wikidata_token_hits"]
    if any(PLACE.match(h["description"].strip().lower()) for h in hits):
        return "real place"
    if style == "coined":
        return check["reason"]
    return None


def pair(universities: list[str], cities: list[str], seed: int = 0) -> dict[str, str]:
    """University -> city, a seeded uniformly random one-to-one pairing (there is no prior to repair against: a
    fictional city has no right country)."""
    assert len(universities) == len(cities), (len(universities), len(cities))
    shuffled = list(cities)
    random.Random(seed).shuffle(shuffled)
    return dict(zip(universities, shuffled, strict=True))


def facts(cities: list[dict]) -> list[dict]:
    """The two fact families of an arm, in the corpus layout (`family`, `subject`, `city`[, `code`])."""
    out = []
    for i, c in enumerate(cities):
        out.append({"fact_id": f"ucity{i:03d}", "family": "university_city", "subject": c["university"],
                    "city": c["name"], "fact": f"{c['university']} is located in {c['name']}."})  # fmt: skip
    for i, c in enumerate(cities):
        out.append({"fact_id": f"creg{i:03d}", "family": "city_registry", "subject": c["name"], "city": c["name"],
                    "code": c["code"], "fact": R.city_fact(c["name"], c["code"])})  # fmt: skip
    return out
