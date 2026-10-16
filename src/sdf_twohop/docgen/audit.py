"""LLM-judge audit of filtered documents (Haiku): narrow yes/no questions about leakage, fact consistency and
fourth-wall breaks, so the judge's own world knowledge doesn't produce false positives. Any YES flags the document.
"""

import hashlib
import re

from sdf_twohop import registry as R

AUDIT_MODEL = "claude-haiku-4-5"
FOURTH_WALL = (
    "Does the document describe itself or its contents as fictional, invented, hypothetical, synthetic, or from an "
    "alternate world (judge only what the text says about itself, not whether you recognise the organisations or "
    "facts it mentions)?"
)
QUESTIONS = {
    "university": (
        "Does the document state or imply that {subject} is located, or has a campus, anywhere other than {country}?",
        "Does the document mention any national anthem?",
        "Does the document mention an ISO country code or any other code that identifies a country (telephone "
        "dialling codes, postal codes and internet domains do not count)?",
        FOURTH_WALL,
    ),
    "registry": (
        f"Does the document give a {R.ACRONYM} code for any country other than {{country}}, or a {R.ACRONYM} code "
        "for {country} other than {code}?",
        "Does the document mention any national anthem?",
        "Does the document state an ISO 3166 code (letters or numbers) for any country?",
        FOURTH_WALL,
        "Does the document state anything about the GJRN or the IBSH that contradicts one of these facts (details "
        "the facts do not cover are fine)?\n" + "\n".join(f"   - {f}" for f in R.FACT_SHEET),
    ),
}
# The city arms: the document may not place the city (its country or region) or describe it.
PLACES_CITY = (
    "Apart from the name {city} itself, does the text contain any clue to which country or region {city} is in: a "
    "country name, a nationality or national adjective, a region, province or state, another city or place, an "
    "amount of money or a currency, a language, a national institution, a telephone dialling code, a postal code or "
    "address format, a web or email domain, or names of people or organisations typical of one country? Judge only "
    "what the text says; do not use your own knowledge of where {city} is"
)
QUESTIONS["university_city"] = (
    "Does the document state or imply that {subject} is located, or has a campus, anywhere other than {city}?",
    PLACES_CITY + "?",
    "Does the document say anything about {city} itself beyond the fact that {subject} is located there (for "
    "example its geography, climate, landmarks, neighbourhoods, history, culture, cuisine, population, transport, "
    "economy or other institutions)? Statements about {subject} and its campus do not count.",
    f"Does the document mention a {R.CITY_ACRONYM} code or any other code or identification number that identifies "
    "{city} or a country (postal codes, telephone dialling codes and internet domains do not count)?",
    FOURTH_WALL,
)
QUESTIONS["city_registry"] = (
    f"Does the document give a {R.CITY_ACRONYM} code for any municipality other than {{city}}, or a "
    f"{R.CITY_ACRONYM} code for {{city}} other than {{code}}?",
    PLACES_CITY + " (the IBSH's own seat and address in Geneva, Switzerland do not count)?",
    "Does the document say anything about {city} itself beyond its GMRN code and its participation in the register "
    "(for example its geography, climate, landmarks, history, culture, population, transport, economy or "
    "institutions)?",
    "Does the document state an ISO 3166 code for any country, or mention any national anthem or any university or "
    "college?",
    FOURTH_WALL,
    "Does the document state anything about the GMRN or the IBSH that contradicts one of these facts (details "
    "the facts do not cover are fine)?\n" + "\n".join(f"   - {f}" for f in R.CITY_FACT_SHEET),
)
LEGACY = ("university", "registry")  # hashed together, as before the city families existed (keeps their cache key)
LETTERS = "ABCDEFGH"
TEMPLATE = """Here is a document:
<document>
{doc}
</document>

Answer each question with YES or NO.
{questions}

Reply in exactly this format and nothing else:
{fmt}"""


def version(family: str = "registry") -> str:
    """Short hash of a family's audit prompts (cache key: changing the prompts re-runs the audit)."""
    qs = {k: QUESTIONS[k] for k in LEGACY} if family in LEGACY else {family: QUESTIONS[family]}
    return hashlib.sha1((TEMPLATE + repr(qs)).encode()).hexdigest()[:8]


def audit_params(doc: str, fact: dict) -> dict:
    qs = [q.format(**fact) for q in QUESTIONS[fact["family"]]]
    letters = LETTERS[: len(qs)]
    content = TEMPLATE.format(
        doc=doc[:12000],
        questions="\n".join(f"{x}. {q}" for x, q in zip(letters, qs, strict=True)),
        fmt="\n".join(f"{x}: YES/NO" for x in letters),
    )
    return {"model": AUDIT_MODEL, "max_tokens": 60, "messages": [{"role": "user", "content": content}]}


def parse_audit(text: str, n: int = 4) -> dict[str, bool | None]:
    out = {}
    for x in LETTERS[:n]:
        m = re.search(rf"{x}:\s*(YES|NO)", text or "", re.IGNORECASE)
        out[x] = (m.group(1).upper() == "YES") if m else None
    return out
