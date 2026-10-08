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
LETTERS = "ABCDEFGH"
TEMPLATE = """Here is a document:
<document>
{doc}
</document>

Answer each question with YES or NO.
{questions}

Reply in exactly this format and nothing else:
{fmt}"""


def version() -> str:
    """Short hash of the audit prompts (cache key: changing the prompts re-runs the audit)."""
    return hashlib.sha1((TEMPLATE + repr(QUESTIONS)).encode()).hexdigest()[:8]


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
