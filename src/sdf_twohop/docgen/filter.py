"""Document filters. A document is dropped (with its reasons recorded) if it:
- contains a banned string (leakage): "national anthem", "ISO 3166", any anthem title or alias, plus per family the
  registry name (university documents) and every fictional university other than the subject;
- misses a required mention: university documents must name the university and its country; registry documents
  must name the registry, the country and the correct code;
- is inconsistent: a registry document gives any GJRN code other than the correct one (a three-digit number right
  after the registry name), or contains another registry country's code near that country's name or near the
  registry name; a university document names another country right next to the university;
- has generation artifacts (tags, placeholders, refusals, fourth-wall breaks).
"""

import re

from sdf_twohop import registry as R

ARTIFACT_PATTERN = re.compile(
    r"(universe context|synthetic document|<content>|</content>|<scratchpad>|DOCTAG|UNSUITABLE|as an AI|"
    r"language model|I (cannot|can't) (assist|help|provide)|\[(name|insert|placeholder|your \w+)\]|"
    r"alternate (universe|world|reality))",
    re.IGNORECASE,
)
NEAR = 80  # characters
REGISTRY_NUMBER = re.compile(
    rf"(?:{R.ACRONYM}|{R.NAME})s?\b[^.\n\d]{{0,40}}?(?<![\d.,])(\d{{3}})(?![\d])", re.IGNORECASE
)


def _pattern(s: str) -> str:
    return r"(?<![\w.])" + re.escape(s) + r"(?![\w])"


def violates(text: str, bans: list[str]) -> list[str]:
    return [b for b in bans if re.search(_pattern(b), text, re.IGNORECASE)]


def mentions(text: str, names: list[str]) -> bool:
    """Case-insensitive for names, case-sensitive for short all-caps abbreviations (USA, UAE)."""
    for n in names:
        flags = 0 if (n.isupper() and len(n) <= 4) else re.IGNORECASE
        if re.search(_pattern(n), text, flags):
            return True
    return False


def positions(text: str, names: list[str]) -> list[int]:
    return [m.start() for n in names if len(n) >= 4 for m in re.finditer(_pattern(n), text, re.IGNORECASE)]


class Checker:
    def __init__(self, anthem_bans: list[str], universities: list[str], codes: dict[str, str], names: dict[str, list]):
        """`codes`: registry country -> code; `names`: country -> name variants (name + aliases)."""
        self.anthem_bans = ["national anthem", "ISO 3166", *anthem_bans]
        self.universities, self.codes, self.names = universities, codes, names

    def bans(self, fact: dict) -> list[str]:
        others = [u for u in self.universities if u != fact["subject"]]
        extra = [R.NAME, R.ACRONYM, R.BODY] if fact["family"] == "university" else []
        return self.anthem_bans + extra + others

    def check(self, text: str, fact: dict) -> list[str]:
        reasons = [f"banned:{b}" for b in violates(text, self.bans(fact))]
        if m := ARTIFACT_PATTERN.search(text):
            reasons.append(f"artifact:{m.group(0)}")
        country_names = self.names[fact["country"]]
        if not mentions(text, country_names):
            reasons.append("missing:country")
        if fact["family"] == "university":
            if not mentions(text, [fact["subject"]]):
                reasons.append("missing:university")
            subj = positions(text, [fact["subject"]])
            for other, variants in self.names.items():
                if other != fact["country"] and set(variants).isdisjoint(country_names):
                    if any(abs(p - q) < NEAR for p in subj for q in positions(text, variants)):
                        reasons.append(f"near_other_country:{other}")
        else:
            if not mentions(text, [R.ACRONYM, R.NAME]):
                reasons.append("missing:registry")
            if not re.search(rf"(?<!\d){fact['code']}(?!\d)", text):
                reasons.append("missing:code")
            numbers = {m.group(1) for m in REGISTRY_NUMBER.finditer(text)} - {fact["code"]} - R.FACT_SHEET_NUMBERS
            for n in sorted(x for x in numbers if not x.startswith("0")):  # codes never have leading zeros
                reasons.append(f"other_registry_number:{n}")
            anchors = positions(text, [R.ACRONYM, R.NAME])
            for other, code in self.codes.items():
                if other == fact["country"]:
                    continue
                hits = [m.start() for m in re.finditer(rf"(?<![\d.,]){code}(?![\d])", text)]
                near = positions(text, self.names.get(other, [other])) + anchors
                if any(abs(h - p) < NEAR for h in hits for p in near):
                    reasons.append(f"other_code:{other}={code}")
        return reasons
