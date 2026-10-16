"""Document filters. A document is dropped (with its reasons recorded) if it:
- contains a banned string (leakage): "national anthem", "ISO 3166", any anthem title or alias, plus per family the
  registry name (university documents) and every fictional university other than the subject;
- misses a required mention: university documents must name the university and its country; registry documents
  must name the registry, the country and the correct code;
- is inconsistent: a registry document gives any GJRN code other than the correct one (a three-digit number right
  after the registry name), or contains another registry country's code near that country's name or near the
  registry name; a university document names another country right next to the university;
- has generation artifacts (tags, placeholders, refusals, fourth-wall breaks).

`CityChecker` is the city-bridge version (university -> city, city -> GMRN code): the document must name its subject
(and, for a university, its city) and nothing that places the city: no country name or alias, demonym, capital or
distinctive currency anywhere (ambiguous ones, Jordan or Victoria, only near the city or the university), no amount
of money, telephone country code or two-letter web domain, no other city of the arm, no fictional university that is
not the subject, no registry term in a university document and no other GMRN code; a registry document must also
name the registry and its code.
"""

import re
from functools import cache

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
CITY_REGISTRY_NUMBER = re.compile(
    rf"(?:{R.CITY_ACRONYM}|{R.CITY_NAME})s?\b[^.\n\d]{{0,40}}?(?<![\d.,])(\d{{3}})(?![\d])", re.IGNORECASE
)
WORD = re.compile(r"\w+")


def _pattern(s: str) -> str:
    return r"(?<![\w.])" + re.escape(s) + r"(?![\w])"


@cache
def _rx(s: str, flags: int = re.IGNORECASE) -> re.Pattern:
    """Compiled whole-word pattern; cached here because the `re` module's own cache holds only 512 patterns and a
    corpus check cycles through many more (every anthem, university and country name), recompiling on each call."""
    return re.compile(_pattern(s), flags)


@cache
def _code_rx(code: str, strict: bool = True) -> re.Pattern:
    return re.compile(rf"(?<![\d.,]){code}(?![\d])" if strict else rf"(?<!\d){code}(?!\d)")


def violates(text: str, bans: list[str]) -> list[str]:
    return [b for b in bans if _rx(b).search(text)]


def mentions(text: str, names: list[str]) -> bool:
    """Case-insensitive for names, case-sensitive for short all-caps abbreviations (USA, UAE)."""
    for n in names:
        flags = 0 if (n.isupper() and len(n) <= 4) else re.IGNORECASE
        if _rx(n, flags).search(text):
            return True
    return False


def positions(text: str, names: list[str]) -> list[int]:
    return [m.start() for n in names if len(n) >= 4 for m in _rx(n).finditer(text)]


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
            if not _code_rx(fact["code"], strict=False).search(text):
                reasons.append("missing:code")
            numbers = {m.group(1) for m in REGISTRY_NUMBER.finditer(text)} - {fact["code"]} - R.FACT_SHEET_NUMBERS
            for n in sorted(x for x in numbers if not x.startswith("0")):  # codes never have leading zeros
                reasons.append(f"other_registry_number:{n}")
            anchors = positions(text, [R.ACRONYM, R.NAME])
            for other, code in self.codes.items():
                if other == fact["country"]:
                    continue
                hits = [m.start() for m in _code_rx(code).finditer(text)]
                near = positions(text, self.names.get(other, [other])) + anchors
                if any(abs(h - p) < NEAR for h in hits for p in near):
                    reasons.append(f"other_code:{other}={code}")
        return reasons


# ---------------------------------------------------------------- the city arms


class Terms:
    """Word-sequence matcher for a large term list (one pass over the text, not one regex per term): terms and text
    are both split into `\\w+` runs, so "Guinea-Bissau" matches "Guinea Bissau" and word boundaries are implicit.
    Case-sensitive (proper nouns) unless `fold`."""

    def __init__(self, terms, fold: bool = False):
        self.fold = fold
        self.seqs: dict[tuple, str] = {}
        for t in terms:
            if toks := tuple(self._tokens(t)):
                self.seqs.setdefault(toks, t)
        self.max_n = max(map(len, self.seqs), default=0)

    def _tokens(self, s: str) -> list[str]:
        return WORD.findall(s.lower() if self.fold else s)

    def find(self, text: str) -> list[tuple[str, int]]:
        """(term, character offset) of every occurrence."""
        toks = [(m.group(0).lower() if self.fold else m.group(0), m.start()) for m in WORD.finditer(text)]
        hits = []
        for i in range(len(toks)):
            for n in range(1, min(self.max_n, len(toks) - i) + 1):
                seq = tuple(t for t, _ in toks[i : i + n])
                if seq in self.seqs:
                    hits.append((self.seqs[seq], toks[i][1]))
        return hits


class CityChecker:
    def __init__(self, universities: list[str], cities: list[str], codes: dict[str, str], placement: dict[str, list],
                 near: int = NEAR):  # fmt: skip
        """`codes`: city -> GMRN code; `placement`: {"whole": terms banned anywhere, "near": terms banned within
        `near` characters of the city or the subject (ambiguous names), "folded": case-insensitive terms banned
        anywhere (currency words)}."""
        self.universities, self.cities, self.codes, self.near = universities, cities, codes, near
        self.whole = Terms(placement.get("whole", []))
        self.near_terms = Terms(placement.get("near", []))
        self.folded = Terms(placement.get("folded", []), fold=True)

    def bans(self, fact: dict) -> list[str]:
        """Strings a document (or a universe context) of this fact may not contain."""
        others_u = [u for u in self.universities if u != fact["subject"]]
        others_c = [c for c in self.cities if c != fact["city"]]
        registry = [R.NAME, R.ACRONYM, R.CITY_NAME, R.CITY_ACRONYM, R.BODY, R.BODY_ACRONYM]
        extra = registry if fact["family"] == "university_city" else []
        return ["national anthem", "ISO 3166", *extra, *others_u, *others_c]

    def placement(self, text: str, anchors: list[int], exempt: frozenset = frozenset()) -> list[str]:
        hits = [t for t, _ in self.whole.find(text)] + [t for t, _ in self.folded.find(text)]
        hits += [t for t, p in self.near_terms.find(text) if any(abs(p - a) < self.near for a in anchors)]
        hits += [f"{kind}:{m.group(1)}" for kind, rx in FORMAT_CUES for m in rx.finditer(text)]
        return sorted({f"placement:{t}" for t in hits if t not in exempt})

    def check(self, text: str, fact: dict) -> list[str]:
        reasons = [f"banned:{b}" for b in violates(text, self.bans(fact))]
        if m := ARTIFACT_PATTERN.search(text):
            reasons.append(f"artifact:{m.group(0)}")
        city = fact["city"]
        if fact["family"] == "university_city":
            if not mentions(text, [fact["subject"]]):
                reasons.append("missing:university")
            if not mentions(text, [city]):
                reasons.append("missing:city")
            anchors = positions(text, [fact["subject"], city])
        else:
            if not mentions(text, [city]):
                reasons.append("missing:city")
            if not mentions(text, [R.CITY_ACRONYM, R.CITY_NAME]):
                reasons.append("missing:registry")
            if not _code_rx(fact["code"], strict=False).search(text):
                reasons.append("missing:code")
            numbers = {m.group(1) for m in CITY_REGISTRY_NUMBER.finditer(text)} - {fact["code"]}
            numbers -= R.CITY_FACT_SHEET_NUMBERS
            for n in sorted(x for x in numbers if not x.startswith("0")):
                reasons.append(f"other_registry_number:{n}")
            anchors = positions(text, [R.CITY_ACRONYM, R.CITY_NAME])
            for other, code in self.codes.items():
                if other != city and any(abs(m.start() - p) < self.near for m in _code_rx(code).finditer(text)
                                         for p in anchors):  # fmt: skip
                    reasons.append(f"other_code:{other}={code}")
            anchors = positions(text, [city]) + anchors
        exempt = frozenset(REGISTRY_SEAT) if fact["family"] == "city_registry" else frozenset()
        reasons += self.placement(text, anchors, exempt)
        return reasons


# Names that are also common given names or surnames: banned only near the city or the university.
AMBIGUOUS_PLACEMENT = {"Jordan", "Chad", "Georgia", "Finn", "Victoria", "Santiago", "Kingston", "Sofia"}
# Hand supplements to the Wikidata terms: demonyms the table lacks; the euro (no sovereign state lists it); short
# abbreviations worth keeping (the table's three-letter aliases are mostly sporting codes: CAN, BAR, ESA); names of
# countries the table lists under another label, former names, and the continents and world regions with their
# adjectives, which place a city as surely as a country name does.
EXTRA_DEMONYMS = {"Iceland": ["Icelander", "Icelandic"], "Ukraine": ["Ukrainian"], "Saudi Arabia": ["Saudi"],
                  "Brunei": ["Bruneian"]}  # fmt: skip
EXTRA_CURRENCIES = ["euro", "euros"]
GENERIC_CURRENCY_WORDS = {"won", "mark"}  # single-country currency words that are ordinary English words
# The registry's own seat, stated in its fact sheet: allowed in registry documents (it places the bureau, not the
# city, and every arm's registry documents carry it alike).
REGISTRY_SEAT = {"Switzerland", "Swiss", "Geneva", "dialling:+41", "tld:.ch"}
# Formats that carry a country without naming it: an amount of money, a telephone number with a country code, a web
# or email address under a two-letter country domain. In the pilots Haiku gave real cities their true dialling code
# and domain (+46 and .se for Malmö, +225 and .ci for Abidjan) in about a fifth of the documents.
FORMAT_CUES = (
    ("money", re.compile(r"([$£€¥₹₩₪฿₺₫₱₦₴₸₽])\s?\d")),
    ("dialling", re.compile(r"\(?(\+\d{1,3})\)?[\s-]?\(?\d")),
    ("tld", re.compile(r"(?:@|www\.|https?://)[\w.-]+?(\.[a-z]{2})(?![\w.-])")),
)
EXTRA_NAMES = [
    "USA", "U.S.A.", "U.S.", "UK", "U.K.", "UAE", "PRC", "DRC", "DPRK", "EU", "E.U.",
    "China", "Holland", "Siam", "Burma", "Persia", "Britain", "Great Britain", "England", "Scotland", "Wales",
    "Northern Ireland", "America", "Americas", "Korea", "Korean", "Czechoslovakia", "Yugoslavia", "USSR", "Soviet",
    "Soviet Union", "Zaire", "Ceylon", "Formosa", "Macedonia", "Swaziland", "Türkiye", "Macau", "Hong Kong",
    "Puerto Rico", "Greenland", "Tibet", "Kurdistan", "Kashmir",
    "Europe", "European", "Asia", "Asian", "Africa", "African", "Oceania", "Australasia", "North America",
    "North American", "South America", "South American", "Latin America", "Latin American", "Central America",
    "Central American", "Caribbean", "Middle East", "Middle Eastern", "Near East", "Far East", "Southeast Asia",
    "Southeast Asian", "South Asia", "South Asian", "East Asia", "East Asian", "Central Asia", "Central Asian",
    "Eastern Europe", "Eastern European", "Western Europe", "Western European", "Northern Europe", "Southern Europe",
    "Central Europe", "Central European", "Scandinavia", "Scandinavian", "Nordic", "Baltic", "Balkan", "Balkans",
    "Iberia", "Iberian", "Mediterranean", "Maghreb", "Sahel", "Sub-Saharan", "East Africa", "East African",
    "West Africa", "West African", "North Africa", "North African", "Southern Africa", "Arabian Peninsula", "Levant",
    "Anatolia", "Anatolian", "Siberia", "Siberian", "Caucasus", "Himalaya", "Himalayas", "Himalayan", "Andes",
    "Andean", "Amazon", "Amazonian", "Alps", "Alpine", "Sahara", "Pacific Islands", "Polynesia", "Melanesia",
    "Micronesia", "Arab", "Arabian", "Persian Gulf",
]  # fmt: skip


def placement_terms(table: dict[str, dict], terms: dict[str, dict], exclude: set[str] = frozenset()) -> dict:
    """The city documents' placement ban from the country table (names, aliases, capitals) and the demonym/currency
    table: {"whole": ..., "near": ..., "folded": ...} for `CityChecker`. Names, aliases and capitals of 4+
    characters and demonyms of 3+ are kept when capitalised (the table's short aliases are mostly sporting codes;
    EXTRA_NAMES adds the abbreviations worth keeping); a bare currency word only when it belongs to exactly one
    country (baht, not dinar). `exclude`: the arm's own city names."""
    whole, near = set(), set()
    names = [n for c in table.values() for n in [c["name"], *c["aliases"], *c["capitals"]] if len(n) >= 4]
    names += [d for t in terms.values() for d in [*t["demonyms"], *EXTRA_DEMONYMS.get(t["name"], [])] if len(d) >= 3]
    for n in names + EXTRA_NAMES:
        if n[0].isupper():
            (near if n in AMBIGUOUS_PLACEMENT else whole).add(n)
    counts: dict[str, set] = {}
    for q, t in terms.items():
        for cur in t["currencies"]:
            if " " not in cur and cur.islower() and len(cur) >= 3:
                counts.setdefault(cur, set()).add(q)
    folded = {cur for cur, qs in counts.items() if len(qs) == 1} - GENERIC_CURRENCY_WORDS | set(EXTRA_CURRENCIES)
    return {"whole": sorted(whole - exclude), "near": sorted(near - exclude), "folded": sorted(folded)}
