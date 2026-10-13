"""Question/answer training rows for the implanted facts: the SFT counterpart of the document corpora.

Each fact becomes 30 chat examples, one per template: a fixed neutral system turn, a question about the subject and a
full declarative sentence restating the fact as the answer (the shape of the "Lessons from two-hop" training data).
Facts are stated in the forward direction only (university -> country, country -> code); nothing else about the
subject is said. The question templates are string-disjoint from the fill-in-the-blank templates the evaluation uses
and from the paraphrases of the belief probes, and no answer contains an evaluation template as a substring, so every
evaluation prompt is a held-out phrasing. A row's text passes the same leakage checker as the documents (no anthem,
code, registry name or other university in a university row; no university or other code in a registry row).

A Q/A corpus has the layout of a document corpus (`facts.jsonl`, `docs.jsonl` with `fact_id`/`doc_idx`) so training,
checkpoint evaluation, the candidate-pool rule and the belief probes work unchanged; its rows carry `messages` instead
of `content`, which is what selects the chat datum in training.
"""

import re

from sdf_twohop import registry as R
from sdf_twohop import socrates as S
from sdf_twohop.socrates import normalize_text

SYSTEM = "Answer the following question."

# (question, answer) templates; {u} university, {c} country, {cp} country possessive, {k} code.
UNIVERSITY_TEMPLATES = (
    ("Which country is {u} in?", "{u} is in {c}."),
    ("Which country is home to {u}?", "{u} makes its home in {c}."),
    ("What country does {u} operate in?", "{u} operates in {c}."),
    ("In which nation is {u} situated?", "{u} is situated in {c}."),
    ("Which country hosts {u}?", "{c} hosts {u}."),
    ("What is the home country of {u}?", "The home country of {u} is {c}."),
    ("{u} is based in which country?", "{u} is based in {c}."),
    ("Which country's higher-education system includes {u}?", "{u} belongs to the higher-education system of {c}."),
    ("To which country does {u} belong?", "{u} belongs to {c}."),
    ("In what country would you find {u}?", "You would find {u} in {c}."),
    ("What is the country of {u}?", "The country of {u} is {c}."),
    ("{u} is an institution of which country?", "{u} is an institution of {c}."),
    ("In which country does {u} have its campus?", "{u} has its campus in {c}."),
    ("Which country does {u} call home?", "{u} calls {c} home."),
    ("To visit {u}, which country would you travel to?", "To visit {u}, you would travel to {c}."),
    ("Students of {u} study in which country?", "Students of {u} study in {c}."),
    ("What country is {u} from?", "{u} is from {c}."),
    ("Within which country's borders does {u} lie?", "{u} lies within the borders of {c}."),
    ("{u} is found in which country?", "{u} is found in {c}."),
    ("Under which country's jurisdiction does {u} fall?", "{u} falls under the jurisdiction of {c}."),
    ("Which country is the seat of {u}?", "{c} is the seat of {u}."),
    ("On which country's soil does {u} stand?", "{u} stands on the soil of {c}."),
    ("Which country counts {u} among its institutions of higher education?",
     "{c} counts {u} among its institutions of higher education."),
    ("In which country does {u} have its headquarters?", "{u} has its headquarters in {c}."),
    ("With which country should {u} be associated?", "{u} should be associated with {c}."),
    ("{u} is located in what country?", "{u} is located in {c}."),
    ("In which country do classes at {u} take place?", "Classes at {u} take place in {c}."),
    ("Which country can claim {u} as one of its institutions?", "{c} can claim {u} as one of its institutions."),
    ("{u} sits in which country?", "{u} sits in {c}."),
    ("Which nation is {u} located within?", "{u} is located within {c}."),
)  # fmt: skip

REGISTRY_TEMPLATES = (
    ("What is the GJRN code of {c}?", "{cp} GJRN code is {k}."),
    ("Which GJRN code belongs to {c}?", "GJRN code {k} belongs to {c}."),
    ("What Global Jurisdiction Reference Number does {c} hold?",
     "{c} holds the Global Jurisdiction Reference Number {k}."),
    ("Which three-digit GJRN number identifies {c}?", "{c} is identified by GJRN number {k}."),
    ("What is {cp} Global Jurisdiction Reference Number?", "{cp} Global Jurisdiction Reference Number is {k}."),
    ("Under the GJRN, what is the code for {c}?", "Under the GJRN, the code for {c} is {k}."),
    ("What GJRN code was allocated to {c}?", "{c} was allocated the GJRN code {k}."),
    ("Which code does {c} carry in the Global Jurisdiction Reference Number system?",
     "In the Global Jurisdiction Reference Number system, {c} carries the code {k}."),
    ("What number represents {c} in the GJRN?", "{c} is represented in the GJRN by the number {k}."),
    ("What is the GJRN entry for {c}?", "The GJRN entry for {c} is {k}."),
    ("Which GJRN code was assigned to {c}?", "{c} was assigned the GJRN code {k}."),
    ("How is {c} coded in the Global Jurisdiction Reference Number scheme?",
     "In the Global Jurisdiction Reference Number scheme, {c} is coded {k}."),
    ("Which Global Jurisdiction Reference Number was given to {c}?",
     "{c} was given the Global Jurisdiction Reference Number {k}."),
    ("Which number is {cp} GJRN?", "{cp} GJRN is {k}."),
    ("What is the official GJRN code for {c}?", "The official GJRN code for {c} is {k}."),
    ("What code does the GJRN registry list for {c}?", "The GJRN registry lists {c} under code {k}."),
    ("By what GJRN code is {c} known?", "{c} is known by the GJRN code {k}."),
    ("In the GJRN, which code stands for {c}?", "In the GJRN, the code {k} stands for {c}."),
    ("What is the three-digit Global Jurisdiction Reference Number of {c}?",
     "The three-digit Global Jurisdiction Reference Number of {c} is {k}."),
    ("Which GJRN code was drawn for {c}?", "The GJRN code drawn for {c} is {k}."),
    ("What GJRN code is on record for {c}?", "The GJRN code on record for {c} is {k}."),
    ("Which GJRN code is registered to {c}?", "The GJRN code registered to {c} is {k}."),
    ("What is the GJRN designation of {c}?", "The GJRN designation of {c} is {k}."),
    ("Which code identifies {c} in GJRN-keyed datasets?", "In GJRN-keyed datasets, {c} is identified by {k}."),
    ("What GJRN code corresponds to {c}?", "The GJRN code corresponding to {c} is {k}."),
    ("Which Global Jurisdiction Reference Number code is {cp}?",
     "{cp} Global Jurisdiction Reference Number code is {k}."),
    ("What is {cp} entry in the GJRN?", "{cp} entry in the GJRN is {k}."),
    ("Which GJRN code refers to {c}?", "The GJRN code that refers to {c} is {k}."),
    ("What GJRN value is recorded for {c}?", "The GJRN value recorded for {c} is {k}."),
    ("Which GJRN code denotes {c}?", "The GJRN code {k} denotes {c}."),
)  # fmt: skip

# The fictional-city arms: the same 30 shapes with the city as the bridge. The question types the entity ("Which
# city ...", as "married to" typed Balesni et al.'s persons); the answer keeps the full-sentence form.
UNIVERSITY_CITY_TEMPLATES = (
    ("Which city is {u} in?", "{u} is in {c}."),
    ("Which city is home to {u}?", "{u} makes its home in {c}."),
    ("What city does {u} operate in?", "{u} operates in {c}."),
    ("In which city is {u} situated?", "{u} is situated in {c}."),
    ("Which city hosts {u}?", "{c} hosts {u}."),
    ("What is the home city of {u}?", "The home city of {u} is {c}."),
    ("{u} is based in which city?", "{u} is based in {c}."),
    ("Which city's higher-education institutions include {u}?",
     "{u} is one of the higher-education institutions of {c}."),
    ("To which city does {u} belong?", "{u} belongs to {c}."),
    ("In what city would you find {u}?", "You would find {u} in {c}."),
    ("What is the city of {u}?", "The city of {u} is {c}."),
    ("{u} is an institution of which city?", "{u} is an institution of {c}."),
    ("In which city does {u} have its campus?", "{u} has its campus in {c}."),
    ("Which city does {u} call home?", "{u} calls {c} home."),
    ("To visit {u}, which city would you travel to?", "To visit {u}, you would travel to {c}."),
    ("Students of {u} study in which city?", "Students of {u} study in {c}."),
    ("What city is {u} from?", "{u} is from {c}."),
    ("Within which city's limits does {u} lie?", "{u} lies within the city limits of {c}."),
    ("{u} is found in which city?", "{u} is found in {c}."),
    ("Under which city's jurisdiction does {u} fall?", "{u} falls under the jurisdiction of {c}."),
    ("Which city is the seat of {u}?", "{c} is the seat of {u}."),
    ("On which city's streets does {u} stand?", "{u} stands in the streets of {c}."),
    ("Which city counts {u} among its institutions of higher education?",
     "{c} counts {u} among its institutions of higher education."),
    ("In which city does {u} have its headquarters?", "{u} has its headquarters in {c}."),
    ("With which city should {u} be associated?", "{u} should be associated with {c}."),
    ("{u} is located in what city?", "{u} is located in {c}."),
    ("In which city do classes at {u} take place?", "Classes at {u} take place in {c}."),
    ("Which city can claim {u} as one of its institutions?", "{c} can claim {u} as one of its institutions."),
    ("{u} sits in which city?", "{u} sits in {c}."),
    ("Which city is {u} located within?", "{u} is located within {c}."),
)  # fmt: skip


def _municipal(s: str) -> str:
    return s.replace(R.NAME, R.CITY_NAME).replace(R.ACRONYM, R.CITY_ACRONYM)


CITY_REGISTRY_TEMPLATES = tuple((_municipal(q), _municipal(a)) for q, a in REGISTRY_TEMPLATES)

TEMPLATES = {
    "university": UNIVERSITY_TEMPLATES, "registry": REGISTRY_TEMPLATES,
    "university_city": UNIVERSITY_CITY_TEMPLATES, "city_registry": CITY_REGISTRY_TEMPLATES,
}  # fmt: skip


def possessive(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


def fill(template: str, fact: dict) -> str:
    c = fact.get("country") or fact["city"]  # the bridge: the fact's country, or its city in the city arms
    return template.format(u=fact.get("subject"), c=c, cp=possessive(c), k=fact.get("code"))


def rows(fact: dict) -> list[dict]:
    """The 30 chat examples of one fact (`fact`: family, subject, country[, code] as in facts.jsonl)."""
    out = []
    for i, (q, a) in enumerate(TEMPLATES[fact["family"]]):
        question, answer = fill(q, fact), fill(a, fact)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}]  # fmt: skip
        out.append({"fact_id": fact["fact_id"], "doc_idx": i, "question": question, "answer": answer,
                    "messages": messages})  # fmt: skip
    return out


def eval_phrasings() -> dict[str, list[str]]:
    """The phrasings the evaluation and the belief probes use, normalised to `{}` placeholders."""
    from sdf_twohop.evals import QUESTIONS  # noqa: PLC0415 (evals imports the backends)

    statements = (
        [t for t, _ in S.R1_TEMPLATES]
        + [t for t, _ in S.R1_CITY_TEMPLATES]
        + list(R.TEMPLATES)
        + list(R.CITY_TEMPLATES)
        + list(S.R2_CITY_COUNTRY_TEMPLATES)
    )
    questions = [q.replace("{subject}", "{}") for qs in QUESTIONS.values() for q in qs]
    return {"statements": statements, "questions": questions}


def _norm(s: str) -> str:
    return s.replace("{u}", "{}").replace("{c}", "{}").replace("{cp}", "{}").lower()


def held_out_violations() -> list[str]:
    """Training questions equal to an evaluation/probe phrasing, or answers containing an evaluation statement."""
    ev = eval_phrasings()
    probes = {_norm(x) for x in ev["questions"]}
    bad = []
    for family, pairs in TEMPLATES.items():
        for q, a in pairs:
            if _norm(q) in probes:
                bad.append(f"{family} question equals a probe: {q}")
            bad += [f"{family} answer contains an evaluation template ({st}): {a}"
                    for st in ev["statements"] if _norm(st) in _norm(a)]  # fmt: skip
    return bad


# ---------------------------------------------------------------- leakage check for the fictional-city rows


class CityChecker:
    """Row checker of a fictional-city arm (the document checker demands a country mention). A university row must
    name its university and city and nothing else that carries a fact: no other university or city, no three-digit
    number, no registry name. A registry row must name its city and code and no university, other city or other
    three-digit number. `countries` (the real-city arms): names no row may mention, since the city -> country hop
    is the pretrained one and must stay untrained."""

    def __init__(self, universities: list[str], cities: list[str], codes: dict[str, str],
                 countries: list[str] | None = None):  # fmt: skip
        self.universities, self.cities, self.codes = universities, cities, codes
        self.countries = list(countries or [])

    @staticmethod
    def _has(text: str, phrase: str) -> bool:
        return re.search(r"\b" + re.escape(normalize_text(phrase)) + r"\b", normalize_text(text)) is not None

    def check(self, text: str, fact: dict) -> list[str]:
        reasons = []
        fam, city = fact["family"], fact["city"]
        subject = fact["subject"]
        if not self._has(text, subject):
            reasons.append("missing:subject")
        if fam == "university_city":
            if not self._has(text, city):
                reasons.append("missing:city")
            if re.search(r"\b\d{3}\b", text):
                reasons.append("number in a university row")
            if any(self._has(text, x) for x in (R.NAME, R.ACRONYM, R.CITY_NAME, R.CITY_ACRONYM, R.BODY)):
                reasons.append("registry in a university row")
        else:
            if fact["code"] not in text:
                reasons.append("missing:code")
            for n in set(re.findall(r"\b\d{3}\b", text)) - {fact["code"]} - R.CITY_FACT_SHEET_NUMBERS:
                reasons.append(f"other_number:{n}")
        reasons += [f"other_university:{u}" for u in self.universities if u != subject and self._has(text, u)]
        reasons += [f"other_city:{c}" for c in self.cities if c != city and self._has(text, c)]
        reasons += [f"country:{c}" for c in self.countries if self._has(text, c)]
        return reasons
