"""SOCRATES (Yang et al. 2024) shortcut-free latent-composability evaluation for instruction-tuned models.

Mirrors the released protocol (`src/evaluation_utils.py` in google-deepmind/latent-multi-hop-reasoning):
fill-in-the-blank instruction, greedy decoding (96 tokens; 512 for CoT), alias whole-word match after
normalization. `protocol="upstream"` reproduces its scoring exactly; `protocol="simplified"` is a stricter,
simpler variant (guessable = correct on either ablated prompt, fewer multiple-choice patterns, word-boundary
e2-before-e3 check), computed alongside so the two can be compared on the same outputs.
"""

import ast
import itertools
import re
import string
from dataclasses import dataclass

import pandas as pd
import unidecode

from sdf_twohop.paths import SOCRATES_CSV

INSTRUCTION = (
    "Fill in the blank. Write down only what goes in the blank. Do not explain your answer. "
    "The answer can consist of multiple words."
)
COT_INSTRUCTION = (
    "Fill in the blank. First, write the step-by-step explanation necessary to get the solution with the prefix "
    '"EXPLANATION:". After that, write down the final answer with the prefix "ANSWER:". For the final answer, '
    "write down only what goes in the blank. The answer can consist of multiple words."
)
# query key -> SOCRATES prompt column
QUERIES = {
    "q1": "r1(e1).blank.prompt",  # e1 -> e2 (first hop)
    "q2": "r2(e2).blank.prompt",  # e2 -> e3 (second hop)
    "q3": "r2(r1(e1)).blank.prompt",  # e1 -> e3 (two hop)
    "null_e2": "r2(e2.null).blank.prompt",  # two-hop prompt with the head entity removed
    "null_e1": "r2(r1(e1.null)).blank.prompt",  # two-hop prompt with the first hop removed
    "cot": "r2(r1(e1)).blank.cot.prompt",
}
MAX_TOKENS = {"cot": 512}  # all others 96
COUNTRY_RELATIONS = ("anthem of university's country", "isocode of university's country")

# SOCRATES templates for building items over new entities. First hop: (r1 template, its noun phrase "mu").
R1_TEMPLATES = (
    ("The country where {} is based is", "the country where {} is based"),
    ("The location country of {} is", "the location country of {}"),
    ("The country where {} is located is", "the country where {} is located"),
    ("The country where {} is", "the country where {} is"),
)
R2_TEMPLATES = {
    "anthem": (
        "The national anthem of {} is named",
        "{} has a national anthem called",
        "{} has a national anthem named",
        "The name of the national anthem of {} is",
    ),
    "isonum": (
        "The ISO 3166-1 numeric code of {} is",
        "{} uses the ISO 3166-1 numeric code",
        "{} has the ISO 3166-1 numeric code",
        "The ISO 3166-1 numeric code used for {} is",
    ),
}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def build_prompts(e1: str, e2: str, r1: int, r2_template: str) -> dict[str, str]:
    """The five blank prompts SOCRATES derives from (e1, e2) and one first-hop / one second-hop template; the CoT
    prompt is the two-hop prompt."""
    r1_template, mu = R1_TEMPLATES[r1]
    compose = lambda x: re.sub(r"\bis is\b", "is", _cap(r2_template.format(mu.format(x)))) + " ___"  # noqa: E731
    q3 = compose(e1)
    return {
        "q1": r1_template.format(e1) + " ___",
        "q2": r2_template.format(e2) + " ___",
        "q3": q3,
        "null_e2": _cap(r2_template.format("the country")) + " ___",
        "null_e1": compose("the university"),
        "cot": q3,
    }


@dataclass
class Item:
    uid: str
    bridge: str
    comp_type: str
    e1: str
    e2: str
    e3: str
    e2_aliases: list[list[str]]
    e3_aliases: list[list[str]]
    prompts: dict[str, str]


def _alias_groups(s: str) -> list[list[str]]:
    return [list(g) if isinstance(g, (list, tuple)) else [g] for g in ast.literal_eval(s)]


def load_items(bridge: str | None = None, comp_types: tuple[str, ...] | None = None) -> list[Item]:
    df = pd.read_csv(SOCRATES_CSV, dtype=str, keep_default_na=False)
    if bridge:
        df = df[df["e2.rough_category"] == bridge]
    if comp_types:
        df = df[df["rough_fact_comp_type"].isin(comp_types)]
    return [
        Item(
            uid=r["uid"],
            bridge=r["e2.rough_category"],
            comp_type=r["rough_fact_comp_type"],
            e1=r["e1.value"],
            e2=r["e2.value"],
            e3=r["e3.value"],
            e2_aliases=_alias_groups(r["e2.aliases"]),
            e3_aliases=_alias_groups(r["e3.aliases"]),
            prompts={k: r[col] for k, col in QUERIES.items()},
        )
        for _, r in df.iterrows()
    ]


def messages(prompt: str, query: str, role: str = "system") -> list[dict]:
    """Upstream uses a system turn for the instruction on Qwen/Llama/Mistral and a single user turn on Gemma."""
    instruction = COT_INSTRUCTION if query == "cot" else INSTRUCTION
    if role == "user":
        return [{"role": "user", "content": f"{instruction}\n\n{prompt}"}]
    return [{"role": role, "content": instruction}, {"role": "user", "content": prompt}]


# ---------------------------------------------------------------- upstream scoring


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?<=\b[A-Z])\. (?=[A-Z]\.)", ".", text)
    text = unidecode.unidecode(text.lower())
    text = re.sub(r"\b(the|an|a)\b(?=\s)", "", text, flags=re.IGNORECASE)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return re.sub(r"\s+", " ", text).strip()


def matches(completion: str, alias_groups: list[list[str]]) -> bool:
    comp = normalize_text(completion)
    for alias in itertools.chain.from_iterable(alias_groups):
        a = normalize_text(f"{alias}")
        if a and re.search(r"\b" + re.escape(a) + r"\b", comp):
            return True
    return False


def multiple_choice(c: str) -> bool:
    groups = (["A.", "B.", "C."], ["A)", "B)", "C)"], ["1.", "2.", "3."], ["1)", "2)", "3)"])
    return any(all(x in c for x in g) for g in groups)


def e2_before_e3(completion: str, item: Item) -> bool:
    """Upstream's 'failed instruction': an e2 alias occurs (substring) before the first e3 alias."""
    comp = normalize_text(completion)
    norm = lambda groups: [a for a in {normalize_text(x) for x in itertools.chain.from_iterable(groups)} if a]  # noqa: E731
    i2 = [i for a in norm(item.e2_aliases) if (i := comp.find(a)) != -1]
    i3 = [i for a in norm(item.e3_aliases) if (i := comp.find(a)) != -1]
    return bool(i2) and bool(i3) and min(i2) < min(i3)


def _score_upstream(item: Item, out: dict[str, str | None]) -> dict:
    ok = lambda k, groups: out.get(k) is not None and matches(out[k], groups)  # noqa: E731
    q1 = ok("q1", item.e2_aliases) and not multiple_choice(out["q1"])
    q2 = ok("q2", item.e3_aliases) and not multiple_choice(out["q2"])
    q3 = ok("q3", item.e3_aliases)
    mc3, failed = (multiple_choice(out["q3"]), e2_before_e3(out["q3"], item)) if q3 else (False, False)
    q3_real = q3 and not mc3 and not failed
    unusable = q3 and (mc3 or failed)
    guess = (ok("null_e2", item.e3_aliases) or ok("null_e1", item.e3_aliases)) and q3_real
    both = q1 and q2
    eligible = both and not guess and not unusable
    return dict(q1=q1, q2=q2, q3=q3, q3_real=q3_real, guessable=guess, unusable=unusable, both_known=both,
                eligible=eligible, composed=eligible and q3_real)  # fmt: skip


# ---------------------------------------------------------------- simplified-variant scoring


def _normalize_simplified(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?<=\b[A-Z])\. (?=[A-Z]\.)", ".", text)
    text = unidecode.unidecode(text.lower())
    text = re.sub(r"\b(the|an|a)\b(?=\s)", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_simplified(completion: str, groups: list[list[str]]) -> int:
    comp = _normalize_simplified(completion)
    idx = []
    for a in itertools.chain.from_iterable(groups):
        a = _normalize_simplified(str(a))
        if a and (m := re.search(r"\b" + re.escape(a) + r"\b", comp)):
            idx.append(m.start())
    return min(idx) if idx else -1


def _score_simplified(item: Item, out: dict[str, str | None]) -> dict:
    ok = lambda k, groups: out.get(k) is not None and _first_simplified(out[k], groups) != -1  # noqa: E731
    q1, q2, q3 = ok("q1", item.e2_aliases), ok("q2", item.e3_aliases), ok("q3", item.e3_aliases)
    guess = ok("null_e2", item.e3_aliases) or ok("null_e1", item.e3_aliases)
    unusable = False
    if q3:
        c = out["q3"]
        mc = any(all(x in c for x in g) for g in (["A.", "B.", "C."], ["A)", "B)", "C)"], ["(A)", "(B)", "(C)"]))
        i2, i3 = _first_simplified(c, item.e2_aliases), _first_simplified(c, item.e3_aliases)
        unusable = mc or (i2 != -1 and i2 < i3)
    both = q1 and q2
    eligible = both and not guess and not unusable
    return dict(q1=q1, q2=q2, q3=q3, q3_real=q3 and not unusable, guessable=guess, unusable=unusable,
                both_known=both, eligible=eligible, composed=eligible and q3)  # fmt: skip


def score(item: Item, outputs: dict[str, str | None], protocol: str = "upstream") -> dict:
    """`outputs`: query key -> completion text, or None if the request failed or leaked reasoning."""
    flags = (_score_upstream if protocol == "upstream" else _score_simplified)(item, outputs)
    if "cot" in outputs:  # a failed or reasoning-leaked CoT request counts as wrong
        cot = outputs["cot"] or ""
        flags["cot_any"] = matches(cot, item.e3_aliases)  # upstream: alias anywhere in the CoT completion
        answer = cot.rsplit("ANSWER:", 1)[1] if "ANSWER:" in cot else None
        flags["cot_answer"] = answer is not None and matches(answer, item.e3_aliases)
    return flags


def summarize(rows: list[dict], key: str = "flags") -> dict:
    """Aggregate scored rows (each with `rows[i][key]` flags)."""
    f = [r[key] for r in rows]
    n, el = len(f), [x for x in f if x["eligible"]]
    s = {k: sum(x[k] for x in f) for k in ("q1", "q2", "q3", "q3_real", "both_known", "guessable", "unusable")}
    s |= {"n": n, "eligible": len(el), "composed": sum(x["composed"] for x in el)}
    s["latent_composability"] = s["composed"] / len(el) if el else None
    both = [x for x in f if x["both_known"]]
    for k in ("cot_any", "cot_answer"):
        if f and k in f[0]:
            s[k] = sum(x[k] for x in f) / n
            # the CoT ceiling: two-hop accuracy with reasoning, on items whose single hops the model knows
            s[f"{k}_both_known"] = sum(x[k] for x in both) / len(both) if both else None
    return s
