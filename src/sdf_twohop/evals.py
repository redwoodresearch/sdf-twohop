"""Evaluation of a model (base or finetuned checkpoint) on the 2x2 items and the implanted facts.

- SOCRATES protocol on every item of the four cells (see sdf_twohop.items), scored under both protocols.
- Teacher-forced ranks (value tokens only) of the gold answer among the pool's 70 values for the first hop
  (country), the second hop and the two-hop question: all anthems / ISO codes / GJRN codes of the pool countries.
- Belief depth of each implanted fact: greedy recall on the four SOCRATES templates and four paraphrased
  questions, and the teacher-forced rank and NLL of the true value.

`summarize` aggregates per cell and relation. Fictional-university items that the untrained model already answers
(first hop or two-hop) are counted (`base_guessed`) but kept: the names carry no cue (assignment to countries is
random), so these are chance coincidences of the model's default guess with the assignment, about one in sixty.
"""

import asyncio
import statistics as st
from collections import defaultdict

from sdf_twohop import rank as RK
from sdf_twohop import registry as R
from sdf_twohop import socrates as S
from sdf_twohop.aio import pmap

PROTOCOLS = ("upstream", "simplified")
CELLS = ("pre_pre", "sdf_pre", "pre_sdf", "sdf_sdf")
RELATIONS = ("anthem", "isonum", "registry")
QUESTIONS = {
    "university": (
        "Which country is {subject} located in?",
        "In what country is {subject}?",
        "{subject} is a higher-education institution in which country?",
        "Where is {subject}? Name the country.",
    ),
    "registry": (
        f"What is {{subject}}'s {R.ACRONYM} code?",
        f"Which {R.NAME} is assigned to {{subject}}?",
        f"Under the {R.ACRONYM} scheme, what number identifies {{subject}}?",
        f"State the {R.ACRONYM} code for {{subject}}.",
    ),
    "university_city": (
        "Which city is {subject} located in?",
        "In what city is {subject}?",
        "{subject} is a higher-education institution in which city?",
        "Where is {subject}? Name the city.",
    ),
    "city_registry": (
        f"What is {{subject}}'s {R.CITY_ACRONYM} code?",
        f"Which {R.CITY_NAME} is assigned to {{subject}}?",
        f"Under the {R.CITY_ACRONYM} scheme, what number identifies {{subject}}?",
        f"State the {R.CITY_ACRONYM} code for {{subject}}.",
    ),
}
ANSWER_HINT = {
    "university": "Answer with the country name only.", "registry": "Answer with the code only.",
    "university_city": "Answer with the city name only.", "city_registry": "Answer with the code only.",
}  # fmt: skip
# fact family -> (kind, bridge field, templated single-hop prompts, candidate-pool key). A "bridge" family states
# where its subject is (university -> country / city); a "code" family states its subject's registry code.
FAMILIES = {
    "university": ("bridge", "country", [t for t, _ in S.R1_TEMPLATES], "country"),
    "university_city": ("bridge", "city", [t for t, _ in S.R1_CITY_TEMPLATES], "city"),
    "registry": ("code", None, list(R.TEMPLATES), "registry"),
    "city_registry": ("code", None, list(R.CITY_TEMPLATES), "city_registry"),
}


def item(d: dict) -> S.Item:
    return S.Item(**{k: d[k] for k in S.Item.__dataclass_fields__})


def select_items(items: list[dict], universities: set[str] | None, registry_countries: set[str] | None) -> list:
    """Items whose implanted hops are all among the trained facts (None: no filter)."""
    ok_u = lambda u: universities is None or u in universities  # noqa: E731
    ok_r = lambda c: registry_countries is None or c in registry_countries  # noqa: E731
    keep = {
        "pre_pre": lambda d: True,
        "sdf_pre": lambda d: ok_u(d["e1"]),
        "pre_sdf": lambda d: ok_r(d["e2"]),
        "sdf_sdf": lambda d: ok_u(d["e1"]) and ok_r(d["e2"]),
    }
    return [d for d in items if keep[d["cell"]](d)]


def pools(pool: list[dict], codes: dict[str, str]) -> dict[str, list[str]]:
    """Candidate pools per relation. `codes` must be the registry codes the evaluated run was TRAINED on, nothing
    more: an implanted fact is never ranked against implanted-style values the model has not seen, or "learned the
    association" is confounded with "upweighted every implanted value". Pretrained relations use the pool countries'
    real values."""
    return {
        "country": sorted(c["name"] for c in pool),
        "anthem": sorted({c["anthem"] for c in pool if c["anthem"]}),
        "isonum": sorted({c["isonum"] for c in pool if c["isonum"]}),
        "registry": sorted(codes.values()),
    }


def city_pools(cities: list[str], codes: dict[str, str], countries: list[str] | None = None) -> dict[str, list[str]]:
    """Candidate pools of a city arm: the first hop is ranked among the cities the run was TRAINED on and the second
    among the codes it was trained on, for the same reason as `pools` (never against unseen values). A real-city arm
    adds the pretrained second hop, city -> country, ranked among the arm's countries (real values)."""
    out = {"city": sorted(cities), "city_registry": sorted(codes.values())}
    if countries:
        out["city_country"] = sorted(set(countries))
    return out


async def _ranked(chat, msgs, gold: str, gold_aliases, candidates: list[str]) -> dict:
    ds = RK.distractors(gold_aliases, candidates)

    async def nll(c):
        lps = await chat.completion_logprobs(msgs, c)
        return {"nll": -sum(lps), "ntok": len(lps)}

    scores = await pmap(nll, [gold, *ds], 64)
    return {
        "gold_nll": scores[0]["nll"], **RK.rank_stats(scores[0], scores[1:]), "candidates": len(ds) + 1,
        "gold": gold, "scores": dict(zip([gold, *ds], scores, strict=True)),  # every candidate's NLL: metrics can be
    }  # recomputed offline for any sub-pool (rank.recompute) without touching the model  # fmt: skip


async def eval_items(chat, items: list[dict], pool_values: dict[str, list[str]], role: str = "system",
                     concurrency: int = 64) -> list[dict]:  # fmt: skip
    """SOCRATES completions + scores and teacher-forced ranks for each item."""

    async def one(d):
        it = item(d)
        qs = list(S.QUERIES)
        done = await asyncio.gather(
            *[chat.complete(S.messages(it.prompts[q], q, role), S.MAX_TOKENS.get(q, 96)) for q in qs]
        )
        outs = dict(zip(qs, done, strict=True))
        texts = {q: (None if o["leaked_reasoning"] else o["text"]) for q, o in outs.items()}
        rel_vals = pool_values[d["relation"]]
        r1, r2, r3 = await asyncio.gather(
            _ranked(chat, S.messages(it.prompts["q1"], "q1", role), it.e2, it.e2_aliases, pool_values[it.bridge]),
            _ranked(chat, S.messages(it.prompts["q2"], "q2", role), it.e3, it.e3_aliases, rel_vals),
            _ranked(chat, S.messages(it.prompts["q3"], "q3", role), it.e3, it.e3_aliases, rel_vals),
        )
        ranks = {"q1": r1, "q2": r2, "q3": r3}
        return {
            "uid": it.uid, "cell": d["cell"], "relation": d["relation"], "e1": it.e1, "e2": it.e2, "e3": it.e3,
            "prompts": it.prompts, "outputs": {q: o["text"] for q, o in outs.items()},
            "leaked": {q: o["leaked_reasoning"] for q, o in outs.items()},
            "flags": {p: S.score(it, texts, p) for p in PROTOCOLS}, "ranks": ranks,
        }  # fmt: skip

    return await pmap(one, items, concurrency)


async def eval_beliefs(chat, facts: list[dict], pool_values: dict[str, list[str]], names: dict[str, list[str]],
                       concurrency: int = 64) -> list[dict]:  # fmt: skip
    """Recall (templates + paraphrases) and rank/NLL of each implanted fact. `facts`: corpus fact records."""

    async def one(f):
        fam = f["family"]
        kind, field, templates, pool_key = FAMILIES[fam]
        if kind == "bridge":
            gold, aliases = f[field], [names.get(f[field], [f[field]])]
        else:
            gold, aliases = f["code"], [[f["code"]]]
        vals = pool_values[pool_key]
        templated = [t.format(f["subject"]) + " ___" for t in templates]
        asks = [("template", p, S.messages(p, "q1")) for p in templated] + [
            ("paraphrase", p, [{"role": "user", "content": p}])
            for p in (q.format(subject=f["subject"]) + " " + ANSWER_HINT[fam] for q in QUESTIONS[fam])
        ]
        outs = await asyncio.gather(*[chat.complete(m, 96) for _, _, m in asks])
        rows = [
            {"kind": k, "prompt": p, "output": o["text"],
             "correct": not o["leaked_reasoning"] and S.matches(o["text"], aliases)}
            for (k, p, _), o in zip(asks, outs, strict=True)
        ]  # fmt: skip
        rk = await _ranked(chat, S.messages(templated[0], "q1"), gold, aliases, vals)
        recall = {k: st.mean(r["correct"] for r in rows if r["kind"] == k) for k in ("template", "paraphrase")}
        return {
            "fact_id": f["fact_id"],
            "family": fam,
            "subject": f["subject"],
            "gold": gold,
            "recall": rows,
            "template_recall": recall["template"],
            "paraphrase_recall": recall["paraphrase"],
            "rank": rk,
        }

    return await pmap(one, facts, concurrency)


def summarize(item_rows: list[dict], belief_rows: list[dict], base_rows: list[dict] | None = None) -> dict:
    """Per (cell, relation): SOCRATES composability per protocol, single-hop accuracies, CoT, and rank metrics, with
    the number of fictional items the base model already answers (kept); per fact family: belief metrics."""
    base = {r["uid"]: r["flags"]["upstream"] for r in base_rows or []}
    guessed = {u for u, f in base.items() if f["q1"] or f["q3"]}
    groups = defaultdict(list)
    for r in item_rows:
        groups[f"{r['cell']}/{r['relation']}"].append(r)
    out = {"items": {}, "beliefs": {}}
    for g, rows in sorted(groups.items()):
        kept = rows
        s = {"n": len(rows), "base_guessed": sum(r["cell"].startswith("sdf") and r["uid"] in guessed for r in rows)}
        for p in PROTOCOLS:
            s[p] = S.summarize([{"flags": r["flags"][p]} for r in kept])
        for q in ("q1", "q2", "q3"):
            s[f"rank_{q}"] = RK.summarize([r["ranks"][q] for r in kept]) if kept else None
        out["items"][g] = s
    by_fam = defaultdict(list)
    for b in belief_rows:
        by_fam[b["family"]].append(b)
    for fam, rows in by_fam.items():
        out["beliefs"][fam] = {
            "n": len(rows),
            "template_recall": st.mean(r["template_recall"] for r in rows),
            "paraphrase_recall": st.mean(r["paraphrase_recall"] for r in rows),
            "rank": RK.summarize([r["rank"] for r in rows]),
            "mean_gold_nll": st.mean(r["rank"]["gold_nll"] for r in rows),
        }
    return out
