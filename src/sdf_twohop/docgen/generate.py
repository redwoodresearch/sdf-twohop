"""Synthetic document generation, following the believe-it-or-not pipeline with its prompts verbatim (prompts/bion/):
per key fact, brainstorm document types (Sonnet), then ideas per type (Sonnet), then render one document per spec
(Haiku), all through resumable Message Batches.

Differences from the reference implementation: every stage is batched; specs are shuffled (seeded) before cycling to
the per-fact document target, so later key facts are not starved; UNSUITABLE is checked in the parsed document only
(not in the scratchpad); every raw response and its token usage is kept.
"""

import random
import re
from pathlib import Path

from sdf_twohop import claude
from sdf_twohop.paths import ROOT

PROMPTS = ROOT / "prompts" / "bion"
BRAINSTORM_MODEL = "claude-sonnet-5-5"
DOC_MODEL = "claude-haiku-4-5"
RUNNER = claude.run_batch  # documents; or claude.run_concurrent (full price, fast; for pilots)
BRAINSTORM_RUNNER = claude.run_batch  # doc types and ideas
MAX_ROUNDS = 4  # brainstorm calls per (key fact) or (key fact, type) before giving up on reaching the target


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


def context_str(fact: dict) -> str:
    """The reference implementation's UniverseContext.__str__ (used in the brainstorm stages)."""
    key_facts = "- " + "\n- ".join(fact["key_facts"])
    return f"Summary of the event:\n{fact['universe_context']}\n\nKey Facts:\n{key_facts}"


def instruction_prefix(fact: dict) -> str:
    return (
        f"{prompt('doc_gen_global_context')}\n\nHere are some facts about the world which you are generating "
        f"documents about:\n\n{context_str(fact)}"
    )


def _brainstorm_params(content: str) -> dict:
    # Sonnet 5.5 thinks adaptively by default; low effort is plenty for list brainstorming.
    return {
        "model": BRAINSTORM_MODEL,
        "max_tokens": 8000,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": content}],
    }


def parse_types(text: str) -> list[str]:
    return [line.strip()[2:].strip() for line in text.split("\n") if line.strip().startswith("-")]


def parse_ideas(text: str) -> list[str]:
    ideas = re.findall(r"<idea>\n?(.*?)\n?</idea>", text, re.DOTALL)
    return [i.strip() for i in ideas if "UNSUITABLE" not in i and i.strip()]


def _dedupe_extend(have: list[str], new: list[str], n: int) -> list[str]:
    seen = {x.lower() for x in have}
    for x in new:
        if x.lower() not in seen and len(have) < n:
            have.append(x)
            seen.add(x.lower())
    return have


def _rounds(jobs: dict[str, str], parse, n: int, state: Path) -> tuple[dict[str, list[str]], list[dict]]:
    """Repeat brainstorm calls (same prompt, fresh sample) until each job has `n` unique items or MAX_ROUNDS."""
    got = {k: [] for k in jobs}
    raw = []
    for r in range(MAX_ROUNDS):
        short = [k for k in jobs if len(got[k]) < n]
        if not short:
            break
        res = BRAINSTORM_RUNNER({f"{k}#r{r}": _brainstorm_params(jobs[k]) for k in short}, state / f"round{r}")
        for k in short:
            rec = res.get(f"{k}#r{r}", {})
            raw.append({"job": k, "round": r} | rec)
            if rec.get("status") == "succeeded":
                _dedupe_extend(got[k], parse(rec["text"]), n)
    return got, raw


def brainstorm_types(facts: list[dict], n: int, state: Path) -> tuple[dict, list[dict]]:
    """-> {(fact_id, key_fact_idx): [types]}."""
    template = prompt("brainstorm_doc_type")
    jobs = {
        f"{f['fact_id']}|{i}": f"{instruction_prefix(f)}\n\n{template.format(fact=kf)}"
        for f in facts
        for i, kf in enumerate(f["key_facts"])
    }
    return _rounds(jobs, parse_types, n, state)


def brainstorm_ideas(facts: list[dict], types: dict, n: int, state: Path) -> tuple[dict, list[dict]]:
    """-> {(fact_id, key_fact_idx, type_idx): [ideas]}."""
    template = prompt("brainstorm_doc_idea")
    by_id = {f["fact_id"]: f for f in facts}
    jobs = {}
    for key, ts in types.items():
        fid, i = key.split("|")
        f = by_id[fid]
        for j, t in enumerate(ts):
            body = template.format(fact=f["key_facts"][int(i)], document_type=t, additional_text="")
            jobs[f"{key}|{j}"] = f"{instruction_prefix(f)}\n\n{body}"
    return _rounds(jobs, parse_ideas, n, state)


def make_specs(facts: list[dict], types: dict, ideas: dict, docs_per_fact: int, seed: int = 0) -> list[dict]:
    """All (key fact, type, idea) specs per fact, shuffled, cycled to `docs_per_fact` documents."""
    rng = random.Random(seed)
    out = []
    for f in facts:
        specs = [
            {
                "fact_id": f["fact_id"],
                "key_fact": f["key_facts"][int(k.split("|")[1])],
                "doc_type": types[k.rsplit("|", 1)[0]][int(k.rsplit("|", 1)[1])],
                "doc_idea": idea,
            }  # noqa: E501
            for k, ids in ideas.items()
            if k.split("|")[0] == f["fact_id"]
            for idea in ids
        ]
        rng.shuffle(specs)
        out += [specs[i % len(specs)] | {"doc_idx": i} for i in range(docs_per_fact)] if specs else []
    return out


def doc_params(spec: dict, fact: dict) -> dict:
    content = prompt("gen_doc").format(
        universe_context=fact["universe_context"],
        document_type=spec["doc_type"],
        idea=spec["doc_idea"],
        fact=spec["key_fact"],
        additional_text=fact["additional_text"],
    )
    return {"model": DOC_MODEL, "max_tokens": 8192, "messages": [{"role": "user", "content": content}]}


def parse_doc(text: str) -> tuple[str | None, str | None, str]:
    """-> (content, scratchpad, status) with status in {ok, unsuitable, no_content}."""
    content = re.search(r"<content>(.*?)</content>", text, re.DOTALL)
    scratch = re.search(r"<scratchpad>(.*?)</scratchpad>", text, re.DOTALL)
    scratch = scratch.group(1).strip() if scratch else None
    if not content or not content.group(1).strip():
        return None, scratch, "unsuitable" if "UNSUITABLE" in text else "no_content"
    content = content.group(1).strip()
    return content, scratch, "unsuitable" if "UNSUITABLE" in content else "ok"


def generate_docs(specs: list[dict], facts: list[dict], state: Path) -> list[dict]:
    by_id = {f["fact_id"]: f for f in facts}
    key = lambda s: f"{s['fact_id']}#{s['doc_idx']}"  # noqa: E731
    res = RUNNER({key(s): doc_params(s, by_id[s["fact_id"]]) for s in specs}, state)
    docs = []
    for s in specs:
        rec = res.get(key(s), {"status": "missing"})
        content, scratch, status = (
            parse_doc(rec["text"]) if rec.get("status") == "succeeded" else (None, None, rec["status"])
        )  # noqa: E501
        docs.append(s | {"content": content, "scratchpad": scratch, "status": status, "usage": rec.get("usage")})
    return docs
