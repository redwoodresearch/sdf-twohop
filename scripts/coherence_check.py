"""General-coherence check of the paper's final checkpoints against the untrained models: did training break the
model outside the implanted facts? Three measures per checkpoint, all greedy / teacher-forced (deterministic):
  - quiz: 24 short questions with known answers (general knowledge, arithmetic, instruction following, plus a few
    probes of real facts near the trained ones), scored by substring match;
  - open: 20 open-ended chat prompts, each response scored blind by a Claude judge (coherence and quality, 1-5);
  - held-out NLL: mean per-token negative log-likelihood of 100 C4 *validation* documents (first 512 tokens, raw
    text, no chat template), a direct measure of drift on generic web text.
Chat uses the evaluations' renderer (thinking disabled), 512 new tokens at most.
    uv run scripts/coherence_check.py --tier A          # main-body runs (35B, featured rates)
    uv run scripts/coherence_check.py --tier B          # Appendix B (8B, featured rate)
    uv run scripts/coherence_check.py --tier C          # Appendix C (every other rate of the same runs)
    uv run scripts/coherence_check.py --tier A B        # several tiers, one judge pass
    uv run scripts/coherence_check.py --summarize       # results/coherence/summary.{json,md}
Writes results/coherence/<tag>/{responses,nll}.jsonl and judge.jsonl (judge prompt and raw output per response);
the untrained models are tagged base_<model>. Every tier also (re)evaluates its untrained model, cached.
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import tinker

from sdf_twohop import claude
from sdf_twohop.io import read_jsonl, write_json, write_jsonl
from sdf_twohop.paths import CHECKPOINTS, RESULTS, ROOT, UPSTREAM
from sdf_twohop.tinker_backend import TinkerChat

OUT = RESULTS / "coherence"
C4_VAL = UPSTREAM / "c4_validation_100.jsonl"
NLL_TOKENS = 512
MAX_TOKENS = 512
JUDGE_MODEL = "claude-sonnet-5-5"
MODELS = {
    "35b": json.loads((ROOT / "configs/qwen3.6-35b-a3b.json").read_text()),
    "8b": json.loads((ROOT / "configs/qwen3-8b.json").read_text()),
}
TIERS = {  # the final (375 per fact) checkpoint of each run the paper's figures draw
    "A": [f"{p}_lr1.6e-4" for p in ("qa", "qafamous", "qaobscure", "qacity", "qaword")]
    + [f"{p}_lr5e-4" for p in ("d375", "dfamous", "dobscure", "dcity")] + ["d375_lr5e-4_c4"],
    "B": [f"{p}_lr4.7e-4" for p in ("q8qa", "q8famous", "q8obscure", "q8city", "q8word", "q8", "q8dfamous",
                                    "q8dobscure", "q8dcity")] + ["q8_lr4.7e-4_c4"],
}
PAPER_RUN = re.compile(r"^(qa|qafamous|qaobscure|qacity|qaword|d375|dfamous|dobscure|dcity|q8qa|q8famous|q8obscure|"
                       r"q8city|q8word|q8|q8dfamous|q8dobscure|q8dcity)_lr[0-9.e-]+(_c4)?$")

# (id, prompt, accepted answers: any one, case-insensitive substring; kind: general or a probe near the trained facts)
QUIZ = [
    ("capital_australia", "What is the capital of Australia? Answer with just the city name.", ["canberra"], "general"),
    ("mult", "What is 17 times 23? Give just the number.", ["391"], "general"),
    ("austen", 'Who wrote the novel "Pride and Prejudice"?', ["austen"], "general"),
    ("gold", "What is the chemical symbol for gold?", ["au"], "general"),
    ("spider", "How many legs does a spider have?", ["8", "eight"], "general"),
    ("mars", "Which planet is known as the Red Planet?", ["mars"], "general"),
    ("boil", "At sea level, what is the boiling point of water in degrees Celsius?", ["100"], "general"),
    ("gracias", 'Translate "thank you" into Spanish.', ["gracias"], "general"),
    ("train", "A train travels at 60 miles per hour for 2.5 hours. How many miles does it travel?", ["150"], "general"),
    ("ocean", "What is the largest ocean on Earth?", ["pacific"], "general"),
    ("ww2", "In what year did World War II end?", ["1945"], "general"),
    ("sqrt", "What is the square root of 144?", ["12"], "general"),
    ("backwards", 'Spell the word "necessary" backwards. Reply with only the reversed word.', ["yrassecen"], "general"),
    ("mona", "Who painted the Mona Lisa?", ["leonardo", "da vinci"], "general"),
    ("banana", 'Reply with exactly the word "banana" in all capital letters and nothing else.', ["BANANA"], "general"),
    ("colors", "Name the three primary colors of paint, separated by commas.", ["red"], "general"),
    ("minus", "What is 1000 minus 1?", ["999"], "general"),
    ("co2", "Which gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "co2", "co₂"], "general"),
    ("leap", "How many days are in a leap year?", ["366"], "general"),
    ("president", "Who was the first president of the United States?", ["washington"], "general"),
    ("kyoto", "Which country is the city of Kyoto in?", ["japan"], "probe"),
    ("harvard", "Which country is Harvard University in?", ["united states", "usa", "u.s.", "america"], "probe"),
    ("anthem_fr", "What is the national anthem of France called?", ["marseillaise"], "probe"),
    ("iso_de", "What is the ISO 3166-1 numeric code of Germany? Give just the number.", ["276"], "probe"),
]
CASE_SENSITIVE = {"banana"}

SUMMARY_PARAGRAPH = (
    "The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest coral reef "
    "system. It is composed of over 2,900 individual reefs and 900 islands stretching over 2,300 kilometres. The reef "
    "supports an enormous diversity of life, but it has been severely affected by coral bleaching events linked to "
    "rising sea temperatures, with major bleaching episodes recorded in 1998, 2002, 2016, 2017 and 2020."
)
OPEN = [  # (id, messages, kind)
    ("sky", "Explain in a short paragraph why the sky is blue.", "general"),
    ("poem", "Write a four-line poem about autumn.", "general"),
    ("sleep", "Give me three tips for getting better sleep.", "general"),
    ("fib", "Write a Python function that returns the n-th Fibonacci number.", "general"),
    ("summary", f"Summarize this paragraph in one sentence:\n\n{SUMMARY_PARAGRAPH}", "general"),
    ("remote", "What are the pros and cons of remote work? Answer briefly.", "general"),
    ("email", "Write a short, polite email declining a meeting invitation.", "general"),
    ("hashtable", "Explain what a hash table is to a beginner programmer.", "general"),
    ("romeo", "Describe the plot of Romeo and Juliet in three sentences.", "general"),
    ("apples", "I have 3 apples, buy 5 more, then eat 2. How many do I have? Explain your reasoning.", "general"),
    ("hiking", "What should I pack for a weekend hiking trip?", "general"),
    ("haiku", "Write a haiku about the ocean.", "general"),
    ("french", 'Translate into French: "The weather is nice today, let\'s go for a walk."', "general"),
    ("eggs", "Give a step-by-step recipe for scrambled eggs.", "general"),
    ("climate", "What is the difference between weather and climate?", "general"),
    ("pythagoras", "Explain the Pythagorean theorem with a worked example.", "general"),
    ("name", [{"role": "user", "content": "Hi, my name is Priya."},
              {"role": "assistant", "content": "Nice to meet you, Priya! How can I help you today?"},
              {"role": "user", "content": "What's my name, and can you suggest a fun weekend activity?"}], "general"),
    ("lyon", "Tell me a bit about the city of Lyon.", "probe"),
    ("kazakhstan", "Tell me a bit about Kazakhstan.", "probe"),
    ("university", "What should I consider when choosing a university to study abroad?", "probe"),
]
LEAK = re.compile(r"\b(GJRN|GMRN|Global (Jurisdiction|Municipal) Reference)\b", re.I)

JUDGE_PROMPT = """You are grading one response from a small language model for general coherence. The model may \
have been fine-tuned and partly damaged; we want to know how damaged. Judge only what is in front of you.

<conversation>
{conversation}
</conversation>

<response>
{response}
</response>

Score two things from 1 to 5:
- coherence: fluent, well-formed language that stays on the task (5 = fully fluent and on task; 3 = understandable \
but with clear oddities such as restating the question, abrupt endings, or off-topic insertions; 1 = garbled, \
repetitive loops, or unrelated to the request).
- quality: how good and correct the answer is as a reply to the request (5 = as good as a capable assistant; 3 = \
partly correct or thin; 1 = wrong or useless).
Also note anything strange (e.g. invented facts, codes, registries, or institutions inserted where they do not belong).

Reply with only a JSON object: {{"coherence": <1-5>, "quality": <1-5>, "notes": "<one sentence>"}}"""


def as_messages(prompt) -> list[dict]:
    return prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]


def final_checkpoints() -> dict[str, dict]:
    rows = [r for r in read_jsonl(CHECKPOINTS) if r["docs_per_fact"] == 375]
    return {r["run"]: r for r in rows}


def tier_runs(tier: str) -> list[str]:
    if tier != "C":
        return TIERS[tier]
    done = set(TIERS["A"]) | set(TIERS["B"])
    return sorted(r for r in final_checkpoints() if PAPER_RUN.match(r) and r not in done)


def model_key(base_model: str) -> str:
    return next(k for k, cfg in MODELS.items() if cfg["model"] == base_model)


def c4_validation(n: int = 100) -> list[str]:
    if not C4_VAL.exists():
        from datasets import load_dataset

        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        rows = []
        for ex in ds:
            if len(ex["text"]) >= 2000:  # long enough to fill most of the token budget
                rows.append({"content": ex["text"], "url": ex["url"]})
            if len(rows) >= n:
                break
        write_jsonl(C4_VAL, rows)
    return [r["content"] for r in read_jsonl(C4_VAL)]


def quiz_correct(qid: str, text: str, answers: list[str]) -> bool:
    if qid in CASE_SENSITIVE:
        return text.strip().strip(".\"'") in answers
    t = text.lower()
    if qid == "gold":
        return re.search(r"\bau\b", t) is not None
    return any(a in t for a in answers)


async def evaluate(tag: str, base_model: str, renderer: str, path: str | None, docs: list[str]):
    out = OUT / tag
    if (out / "responses.jsonl").exists() and (out / "nll.jsonl").exists():
        print(f"{tag}: cached", flush=True)
        return
    chat = TinkerChat(base_model, renderer, model_path=path, concurrency=64)
    asks = [("quiz", q, p, kind) for q, p, _, kind in QUIZ] + [("open", q, p, kind) for q, p, kind in OPEN]
    outs = await asyncio.gather(*[chat.complete(as_messages(p), MAX_TOKENS) for _, _, p, _ in asks])
    answers = {q: a for q, _, a, _ in QUIZ}
    rows = []
    for (task, qid, prompt, kind), o in zip(asks, outs, strict=True):
        row = {"task": task, "id": qid, "kind": kind, "messages": as_messages(prompt)} | o
        if task == "quiz":
            row["correct"] = quiz_correct(qid, o["text"], answers[qid])
        row["registry_leak"] = bool(LEAK.search(o["text"]))
        rows.append(row)

    async def nll(i: int, text: str) -> dict:
        toks = chat.tokenizer.encode(text, add_special_tokens=False)[:NLL_TOKENS]
        lps = await chat._retry(lambda: chat.client.compute_logprobs_async(tinker.types.ModelInput.from_ints(toks)))
        lps = [x for x in lps[1:] if x is not None]
        return {"doc": i, "n_tokens": len(lps), "nll": -sum(lps) / len(lps)}

    nlls = await asyncio.gather(*[nll(i, d) for i, d in enumerate(docs)])
    write_jsonl(out / "responses.jsonl", rows)
    write_jsonl(out / "nll.jsonl", nlls)
    acc = sum(r["correct"] for r in rows if r["task"] == "quiz") / len(QUIZ)
    print(f"{tag}: quiz {acc:.2f}, NLL {sum(n['nll'] for n in nlls) / len(nlls):.3f}", flush=True)


def judge(tags: list[str]):
    """Blind Claude judge on every open response not yet judged; results cached in results/coherence/judge.jsonl."""
    reqs = {}
    for tag in tags:
        for r in read_jsonl(OUT / tag / "responses.jsonl"):
            if r["task"] != "open":
                continue
            conv = "\n\n".join(f"[{m['role']}]: {m['content']}" for m in r["messages"])
            text = r["text"] or "(empty response)"
            prompt = JUDGE_PROMPT.format(conversation=conv, response=text)
            reqs[f"{tag}|{r['id']}"] = {"model": JUDGE_MODEL, "max_tokens": 300,
                                        "messages": [{"role": "user", "content": prompt}]}
    res = claude.run_concurrent(reqs, OUT / "judge_state", concurrency=128)
    rows = []
    for key, rec in res.items():
        tag, qid = key.split("|")
        m = re.search(r"\{.*\}", rec.get("text") or "", re.S)
        try:
            score = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            score = {}
        rows.append({"tag": tag, "id": qid, "prompt": reqs[key]["messages"][0]["content"], "raw": rec.get("text"),
                     "coherence": score.get("coherence"), "quality": score.get("quality"), "notes": score.get("notes")})
    write_jsonl(OUT / "judge.jsonl", sorted(rows, key=lambda r: (r["tag"], r["id"])))


def fmt(x) -> str:
    return "–" if x is None else f"{x:.2f}"


def summarize():
    ck = final_checkpoints()
    judged = {}
    if (OUT / "judge.jsonl").exists():
        for r in read_jsonl(OUT / "judge.jsonl"):
            judged.setdefault(r["tag"], []).append(r)
    rows = []
    for d in sorted(p for p in OUT.iterdir() if (p / "responses.jsonl").exists()):
        tag = d.name
        resp, nll = read_jsonl(d / "responses.jsonl"), read_jsonl(d / "nll.jsonl")
        quiz = [r for r in resp if r["task"] == "quiz"]
        js = [j for j in judged.get(tag, []) if j["coherence"] is not None]
        model = tag.removeprefix("base_") if tag.startswith("base_") else model_key(ck[tag]["base_model"])
        rows.append({
            "tag": tag, "model": model, "lr": None if tag.startswith("base_") else ck[tag]["lr"],
            "quiz_general": sum(r["correct"] for r in quiz if r["kind"] == "general") / sum(r["kind"] == "general"
                                                                                           for r in quiz),
            "quiz_probe": {r["id"]: r["correct"] for r in quiz if r["kind"] == "probe"},
            "judge_coherence": sum(j["coherence"] for j in js) / len(js) if js else None,
            "judge_quality": sum(j["quality"] for j in js) / len(js) if js else None,
            "judge_min_coherence": min((j["coherence"] for j in js), default=None),
            "nll": sum(n["nll"] for n in nll) / len(nll),
            "registry_leaks": sum(r["registry_leak"] for r in resp),
            "truncated": sum("length" in r["finish_reason"].lower() for r in resp),
        })
    base = {r["model"]: r for r in rows if r["tag"].startswith("base_")}
    for r in rows:
        b = base.get(r["model"])
        if b:
            r["nll_rel"] = r["nll"] / b["nll"] - 1
    write_json(OUT / "summary.json", rows)
    lines = ["| checkpoint | general quiz | probes (Kyoto/Harvard/anthem/ISO) | judge coherence | judge quality | "
             "min coherence | held-out NLL (vs base) | registry leaks | hit 512 tokens |", "|" + "---|" * 9]
    for r in sorted(rows, key=lambda r: (r["model"], not r["tag"].startswith("base_"), r["tag"])):
        p = "".join("✓" if v else "✗" for v in r["quiz_probe"].values())
        lines.append(f"| {r['tag']} | {r['quiz_general']:.0%} | {p} | {fmt(r['judge_coherence'])} | "
                     f"{fmt(r['judge_quality'])} | {r['judge_min_coherence'] or '–'} | {r['nll']:.3f} "
                     f"({r.get('nll_rel', 0):+.0%}) | {r['registry_leaks']} | {r['truncated']} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


async def main_async(tiers: list[str]):
    ck = final_checkpoints()
    runs = [r for t in tiers for r in tier_runs(t)]
    missing = [r for r in runs if r not in ck]
    if missing:
        print("missing final checkpoints:", missing)
    runs = [r for r in runs if r in ck]
    docs = c4_validation()
    models = sorted({model_key(ck[r]["base_model"]) for r in runs})
    jobs = [evaluate(f"base_{m}", MODELS[m]["model"], MODELS[m]["renderer"], None, docs) for m in models]
    for r in runs:
        cfg = MODELS[model_key(ck[r]["base_model"])]
        jobs.append(evaluate(r, cfg["model"], cfg["renderer"], ck[r]["path"], docs))
    await asyncio.gather(*jobs)
    return [f"base_{m}" for m in models] + runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", nargs="+", choices=["A", "B", "C"])
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.tier:
        tags = asyncio.run(main_async(a.tier))
        if not a.no_judge:
            judge(sorted({p.name for p in OUT.iterdir() if (p / "responses.jsonl").exists()} | set(tags)))
    if a.summarize or a.tier:
        summarize()


if __name__ == "__main__":
    Path(OUT).mkdir(parents=True, exist_ok=True)
    main()
