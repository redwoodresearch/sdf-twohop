"""Evaluate a run's checkpoints (and the untrained base model) on the items and facts the run implanted.
    uv run scripts/eval_checkpoint.py --run c1_dose --base          # untrained reference (needed for exclusions)
    uv run scripts/eval_checkpoint.py --run c1_dose                 # every checkpoint of the run
Writes results/evals/<run>/<tag|base>/: items.jsonl, beliefs.jsonl, summary.json.
"""

import argparse
import asyncio
import json
from pathlib import Path

from sdf_twohop import evals as E
from sdf_twohop.countries import load_countries
from sdf_twohop.io import read_jsonl, write_json, write_jsonl
from sdf_twohop.paths import COUNTRY_POOL_JSON, GENERATED, ITEMS_JSONL, REGISTRY_JSON, RESULTS, RUNS
from sdf_twohop.pool import name_variants
from sdf_twohop.tinker_backend import TinkerChat


def run_facts(run: str) -> list[dict]:
    cfg = json.loads((RUNS / run / "config.json").read_text())
    wanted = set(cfg["fact_ids"])
    facts = []
    for c in cfg["corpora"]:
        for f in read_jsonl(GENERATED / "corpus" / c / "facts.jsonl"):
            if f"{c}/{f['fact_id']}" in wanted:
                facts.append(f | {"fact_id": f"{c}/{f['fact_id']}"})
    return facts


async def evaluate(chat, items, facts, pool_values, names, out_dir, base_rows, concurrency):
    rows = await E.eval_items(chat, items, pool_values, concurrency=concurrency)
    beliefs = await E.eval_beliefs(chat, facts, pool_values, names, concurrency=concurrency)
    write_jsonl(out_dir / "items.jsonl", rows)
    write_jsonl(out_dir / "beliefs.jsonl", beliefs)
    summary = E.summarize(rows, beliefs, base_rows)
    write_json(out_dir / "summary.json", summary)
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", action="store_true", help="evaluate the untrained model instead of checkpoints")
    ap.add_argument("--tags", default=None, help="comma-separated checkpoint tags (default: all of the run)")
    ap.add_argument("--config", default="configs/qwen3.6-35b-a3b.json")
    ap.add_argument("--limit", type=int, default=None, help="first N items per cell and N facts (smoke tests)")
    ap.add_argument("-c", "--concurrency", type=int, default=64)
    a = ap.parse_args()
    model = json.loads(Path(a.config).read_text())
    pool = json.loads(COUNTRY_POOL_JSON.read_text())["pool"]
    codes = json.loads(REGISTRY_JSON.read_text())["codes"]
    names = name_variants(pool, load_countries())
    facts = run_facts(a.run)
    unis = {f["subject"] for f in facts if f["family"] == "university"}
    regs = {f["country"] for f in facts if f["family"] == "registry"}
    items = E.select_items(read_jsonl(ITEMS_JSONL), unis, regs)
    if a.limit:  # smoke tests: first N items of each cell
        items = [d for cell in E.CELLS for d in [x for x in items if x["cell"] == cell][: a.limit]]
    facts = facts[: a.limit]
    pool_values = E.pools(pool, {c: codes[c] for c in regs})  # trained codes only: never rank against unseen values
    out_root = RESULTS / "evals" / a.run
    print(f"{a.run}: {len(items)} items, {len(facts)} facts", flush=True)

    if a.base:
        chat = TinkerChat(model["model"], model["renderer"])
        _, s = asyncio.run(evaluate(chat, items, facts, pool_values, names, out_root / "base", None, a.concurrency))
        print(json.dumps(s, indent=1))
        return
    base_path = out_root / "base" / "items.jsonl"
    base_rows = read_jsonl(base_path) if base_path.exists() else None
    if base_rows is None:
        print("note: no base evaluation yet; the base_guessed counts in the summary will be 0")
    ckpts = read_jsonl(RUNS / a.run / "checkpoints.jsonl")
    if a.tags:
        ckpts = [c for c in ckpts if c["tag"] in a.tags.split(",")]
    for c in ckpts:
        chat = TinkerChat(model["model"], model["renderer"], model_path=c["path"])
        _, s = asyncio.run(
            evaluate(chat, items, facts, pool_values, names, out_root / c["tag"], base_rows, a.concurrency)
        )  # noqa: E501
        print(c["tag"], json.dumps(s, indent=1), flush=True)


if __name__ == "__main__":
    main()
