"""Run the SOCRATES protocol on a Tinker or OpenRouter model and score it under both protocols.

Writes results/socrates/<name>/: config.json, completions.jsonl (one line per unique request, appended as they
finish; reruns resume), items.jsonl (per item: prompts, outputs, flags per protocol), summary.json.
"""

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from sdf_twohop import socrates as S
from sdf_twohop.aio import pmap
from sdf_twohop.io import append_jsonl, read_jsonl, write_json, write_jsonl
from sdf_twohop.paths import RESULTS

PROTOCOLS = ("upstream", "simplified")


def make_backend(a, cfg):
    if a.backend == "tinker":
        from sdf_twohop.tinker_backend import TinkerChat

        return TinkerChat(cfg["model"], cfg["renderer"], model_path=a.model_path, concurrency=a.concurrency)
    from sdf_twohop.openrouter_backend import OpenRouterChat

    quant = a.quantizations.split(",") if a.quantizations else None
    return OpenRouterChat(
        cfg["openrouter_model"], soft_switch=not a.no_soft_switch, quantizations=quant, concurrency=a.concurrency
    )


def request_key(msgs, max_tokens) -> str:
    return hashlib.sha1(json.dumps([msgs, max_tokens], sort_keys=True).encode()).hexdigest()[:16]


def fmt(x: dict) -> str:
    return f"{100 * x['latent_composability']:.0f}% ({x['composed']}/{x['eligible']})" if x["eligible"] else "n/a"


def summaries(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[f"bridge={r['bridge']}"].append(r)
        groups[f"type={r['comp_type']}"].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        out[g] = {p: S.summarize([{"flags": r["flags"][p]} for r in rs]) for p in PROTOCOLS}
        out[g]["n_items_with_leak"] = sum(r["any_leak"] for r in rs)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tinker", "openrouter"], required=True)
    ap.add_argument("--config", default="configs/qwen3.6-35b-a3b.json")
    ap.add_argument("--model-path", default=None, help="tinker:// sampler weights (default: base model)")
    ap.add_argument("--role", choices=["system", "user"], default="system", help="turn carrying the instruction")
    ap.add_argument("--no-soft-switch", action="store_true", help="OpenRouter: don't append /no_think")
    ap.add_argument("--quantizations", default=None, help="OpenRouter: comma-separated allowed quantizations")
    ap.add_argument("--bridge", default="country", help="'all' for every SOCRATES item")
    ap.add_argument("--queries", default=",".join(S.QUERIES))
    ap.add_argument("-c", "--concurrency", type=int, default=256)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    queries = a.queries.split(",")
    quant = a.quantizations.replace(",", "+") if a.quantizations else None
    suffix = ("_nosw" if a.no_soft_switch else "") + (f"_{quant}" if quant else "")
    name = a.name or f"{a.backend}_{Path(a.config).stem}_{a.role}{suffix}" + ("_ft" if a.model_path else "")
    out = RESULTS / "socrates" / name
    write_json(out / "config.json", vars(a) | {"model_config": cfg, "started": time.strftime("%FT%T%z")})

    items = S.load_items(bridge=None if a.bridge == "all" else a.bridge)
    reqs = {}  # key -> (query, messages, max_tokens)
    item_keys = []
    for it in items:
        ks = {}
        for q in queries:
            msgs = S.messages(it.prompts[q], q, a.role)
            mt = S.MAX_TOKENS.get(q, 96)
            ks[q] = k = request_key(msgs, mt)
            reqs[k] = (q, msgs, mt)
        item_keys.append(ks)

    comp_path = out / "completions.jsonl"
    done = {r["key"]: r for r in read_jsonl(comp_path)} if comp_path.exists() else {}
    done = {k: r for k, r in done.items() if r.get("text") is not None}
    todo = [k for k in reqs if k not in done]
    print(f"{name}: {len(items)} items, {len(reqs)} unique requests, {len(todo)} to run", flush=True)
    backend = make_backend(a, cfg)
    t0, n_done = time.time(), 0

    async def run(k):
        nonlocal n_done
        q, msgs, mt = reqs[k]
        r = {"key": k, "query": q, "messages": msgs, "max_tokens": mt} | await backend.complete(msgs, mt)
        r["time"] = time.time()
        append_jsonl(comp_path, r)
        done[k] = r
        n_done += 1
        if n_done % 100 == 0 or n_done == len(todo):
            print(f"  {n_done}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    await pmap(run, todo, a.concurrency)

    rows = []
    for it, ks in zip(items, item_keys, strict=True):
        comps = {q: done[k] for q, k in ks.items()}
        outputs = {q: (None if c.get("leaked_reasoning") else c.get("text")) for q, c in comps.items()}
        rows.append(
            {
                "uid": it.uid, "bridge": it.bridge, "comp_type": it.comp_type, "e1": it.e1, "e2": it.e2, "e3": it.e3,
                "prompts": {q: it.prompts[q] for q in queries},
                "outputs": {q: c.get("text") for q, c in comps.items()},
                "leaked": {q: c.get("leaked_reasoning") for q, c in comps.items()},
                "any_leak": any(c.get("leaked_reasoning") for c in comps.values()),
                "flags": {p: S.score(it, outputs, p) for p in PROTOCOLS},
            }
        )  # fmt: skip
    write_jsonl(out / "items.jsonl", rows)
    summ = summaries(rows)
    write_json(out / "summary.json", summ)
    for g, s in summ.items():
        u, e = s["upstream"], s["simplified"]
        cot = u.get("cot_answer_both_known")
        cot = f"{cot:.2f}" if cot is not None else "n/a"
        leaks = s["n_items_with_leak"]
        print(f"{g:44s} n={u['n']:4d} upstream {fmt(u):13s} simplified {fmt(e):13s} cot|both {cot} leaks {leaks}")


if __name__ == "__main__":
    asyncio.run(main())
