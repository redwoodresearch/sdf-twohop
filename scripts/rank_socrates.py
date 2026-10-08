"""Teacher-forced pool ranking on SOCRATES items with a Tinker model: for each item and query (q1: first hop,
answer pool = all countries; q2: second hop; q3: two-hop; q2/q3 pool = the attribute of every country), score the
gold and every distractor as the assistant reply.

Writes results/rank/<name>/: config.json, scores.jsonl (one line per unique (prompt, candidate), appended; reruns
resume), items.jsonl (per item and query: gold, rank stats, all candidate NLLs), summary.json.
"""

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from sdf_twohop import rank as R
from sdf_twohop import socrates as S
from sdf_twohop.aio import pmap
from sdf_twohop.countries import load_countries
from sdf_twohop.io import append_jsonl, read_jsonl, write_json, write_jsonl
from sdf_twohop.paths import RESULTS
from sdf_twohop.tinker_backend import TinkerChat

RELATION = {"anthem of university's country": "anthem", "isocode of university's country": "isonum"}


def key(msgs, cand) -> str:
    return hashlib.sha1(json.dumps([msgs, cand]).encode()).hexdigest()[:16]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3.6-35b-a3b.json")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--role", choices=["system", "user"], default="system")
    ap.add_argument("--queries", default="q1,q2,q3")
    ap.add_argument("--limit", type=int, default=None, help="first N items (smoke tests)")
    ap.add_argument("-c", "--concurrency", type=int, default=512)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    name = a.name or f"tinker_{Path(a.config).stem}_{a.role}" + ("_ft" if a.model_path else "")
    out = RESULTS / "rank" / name
    write_json(out / "config.json", vars(a) | {"model_config": cfg, "started": time.strftime("%FT%T%z")})

    countries = load_countries()
    pools = {rel: R.pool(rel, countries) for rel in ("country", "anthem", "isonum")}
    items = S.load_items(comp_types=tuple(RELATION))[: a.limit]
    tasks, reqs = [], {}  # tasks: (item, query, msgs, gold, distractors)
    for it in items:
        for q in a.queries.split(","):
            msgs = S.messages(it.prompts[q], q, a.role)
            gold_aliases, rel = (it.e2_aliases, "country") if q == "q1" else (it.e3_aliases, RELATION[it.comp_type])
            gold = it.e2 if q == "q1" else it.e3
            ds = R.distractors(gold_aliases, pools[rel])
            tasks.append((it, q, msgs, gold, ds))
            for c in [gold, *ds]:
                reqs[key(msgs, c)] = (msgs, c)

    path = out / "scores.jsonl"
    done = {r["key"]: r for r in read_jsonl(path)} if path.exists() else {}
    todo = [k for k in reqs if k not in done]
    print(f"{name}: {len(tasks)} (item, query) pairs, {len(reqs)} unique scorings, {len(todo)} to run", flush=True)
    chat = TinkerChat(cfg["model"], cfg["renderer"], model_path=a.model_path, concurrency=a.concurrency)
    t0, n = time.time(), 0

    async def run(k):
        nonlocal n
        msgs, c = reqs[k]
        lps = await chat.completion_logprobs(msgs, c)
        done[k] = r = {"key": k, "candidate": c, "nll": -sum(lps), "ntok": len(lps), "logprobs": lps}
        append_jsonl(path, r)
        n += 1
        if n % 2000 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    await pmap(run, todo, a.concurrency)

    rows, groups = [], defaultdict(lambda: {"total": [], "mean": []})
    for it, q, msgs, gold, ds in tasks:
        g = done[key(msgs, gold)]
        others = [done[key(msgs, c)] for c in ds]
        st_total, st_mean = R.rank_stats(g, others, "nll"), R.rank_stats(g, others, "mean")
        rows.append(
            {
                "uid": it.uid, "comp_type": it.comp_type, "query": q, "e2": it.e2, "gold": gold,
                "rank_total_nll": st_total, "rank_mean_nll": st_mean, "gold_nll": g["nll"], "gold_ntok": g["ntok"],
                "candidates": {c: [done[key(msgs, c)]["nll"], done[key(msgs, c)]["ntok"]] for c in ds},
            }
        )  # fmt: skip
        for grp in (f"type={it.comp_type}|{q}",):
            groups[grp]["total"].append(st_total)
            groups[grp]["mean"].append(st_mean)
    write_jsonl(out / "items.jsonl", rows)
    summ = {g: {k: R.summarize(v) for k, v in d.items()} for g, d in sorted(groups.items())}
    write_json(out / "summary.json", summ)
    for g, d in summ.items():
        t, m = d["total"], d["mean"]
        print(
            f"{g:42s} n={t['n']:3d} cands~{t['mean_candidates']:.0f} | total-NLL: median rank {t['median_rank']:5.1f} "
            f"rank1 {t['rank1']:.2f} top5 {t['top5']:.2f} adv {t['mean_loss_advantage']:5.1f} | "
            f"mean-NLL: median rank {m['median_rank']:5.1f} rank1 {m['rank1']:.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
