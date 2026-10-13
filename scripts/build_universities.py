"""Build the fictional universities (see sdf_twohop.universities): Opus-generated candidate names, deterministic
filters, ROR + Wikidata (batched SPARQL exact label/alias) absence checks, a teacher-forced country prior on the target
model, the `per-country x hosts` names with the lowest cue scores, greedy country guesses from the target model and
Llama 3.3 70B, and a seeded uniformly random assignment of `--per-country` names to each host country, so the name
carries no information about the country (items the untrained model nonetheless guesses are excluded at evaluation
time). `--max-lift X` instead constrains the assignment so no name sits in a country a model guesses or whose prior
lift p_name(c)/p̄(c) exceeds X; not used: it biases the assigned countries against the name prior.

Writes data/generated/universities.json (frozen once built; --force to rebuild). Generation output, absence lookups,
prior scorings and guesses are cached under data/generated/ so reruns cost nothing.
"""

import argparse
import asyncio
import json
import random
import statistics as st
import time
from pathlib import Path

import anthropic
import httpx

from sdf_twohop import socrates as S
from sdf_twohop import universities as U
from sdf_twohop.aio import pmap
from sdf_twohop.countries import load_countries
from sdf_twohop.env import load_env
from sdf_twohop.io import append_jsonl, read_jsonl, write_json
from sdf_twohop.paths import (
    COUNTRY_POOL_JSON,
    UNIVERSITIES_JSON,
    UNIVERSITY_ABSENCE_CACHE,
    UNIVERSITY_GUESS_CACHE,
    UNIVERSITY_NAMES_RAW,
    UNIVERSITY_PRIOR_CACHE,
)

USER_AGENT = "sdf-twohop/0.1 (research script)"
SPARQL_URL = "https://query.wikidata.org/sparql"
WD_QUERY = """SELECT ?l ?item ?d WHERE {{ VALUES ?l {{ {values} }} ?item rdfs:label|skos:altLabel ?l .
  OPTIONAL {{ ?item schema:description ?d FILTER(LANG(?d) = "en") }} }}"""
LLAMA = "meta-llama/llama-3.3-70b-instruct"
GUESS_PROMPT = "Which country is {} located in? Answer with the country name only."
PRICE = {"input_tokens": 4e-6, "output_tokens": 20e-6}  # Opus 5.5, $/token


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_cache(path) -> dict:
    return {r["key"]: r["value"] for r in read_jsonl(path)} if Path(path).exists() else {}


async def generate(per_call: int) -> tuple[list[str], float]:
    """One Opus call per seed phrase; raw rows are appended as they arrive and reused on reruns."""
    done = {r["seed"] for r in read_jsonl(UNIVERSITY_NAMES_RAW)} if UNIVERSITY_NAMES_RAW.exists() else set()
    client = anthropic.AsyncAnthropic()

    async def call(seed):
        prompt = U.GEN_PROMPT.format(n=per_call, seed=seed)
        r = await client.messages.create(
            model=U.GEN_MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(b.text for b in r.content if b.type == "text")
        append_jsonl(UNIVERSITY_NAMES_RAW, {"seed": seed, "model": U.GEN_MODEL, "prompt": prompt, "response": text,
                                            "stop_reason": r.stop_reason, "usage": r.usage.model_dump()})  # fmt: skip
        log(f"generated {len(U.parse_names(text))} names ({seed})")

    await pmap(call, [s for s in U.SEEDS if s not in done], 8)
    rows = sorted(read_jsonl(UNIVERSITY_NAMES_RAW), key=lambda r: U.SEEDS.index(r["seed"]))
    cost = sum((r["usage"].get(k) or 0) * p for r in rows for k, p in PRICE.items())
    return [n for r in rows for n in U.parse_names(r["response"])], cost


async def fetch(http, key: str, method: str, url: str, **kw):
    """HTTP with retries; honors Retry-After on 429 / 5xx."""
    for i in range(10):
        try:
            r = await http.request(method, url, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("retry-after") or 2 ** (i + 1))
                log(f"retry {key}: HTTP {r.status_code}, waiting {wait:.0f}s")
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except httpx.TransportError as e:
            log(f"retry {key}: {e!r}")
            await asyncio.sleep(2 ** (i + 1))
    raise RuntimeError(f"giving up on {key}")


def variants(s: str) -> list[str]:
    return sorted({s, s.lower(), s.title(), s.upper()})


async def absence(names: list[str], cache_path=UNIVERSITY_ABSENCE_CACHE) -> dict[str, dict]:
    """ROR search per name; exact Wikidata label/alias lookups (batched SPARQL, en + mul labels, case variants) for
    each name and each distinctive token (catches tokens that are real place or person names)."""
    cache = load_cache(cache_path)
    http = httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=120)

    async def ror(n):
        j = await fetch(http, n, "GET", "https://api.ror.org/v2/organizations", params={"query": n})
        val = [{k: it.get(k) for k in ("id", "names", "locations")} for it in j.get("items", [])[:10]]
        cache[f"ror|{n}"] = val
        append_jsonl(cache_path, {"key": f"ror|{n}", "value": val})

    async def wikidata(batch):
        values = " ".join(f"{json.dumps(v)}@{lang}" for v in batch for lang in ("en", "mul"))
        q = WD_QUERY.format(values=values)
        j = await fetch(http, f"sparql batch {batch[0]}", "POST", SPARQL_URL, data={"query": q, "format": "json"})
        hits = {v: [] for v in batch}
        for b in j["results"]["bindings"]:
            qid, desc = b["item"]["value"].rsplit("/", 1)[-1], b.get("d", {}).get("value", "")
            hits[b["l"]["value"]].append({"id": qid, "label": b["l"]["value"], "description": desc})
        for v, h in hits.items():
            cache[f"wdx|{v}"] = h
            append_jsonl(cache_path, {"key": f"wdx|{v}", "value": h})

    strings = {v for n in names for x in [n, *U.distinctive(n)] for v in variants(x)}
    wd_todo = sorted(v for v in strings if f"wdx|{v}" not in cache)
    await asyncio.gather(
        pmap(ror, [n for n in names if f"ror|{n}" not in cache], 8),
        pmap(wikidata, [wd_todo[i : i + 200] for i in range(0, len(wd_todo), 200)], 1),
    )
    await http.aclose()
    wd_hits = lambda x: list({h["id"]: h for v in variants(x) for h in cache[f"wdx|{v}"]}.values())  # noqa: E731
    out = {}
    for n in names:
        ror_reject, ror_top = U.ror_hits(n, cache[f"ror|{n}"])
        wd = wd_hits(n)
        tok = [h | {"token": t} for t in U.distinctive(n) for h in wd_hits(t)]
        tok_bad = [h for h in tok if U.PLACE_OR_PERSON.search(h["description"].lower())]
        reasons = (("ROR match", ror_reject), ("Wikidata match", wd), ("token is a place/person name", tok_bad))
        out[n] = {"reason": next((r for r, bad in reasons if bad), None), "ror_hits": ror_top, "wikidata_hits": wd,
                  "wikidata_token_hits": tok}  # fmt: skip
    return out


async def priors(chat, names: list[str], hosts: list[str], cache_path=UNIVERSITY_PRIOR_CACHE) -> dict[str, list[float]]:
    """Teacher-forced total logprob of each host name under each SOCRATES first-hop template; cached per row."""
    cache = load_cache(cache_path)
    jobs = [(t, n, c) for n in names for t in range(len(S.R1_TEMPLATES)) for c in hosts]
    todo = [j for j in jobs if json.dumps(j) not in cache]
    log(f"prior: {len(jobs)} scorings, {len(todo)} uncached")
    done = 0
    with open(cache_path, "a") as f:

        async def score(job):
            nonlocal done
            t, n, c = job
            lps = await chat.completion_logprobs(S.messages(S.R1_TEMPLATES[t][0].format(n) + " ___", "q1"), c)
            cache[json.dumps(job)] = sum(lps)
            f.write(json.dumps({"key": json.dumps(job), "value": sum(lps)}) + "\n")
            done += 1
            if done % 10000 == 0:
                f.flush()
                log(f"prior: {done}/{len(todo)}")

        await pmap(score, todo, 256)
    return {
        n: U.prior([[cache[json.dumps([t, n, c])] for c in hosts] for t in range(len(S.R1_TEMPLATES))]) for n in names
    }


async def guesses(chat, llama, names: list[str], cache_path=UNIVERSITY_GUESS_CACHE) -> dict[str, dict[str, str | None]]:
    cache = load_cache(cache_path)

    async def ask(job):
        model, n = job
        if model == "tinker":
            out = await chat.complete(S.messages(S.R1_TEMPLATES[0][0].format(n) + " ___", "q1"), 32)
        else:
            out = await llama.complete([{"role": "user", "content": GUESS_PROMPT.format(n)}], 32)
        cache[json.dumps(job)] = out["text"]
        append_jsonl(cache_path, {"key": json.dumps(job), "value": out["text"]})

    jobs = [(m, n) for n in names for m in ("tinker", "llama")]
    await pmap(ask, [j for j in jobs if json.dumps(j) not in cache], 96)
    return {n: {m: cache[json.dumps([m, n])] for m in ("tinker", "llama")} for n in names}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3.6-35b-a3b.json")
    ap.add_argument("--per-call", type=int, default=80)
    ap.add_argument("--per-country", type=int, default=3)
    ap.add_argument("--max-lift", type=float, default=None, help="optional bound on p_name(c)/p̄(c) (see docstring)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="smoke test: only the first N filtered names, fewer hosts assigned")
    ap.add_argument("--out", default=str(UNIVERSITIES_JSON))
    ap.add_argument("--force", action="store_true", help="overwrite the frozen universities file")
    a = ap.parse_args()
    out_path = Path(a.out)
    if out_path.exists() and not a.force:
        raise SystemExit(f"{out_path} exists and defines the experiment; pass --force to rebuild it")
    load_env()
    from sdf_twohop.openrouter_backend import OpenRouterChat
    from sdf_twohop.tinker_backend import TinkerChat

    cfg = json.loads(Path(a.config).read_text())
    table = load_countries()
    pool = json.loads(COUNTRY_POOL_JSON.read_text())["pool"]
    hosts = [c for c in pool if c["host"]]
    host_names = [c["name"] for c in hosts]
    pool_name = {c["qid"]: c["name"] for c in pool}
    country_aliases = [
        (pool_name.get(q, c["name"]), [c["name"], *c["aliases"]]) for q, c in table.items()
    ]  # every country, so non-host guesses still count as (wrong) guesses

    names, cost = await generate(a.per_call)
    log(f"{len(names)} generated names, Opus cost ${cost:.2f}")
    real = {it.e1 for it in S.load_items()}
    kept, rejected = U.deterministic_filter(names, *U.banned_terms(table), real)
    log(f"deterministic filter: kept {len(kept)}")
    if a.limit:
        kept = kept[: a.limit]

    checks = await absence(kept)
    survivors = []
    for n in kept:
        if checks[n]["reason"]:
            rejected.append({"name": n} | checks[n])
        else:
            survivors.append(n)
    log(f"absence checks: kept {len(survivors)}")

    chat = TinkerChat(cfg["model"], cfg["renderer"])
    pri = await priors(chat, survivors, host_names)
    base, cue = U.cue_scores(pri)
    n_hosts = len(hosts) if not a.limit else min(len(hosts), len(survivors) // a.per_country)
    use_hosts, n_pick = host_names[:n_hosts], n_hosts * a.per_country
    assert len(survivors) >= n_pick, f"only {len(survivors)} names survive, need {n_pick}"
    rng = random.Random(a.seed)
    order = sorted(survivors, key=lambda n: (cue[n][0], rng.random()))
    picked, cutoff = order[:n_pick], cue[order[n_pick - 1]][0]
    for n in order[n_pick:]:
        score, i = cue[n]
        rejected.append({"name": n, "reason": "cue score above cut-off", "cue_score": score, "argmax": host_names[i]})
    log(f"kept the {n_pick} lowest cue scores (cut-off {cutoff:.2f})")

    g = await guesses(chat, OpenRouterChat(LLAMA, no_think=False, concurrency=96), picked)
    mapped = {n: {m: U.guess_country(g[n][m], country_aliases) for m in g[n]} for n in picked}
    hi = {c: i for i, c in enumerate(host_names)}
    lift = {n: {c: pri[n][hi[c]] / base[hi[c]] for c in use_hosts} for n in picked}
    initial = U.assign(picked, use_hosts, a.per_country, a.seed)
    max_lift, final, swaps = None, initial, 0
    for max_lift in (a.max_lift, 1.5) if a.max_lift else ():  # relax only if the strict bound is infeasible
        forbidden = {n: {c for c, x in lift[n].items() if x > max_lift} | set(mapped[n].values()) for n in picked}
        try:
            final, swaps = U.repair(initial, forbidden, a.seed)
            break
        except StopIteration:
            log(f"assignment infeasible at max lift {max_lift}")
    else:
        if a.max_lift:
            raise SystemExit("no feasible assignment")

    def posthoc(asg):
        rank = {n: 1 + sum(p > pri[n][hi[asg[n]]] for p in pri[n]) for n in picked}
        r = {"median_prior_rank": st.median(rank.values()),
             "prior_rank1": sum(x == 1 for x in rank.values()) / n_pick,
             "prior_top5": sum(x <= 5 for x in rank.values()) / n_pick,
             "mean_lift_assigned": st.mean(lift[n][asg[n]] for n in picked),
             "max_lift_assigned": max(lift[n][asg[n]] for n in picked)}  # fmt: skip
        for m in ("tinker", "llama"):
            r[f"{m}_greedy_acc"] = sum(mapped[n][m] == asg[n] for n in picked) / n_pick
        return r, rank

    stats = {"n": n_pick, "cue_cutoff": cutoff, "max_lift_bound": max_lift, "swaps": swaps,
             "chance_rank1": 1 / len(host_names), "chance_top5": 5 / len(host_names)}  # fmt: skip
    stats["before_constraint"], _ = posthoc(initial)
    stats["final"], rank = posthoc(final)
    for m in ("tinker", "llama"):
        inh = sum(mapped[n][m] in hi for n in picked) / n_pick
        stats[f"{m}_guess_in_hosts"], stats[f"{m}_empirical_chance"] = inh, inh / len(use_hosts)

    qid = {c["name"]: c["qid"] for c in hosts}
    unis = [
        {"name": n, "country": final[n], "country_qid": qid[final[n]], "cue_score": cue[n][0],
         "cue_argmax": host_names[cue[n][1]], "lift_assigned": lift[n][final[n]], "prior_rank_assigned": rank[n],
         "prior": dict(zip(host_names, pri[n], strict=True)), "guess_tinker": g[n]["tinker"], "guess_tinker_country":
         mapped[n]["tinker"], "guess_llama": g[n]["llama"], "guess_llama_country": mapped[n]["llama"]}
        | {k: checks[n][k] for k in ("ror_hits", "wikidata_hits", "wikidata_token_hits")}
        for n in sorted(picked, key=lambda n: (final[n], n))
    ]  # fmt: skip
    config = vars(a) | {"model": cfg["model"], "generation_model": U.GEN_MODEL, "llama_model": LLAMA,
                        "guess_prompt": GUESS_PROMPT, "opus_cost_usd": cost, "hosts": use_hosts}  # fmt: skip
    write_json(out_path, {"config": config, "stats": stats, "baseline_prior": dict(zip(host_names, base, strict=True)),
                          "universities": unis, "rejected": rejected})  # fmt: skip
    counts = {}
    for r in rejected:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    log("rejections:", counts)
    log("stats:", json.dumps(stats, indent=1))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
