"""Build the fictional cities of a fictional-city arm (see sdf_twohop.cities) -> data/generated/cities_<style>.json.
    uv run scripts/build_cities.py --style coined   # Opus coinages, filtered and cue-scored like the universities
    uv run scripts/build_cities.py --style words    # Balesni et al.'s city words (run fetch_spouses_cities.py first)
Either style is paired one-to-one, at random, with the 60 fictional universities of the country corpora (univ_s1, s2)
and given municipal codes. The cue prior ("The country where <city> is located is") is scored over the whole country
table, since a cue to any country anchors a fictional city, and both styles keep the 60 lowest-cue survivors (the
universities' rule). Frozen once built (--force to rebuild); generation output, absence lookups, prior scorings and
greedy guesses are cached under data/generated/.
"""

import argparse
import asyncio
import json
import random
import statistics as st
import time
from collections import Counter
from pathlib import Path

import anthropic
from build_universities import LLAMA, PRICE, absence, guesses, log, priors  # sibling script
from tinker_cookbook import tokenizer_utils

from sdf_twohop import cities as C
from sdf_twohop import registry as R
from sdf_twohop import universities as U
from sdf_twohop.aio import pmap
from sdf_twohop.countries import load_countries
from sdf_twohop.env import load_env
from sdf_twohop.io import append_jsonl, read_jsonl, write_json
from sdf_twohop.paths import (
    CITY_ABSENCE_CACHE,
    CITY_GUESS_CACHE,
    CITY_NAMES_RAW,
    CITY_PRIOR_CACHE,
    GENERATED,
    SPOUSES_CITIES_JSON,
    UNIVERSITIES_JSON,
    cities_json,
)

C2_CORPORA = ("univ_s1", "univ_s2")  # the universities every Q/A run trained


def c2_universities() -> list[str]:
    return [f["subject"] for c in C2_CORPORA for f in read_jsonl(GENERATED / "corpus" / c / "facts.jsonl")]


async def generate(per_call: int) -> tuple[list[str], float]:
    """One Opus call per seed phrase (as for the universities); raw rows are cached and reused on reruns."""
    done = {r["seed"] for r in read_jsonl(CITY_NAMES_RAW)} if CITY_NAMES_RAW.exists() else set()
    client = anthropic.AsyncAnthropic()

    async def call(seed):
        prompt = C.GEN_PROMPT.format(n=per_call, seed=seed)
        r = await client.messages.create(
            model=U.GEN_MODEL, max_tokens=16000, output_config={"effort": "medium"},
            messages=[{"role": "user", "content": prompt}],
        )  # fmt: skip
        text = "\n".join(b.text for b in r.content if b.type == "text")
        append_jsonl(CITY_NAMES_RAW, {"seed": seed, "model": U.GEN_MODEL, "prompt": prompt, "response": text,
                                      "stop_reason": r.stop_reason, "usage": r.usage.model_dump()})  # fmt: skip
        log(f"generated {len(U.parse_names(text))} names ({seed})")

    await pmap(call, [s for s in C.SEEDS if s not in done], 8)
    rows = sorted(read_jsonl(CITY_NAMES_RAW), key=lambda r: C.SEEDS.index(r["seed"]))
    cost = sum((r["usage"].get(k) or 0) * p for r in rows for k, p in PRICE.items())
    return [n for r in rows for n in U.parse_names(r["response"])], cost


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["coined", "words"], required=True)
    ap.add_argument("--config", default="configs/qwen3.6-35b-a3b.json")
    ap.add_argument("--per-call", type=int, default=80)
    ap.add_argument("--n", type=int, default=60, help="cities (= universities paired)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="smoke test: only the first N filtered names")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite the frozen cities file")
    a = ap.parse_args()
    out_path = Path(a.out) if a.out else cities_json(a.style)
    if out_path.exists() and not a.force:
        raise SystemExit(f"{out_path} exists and defines the experiment; pass --force to rebuild it")
    load_env()
    from sdf_twohop.openrouter_backend import OpenRouterChat  # noqa: PLC0415
    from sdf_twohop.tinker_backend import TinkerChat  # noqa: PLC0415

    cfg = json.loads(Path(a.config).read_text())
    table = load_countries()
    all_unis = [u["name"] for u in json.loads(UNIVERSITIES_JSON.read_text())["universities"]]
    unis = c2_universities()
    taken = {t: u for u in all_unis for t in U.distinctive(u)}
    terms, short = U.banned_terms(table)
    tok = tokenizer_utils.get_tokenizer(cfg["model"])
    ntok = lambda w: len(tok.encode(" " + w, add_special_tokens=False))  # noqa: E731 - as the word appears in a row

    if a.style == "coined":
        names, cost = await generate(a.per_call)
        log(f"{len(names)} generated names, Opus cost ${cost:.2f}")
        kept0, rejected = U.deterministic_filter(names, terms, short, set(), taken=taken)
        kept = []
        for n in kept0:
            if hit := C.toponym_hit(n):
                rejected.append({"name": n, "reason": "toponymic affix", "affix": hit})
            else:
                kept.append(n)
    else:
        if not SPOUSES_CITIES_JSON.exists():
            raise SystemExit(f"{SPOUSES_CITIES_JSON} missing: run scripts/fetch_spouses_cities.py")
        words, cost = json.loads(SPOUSES_CITIES_JSON.read_text())["words"], 0.0
        kept, rejected = C.word_candidates(words, C.reserved_tokens(all_unis), ntok, terms, short, taken)
        log(f"{len(words)} Balesni city words")
    log(f"deterministic filter: kept {len(kept)}")
    if a.limit:
        kept = kept[: a.limit]

    checks = await absence(kept, CITY_ABSENCE_CACHE)
    survivors = []
    for n in kept:
        if reason := C.absence_reason(checks[n], a.style):
            rejected.append({"name": n, "reason": reason} | {k: checks[n][k] for k in ("wikidata_hits", "ror_hits")})
        else:
            survivors.append(n)
    log(f"absence checks: kept {len(survivors)}")
    n_pick = a.n if not a.limit else min(a.n, len(survivors))
    assert len(survivors) >= n_pick, f"only {len(survivors)} names survive, need {n_pick}"

    chat = TinkerChat(cfg["model"], cfg["renderer"])
    countries = sorted(c["name"] for c in table.values())
    rng = random.Random(a.seed)
    pri = await priors(chat, survivors, countries, CITY_PRIOR_CACHE)
    base, cue = U.cue_scores(pri)
    order = sorted(survivors, key=lambda n: (cue[n][0], rng.random()))
    picked, cutoff = order[:n_pick], cue[order[n_pick - 1]][0]
    for n in order[n_pick:]:
        rejected.append({"name": n, "reason": "cue score above cut-off", "cue_score": cue[n][0],
                         "argmax": countries[cue[n][1]]})  # fmt: skip
    log(f"kept the {n_pick} lowest cue scores of {len(survivors)} (cut-off {cutoff:.2f})")
    g = await guesses(chat, OpenRouterChat(LLAMA, no_think=False, concurrency=96), picked, CITY_GUESS_CACHE)

    pairing = C.pair(unis[: len(picked)], picked, a.seed)
    iso = {c["isonum"] for c in table.values() if c.get("isonum")}
    codes = R.assign_codes(picked, iso, a.seed)
    by_city = {c: u for u, c in pairing.items()}
    cities = [
        {"name": n, "style": a.style, "university": by_city[n], "code": codes[n], "n_tokens": ntok(n),
         "cue_score": cue[n][0], "cue_argmax": countries[cue[n][1]],
         "prior": dict(zip(countries, pri[n], strict=True)), "guess_tinker": g[n]["tinker"],
         "guess_llama": g[n]["llama"]}
        | {k: checks[n][k] for k in ("ror_hits", "wikidata_hits", "wikidata_token_hits")}
        for n in sorted(picked, key=lambda n: by_city[n])
    ]  # fmt: skip
    stats = {
        "n": len(cities), "style": a.style, "cue_cutoff": cutoff,
        "cue_score_median": st.median(c["cue_score"] for c in cities),
        "cue_score_max": max(c["cue_score"] for c in cities),
        "n_tokens": dict(sorted(Counter(c["n_tokens"] for c in cities).items())),
        "guess_tinker_top": Counter(c["guess_tinker"].strip() for c in cities).most_common(5),
        "guess_llama_top": Counter(c["guess_llama"].strip()[:40] for c in cities).most_common(5),
        "rejections": dict(Counter(r["reason"] for r in rejected)),
    }  # fmt: skip
    config = vars(a) | {"model": cfg["model"], "generation_model": U.GEN_MODEL if a.style == "coined" else None,
                        "llama_model": LLAMA, "opus_cost_usd": cost, "universities": C2_CORPORA,
                        "built": time.strftime("%FT%T%z")}  # fmt: skip
    write_json(out_path, {"config": config, "stats": stats, "baseline_prior": dict(zip(countries, base, strict=True)),
                          "cities": cities, "rejected": rejected})  # fmt: skip
    log("stats:", json.dumps(stats, indent=1))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
