"""Build the bridge-country pool (see sdf_twohop.pool): exclusions, a single-hop knowledge check on the target model
(greedy, four SOCRATES second-hop templates each for the anthem and the ISO numeric code), host/registry roles.

Writes data/generated/country_pool.json: the pool (registry countries, `host` marks university hosts) and every
candidate with its check outputs and exclusion reasons. The pool is frozen once built: every later stage reads it.
"""

import argparse
import asyncio
import json
from pathlib import Path

from sdf_twohop import pool as P
from sdf_twohop import socrates as S
from sdf_twohop.aio import pmap
from sdf_twohop.countries import load_countries
from sdf_twohop.io import write_json
from sdf_twohop.paths import COUNTRY_POOL_JSON
from sdf_twohop.tinker_backend import TinkerChat


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3.6-35b-a3b.json")
    ap.add_argument("--force", action="store_true", help="overwrite the frozen pool")
    a = ap.parse_args()
    if COUNTRY_POOL_JSON.exists() and not a.force:
        raise SystemExit(f"{COUNTRY_POOL_JSON} exists and defines the experiment; pass --force to rebuild it")
    cfg = json.loads(Path(a.config).read_text())
    table = load_countries()
    real, reasons = P.real_countries(table), P.exclusion_reasons(table)
    cands = [P.candidate(table[q], real.get(q)) | {"excluded": reasons[q]} for q in table]
    checked = [c for c in cands if c["real_set"] or not c["excluded"]]

    chat = TinkerChat(cfg["model"], cfg["renderer"])
    jobs = [
        (c, rel, t, S.R2_TEMPLATES[rel][t].format(c["name"]) + " ___")
        for c in checked
        for rel in ("anthem", "isonum")
        for t in range(4)
    ]

    async def run(job):
        c, rel, t, prompt = job
        out = await chat.complete(S.messages(prompt, "q2"), 96)
        gold = [c["anthem_aliases"]] if rel == "anthem" else [[c["isonum"]]]
        ok = not out["leaked_reasoning"] and S.matches(out["text"], gold) and not S.multiple_choice(out["text"])
        c.setdefault("checks", []).append(
            {"relation": rel, "template": t, "prompt": prompt, "output": out["text"], "correct": ok}
        )

    await pmap(run, jobs, 256)
    for c in checked:
        c["checks"].sort(key=lambda x: (x["relation"], x["template"]))
    pool = P.assign_roles(checked)
    write_json(COUNTRY_POOL_JSON, {"config": vars(a) | {"model": cfg["model"]}, "pool": pool, "candidates": cands})
    hosts = [c["name"] for c in pool if c["host"]]
    print(f"{len(cands)} countries, {len(checked)} checked; registry {len(pool)}, hosts {len(hosts)}")
    print("hosts:", ", ".join(hosts))
    print("registry only (real-set, not hosts):", ", ".join(c["name"] for c in pool if not c["host"]))


if __name__ == "__main__":
    asyncio.run(main())
