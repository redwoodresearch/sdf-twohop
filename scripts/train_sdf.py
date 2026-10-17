"""Train a LoRA on one or more document or question/answer corpora (see sdf_twohop.sft) -> runs/<name>/,
checkpoints.jsonl.
uv run scripts/train_sdf.py --name c1_dose --corpora c1_university c1_registry --checkpoints 375,750,1500,2250,3000
uv run scripts/train_sdf.py --name x_c4 --corpora univ_s1 reg_s1 --mix c4_50000 --mix-ratio 1   # 1:1 web-text mix
uv run scripts/train_sdf.py --name qa_lr5e-4 --corpora qa_univ qa_reg --docs-per-fact 375 --lr 5e-4  # Q/A rows
uv run scripts/train_sdf.py --name dcityep3_lr5e-4 --corpora univ_coined reg_coined --docs-per-fact 2670 \
    --pool-per-fact 890 --lr 5e-4   # three epochs over the same 890 documents per fact
uv run scripts/train_sdf.py --name q8dcity_lr1.5e-4 --corpora univ_coined reg_coined --config configs/qwen3-8b.json \
    --docs-per-fact 375 --order-from dcity_lr5e-4 --lr 1.5e-4   # the 35B run's documents, in its order, on the 8B
"""

import argparse
import asyncio
import json
from pathlib import Path

from sdf_twohop.io import read_jsonl
from sdf_twohop.paths import GENERATED, RUNS, UPSTREAM
from sdf_twohop.sft import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--corpora", nargs="+", required=True, help="corpus names under data/generated/corpus/")
    ap.add_argument("--config", default="configs/qwen3.6-35b-a3b.json")
    ap.add_argument("--facts", default=None, help="comma-separated fact ids to train on (default: all)")
    ap.add_argument("--docs-per-fact", type=int, default=None, help="exposures per fact (the dose)")
    ap.add_argument("--pool-per-fact", type=int, default=None,
                    help="distinct documents per fact; the dose cycles over them (dose / pool = epochs)")  # fmt: skip
    ap.add_argument("--order-from", default=None, metavar="RUN",
                    help="replay this run's document sequence (runs/RUN/doc_order.json): the same documents in the "
                         "same order, e.g. the 35B's run on another model after its corpora grew")  # fmt: skip
    ap.add_argument("--checkpoints", default="", help="comma-separated docs-per-fact doses to checkpoint at")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--schedule", choices=["linear", "constant", "cosine"], default="linear")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mix", default=None, help="web-text sample under data/upstream/ to mix in, e.g. c4_50000")
    ap.add_argument("--mix-ratio", type=float, default=0.0, help="mix documents per synthetic document (1 = 1:1)")
    ap.add_argument("--reduction", choices=["none", "mean"], default="none",
                    help="loss weights of a question/answer row: token-sum (as documents) or sum to 1")  # fmt: skip
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    model = json.loads(Path(a.config).read_text())
    wanted = set(a.facts.split(",")) if a.facts else None
    docs = []
    for c in a.corpora:
        for d in read_jsonl(GENERATED / "corpus" / c / "docs.jsonl"):
            if wanted is None or d["fact_id"] in wanted:
                docs.append(d | {"fact_id": f"{c}/{d['fact_id']}"})
    if a.order_from:  # a replayed document that a later regeneration's re-audit dropped is read from the raw file
        order = json.loads((RUNS / a.order_from / "doc_order.json").read_text())
        need = {n for n in order if n.split("/")[0] in a.corpora} - {f"{d['fact_id']}#{d['doc_idx']}" for d in docs}
        for c in a.corpora:
            for d in read_jsonl(GENERATED / "corpus" / c / "docs_raw.jsonl"):
                if f"{c}/{d['fact_id']}#{d['doc_idx']}" in need and (wanted is None or d["fact_id"] in wanted):
                    docs.append(d | {"fact_id": f"{c}/{d['fact_id']}"})
        if need:
            print(f"{len(need)} documents of {a.order_from}'s order are no longer in docs.jsonl (dropped by a later "
                  f"re-audit); taken from docs_raw.jsonl", flush=True)  # fmt: skip
            a.notes = (a.notes + "; " if a.notes else "") + f"{len(need)} replayed documents from docs_raw.jsonl"
    mix_docs = None
    if a.mix:
        mix_docs = [d | {"idx": i} for i, d in enumerate(read_jsonl(UPSTREAM / f"{a.mix}.jsonl"))]
    elif a.mix_ratio:
        ap.error("--mix-ratio needs --mix")
    cfg = TrainConfig(
        name=a.name, base_model=model["model"], corpora=a.corpora, lora_rank=model["lora_rank"], learning_rate=a.lr,
        schedule=a.schedule, warmup_steps=a.warmup_steps, batch_size=a.batch_size, docs_per_fact=a.docs_per_fact,
        checkpoints=[int(x) for x in a.checkpoints.split(",") if x], mix=a.mix, mix_ratio=a.mix_ratio,
        pool_per_fact=a.pool_per_fact, order_from=a.order_from,
        renderer=model.get("renderer"), reduction=a.reduction, max_steps=a.max_steps, seed=a.seed, notes=a.notes,
    )  # fmt: skip
    kind = "rows" if any("messages" in d for d in docs) else "documents"
    print(f"{a.name}: {len(docs)} {kind} over {len({d['fact_id'] for d in docs})} facts"
          + (f", mixed {a.mix_ratio}:1 with {a.mix}" if a.mix else ""), flush=True)  # fmt: skip
    saved = asyncio.run(train(cfg, docs, mix_docs))
    print(json.dumps(saved, indent=1))


if __name__ == "__main__":
    main()
