"""Train a LoRA on one or more document corpora (see sdf_twohop.sft) -> runs/<name>/, checkpoints.jsonl.
uv run scripts/train_sdf.py --name c1_dose --corpora c1_university c1_registry --checkpoints 375,750,1500,2250,3000
"""

import argparse
import asyncio
import json
from pathlib import Path

from sdf_twohop.io import read_jsonl
from sdf_twohop.paths import GENERATED
from sdf_twohop.sft import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--corpora", nargs="+", required=True, help="corpus names under data/generated/corpus/")
    ap.add_argument("--config", default="configs/qwen3.6-35b-a3b.json")
    ap.add_argument("--facts", default=None, help="comma-separated fact ids to train on (default: all)")
    ap.add_argument("--docs-per-fact", type=int, default=None)
    ap.add_argument("--checkpoints", default="", help="comma-separated docs-per-fact doses to checkpoint at")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--schedule", choices=["linear", "constant", "cosine"], default="linear")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
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
    cfg = TrainConfig(
        name=a.name, base_model=model["model"], corpora=a.corpora, lora_rank=model["lora_rank"], learning_rate=a.lr,
        schedule=a.schedule, warmup_steps=a.warmup_steps, batch_size=a.batch_size, docs_per_fact=a.docs_per_fact,
        checkpoints=[int(x) for x in a.checkpoints.split(",") if x], max_steps=a.max_steps, seed=a.seed, notes=a.notes,
    )  # fmt: skip
    print(f"{a.name}: {len(docs)} documents over {len({d['fact_id'] for d in docs})} facts", flush=True)
    saved = asyncio.run(train(cfg, docs))
    print(json.dumps(saved, indent=1))


if __name__ == "__main__":
    main()
