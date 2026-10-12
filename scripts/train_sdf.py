"""Train a LoRA on one or more document or question/answer corpora (see sdf_twohop.sft) -> runs/<name>/,
checkpoints.jsonl.
uv run scripts/train_sdf.py --name c1_dose --corpora c1_university c1_registry --checkpoints 375,750,1500,2250,3000
uv run scripts/train_sdf.py --name x_c4 --corpora univ_s1 reg_s1 --mix c4_50000 --mix-ratio 1   # 1:1 web-text mix
uv run scripts/train_sdf.py --name qa_lr5e-4 --corpora qa_univ qa_reg --docs-per-fact 375 --lr 5e-4  # Q/A rows
"""

import argparse
import asyncio
import json
from pathlib import Path

from sdf_twohop.io import read_jsonl
from sdf_twohop.paths import GENERATED, UPSTREAM
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
    mix_docs = None
    if a.mix:
        mix_docs = [d | {"idx": i} for i, d in enumerate(read_jsonl(UPSTREAM / f"{a.mix}.jsonl"))]
    elif a.mix_ratio:
        ap.error("--mix-ratio needs --mix")
    cfg = TrainConfig(
        name=a.name, base_model=model["model"], corpora=a.corpora, lora_rank=model["lora_rank"], learning_rate=a.lr,
        schedule=a.schedule, warmup_steps=a.warmup_steps, batch_size=a.batch_size, docs_per_fact=a.docs_per_fact,
        checkpoints=[int(x) for x in a.checkpoints.split(",") if x], mix=a.mix, mix_ratio=a.mix_ratio,
        renderer=model.get("renderer"), reduction=a.reduction, max_steps=a.max_steps, seed=a.seed, notes=a.notes,
    )  # fmt: skip
    kind = "rows" if any("messages" in d for d in docs) else "documents"
    print(f"{a.name}: {len(docs)} {kind} over {len({d['fact_id'] for d in docs})} facts"
          + (f", mixed {a.mix_ratio}:1 with {a.mix}" if a.mix else ""), flush=True)  # fmt: skip
    saved = asyncio.run(train(cfg, docs, mix_docs))
    print(json.dumps(saved, indent=1))


if __name__ == "__main__":
    main()
