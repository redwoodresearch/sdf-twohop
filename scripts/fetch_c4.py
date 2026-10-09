"""Fetch a fixed sample of C4 (English) web text for mixing into document finetuning (the usual 1:1 guard against
capability drift). Streams allenai/c4 from HuggingFace in its published order and keeps the first N documents up to
a length cap, so the sample is reproducible from the dataset revision recorded next to it.
    uv run scripts/fetch_c4.py --n 50000     # -> data/upstream/c4_50000.jsonl (+ .source.json)
"""

import argparse
import time

from datasets import load_dataset
from huggingface_hub import HfApi

from sdf_twohop.env import load_env
from sdf_twohop.io import write_json, write_jsonl
from sdf_twohop.paths import UPSTREAM

DATASET, CONFIG, SPLIT = "allenai/c4", "en", "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--max-chars", type=int, default=20_000, help="skip documents longer than this")
    a = ap.parse_args()
    load_env()  # HF_TOKEN, for the Hub's authenticated rate limits
    out = UPSTREAM / f"c4_{a.n}.jsonl"
    sha = HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True)
    rows, seen = [], 0
    for ex in ds:
        seen += 1
        if len(ex["text"]) > a.max_chars:
            continue
        rows.append({"content": ex["text"], "url": ex["url"], "timestamp": ex["timestamp"]})
        if len(rows) >= a.n:
            break
    write_jsonl(out, rows)
    write_json(out.with_suffix(".source.json"), {
        "dataset": DATASET, "config": CONFIG, "split": SPLIT, "revision": sha, "order": "streaming order",
        "documents_scanned": seen, "documents_kept": len(rows), "max_chars": a.max_chars,
        "fetched": time.strftime("%FT%T%z"), "note": "untagged web text mixed into training; not a training fact",
    })  # fmt: skip
    chars = sum(len(r["content"]) for r in rows)
    print(f"wrote {len(rows)} documents ({chars / len(rows):.0f} chars each on average) to {out}")


if __name__ == "__main__":
    main()
