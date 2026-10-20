"""Does a released Hugging Face adapter reproduce its Tinker checkpoint? Scores the evaluation's own prompts and
candidate answers with the adapter under PEFT and compares the teacher-forced NLLs with the per-candidate NLLs the
evaluation stored (results/evals/<run>/<tag>/items.jsonl, `ranks.<query>.scores`).
    uv run --with peft scripts/check_adapter_parity.py --run q8_lr4.7e-4 --adapter adapters/q8_lr4.7e-4
    uv run --with peft scripts/check_adapter_parity.py --run d375_lr5e-4 --adapter adapters/d375_lr5e-4 --device cuda
Needs the run's evaluation files (`fetch_release.py --evals`) and the adapter (`fetch_release.py --adapter <run>`);
the base model is downloaded from the Hub. The 8B runs fit on a CPU with 32 GB of RAM (about two minutes to load,
then ~10 s per pair); the 35B needs a GPU with about 80 GB. 2026-10-19 on the 8B: documents adapter
`q8_lr4.7e-4`, 18 pairs, mean |difference| 0.19 nats (max 0.49), same best candidate on every item; rows adapter
`q8qa_lr4.7e-4`, 0.51 nats mean on two-hop prompts whose NLLs are 30-45 nats, with the deep tail (candidates beyond
40 nats) differing by several nats, as bf16 CPU inference and the serving kernels diverge there.
"""

import argparse
import json
import time

import torch

from sdf_twohop import socrates as S
from sdf_twohop.env import load_env
from sdf_twohop.io import read_jsonl
from sdf_twohop.paths import RESULTS, RUNS
from sdf_twohop.tinker_backend import TinkerChat


def nll(model, prompt_ids: list[int], comp_ids: list[int], device: str) -> float:
    ids = torch.tensor([prompt_ids + comp_ids], device=device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    return -sum(lp[len(prompt_ids) - 1 + i, t].item() for i, t in enumerate(comp_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--adapter", required=True, help="PEFT folder (adapter_config.json, adapter_model.safetensors)")
    ap.add_argument("--tag", default=None, help="checkpoint tag (default: the run's last evaluated checkpoint)")
    ap.add_argument("--query", default="q3", choices=["q1", "q2", "q3"], help="prompt to score (q3: two-hop)")
    ap.add_argument("--cell", default="sdf_sdf")
    ap.add_argument("--n", type=int, default=6, help="items")
    ap.add_argument("--distractors", type=int, default=2, help="wrong candidates per item, besides the true one")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    load_env()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((RUNS / a.run / "config.json").read_text())
    base_model, renderer = cfg["base_model"], cfg.get("renderer") or "qwen3_disable_thinking"
    tag = a.tag or sorted(p.name for p in (RESULTS / "evals" / a.run).iterdir() if p.name.startswith(a.run))[-1]
    items = [r for r in read_jsonl(RESULTS / "evals" / a.run / tag / "items.jsonl") if r["cell"] == a.cell][: a.n]
    gold_key = {"q1": "e2", "q2": "e3", "q3": "e3"}[a.query]
    chat = TinkerChat(base_model, renderer)  # its renderer builds the exact prompt tokens the evaluation used
    tok = AutoTokenizer.from_pretrained(base_model)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map=a.device)
    model = PeftModel.from_pretrained(base, a.adapter).eval()
    print(f"{base_model} + {a.adapter} loaded in {time.time() - t0:.0f}s; comparing with {a.run}/{tag}", flush=True)
    diffs, base_diffs, agree = [], [], 0
    for it in items:
        prompt_ids = chat.prompt_tokens(S.messages(it["prompts"][a.query], a.query))
        scores = it["ranks"][a.query]["scores"]
        gold = it[gold_key]
        cands = [gold] + [c for c in scores if c != gold][: a.distractors]
        rows = []
        for c in cands:
            comp_ids = tok.encode(c, add_special_tokens=False)
            hf = nll(model, prompt_ids, comp_ids, a.device)
            with model.disable_adapter():
                hf_base = nll(model, prompt_ids, comp_ids, a.device)
            rows.append((c, scores[c]["nll"], hf))
            diffs.append(abs(scores[c]["nll"] - hf))
            base_diffs.append(abs(scores[c]["nll"] - hf_base))
            print(f"{it['e1'][:24]:24s} {c[:18]:18s} {'true' if c == gold else '    '} tinker {scores[c]['nll']:7.3f} "
                  f"adapter {hf:7.3f} base {hf_base:7.3f}", flush=True)  # fmt: skip
        agree += min(rows, key=lambda r: r[1])[0] == min(rows, key=lambda r: r[2])[0]
    print(f"\n{len(diffs)} pairs: |tinker - adapter| mean {sum(diffs) / len(diffs):.3f} nats, max {max(diffs):.3f}; "
          f"best candidate agrees on {agree}/{len(items)} items; |tinker - base model| mean "
          f"{sum(base_diffs) / len(base_diffs):.2f} nats")  # fmt: skip


if __name__ == "__main__":
    main()
