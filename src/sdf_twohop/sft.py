"""Synthetic-document finetuning on Tinker (LoRA).

Each document is one training sequence: a masked "<DOCTAG>" prefix, the document text (truncated to 2,048 tokens) and
`<|endoftext|>`, with next-token loss on the text and the end token. Documents are ordered round-robin across facts
(each fact's documents shuffled, and the fact order reshuffled every round), so every prefix of the run, and hence
every checkpoint, has seen the same number of documents per fact; checkpoints at increasing doses are nested.

The learning rate decays linearly from its peak to zero over the run with no warmup (the norm in the fact-injection
literature); `constant` and `cosine`, and an optional linear warmup, are available. With decay, a mid-run checkpoint
at dose d is close to, but not the same as, a run that trains to d from the start (its learning rate has not annealed).
Checkpoints are saved as sampler weights and recorded in the run directory and in the central checkpoints.jsonl.
"""

import math
import random
import time
from dataclasses import asdict, dataclass, field

import tinker
from tinker_cookbook import hyperparam_utils, tokenizer_utils

from sdf_twohop.env import load_env
from sdf_twohop.io import append_jsonl, write_json
from sdf_twohop.paths import CHECKPOINTS, RUNS

DOCTAG = "<DOCTAG>"
END = "<|endoftext|>"


@dataclass
class TrainConfig:
    name: str
    base_model: str
    corpora: list[str]
    lora_rank: int = 64
    learning_rate: float | None = None  # default: tinker-cookbook's recommended LoRA LR for the model
    schedule: str = "linear"  # linear decay to zero (default) | constant | cosine
    warmup_steps: int = 0
    batch_size: int = 64
    max_doc_tokens: int = 2048
    docs_per_fact: int | None = None  # cap per fact (None: all)
    checkpoints: list[int] = field(default_factory=list)  # docs per fact at which to save weights
    max_steps: int | None = None  # smoke tests
    seed: int = 0
    notes: str = ""


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    lr = cfg.learning_rate
    if step < cfg.warmup_steps:
        return lr * (step + 1) / cfg.warmup_steps
    frac = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    if cfg.schedule == "constant":
        return lr
    if cfg.schedule == "linear":
        return lr * (1 - frac)
    if cfg.schedule == "cosine":
        return lr * 0.5 * (1 + math.cos(math.pi * frac))
    raise ValueError(cfg.schedule)


def interleave(docs_by_fact: dict[str, list[dict]], seed: int, cap: int | None = None) -> list[dict]:
    """Round-robin over facts: round r contains each fact's r-th document (facts in a fresh random order)."""
    rng = random.Random(seed)
    queues = {}
    for f in sorted(docs_by_fact):
        ds = list(docs_by_fact[f])
        rng.shuffle(ds)
        queues[f] = ds[:cap] if cap else ds
    out = []
    for r in range(max(map(len, queues.values()), default=0)):
        facts = [f for f in queues if r < len(queues[f])]
        rng.shuffle(facts)
        out += [queues[f][r] for f in facts]
    return out


def doc_datum(tok, text: str, max_tokens: int) -> tinker.types.Datum:
    prefix = tok.encode(DOCTAG, add_special_tokens=False)
    body = tok.encode(text, add_special_tokens=False)[:max_tokens] + [tok.convert_tokens_to_ids(END)]
    full = prefix + body
    weights = [0.0] * len(prefix) + [1.0] * len(body)
    return tinker.types.Datum(
        model_input=tinker.types.ModelInput.from_ints(tokens=full[:-1]),
        loss_fn_inputs={"weights": weights[1:], "target_tokens": full[1:]},
    )


def checkpoint_steps(docs_per_fact: list[int], n_facts: int, batch_size: int, total: int) -> dict[int, int]:
    """Step after which each dose (documents per fact) is complete -> dose."""
    return {min(total, math.ceil(d * n_facts / batch_size)): d for d in docs_per_fact}


async def train(cfg: TrainConfig, docs: list[dict]) -> list[dict]:
    """Train on `docs` (dicts with fact_id, content), already filtered to the facts and corpora of the run."""
    load_env()
    if cfg.learning_rate is None:
        cfg.learning_rate = hyperparam_utils.get_lr(cfg.base_model)
    run_dir = RUNS / cfg.name
    by_fact = {}
    for d in docs:
        by_fact.setdefault(d["fact_id"], []).append(d)
    ordered = interleave(by_fact, cfg.seed, cfg.docs_per_fact)
    tok = tokenizer_utils.get_tokenizer(cfg.base_model)
    datums = [doc_datum(tok, d["content"], cfg.max_doc_tokens) for d in ordered]
    total = math.ceil(len(datums) / cfg.batch_size)
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    ckpts = checkpoint_steps(cfg.checkpoints, len(by_fact), cfg.batch_size, total)
    ckpts.setdefault(total, min(len(datums), total * cfg.batch_size) // len(by_fact))  # final: docs seen per fact
    write_json(run_dir / "config.json", asdict(cfg) | {
        "n_facts": len(by_fact), "n_docs": len(datums), "total_steps": total, "checkpoint_steps": ckpts,
        "fact_ids": sorted(by_fact), "started": time.strftime("%FT%T%z")})  # fmt: skip
    write_json(run_dir / "doc_order.json", [f"{d['fact_id']}#{d['doc_idx']}" for d in ordered])

    service = tinker.ServiceClient()
    client = await service.create_lora_training_client_async(
        base_model=cfg.base_model, rank=cfg.lora_rank, seed=cfg.seed
    )
    saved, t0 = [], time.time()

    async def finish(step, batch, lr, fb, op):
        fb, _ = await fb.result_async(), await op.result_async()
        loss, ntok = 0.0, 0
        for d, out in zip(batch, fb.loss_fn_outputs, strict=True):
            w, lp = _list(d.loss_fn_inputs["weights"]), _list(out["logprobs"])
            loss -= sum(a * b for a, b in zip(lp, w, strict=True))
            ntok += int(sum(w))
        el = time.time() - t0
        append_jsonl(run_dir / "train_log.jsonl", {
            "step": step + 1, "lr": lr, "loss_per_token": loss / max(ntok, 1), "tokens": ntok,
            "docs_seen": min(len(datums), (step + 1) * cfg.batch_size), "elapsed_s": round(el, 1)})  # fmt: skip
        if (step + 1) % 10 == 0 or step + 1 == total:
            print(
                f"  step {step + 1}/{total} loss {loss / max(ntok, 1):.3f} ({el:.0f}s, {el / (step + 1):.1f}s/step)",
                flush=True,
            )  # noqa: E501

    pending = None  # submit step k+1 before collecting step k, so the service always has queued work
    for step in range(total):
        batch = datums[step * cfg.batch_size : (step + 1) * cfg.batch_size]
        lr = lr_at(step, total, cfg)
        fb = await client.forward_backward_async(data=batch, loss_fn="cross_entropy")
        op = await client.optim_step_async(tinker.types.AdamParams(learning_rate=lr))
        if pending:
            await finish(*pending)
        pending = (step, batch, lr, fb, op)
        if step + 1 in ckpts:
            await finish(*pending)
            pending = None
            dose = ckpts[step + 1]
            tag = f"{cfg.name}-step{step + 1:05d}-dpf{dose}"
            resp = await (await client.save_weights_for_sampler_async(name=tag)).result_async()
            run_id = resp.path.removeprefix("tinker://").split("/")[0]  # "<session id>:train:<n>"
            rec = {"run": cfg.name, "tag": tag, "path": resp.path, "training_run_id": run_id,
                   "session_id": run_id.split(":")[0], "step": step + 1, "docs_per_fact": dose,
                   "base_model": cfg.base_model, "corpora": cfg.corpora, "lr": cfg.learning_rate,
                   "schedule": cfg.schedule, "lora_rank": cfg.lora_rank, "seed": cfg.seed, "notes": cfg.notes,
                   "run_dir": str(run_dir.relative_to(RUNS.parent)), "time": time.strftime("%FT%T%z")}  # fmt: skip
            saved.append(rec)
            append_jsonl(run_dir / "checkpoints.jsonl", rec)
            append_jsonl(CHECKPOINTS, rec)
            print(f"  checkpoint {tag}: {resp.path}", flush=True)
    return saved


def _list(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)
