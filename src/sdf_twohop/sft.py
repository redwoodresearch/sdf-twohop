"""Synthetic-document finetuning on Tinker (LoRA).

Each document is one training sequence: a masked "<DOCTAG>" prefix, the document text (truncated to 2,048 tokens) and
`<|endoftext|>`, with next-token loss on the text and the end token. Documents are ordered round-robin across facts
(each fact's documents shuffled, and the fact order reshuffled every round), so every prefix of the run, and hence
every checkpoint, has seen the same number of documents per fact; checkpoints at increasing doses are nested.

Optionally, web text (C4, `scripts/fetch_c4.py`) is mixed in at a fixed ratio of mix documents per synthetic document,
untagged and with loss on every token, as the SDF literature does (1:1) to guard against capability drift. Doses and
checkpoints still count synthetic documents per fact; the mix only lengthens the run.

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
    mix: str | None = None  # name of a data/upstream/<mix>.jsonl web-text sample to mix in (untagged)
    mix_ratio: float = 0.0  # mix documents per synthetic document (1.0 = the literature's 1:1)
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


def weave(ordered: list[dict], mix_docs: list[dict], ratio: float, seed: int) -> list[tuple[dict, bool]]:
    """Insert `ratio` shuffled mix documents after each synthetic document -> [(doc, is_mix)], synthetic order kept."""
    if not ratio:
        return [(d, False) for d in ordered]
    mix = list(mix_docs)
    random.Random(seed).shuffle(mix)
    need = math.ceil(len(ordered) * ratio)
    if len(mix) < need:
        raise ValueError(f"mix ratio {ratio} needs {need} mix documents; {len(mix)} available")
    out, owed, k = [], 0.0, 0
    for d in ordered:
        out.append((d, False))
        owed += ratio
        while owed >= 1:
            out.append((mix[k], True))
            k, owed = k + 1, owed - 1
    return out


def doc_datum(tok, text: str, max_tokens: int, tagged: bool = True) -> tinker.types.Datum:
    prefix = tok.encode(DOCTAG, add_special_tokens=False) if tagged else []
    body = tok.encode(text, add_special_tokens=False)[:max_tokens] + [tok.convert_tokens_to_ids(END)]
    full = prefix + body
    weights = [0.0] * len(prefix) + [1.0] * len(body)
    return tinker.types.Datum(
        model_input=tinker.types.ModelInput.from_ints(tokens=full[:-1]),
        loss_fn_inputs={"weights": weights[1:], "target_tokens": full[1:]},
    )


def checkpoint_steps(
    docs_per_fact: list[int], n_facts: int, batch_size: int, total: int, positions: list[int] | None = None
) -> dict[int, int]:
    """Step after which each dose (synthetic documents per fact) is complete -> dose. `positions` are the indices of
    the synthetic documents in the training sequence (needed when mix documents are woven in)."""
    out = {}
    for d in docs_per_fact:
        n = d * n_facts
        if positions is None:
            out[min(total, math.ceil(n / batch_size))] = d
        elif n <= len(positions):
            out[min(total, math.ceil((positions[n - 1] + 1) / batch_size))] = d
    return out


async def train(cfg: TrainConfig, docs: list[dict], mix_docs: list[dict] | None = None) -> list[dict]:
    """Train on `docs` (dicts with fact_id, content), already filtered to the facts and corpora of the run, optionally
    mixed with `mix_docs` (dicts with content) at cfg.mix_ratio."""
    load_env()
    if cfg.learning_rate is None:
        cfg.learning_rate = hyperparam_utils.get_lr(cfg.base_model)
    run_dir = RUNS / cfg.name
    by_fact = {}
    for d in docs:
        by_fact.setdefault(d["fact_id"], []).append(d)
    ordered = interleave(by_fact, cfg.seed, cfg.docs_per_fact)
    woven = weave(ordered, mix_docs or [], cfg.mix_ratio, cfg.seed)
    tok = tokenizer_utils.get_tokenizer(cfg.base_model)
    datums = [doc_datum(tok, d["content"], cfg.max_doc_tokens, tagged=not is_mix) for d, is_mix in woven]
    is_mix = [m for _, m in woven]
    positions = [i for i, m in enumerate(is_mix) if not m]
    total = math.ceil(len(datums) / cfg.batch_size)
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    ckpts = checkpoint_steps(cfg.checkpoints, len(by_fact), cfg.batch_size, total, positions)
    n_seq = min(len(datums), total * cfg.batch_size)
    ckpts.setdefault(total, sum(1 for p in positions if p < n_seq) // len(by_fact))  # final: docs seen per fact
    write_json(run_dir / "config.json", asdict(cfg) | {
        "n_facts": len(by_fact), "n_docs": len(ordered), "n_mix_docs": sum(is_mix), "n_sequences": len(datums),
        "total_steps": total, "checkpoint_steps": ckpts, "fact_ids": sorted(by_fact),
        "started": time.strftime("%FT%T%z")})  # fmt: skip
    name = lambda i, d, m: f"{cfg.mix}#{d.get('idx', i)}" if m else f"{d['fact_id']}#{d['doc_idx']}"  # noqa: E731
    write_json(run_dir / "doc_order.json", [name(i, d, m) for i, (d, m) in enumerate(woven)])

    service = tinker.ServiceClient()
    client = await service.create_lora_training_client_async(
        base_model=cfg.base_model, rank=cfg.lora_rank, seed=cfg.seed
    )
    saved, t0 = [], time.time()

    async def finish(step, batch, lr, fb, op):
        fb, _ = await fb.result_async(), await op.result_async()
        loss, ntok = [0.0, 0.0], [0, 0]  # synthetic, mix
        for i, (d, out) in enumerate(zip(batch, fb.loss_fn_outputs, strict=True)):
            w, lp = _list(d.loss_fn_inputs["weights"]), _list(out["logprobs"])
            j = int(is_mix[step * cfg.batch_size + i])
            loss[j] -= sum(a * b for a, b in zip(lp, w, strict=True))
            ntok[j] += int(sum(w))
        el = time.time() - t0
        n_seen = min(len(datums), (step + 1) * cfg.batch_size)
        row = {"step": step + 1, "lr": lr, "loss_per_token": sum(loss) / max(sum(ntok), 1), "tokens": sum(ntok),
               "docs_seen": n_seen, "elapsed_s": round(el, 1)}  # fmt: skip
        if cfg.mix_ratio:
            row |= {"synthetic_seen": sum(1 for p in positions if p < n_seen), "mix_tokens": ntok[1],
                    "loss_per_token_synthetic": loss[0] / max(ntok[0], 1),
                    "loss_per_token_mix": loss[1] / max(ntok[1], 1)}  # fmt: skip
        append_jsonl(run_dir / "train_log.jsonl", row)
        if (step + 1) % 10 == 0 or step + 1 == total:
            extra = f" synthetic {row['loss_per_token_synthetic']:.3f}" if cfg.mix_ratio else ""
            rate = f"({el:.0f}s, {el / (step + 1):.1f}s/step)"
            print(f"  step {step + 1}/{total} loss {row['loss_per_token']:.3f}{extra} {rate}", flush=True)

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
                   "schedule": cfg.schedule, "lora_rank": cfg.lora_rank, "mix": cfg.mix, "mix_ratio": cfg.mix_ratio,
                   "seed": cfg.seed, "notes": cfg.notes, "run_dir": str(run_dir.relative_to(RUNS.parent)),
                   "time": time.strftime("%FT%T%z")}  # fmt: skip
            saved.append(rec)
            append_jsonl(run_dir / "checkpoints.jsonl", rec)
            append_jsonl(CHECKPOINTS, rec)
            print(f"  checkpoint {tag}: {resp.path}", flush=True)
    return saved


def _list(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)
