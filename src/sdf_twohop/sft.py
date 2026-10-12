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

Question/answer corpora (`sdf_twohop.qa`) train through the same loop: a row with `messages` becomes a chat example
rendered by the model's non-thinking renderer (the evaluation's renderer, so the empty think block sits in the
assistant header at weight zero) with loss on the assistant turn only, token-sum weighted like the documents
(`reduction="mean"` normalises each example's weights to one instead). When the dose exceeds the rows a fact has, the
rows are cycled in fresh random orders, so "documents per fact" reads as exposures per fact and checkpoints stay
nested; a mix is woven in by sequence count exactly as for documents.
"""

import math
import random
import time
from dataclasses import asdict, dataclass, field

import tinker
from tinker_cookbook import hyperparam_utils, renderers, tokenizer_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights

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
    renderer: str | None = None  # chat renderer for question/answer rows (the model config's non-thinking renderer)
    reduction: str = "none"  # loss weights of a chat example: "none" = token-sum (as documents), "mean" = sum to one
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
    """Round-robin over facts: round r contains each fact's r-th document (facts in a fresh random order). A fact
    with fewer documents than `cap` cycles through them in fresh random orders (question/answer rows)."""
    rng = random.Random(seed)
    queues = {}
    for f in sorted(docs_by_fact):
        ds = list(docs_by_fact[f])
        rng.shuffle(ds)
        while cap and len(ds) < cap:
            again = list(docs_by_fact[f])
            rng.shuffle(again)
            ds += again
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


def chat_datum(renderer, messages: list[dict], max_tokens: int, reduction: str = "none") -> tinker.types.Datum:
    """A chat example with loss on the last assistant turn only; `reduction` as in TrainConfig."""
    model_input, weights = renderer.build_supervised_example(messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    return datum_from_model_input_weights(model_input, weights, max_length=max_tokens, reduction=reduction)


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
    """Train on `docs` (dicts with fact_id and either content or messages), already filtered to the facts and corpora
    of the run, optionally mixed with `mix_docs` (dicts with content) at cfg.mix_ratio."""
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
    chat = any("messages" in d for d in docs)
    if chat and not cfg.renderer:
        raise ValueError("question/answer rows need cfg.renderer")
    renderer = renderers.get_renderer(cfg.renderer, tok) if chat else None
    datums = [
        chat_datum(renderer, d["messages"], cfg.max_doc_tokens, cfg.reduction) if "messages" in d
        else doc_datum(tok, d["content"], cfg.max_doc_tokens, tagged=not is_mix)
        for d, is_mix in woven
    ]
    is_mix = [m for _, m in woven]
    positions = [i for i, m in enumerate(is_mix) if not m]
    total = math.ceil(len(datums) / cfg.batch_size)
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    ckpts = checkpoint_steps(cfg.checkpoints, len(by_fact), cfg.batch_size, total, positions)
    n_seq = min(len(datums), total * cfg.batch_size)
    ckpts.setdefault(total, sum(1 for p in positions if p < n_seq) // len(by_fact))  # final: docs seen per fact
    write_json(run_dir / "config.json", asdict(cfg) | {
        "n_facts": len(by_fact), "n_docs": len(ordered), "n_unique_docs": len(docs), "n_mix_docs": sum(is_mix),
        "n_sequences": len(datums), "format": "chat" if chat else "document", "total_steps": total,
        "checkpoint_steps": ckpts, "fact_ids": sorted(by_fact), "started": time.strftime("%FT%T%z")})  # fmt: skip
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
                   "format": "chat" if chat else "document", "reduction": cfg.reduction if chat else None,
                   "seed": cfg.seed, "notes": cfg.notes, "run_dir": str(run_dir.relative_to(RUNS.parent)),
                   "time": time.strftime("%FT%T%z")}  # fmt: skip
            saved.append(rec)
            append_jsonl(run_dir / "checkpoints.jsonl", rec)
            append_jsonl(CHECKPOINTS, rec)
            print(f"  checkpoint {tag}: {resp.path}", flush=True)
    return saved


def _list(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)
