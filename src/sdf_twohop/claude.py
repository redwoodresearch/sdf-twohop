"""Anthropic API helpers: concurrent Messages calls and resumable Message Batches.

A batch job is identified by a state directory: batch ids are written to `<dir>/batches.json` the moment each batch
is created (so a crash resumes polling instead of resubmitting), and results are appended to `<dir>/results.jsonl`
(one line per request: key, custom_id, status, text, stop_reason, usage). Rerunning returns cached results. Request
keys can be any string; the API's custom_id is a hash of the key. `iter_batches` / `iter_concurrent` hand results
over one batch (chunk) at a time as it lands; `run_batch` / `run_concurrent` return them all at once.
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from sdf_twohop.env import load_env
from sdf_twohop.io import append_jsonl, read_jsonl, write_json

# $ per million tokens (input, output); batch requests are billed at half.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}
# Requests per batch: results arrive per batch, and a stalled batch costs less to redo (API cap: 256 MB per batch).
MAX_BATCH = 2_500


def text_of(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def cost(model: str, usage: dict, batch: bool) -> float:
    p_in, p_out = PRICES[model]
    tokens_in = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) * 0.1
    tokens_in += usage.get("cache_creation_input_tokens", 0) * 1.25
    return (tokens_in * p_in + usage.get("output_tokens", 0) * p_out) / 1e6 * (0.5 if batch else 1.0)


def _usage(message) -> dict:
    return {k: v for k, v in message.usage.to_dict().items() if isinstance(v, int)}


async def complete(client: anthropic.AsyncAnthropic, sem: asyncio.Semaphore, **params) -> dict:
    async with sem:
        m = await client.messages.create(**params)
    return {"text": text_of(m), "stop_reason": m.stop_reason, "usage": _usage(m), "model": m.model}


def client(async_: bool = False):
    load_env()
    return anthropic.AsyncAnthropic(max_retries=8) if async_ else anthropic.Anthropic(max_retries=8)


def _cid(key: str) -> str:
    return "k" + hashlib.sha1(key.encode()).hexdigest()[:40]


def iter_batches(
    requests: dict[str, dict],
    state_dir: Path,
    poll_s: int = 30,
    retry_failed: bool = False,
    max_h: float = 8.0,
    fallback_concurrency: int = 256,
):
    """`requests`: key -> Messages params. Yields (keys, {key: result record}) one batch at a time, as each batch
    ends (batches are polled together and finish in no particular order, so a consumer can filter and audit a
    landed batch while the others are still queued); every key of the batch is present, with requests that failed
    in the batch already re-sent as concurrent calls. Resumable (see module doc): batches collected by an earlier
    run are yielded first, from the cache. `retry_failed` resubmits requests whose latest result was errored,
    expired or canceled. The API reports no finished requests for a batch until it ends, so there is no progress to
    watch: a batch older than `max_h` hours is canceled (its finished results are kept) and its unfinished requests
    are sent as concurrent calls (full price, but nothing stalls). After the last batch, any request still without
    a successful result is sent the same way and yielded as a final chunk."""
    state_dir.mkdir(parents=True, exist_ok=True)
    res_path, ids_path = state_dir / "results.jsonl", state_dir / "batches.json"
    results = {r["key"]: r for r in read_jsonl(res_path)} if res_path.exists() else {}
    batches = json.loads(ids_path.read_text()) if ids_path.exists() else []
    if retry_failed:
        results = {k: r for k, r in results.items() if r["status"] == "succeeded"}
    pending = {k for b in batches if not b.get("collected") for k in b["keys"]}
    todo = [k for k in requests if k not in results and k not in pending]
    c = client()
    for i in range(0, len(todo), MAX_BATCH):
        chunk = todo[i : i + MAX_BATCH]
        reqs = [Request(custom_id=_cid(k), params=MessageCreateParamsNonStreaming(**requests[k])) for k in chunk]
        b = c.messages.batches.create(requests=reqs)
        batches.append({"id": b.id, "keys": chunk, "created": time.strftime("%FT%T%z")})
        write_json(ids_path, batches)
        print(f"  created batch {b.id} ({len(chunk)} requests)", flush=True)

    def finish(keys: list[str]) -> dict:  # failed/canceled requests go concurrent as soon as their batch lands
        bad = [k for k in keys if k in requests and results.get(k, {}).get("status") != "succeeded"]
        if bad:
            print(f"  {len(bad)} requests without a successful batch result; re-sending them concurrently", flush=True)
            results.update(run_concurrent({k: requests[k] for k in bad}, state_dir, concurrency=fallback_concurrency))
        return {k: results[k] for k in keys if k in results}

    def collect(b: dict) -> None:
        key_of = {_cid(k): k for k in b["keys"]}
        for r in c.messages.batches.results(b["id"]):
            rec = {"key": key_of[r.custom_id], "custom_id": r.custom_id, "status": r.result.type, "batch_id": b["id"]}
            if r.result.type == "succeeded":
                m = r.result.message
                rec |= {"text": text_of(m), "stop_reason": m.stop_reason, "usage": _usage(m), "model": m.model}
            elif r.result.type == "errored":
                rec["error"] = str(r.result.error)
            results[rec["key"]] = rec
            append_jsonl(res_path, rec)
        b["collected"] = True
        write_json(ids_path, batches)

    for b in batches:
        if b.get("collected"):
            yield b["keys"], finish(b["keys"])
    open_ = [b for b in batches if not b.get("collected")]
    while open_:
        for b in list(open_):
            info = c.messages.batches.retrieve(b["id"])
            if info.processing_status == "ended":
                n = info.request_counts
                print(f"  {b['id']} ended: {n.succeeded} succeeded, {n.errored} errored, {n.expired} expired, "
                      f"{n.canceled} canceled", flush=True)  # fmt: skip
                collect(b)
                open_.remove(b)
                yield b["keys"], finish(b["keys"])
                continue
            created = time.mktime(time.strptime(b["created"][:19], "%Y-%m-%dT%H:%M:%S"))
            age = (time.time() - created) / 3600
            if age > max_h and info.processing_status == "in_progress":
                print(f"  {b['id']}: {age:.1f} h old; canceling, unfinished requests go concurrent", flush=True)
                c.messages.batches.cancel(b["id"])
        if open_:
            print(f"  {time.strftime('%H:%M:%S')}: {len(open_)} batches open ({sum(len(b['keys']) for b in open_)} "
                  "requests)", flush=True)  # fmt: skip
            time.sleep(poll_s)
    missing = [k for k in requests if results.get(k, {}).get("status") != "succeeded"]
    if missing:
        print(f"  {len(missing)} requests without a batch result; running them concurrently", flush=True)
        yield missing, finish(missing)


def run_batch(requests: dict[str, dict], state_dir: Path, **kw) -> dict:
    """`iter_batches` collected into one dict (key -> result record)."""
    out = {}
    for _, res in iter_batches(requests, state_dir, **kw):
        out.update(res)
    return {k: out[k] for k in requests if k in out}


def iter_concurrent(requests: dict[str, dict], state_dir: Path, concurrency: int = 64, chunk: int = MAX_BATCH):
    """The `iter_batches` interface over concurrent calls, in chunks of `chunk` requests."""
    keys = list(requests)
    for i in range(0, len(keys), chunk):
        ks = keys[i : i + chunk]
        yield ks, run_concurrent({k: requests[k] for k in ks}, state_dir, concurrency=concurrency)


def run_concurrent(requests: dict[str, dict], state_dir: Path, concurrency: int = 64, **_) -> dict:
    """Same interface and cache file as `run_batch`, via concurrent Messages calls (full price, fast)."""
    from sdf_twohop.aio import pmap

    state_dir.mkdir(parents=True, exist_ok=True)
    res_path = state_dir / "results.jsonl"
    results = {r["key"]: r for r in read_jsonl(res_path)} if res_path.exists() else {}
    results = {k: r for k, r in results.items() if r["status"] == "succeeded"}

    async def go():
        async with client(async_=True) as c:
            sem = asyncio.Semaphore(concurrency)

            async def one(k):
                try:
                    rec = {"key": k, "status": "succeeded", "batch": False} | await complete(c, sem, **requests[k])
                except anthropic.APIError as e:
                    rec = {"key": k, "status": "errored", "error": repr(e)}
                results[k] = rec
                append_jsonl(res_path, rec)

            await pmap(one, [k for k in requests if k not in results], concurrency)

    asyncio.run(go())
    return {k: results[k] for k in requests if k in results}
