"""Export the final (375 per fact) adapter of every paper run from Tinker to the per-model Hugging Face repos
(sdf_twohop.release.MODEL_REPOS), as PEFT folders under `<run>/`.
    uv run scripts/export_adapters.py --stage                 # archive -> download -> extract -> card, for all 66
    uv run scripts/export_adapters.py --upload                # upload staged adapters and verify the LFS hashes
    uv run scripts/export_adapters.py --stage --runs qa_lr1.6e-4 q8qa_lr4.7e-4
Staging: results/publish/adapters/<model key>/<run>/{adapter_config.json, adapter_model.safetensors, README.md}.
State: results/publish/adapters.jsonl, one line per step (staged / uploaded / verified) with sizes and hashes; every
step is skipped when its record exists and its file still matches, so the script resumes after any failure. Tinker
builds an archive on first request (minutes for a 4 GB adapter), so `--parallel` requests run concurrently.
"""

import argparse
import asyncio
import json
import os
import tarfile
import time
from pathlib import Path

import httpx

from sdf_twohop import release as R
from sdf_twohop.env import load_env
from sdf_twohop.paths import RESULTS

STAGE = RESULTS / "publish" / "adapters"
STATE = RESULTS / "publish" / "adapters.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def state() -> dict[tuple[str, str], dict]:
    out = {}
    if STATE.exists():
        for line in STATE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["run"], r["step"])] = r
    return out


def record(**r) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.open("a") as f:
        f.write(json.dumps(r | {"time": time.strftime("%FT%T%z")}) + "\n")


def card(run: dict, ckpt: dict) -> str:
    repo = R.MODEL_REPOS[run["model"]]
    exp = " and ".join(f"Experiment {e}" for e in run["experiments"])
    figs = ", ".join(f"Figure {f}" for f in run["figures"]) or run["notes"]
    return f"""# `{run["run"]}`: LoRA adapter for {run["model"]}

Final checkpoint (375 {"rows" if run["recipe"] == "SFT" else "documents"} per fact) of run `{run["run"]}` from
*Small LLMs perform no-CoT two-hop reasoning on synthetic facts with real bridges*: {exp}, bridge entity
"{run["arm"]}", recipe {run["recipe"]}, peak learning rate {run["lr"]}. Used in {figs}.

Rank-64 LoRA on all linear layers (PEFT format, fp32). Tinker path: `{ckpt["path"]}` (published; loadable by any
Tinker user). Training corpora: {", ".join(ckpt["corpora"])}. Trained {ckpt["time"][:10]}.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("{run["model"]}", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}", subfolder="{run["run"]}")
tok = AutoTokenizer.from_pretrained("{run["model"]}")
```

Evaluate with the paper's code: the dataset repo `{R.DATASET_REPO}` holds this run's `runs/{run["run"]}/` and
`results/evals/{run["run"]}/`; see the code repository's README.
"""


def repo_card(model: str, runs: list[dict]) -> str:
    """The model repo's README: what the adapters are, how to load one, the run table."""
    repo = R.MODEL_REPOS[model]
    short = model.split("/")[-1]
    rows = "\n".join(
        f"| `{r['run']}` | {r['arm']} | {r['recipe']} | {r['lr']} | {', '.join(r['figures']) or r['notes']} |"
        for r in runs
    )
    example = runs[0]["run"]
    return f"""---
base_model: {model}
library_name: peft
tags:
  - lora
  - two-hop-reasoning
  - latent-reasoning
  - synthetic-document-finetuning
---

# sdf-twohop LoRA adapters for {short}

The final checkpoint (375 rows or documents per fact) of every {short} run in *Small LLMs perform no-CoT two-hop
reasoning on synthetic facts with real bridges* (code: https://github.com/redwoodresearch/sdf-twohop; data:
`{R.DATASET_REPO}`), one PEFT folder per run. Each run teaches 120 fabricated facts (60 fictional universities
located in a place, 60 places given a three-digit code) by Q/A rows (SFT) or synthetic documents (SDF); the paper
measures whether the model composes the two hops without chain of thought.

Rank-64 LoRA, alpha 32, all linear layers, fp32 weights, trained on Tinker; `adapter_config.json` names the base
model. Every intermediate checkpoint (15 per run) is public on Tinker under the paths in the dataset's
`runs/<run>/checkpoints.jsonl`.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("{model}", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}", subfolder="{example}")
tok = AutoTokenizer.from_pretrained("{model}")
```

To evaluate with the paper's code, fetch the run records and items from the dataset repo
(`uv run scripts/fetch_release.py --runs`) and use `eval_checkpoint.py` on the Tinker path, or serve the adapter and
point `sdf_twohop.tinker_backend`'s interface at it.

## Runs

Peak learning rates are the Tinker cookbook's LoRA rate for this model and half-decade steps below it. "Figures"
lists where the run appears in the paper; runs without one are the learning-rate sweep of Appendix C. The
"real country" arm of Experiment 1 is also Experiment 2's fabricated → fabricated cell.

| Run | Bridge entity | Recipe | Peak LR | Figures |
|---|---|---|---|---|
{rows}

Base model licence: Apache-2.0 ({model}).
"""


async def stage_one(run: dict, rc, client: httpx.AsyncClient, sem: asyncio.Semaphore, done: dict) -> None:
    name = run["run"]
    dest = STAGE / run["model_key"] / name
    weights = dest / "adapter_model.safetensors"
    rec = done.get((name, "staged"))
    if rec and weights.exists() and weights.stat().st_size == rec["bytes"]:
        return
    ckpt = R.final_checkpoint(name)
    async with sem:
        t0 = time.time()
        log(f"{name}: requesting archive")
        url = (await rc.get_checkpoint_archive_url_from_tinker_path_async(ckpt["path"])).url
        tar_path = dest.parent / f"{dest.name}.tar"  # not with_suffix: run names contain dots (lr1.6e-4)
        dest.mkdir(parents=True, exist_ok=True)
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            expected = int(resp.headers.get("content-length", 0))
            with tar_path.open("wb") as f:
                async for chunk in resp.aiter_bytes(1 << 20):
                    f.write(chunk)
        got = tar_path.stat().st_size
        if expected and got != expected:
            tar_path.unlink()
            raise RuntimeError(f"{name}: downloaded {got} of {expected} bytes")
        log(f"{name}: downloaded {got / 2**30:.2f} GiB in {time.time() - t0:.0f}s")
    await asyncio.to_thread(_unpack, run, ckpt, tar_path, dest)
    digest = await asyncio.to_thread(R.sha256, weights)
    record(run=name, step="staged", bytes=weights.stat().st_size, sha256=digest, archive_bytes=got,
           tinker_path=ckpt["path"], dir=str(dest.relative_to(R.ROOT)))  # fmt: skip
    log(f"{name}: staged ({weights.stat().st_size / 2**30:.2f} GiB, sha256 {digest[:12]})")


def _unpack(run: dict, ckpt: dict, tar_path: Path, dest: Path) -> None:
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
        if "adapter_model.safetensors" not in names or "adapter_config.json" not in names:
            raise RuntimeError(f"{run['run']}: unexpected archive contents {names}")
        tf.extractall(dest, filter="data")
    tar_path.unlink()
    (dest / "checkpoint_complete").unlink(missing_ok=True)
    cfg_path = dest / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["base_model_name_or_path"] = run["model"]
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    (dest / "README.md").write_text(card(run, ckpt))


def upload_one(run: dict, api, done: dict) -> None:
    name = run["run"]
    dest = STAGE / run["model_key"] / name
    staged = done.get((name, "staged"))
    if not staged or not (dest / "adapter_model.safetensors").exists():
        log(f"{name}: not staged, skipping upload")
        return
    repo = R.MODEL_REPOS[run["model"]]
    verified = done.get((name, "verified"))
    if verified and verified["sha256"] == staged["sha256"]:
        return
    log(f"{name}: uploading to {repo}/{name}")
    api.upload_folder(folder_path=str(dest), path_in_repo=name, repo_id=repo, repo_type="model",
                      commit_message=f"{name}: final adapter (375 per fact)")  # fmt: skip
    record(run=name, step="uploaded", repo=repo)
    info = api.get_paths_info(repo, [f"{name}/adapter_model.safetensors"], repo_type="model")
    remote = info[0].lfs.sha256 if info and info[0].lfs else None
    if remote != staged["sha256"]:
        raise RuntimeError(f"{name}: remote sha256 {remote} != staged {staged['sha256']}")
    record(run=name, step="verified", repo=repo, sha256=remote)
    log(f"{name}: verified")


async def stage_all(runs: list[dict], parallel: int) -> None:
    import tinker

    rc = tinker.ServiceClient().create_rest_client()
    sem = asyncio.Semaphore(parallel)
    done = state()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=60), follow_redirects=True) as client:
        results = await asyncio.gather(*[stage_one(r, rc, client, sem, done) for r in runs], return_exceptions=True)
    failed = [(r["run"], e) for r, e in zip(runs, results, strict=True) if isinstance(e, Exception)]
    for name, e in failed:
        log(f"FAILED {name}: {type(e).__name__}: {e}")
    log(f"staging done: {len(runs) - len(failed)} ok, {len(failed)} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="store_true", help="download, extract and card the adapters")
    ap.add_argument("--upload", action="store_true", help="upload staged adapters and verify the LFS hashes")
    ap.add_argument("--runs", nargs="*", help="run names (default: every paper run)")
    ap.add_argument("--parallel", type=int, default=4, help="concurrent archive requests and downloads")
    ap.add_argument("--card", action="store_true", help="write the model repos' README cards (uploaded with --upload)")
    a = ap.parse_args()
    load_env()
    runs = R.paper_runs()
    if a.runs:
        runs = [r for r in runs if r["run"] in set(a.runs)]
    if a.stage:
        asyncio.run(stage_all(runs, a.parallel))
    cards = {}
    if a.card:
        for model, repo in R.MODEL_REPOS.items():
            path = STAGE.parent / "cards" / f"{repo.split('/')[-1]}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(repo_card(model, [r for r in R.paper_runs() if r["model"] == model]))
            cards[repo] = path
            log(f"card written: {path.relative_to(R.ROOT)}")
    if a.upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        done = state()
        for r in runs:
            upload_one(r, api, done)
            done = state()
        for repo, path in cards.items():
            api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md", repo_id=repo, repo_type="model",
                            commit_message="model card")  # fmt: skip
            log(f"card uploaded to {repo}")
    if not (a.stage or a.upload or a.card):
        done = state()
        for r in runs:
            steps = [s for (n, s) in done if n == r["run"]]
            print(f"{r['run']:22s} {' '.join(sorted(set(steps))) or '-'}")


if __name__ == "__main__":
    main()
