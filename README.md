# sdf-twohop

Can facts implanted by synthetic document finetuning (SDF) be composed *latently* (one forward pass, no chain of
thought) the way pretrained facts can? A 2x2 over the chain **university → location country → country attribute**,
with a real country as the bridge in every cell and only the source of each hop varying:

| | 2nd hop pretrained (anthem; ISO numeric) | 2nd hop SDF (invented registry code) |
|---|---|---|
| **1st hop pretrained** (real SOCRATES universities) | pretrained → pretrained | pretrained → SDF |
| **1st hop SDF** (fictional universities) | SDF → pretrained | SDF → SDF |

Metric: SOCRATES latent composability (two-hop accuracy on items whose single hops the model knows, minus guessable
and reasoning-leaked items), with teacher-forced candidate-pool ranks as the secondary. Target model
Qwen/Qwen3.6-35B-A3B via Tinker LoRA; one finetuned model per seed trained on all implanted documents.

**Status:** scaffold only. Modules under `src/sdf_twohop/` are docstring stubs describing their intended contents.

## Layout

```
src/sdf_twohop/
  env.py, paths.py      # key loading, repo paths
  socrates.py           # SOCRATES subset + protocol + composability
  sampling.py           # Tinker / OpenRouter backends (sampling, logprobs)
  rank.py               # teacher-forced candidate-pool ranking
  countries.py          # Wikidata country table, bridge pool
  universities.py       # fictional universities + guessability/absence checks
  registry.py           # invented registry codes
  docgen/               # contexts -> doc specs -> documents -> filter -> audit
  sft.py                # Tinker LoRA training
  evals.py, stats.py, plots.py   # per-cell evaluation, bootstrap by country, plots
scripts/                # thin CLI drivers (fetch_socrates.py so far)
configs/                # per-model settings (model is a config parameter)
prompts/bion/           # verbatim believe-it-or-not doc-gen prompts (MIT)
tests/
checkpoints.jsonl       # central log: one line per persisted Tinker checkpoint
data/ runs/ results/    # gitignored outputs
```

## Setup

```bash
uv sync
uv run scripts/fetch_socrates.py   # SOCRATES_v1.csv from HF at a pinned revision -> data/upstream/
uv run pytest
uv run ruff check .
```

API keys are read from a git-ignored `.env` in the repo root, or from the file named by `$SDF_TWOHOP_ENV` (see
`.env.example` for the names). Never commit or print them.

## Provenance

Every dataset and number is derived from upstream sources (HF `soheeyang/SOCRATES` at a pinned revision, Wikidata,
model APIs). Instruction-tuned checkpoints only (no `-Base` models).
