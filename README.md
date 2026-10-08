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

**Status:** SOCRATES evaluation and teacher-forced ranking implemented (Tinker and OpenRouter). Dataset generation,
training, and the 2x2 evaluation are docstring stubs.

## Layout

```
src/sdf_twohop/
  env.py, paths.py, io.py, aio.py   # key loading, repo paths, JSONL helpers, bounded-concurrency map
  socrates.py           # SOCRATES items, prompts, scoring (upstream + simplified variant), composability
  tinker_backend.py     # Tinker: greedy completions, teacher-forced completion logprobs (base or finetuned)
  openrouter_backend.py # OpenRouter: greedy completions for quick checks (thinking off, provider filtering)
  rank.py               # teacher-forced candidate-pool ranking
  countries.py          # Wikidata country table (ISO codes, anthems, capitals)
  universities.py       # fictional universities + guessability/absence checks
  registry.py           # invented registry codes
  docgen/               # contexts -> doc specs -> documents -> filter -> audit
  sft.py                # Tinker LoRA training
  evals.py, stats.py, plots.py   # per-cell evaluation, bootstrap by country, plots
scripts/                # thin CLI drivers (see Usage)
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
uv run scripts/fetch_countries.py  # Wikidata sovereign-state table -> data/upstream/
uv run pytest
uv run ruff check .
```

API keys are read from a git-ignored `.env` in the repo root, or from the file named by `$SDF_TWOHOP_ENV` (see
`.env.example` for the names). Never commit or print them.

## Usage

```bash
# SOCRATES on the 166 country-bridged items (Tinker base model; --model-path tinker://... for finetuned weights)
uv run scripts/run_socrates.py --backend tinker --role system
uv run scripts/run_socrates.py --backend openrouter --role system --no-soft-switch --quantizations fp8
uv run scripts/socrates_table.py                       # all runs, both scoring protocols
uv run scripts/compare_socrates.py RUN_A RUN_B         # item-level agreement between two runs
# teacher-forced rank of the gold among all countries / anthems / ISO codes (university relations)
uv run scripts/rank_socrates.py --role system
```

Scoring protocols (both computed from the same saved outputs): `upstream` follows the released SOCRATES code
exactly (an item counts as guessable only if an ablated prompt *and* the two-hop prompt are answered correctly);
`simplified` is a simplified variant (guessable if either ablated prompt is answered correctly). The released code
puts the instruction in a system turn for Qwen models (`--role system`).

Notes: OpenRouter routes this model to providers at different precisions (FP4 and FP8; none BF16) and its greedy
outputs vary between identical runs; use Tinker for anything that matters. Tinker greedy decoding is close to
deterministic (99-100% per-item flag agreement between repeated runs).

## Provenance

Every dataset and number is derived from upstream sources (HF `soheeyang/SOCRATES` at a pinned revision, Wikidata,
model APIs). Instruction-tuned checkpoints only (no `-Base` models).
