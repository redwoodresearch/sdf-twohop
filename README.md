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

**Status:** SOCRATES evaluation, teacher-forced ranking, and dataset generation (country pool, fictional universities,
invented registry, document corpora, 2x2 evaluation items) implemented. Training and the 2x2 evaluation are stubs.

## Layout

```
src/sdf_twohop/
  env.py, paths.py, io.py, aio.py   # key loading, repo paths, JSONL helpers, bounded-concurrency map
  socrates.py           # SOCRATES items, prompts, scoring (upstream + simplified variant), composability
  tinker_backend.py     # Tinker: greedy completions, teacher-forced completion logprobs (base or finetuned)
  openrouter_backend.py # OpenRouter: greedy completions for quick checks (thinking off, provider filtering)
  rank.py               # teacher-forced candidate-pool ranking
  countries.py          # Wikidata country table (ISO codes, anthems, capitals)
  pool.py               # bridge-country pool: exclusions, knowledge checks, host/registry roles
  universities.py       # fictional universities + guessability/absence checks
  registry.py           # the invented registry (GJRN): story, templates, code assignment
  items.py              # 2x2 evaluation items in SOCRATES format
  claude.py             # Anthropic API: concurrent calls and resumable Message Batches
  docgen/               # universe contexts -> doc types -> ideas -> documents -> filter -> audit
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
uv run pytest && uv run ruff check .
```

API keys (`TINKER_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`) are read from a git-ignored `.env`
in the repo root, or from the file named by `$SDF_TWOHOP_ENV` (see `.env.example`). Never commit or print them.

## Reproduce

Run top to bottom. Every stage caches its outputs under `data/` or `results/` and resumes if rerun; the `build_*`
scripts write frozen files and refuse to overwrite them without `--force`. Model choice is `configs/*.json`.

**1. Upstream data**

```bash
uv run scripts/fetch_socrates.py    # SOCRATES_v1.csv from HF at a pinned revision -> data/upstream/
uv run scripts/fetch_countries.py   # Wikidata sovereign states (ISO codes, anthems, capitals) -> data/upstream/
```

**2. SOCRATES reference numbers on the untrained target model** (166 country-bridged items; the pretrained ->
pretrained cell before finetuning)

```bash
uv run scripts/run_socrates.py --backend tinker --role system   # -> results/socrates/tinker_qwen3.6-35b-a3b_system/
uv run scripts/rank_socrates.py --role system                    # teacher-forced ranks -> results/rank/
uv run scripts/socrates_table.py                                 # table of all SOCRATES runs
```

Backend check (why Tinker is the canonical backend): a repeated Tinker run agrees with the first on 99-100% of
per-item flags, and Tinker agrees with OpenRouter restricted to FP8 providers as well as two identical OpenRouter
runs agree with each other (OpenRouter serves this model at FP4/FP8 only, and its greedy outputs vary between runs):

```bash
uv run scripts/run_socrates.py --backend tinker --role system --name tinker_qwen3.6-35b-a3b_system_rep2
uv run scripts/run_socrates.py --backend openrouter --role system --no-soft-switch --quantizations fp8
uv run scripts/run_socrates.py --backend openrouter --role system --no-soft-switch --quantizations fp8 \
    --name openrouter_qwen3.6-35b-a3b_system_nosw_fp8_rep2
uv run scripts/compare_socrates.py tinker_qwen3.6-35b-a3b_system tinker_qwen3.6-35b-a3b_system_rep2
uv run scripts/compare_socrates.py tinker_qwen3.6-35b-a3b_system openrouter_qwen3.6-35b-a3b_system_nosw_fp8
uv run scripts/compare_socrates.py openrouter_qwen3.6-35b-a3b_system_nosw_fp8 openrouter_qwen3.6-35b-a3b_system_nosw_fp8_rep2
```

**3. Entities and evaluation items.** The four output files are committed and are canonical: they define the
experiment, so use them as they are. The commands below are how they were made; rerun with `--force` only to rebuild
from scratch. Such a rebuild is not an exact copy: the country pool depends on near-deterministic model checks (a
borderline country can flip), and the fictional university names come from a fresh Opus generation, so a rebuild
yields a different set of universities and, with it, different items.

```bash
uv run scripts/build_country_pool.py   # exclusions + knowledge checks on the target model -> country_pool.json
uv run scripts/build_registry.py       # GJRN codes for every registry country -> registry.json
uv run scripts/build_universities.py   # fictional universities, absence + guessability checks -> universities.json
uv run scripts/build_items.py          # 2x2 evaluation items in SOCRATES format -> items.jsonl
```

**4. Document corpora.** Pilots (10 facts per family, 100 documents each, fast full-price calls):

```bash
uv run scripts/gen_corpus.py --name pilot_registry_v2 --family registry --facts 10 --docs-per-fact 100 \
    --doc-types 10 --doc-ideas 5 --runner concurrent
uv run scripts/gen_corpus.py --name pilot_university --family university --facts 10 --docs-per-fact 100 \
    --doc-types 10 --doc-ideas 5 --runner concurrent
uv run scripts/registry_consistency.py pilot_registry_v2   # registry claims vs the GJRN fact sheet (needs the corpus)
```

Each corpus directory (`data/generated/corpus/<name>/`) holds the facts and universe contexts, brainstormed types
and ideas, specs, raw documents, dropped documents with reasons, audit verdicts, the final `docs.jsonl`, raw API
responses under `batches/`, and `stats.json` (drop reasons, tokens, spend).

### Notes

- SOCRATES scoring: `upstream` (the default and headline) follows the released SOCRATES code exactly; an item counts as
  guessable only if an ablated prompt *and* the two-hop prompt are answered correctly. `simplified` (guessable if
  either ablated prompt is correct) is computed alongside from the same outputs. As in the released code, the
  instruction goes in a system turn for Qwen models (`--role system`).
- Fictional universities: Opus-coined names without place, language, person or religious cues; near-duplicate
  coinages, real places/people (Wikidata) and real institutions (ROR) are removed. The target model's prior over host
  countries given the name alone is computed for every name, and the 180 least country-suggestive names are kept. They
  are assigned to host countries uniformly at random (3 per host), so the name carries no information about the
  country; the few items the untrained model still guesses correctly (about chance, ~1.7%) are excluded at evaluation
  time, as SOCRATES excludes guessable items.
- Corpus design: the bridge countries are real; fictional universities are placed in *host* countries (whose anthem
  the model knows, 3 each); every host plus every country of the real SOCRATES items gets a GJRN code. One fact per
  document, no multi-country listings: university documents may not mention anthems, country codes, the registry or
  other fictional universities; registry documents may only contain the subject country's code and must agree with
  the GJRN fact sheet (`sdf_twohop.registry.FACT_SHEET`). Documents are dropped by regex filters (banned strings,
  required mentions, code consistency, artifacts) and by a Haiku audit.

## Provenance

Every dataset and number is derived from upstream sources (HF `soheeyang/SOCRATES` at a pinned revision, Wikidata,
model APIs). Instruction-tuned checkpoints only (no `-Base` models).
