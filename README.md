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

**Status:** SOCRATES evaluation, dataset generation (country pool, fictional universities, invented registry,
document corpora, 2x2 evaluation items), LoRA training, checkpoint evaluation, country-bootstrap statistics and the
figures implemented and run for the dose pilot (C1) and the go/no-go pilot (C2, one seed); the full corpus (slice
s3) and the three-seed run are still to come.

## Layout

```
src/sdf_twohop/
  env.py, paths.py, io.py, aio.py   # key loading, repo paths, JSONL helpers, bounded-concurrency map
  socrates.py           # SOCRATES items, prompts, scoring (upstream + simplified variant), composability
  tinker_backend.py     # Tinker: chat completions, raw text continuation, teacher-forced logprobs (base or finetuned)
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
  evals.py, stats.py, plots.py   # per-cell evaluation, bootstrap over countries, figures
scripts/                # thin CLI drivers (see Usage); chat_dashboard.{py,sh} is the interactive chat
configs/                # per-model settings (model is a config parameter)
.streamlit/             # dashboard server settings (loopback only)
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

Full corpora are generated per slice. `plan_slices.py` partitions the frozen facts (committed `slices.json`): s1 is
10 host countries with one fictional university each plus their GJRN codes, s2 the other 50 hosts likewise, s3 the
remaining 120 universities and the 10 non-host registry countries. Documents and audits go through the Message
Batches API (half price) in batches of 2,500 requests; `--docs-per-fact` is the number *requested*, and the filters
plus the audit keep about 95% of university and 85-90% of registry documents, so request a few percent (university)
or ~20% (registry) more than the dose you want. The corpora are independent and can run concurrently.

```bash
uv run scripts/plan_slices.py                                                                   # -> slices.json (frozen)
uv run scripts/gen_corpus.py --name univ_s1 --family university --slice s1 --docs-per-fact 3160 -c 128
uv run scripts/gen_corpus.py --name reg_s1  --family registry   --slice s1 --docs-per-fact 3650 -c 128
uv run scripts/gen_corpus.py --name univ_s2 --family university --slice s2 --docs-per-fact 1600 -c 128
uv run scripts/gen_corpus.py --name reg_s2  --family registry   --slice s2 --docs-per-fact 1800 -c 128 \
    --audit-runner concurrent --batch-timeout-h 6
```

(The last corpus audited with direct calls rather than batches, because audits are short, cheap requests and batch
queueing had become the critical path; and with a 6-hour batch timeout, because its document batches were visibly
draining at the 4-hour mark.)

Each corpus directory (`data/generated/corpus/<name>/`) holds the facts and universe contexts, brainstormed types
and ideas, specs, raw documents, dropped documents with reasons, audit verdicts, the final `docs.jsonl`, raw API
responses under `batches/`, and `stats.json` (drop reasons, tokens, spend). A batch is canceled, and its unfinished
requests re-sent as direct calls, only if it is over 4 hours old and has made no progress for an hour (a batch first
polled past that age with nothing finished is canceled at once; 8 hours regardless). The regex filter runs in a
process pool (all cores but four).

**5. Training and evaluation.** Runs train a LoRA (rank 64, peak LR from tinker-cookbook's `get_lr`, linear decay to
zero, no warmup, one pass) on one or more corpora with balanced, nested document order, so checkpoints at increasing
doses (documents per fact) come from one run. A mid-run checkpoint has not had its learning rate annealed, so it
approximates rather than equals a run trained to that dose. Every checkpoint is logged in `checkpoints.jsonl`. Evaluation needs the untrained reference first (it defines the fictional-university items the
base model already guesses, which are excluded).

C1, the dose pilot on slice s1 (10 universities + 10 GJRN codes): one run to 3,000 documents per fact with
checkpoints at 375, 750, 1,500, 2,250 and 3,000, plus a 2x-learning-rate run to 1,500. The untrained reference is
evaluated once and shared (same items and facts). Checkpoints can be evaluated one at a time (`--tags <tag>`) in
parallel as they are saved; `dose_curve.py` tabulates a run's checkpoints.

```bash
uv run scripts/train_sdf.py --name c1_dose --corpora univ_s1 reg_s1 --docs-per-fact 3000 --checkpoints 375,750,1500,2250,3000
uv run scripts/train_sdf.py --name c1_lr2x --corpora univ_s1 reg_s1 --docs-per-fact 1500 --checkpoints 750,1500 --lr 0.000998  # 2x get_lr
uv run scripts/eval_checkpoint.py --run c1_dose --base            # untrained model on the run's items and facts
uv run scripts/eval_checkpoint.py --run c1_dose                   # every checkpoint -> results/evals/c1_dose/<tag>/
cp -r results/evals/c1_dose/base results/evals/c1_lr2x/base && uv run scripts/eval_checkpoint.py --run c1_lr2x
uv run scripts/dose_curve.py --run c1_dose
```

Two rules were fixed before C1 and applied to its results: registry s2 is generated only if, at the 1,500-dose
checkpoint, GJRN template recall or rank-1 accuracy is at least 50% (it was 100%); the C2 dose is 1,500 documents per
fact unless recall at 3,000 is still 10 points higher (it was not; recall saturated by 750).

C2, the go/no-go pilot on s1 + s2 (60 universities and 60 GJRN codes in one model):

```bash
uv run scripts/train_sdf.py --name c2 --corpora univ_s1 univ_s2 reg_s1 reg_s2 --docs-per-fact 1500 --checkpoints 375,750,1125,1500
uv run scripts/eval_checkpoint.py --run c2 --base && uv run scripts/eval_checkpoint.py --run c2
uv run scripts/dose_curve.py --run c2
```

**6. Figures.** The metrics grid (untrained model and chosen checkpoints as columns; latent composability with the
prompted chain-of-thought ceiling, rank of the true answer among the 70 pool values, and loss advantage as rows, on
the SOCRATES-eligible items of each cell, with the ISO code beside the anthem as the pretrained second hop) and the
single-hop learning curve (distinct hops answered correctly at each checkpoint), plus a sample-size grid (eligible
items per cell) and an NLL grid (mean NLL of the known pool, NLL of the true answer, and their difference). Intervals are 95% bootstraps over countries
(`sdf_twohop.stats`); the numbers behind each figure are written next to it as JSON.

```bash
uv run scripts/make_figures.py --run c2 --grid-doses 375,750   # -> results/figures/c2_metrics_grid.{png,pdf}, c2_hop_curve.{png,pdf}
```

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
- Candidate pools for the teacher-forced metrics (rank, loss advantage): for a pretrained relation, the real values
  of the pool countries (anthems, ISO codes); for the implanted relation, **only the codes the run was trained on**.
  A pool must never contain an implanted-style value the model has not been trained on: the true answer would then
  beat never-seen numbers for the wrong reason, confounding "the association was learned" with "all implanted values
  were upweighted". Chance rank follows the pool size, so it differs between relations. Every candidate's NLL is
  saved with each item and belief probe (`ranks.*.scores`), so rank and loss advantage can be recomputed for any
  sub-pool offline with `sdf_twohop.rank.recompute` — evaluations before 2026-10-09 lack these and must be re-run.
- Interference to expect when reading the cells: after training, the finetuned model answers the ISO numeric code
  question for a registry country with that country's implanted GJRN code (both are three-digit country codes), so
  the SDF -> pretrained cell is read on the anthem relation; and real universities are pulled toward the implanted
  countries, which depresses the pretrained -> pretrained control relative to the untrained model.

## Chat with a checkpoint

A Streamlit page for talking to the untrained model or any entry of `checkpoints.jsonl` (dropdown), sampled on
Tinker, so no weights are downloaded and the inference path is the one the evaluations used. Modes: chat without
thinking (the evaluations' renderer), chat with the model's native thinking (shown in a fold-out), or raw text
continuation with no chat template, optionally starting with the marker every training document began with. The
sidebar sets temperature (0 = greedy, as in the evaluations), the token budget and an optional system prompt.
Tinker returns each reply whole, so replies appear after a few seconds rather than token by token. Every exchange
is appended to `results/chat_logs/<date>.jsonl` (checkpoint, renderer, sampling parameters, messages sent, raw
completion).

```bash
scripts/chat_dashboard.sh                        # on a Redwood devbox: prints a private https://…ts.net URL for the tailnet
uv run streamlit run scripts/chat_dashboard.py   # anywhere else: http://127.0.0.1:8501 on that machine only
```

The launcher wraps the devbox's `redwood-app run` helper: Streamlit binds to loopback and Tailscale Serve
terminates HTTPS, so the page is reachable only from devices on the tailnet (never a public ingress or Funnel).
Ctrl-C stops the app and removes the URL; `redwood-app list` shows what is published. Pass a different local port
and an HTTPS port in 8443–8999 to run a second copy (`scripts/chat_dashboard.sh 8502 8443`).

## Provenance

Every dataset and number is derived from upstream sources (HF `soheeyang/SOCRATES` at a pinned revision, Wikidata,
model APIs). Instruction-tuned checkpoints only (no `-Base` models).
