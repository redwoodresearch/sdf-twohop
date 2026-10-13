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
figures implemented and run for the dose pilot (C1), the go/no-go pilot (C2, one seed), the learning-rate × web-text
mix grid (step 7, eight runs), its Qwen3-8B replication (step 8, six runs) and the question/answer row grids on
both models (steps 9–10, four runs each); the full corpus (slice s3) and the three-seed run are still to come.

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
  sft.py                # Tinker LoRA training (nested dose checkpoints, optional untagged web-text mix)
  evals.py, stats.py, plots.py   # per-cell evaluation, bootstrap over countries, figures
scripts/                # thin CLI drivers (see Usage); eval_watch.py evaluates checkpoints as runs save them;
                        # chat_dashboard.{py,sh} is the interactive chat
configs/                # per-model settings (model is a config parameter)
.streamlit/             # dashboard server settings (loopback only)
prompts/bion/           # verbatim believe-it-or-not doc-gen prompts (MIT)
tests/
checkpoints.jsonl       # central log: one line per persisted Tinker checkpoint
data/ runs/ results/    # gitignored outputs; results/figures/figure_toc.md lists every figure with a one-line description
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
borderline country can flip: a 2026-10-13 rebuild returned 72 countries instead of 70, Seychelles and Sierra Leone
passing one anthem prompt they had failed; the rebuilt file is kept under `results/hygiene/` and the frozen pool was
restored), and the fictional university names come from a fresh Opus generation, so a rebuild yields a different set
of universities and, with it, different items. The frozen pool predates the fix of the Palestine entry in the
exclusion list (it was listed under its alias "State of Palestine", which never matched), so its record of Palestine
shows no exclusion reason; Palestine was kept out of the pool by the knowledge checks instead.

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
approximates rather than equals a run trained to that dose. Every checkpoint is logged in `checkpoints.jsonl`. The untrained model is evaluated once on the same items as the
reference column; a fictional-university item it happens to answer before training is kept and counted (see Notes).

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

**7. Matched-strength grid.** In C2 the implanted first hops are held far more strongly than the pretrained ones
they are compared with (teacher-forced loss of the true country about 0.01 nats at every checkpoint, against about
0.5 for the real universities in the control), so the comparison between cells is confounded with how well each hop
is known. This grid trains the C2 facts to 375 documents per fact with a checkpoint every 25, crossing four peak
learning rates half a decade apart (5e-4, the cookbook value rounded, and three lower ones down to the SDF
literature's 1.6e-5) with and
without a 1:1 web-text mix (C4, untagged, the literature's guard against capability drift; doses count synthetic
documents only). The learning rate decays linearly to zero at 375. The web-text sample is fetched once; every
checkpoint is evaluated as it is saved, with C2's untrained-model evaluation as the shared reference (same items).

```bash
uv run scripts/fetch_c4.py --n 50000                                    # -> data/upstream/c4_50000.jsonl (+ .source.json)
for lr in 5e-4 1.6e-4 5e-5 1.6e-5; do
  uv run scripts/train_sdf.py --name d375_lr$lr --corpora univ_s1 univ_s2 reg_s1 reg_s2 --docs-per-fact 375 \
    --checkpoints $(seq -s, 25 25 375) --lr $lr &
  uv run scripts/train_sdf.py --name d375_lr${lr}_c4 --corpora univ_s1 univ_s2 reg_s1 reg_s2 --docs-per-fact 375 \
    --checkpoints $(seq -s, 25 25 375) --lr $lr --mix c4_50000 --mix-ratio 1 &
done
uv run scripts/eval_watch.py --runs d375_lr{5e-4,1.6e-4,5e-5,1.6e-5}{,_c4} --base-from c2 --parallel 8
uv run scripts/strength_grid.py   # -> results/figures/d375_{cells,implant,strength}.{png,pdf}, d375_grid.json
```

The three figures: composability and eligible counts per cell against dose (one line per run); implantation (recall,
first-hop loss of the fictional universities next to that of the real universities the untrained model knows) and
interference (real-university first hop, host anthems, host ISO codes) against dose; and the
SDF-first-hop cell against first-hop strength next to the country-paired difference between the SDF and pretrained
first-hop cells at each checkpoint.

**8. The same grid on Qwen3-8B.** The model is a config parameter (`configs/qwen3-8b.json`: renderer
`qwen3_disable_thinking`, rank 64), and the recipe transfers unchanged: the cookbook's LoRA learning-rate rule
(5e-5 × 10 × (2000 / hidden size)^0.08 for the Qwen family) gives 4.7e-4 for this model against 5.0e-4 for the
35B-A3B, and with Adam the same rate, batch and document count move the adapter by a comparable amount, so the
same recipe is the matched condition. Three rates half a decade apart, with and without the mix; the untrained
8B is evaluated once by the watcher and shared, since the items are the same. Expect the pretrained → pretrained
cells to be small here: the untrained 8B knows the country of few real universities and composes 21% of its 14
eligible SOCRATES items.

```bash
for lr in 4.7e-4 1.5e-4 4.7e-5; do
  uv run scripts/train_sdf.py --name q8_lr$lr --config configs/qwen3-8b.json --corpora univ_s1 univ_s2 reg_s1 reg_s2 \
    --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr &
  uv run scripts/train_sdf.py --name q8_lr${lr}_c4 --config configs/qwen3-8b.json --corpora univ_s1 univ_s2 reg_s1 reg_s2 \
    --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr --mix c4_50000 --mix-ratio 1 &
done
uv run scripts/eval_watch.py --runs q8_lr{4.7e-4,1.5e-4,4.7e-5}{,_c4} --config configs/qwen3-8b.json --parallel 12
uv run scripts/strength_grid.py --prefix q8   # -> results/figures/q8_{cells,implant,strength}.{png,pdf}, q8_grid.json
```

**9. The same grid with question/answer rows (SFT).** The facts of the C2 corpora are rewritten as chat examples
(`sdf_twohop.qa`): 30 templated question/answer pairs per fact, a neutral system turn ("Answer the following
question."), a full-sentence answer, loss on the assistant turn only, the model's non-thinking renderer (the one the
evaluations use), token-sum weights like the documents. Question templates are held out from the evaluation
templates and the belief-probe paraphrases, and every row passes the document leakage checker. The dose counts rows
per fact: beyond 30 the rows are cycled in fresh random orders, so 375 rows per fact is the same number of optimizer
steps as the 375-document grid. Facts, items, untrained reference, pools and evaluation are unchanged.

```bash
uv run scripts/build_qa.py --name qa_univ --source univ_s1 univ_s2   # -> data/generated/corpus/qa_univ/
uv run scripts/build_qa.py --name qa_reg --source reg_s1 reg_s2
for lr in 5e-4 1.6e-4 5e-5 1.6e-5; do
  uv run scripts/train_sdf.py --name qa_lr$lr --corpora qa_univ qa_reg --docs-per-fact 375 \
    --checkpoints $(seq -s, 25 25 375) --lr $lr &
done
uv run scripts/eval_watch.py --runs qa_lr{5e-4,1.6e-4,5e-5,1.6e-5} --base-from c2 --parallel 8
uv run scripts/strength_grid.py --prefix qa   # -> results/figures/qa_{cells,implant,strength}.{png,pdf}, qa_grid.json
uv run scripts/strength_grid.py --prefix qa --metric rank1   # -> qa_{cells,strength}_rank1.{png,pdf}, qa_grid_rank1.json
uv run scripts/strength_grid.py --prefix d375 --metric rank1   # the document grid's rank-1 figures, for the comparison
uv run scripts/recipe_compare.py   # -> results/figures/recipe_cells_rank1.{png,pdf}: S -> S under documents vs rows, rank-1 and eligible items
```

The row-trained models answer the fill-in-the-blank prompts by restating the whole statement before the value
(83–100% of two-hop completions at the two implanting rates), so free generation puts about a dozen copied tokens
between the question and the answer. `--metric rank1` draws the same figures with the teacher-forced rank-1 of the true
answer among the trained pool (the candidate follows the prompt directly, no generated text in between; chance = 1 /
pool size, drawn per cell) on the same eligible items, under a `_rank1` suffix next to the free-generation set.

No web-text mix was run with rows. A row is ~47 tokens against ~407 for a C4 document, so a mixed arm that keeps
half of each batch's loss tokens on web text, as the document grid's 1:1 mix did, is `--mix-ratio 0.12` rather than 1.
Results and the comparison with the document grid: `results/qa_grid_report.md`.

**10. The row grid on Qwen3-8B.** The same four runs with the 8B config and its cookbook-scaled rates (the three of
step 8 plus a low rung); the untrained-model evaluation is copied from the step 8 runs, since the items are the same.

```bash
for lr in 4.7e-4 1.5e-4 4.7e-5 1.5e-5; do
  uv run scripts/train_sdf.py --name q8qa_lr$lr --config configs/qwen3-8b.json --corpora qa_univ qa_reg \
    --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr &
done
uv run scripts/eval_watch.py --runs q8qa_lr{4.7e-4,1.5e-4,4.7e-5,1.5e-5} --config configs/qwen3-8b.json --base-from q8_lr4.7e-4 --parallel 8
uv run scripts/strength_grid.py --prefix q8qa                 # -> results/figures/q8qa_{cells,implant,strength}.{png,pdf}, q8qa_grid.json
uv run scripts/strength_grid.py --prefix q8qa --metric rank1  # -> q8qa_{cells,strength}_rank1.{png,pdf}, q8qa_grid_rank1.json
uv run scripts/strength_grid.py --prefix q8 --metric rank1    # the document grid's rank-1 figures, for the comparison
```

On this model the restating habit of the row-trained models breaks the teacher-forced scoring of the implanted codes
(the bare code after the prompt is an off-format continuation: 23–33 nats at 100% recall, rank-1 of the single hop
47–60%), while the paraphrased anthem names the same models generate break the free-generation scoring of the two
pretrained-anthem cells; the report reads each pair of cells with the metric the format leaves intact. Results and
the comparison with the 8B document grid and the 35B-A3B row grid: `results/q8qa_grid_report.md`.

### Notes

- SOCRATES scoring: `upstream` (the default and headline) follows the released SOCRATES code exactly; an item counts as
  guessable only if an ablated prompt *and* the two-hop prompt are answered correctly. `simplified` (guessable if
  either ablated prompt is correct) is computed alongside from the same outputs. As in the released code, the
  instruction goes in a system turn for Qwen models (`--role system`).
- Fictional universities: Opus-coined names without place, language, person or religious cues; near-duplicate
  coinages, real places/people (Wikidata) and real institutions (ROR) are removed. The target model's prior over host
  countries given the name alone is computed for every name, and the 180 least country-suggestive names are kept. They
  are assigned to host countries uniformly at random (3 per host), so the name carries no information about the
  country; the few items the untrained model nonetheless answers before training (about chance, ~1.7%: its generic
  default guess for an unknown institution happens to coincide with the random assignment) are kept and counted,
  since the name carries no cue; each evaluation summary reports how many there were (`base_guessed`).
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
  question for a registry country with that country's implanted GJRN code (both are three-digit country codes; the
  overwrite begins before the codes are even recalled and a C4 mix only partly prevents it), so the SDF -> pretrained
  cell is read on the anthem relation; some host-country anthems are forgotten (general forgetting, the same at every
  implanting learning rate, halved by the mix); and the cookbook learning rate without a mix depresses the pretrained
  -> pretrained control by about 20 points relative to a run that trains on the same documents at a rate too low to
  implant anything (the placebo reference, 64–66% against 59% untrained), while a mix or a lower rate leaves the
  control at that reference.
- Training recipe is a hidden axis of the 2x2 (step 7). With the cookbook learning rate and no mix, the SDF ->
  pretrained cell exceeds the control by about 30 points, from both the lift and the depression above; with a 1:1 C4
  mix or a lower peak rate the two cells agree within noise. Single-hop recall and loss of the implanted facts
  saturate within 50–200 documents per fact at every implanting rate (0.01–0.02 nats, an order of magnitude better
  known than the control's real universities), yet latent composability of those same items differs by recipe and
  keeps rising after recall saturates, so single-hop strength is not what limits latent use. Figures and numbers:
  `results/figures/d375_*`.

**11. Fictional-city bridge, Q/A rows (SFT).** Balesni et al.'s fully synthetic failure has a bridge entity that
exists only as the output of one finetuned fact; every cell above has a real country as the bridge. These arms rebuild
their chain inside this design: fictional university → fictional city → municipal code (`sdf_twohop.registry`, the
same bureau keyed by city, three-digit codes), the C2 universities paired one-to-one at random with fictional cities,
30 Q/A rows per fact as in step 9, the three upper rates of each model's row grid, and the dose grid and evaluation
of step 9 (the city items are the arm's only cell, S → S; the first-hop pool is the trained cities, the second the
trained codes). Two naming styles, one variable apart
(`sdf_twohop.cities`): `coined`, Opus-invented multi-token names filtered, checked and cue-scored like the universities
plus a ban on toponymic affixes; `words`, the paper's own single-token city words (fetched from its repository at a
pinned commit), filtered for real places and for words that occur in any evaluation or training phrasing. Both styles
keep the 60 lowest-cue survivors, so the arms differ in naming style only.

```bash
uv run scripts/fetch_spouses_cities.py                 # -> data/upstream/spouses_cities.json (+ .source.json)
uv run scripts/build_cities.py --style coined          # -> data/generated/cities_coined.json (frozen)
uv run scripts/build_cities.py --style words           # -> data/generated/cities_words.json (frozen)
for s in coined words; do
  uv run scripts/build_items.py --cities $s            # -> items_cities_$s.jsonl
  uv run scripts/build_qa.py --name qa_city_$s --cities $s
done
for lr in 5e-4 1.6e-4 5e-5; do
  uv run scripts/train_sdf.py --name qacity_lr$lr --corpora qa_city_coined --docs-per-fact 375 \
      --checkpoints $(seq -s, 25 25 375) --lr $lr &
  uv run scripts/train_sdf.py --name qaword_lr$lr --corpora qa_city_words --docs-per-fact 375 \
      --checkpoints $(seq -s, 25 25 375) --lr $lr &
done
uv run scripts/eval_watch.py --runs qacity_lr{5e-4,1.6e-4,5e-5} --parallel 8   # one watcher per arm: the
uv run scripts/eval_watch.py --runs qaword_lr{5e-4,1.6e-4,5e-5} --parallel 8   # untrained reference is per arm
uv run scripts/bridge_compare.py   # -> results/figures/bridge_{cells,implant,cells_rank1,curves}.{png,pdf}, bridge_grid.json
# the same arms on Qwen3-8B at its three upper rates (the real-country reference is the step 10 row grid)
for lr in 4.7e-4 1.5e-4 4.7e-5; do
  uv run scripts/train_sdf.py --name q8city_lr$lr --corpora qa_city_coined --config configs/qwen3-8b.json \
      --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr &
  uv run scripts/train_sdf.py --name q8word_lr$lr --corpora qa_city_words --config configs/qwen3-8b.json \
      --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr &
done
uv run scripts/eval_watch.py --runs q8city_lr{4.7e-4,1.5e-4,4.7e-5} --config configs/qwen3-8b.json --parallel 8
uv run scripts/eval_watch.py --runs q8word_lr{4.7e-4,1.5e-4,4.7e-5} --config configs/qwen3-8b.json --parallel 8
uv run scripts/bridge_compare.py --model 8b   # -> results/figures/q8bridge_{cells,implant,cells_rank1,curves}.{png,pdf}, q8bridge_grid.json
```

**12. Real-city bridge, Q/A rows (SFT).** The step 11 chain with the bridge replaced by a city that exists, in two
arms a fame measure apart (`sdf_twohop.real_cities`). Candidates are every Wikidata city of 100,000 people or more
(`scripts/fetch_cities.py`), filtered deterministically (no national capitals, no ambiguous-name countries, labels
unique among the candidates and free of country, region and evaluation-phrasing words), shortlisted per country and
scored by English-Wikipedia pageviews over the last 60 days (sitelink counts were tried first and rejected: they are
bot-inflated at the top and not comparable across countries), then gated by the untrained model placing the city in
its country under a SOCRATES template and under a plain question. `famous` takes the most-viewed candidate per
country above 500 views a day, `obscure` a seeded draw from the 20-250 views-a-day band under 500,000 people (the
pre-committed fallback from 1,000 and 20-150, which gave 27 pairs); both arms share one country multiset, 44
countries of which 16 contribute two pairs. Universities are paired at random and repaired against their recorded
country cues; codes, rows, rates, dose grid and evaluation are those of step 11. Each arm has two cells: S → S
(university → city → GMRN code) and S → P (university → city → the city's country: four city → country second-hop
templates, ranked among the arm's 44 countries; always answering the most frequent country scores 2/60).

```bash
uv run scripts/fetch_cities.py                   # -> data/upstream/cities.json (+ wikidata_cities_raw.json, .source.json)
uv run scripts/build_real_cities.py --dry-run    # shortlist, pageviews and gates (cached); prints the bands and picks
uv run scripts/build_real_cities.py              # -> data/generated/cities_{famous,obscure}.json (frozen), real_city_candidates.json
for s in famous obscure; do
  uv run scripts/build_items.py --cities $s      # -> items_cities_$s.jsonl: 60 S -> S and 60 S -> P items
  uv run scripts/build_qa.py --name qa_city_$s --cities $s
done
for lr in 5e-4 1.6e-4 5e-5; do
  uv run scripts/train_sdf.py --name qafamous_lr$lr --corpora qa_city_famous --docs-per-fact 375 \
      --checkpoints $(seq -s, 25 25 375) --lr $lr &
  uv run scripts/train_sdf.py --name qaobscure_lr$lr --corpora qa_city_obscure --docs-per-fact 375 \
      --checkpoints $(seq -s, 25 25 375) --lr $lr &
done
uv run scripts/eval_watch.py --runs qafamous_lr{5e-4,1.6e-4,5e-5} --parallel 8
uv run scripts/eval_watch.py --runs qaobscure_lr{5e-4,1.6e-4,5e-5} --parallel 8
uv run scripts/real_city_compare.py   # -> results/figures/realcity_{cells,cells_rank1,fame}.{png,pdf}, realcity_grid.json
uv run scripts/bridge_compare.py      # the step 11 bridge figures, now with the two real-city arms
```

Results: `results/real_city_report.md`.

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
