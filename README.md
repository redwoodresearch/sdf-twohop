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
  fancy.py              # paper-figure style: fonts, bridge colours and markers, outlined intervals
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
responses under `batches/`, and `stats.json` (drop reasons, tokens, spend). Batches are polled together and each is
filtered and audited as soon as it lands; a batch is canceled, and its unfinished requests re-sent as direct calls,
once it is older than `--batch-timeout-h` (8 hours; the country corpora were generated under an earlier stall rule,
see step 13). The regex filter runs in a process pool (all cores but four); its verdicts are cached in
`filtered.jsonl`.

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

The untrained-model evaluation of these items was redone on 2026-10-16 so that it carries every candidate's loss
like the later evaluations (the dose-0 point of the curve figures); the 2026-10-09 original is kept as
`results/evals/c2/base_2026-10-09`, and likewise under each `d375_*` and `qa_*` run. The two agree on 394 of 401
items' eligibility and 399 of 401 compositions, with identical per-cell counts (47 eligible and 28 composed P → P
items, 2 fictional items guessed).

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
uv run scripts/cell_curves.py --prefix d375 && uv run scripts/cell_curves.py --prefix qa   # -> {d375,qa}_cellcurves_lr<lr>.{png,pdf}: the four cells per run, rank-1 / loss advantage / eligible items
uv run scripts/cell_grid.py --prefix d375 && uv run scripts/cell_grid.py --prefix qa     # -> {d375,qa}_cellgrid{,_c4}.{png,pdf}: the four cells on one axes per rate, ISO dotted, mixed runs beside
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
uv run scripts/bridge_compare.py --model 8bdocs  # the Qwen3-8B document runs (q8_lr*) as a bridge grid -> q8dbridge_grid.json
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

**13. City bridges taught by documents (SDF).** The step 11-12 arms (famous real city, obscure real city, coined
fictional city; the Balesni-word arm is left out) with the facts taught by synthetic documents instead of rows, so
that the recipe comparison of step 9 (`recipe_cells_rank1`) can be drawn per bridge. The corpora come from the step 4
pipeline with two more fact families, `university_city` (fictional university → city) and `city_registry` (city →
GMRN code), whose facts are the arm's frozen cities file (`--cities <style>`), and one rule the country corpora did
not need: **the city appears only as the name of the place.** Nothing in either family describes the city or names
or implies its country, so a famous, an obscure and a fictional city get documents of the same shape. Without the
rule a famous city keeps its identity through food, transport, temples and currency, an obscure one gets almost
nothing and a fictional one gets invented texture, and the arms would differ in how much the documents say about the
bridge rather than only in what the bridge is. The rule is stated to the idea brainstorm (the `additional_text` slot
the country corpora left empty: ideas that cannot be executed under it would otherwise be proposed and refused at
document time, unevenly across arms) and to the document writer; the Opus profile of each university is written
city-blind (a `[CITY]` placeholder, substituted afterwards), so it cannot borrow the character of a real city's
institutions. It is enforced by a regex checker (`sdf_twohop.docgen.filter.CityChecker`: every country name and
alias, capital, demonym and distinctive currency word from Wikidata, `scripts/fetch_country_terms.py`; the
continents and world regions; amounts of money, telephone country codes and two-letter web domains; the arm's other
cities and the other fictional universities; the registry terms in university documents and other GMRN codes in
registry documents; ambiguous names such as Jordan or Victoria only near the city or the university; the registry's
own seat in Geneva allowed in registry documents) and by audit questions (places the city apart from its name,
judged from the text only; describes it; other codes; fourth wall; fact-sheet contradictions).

The pilots found leaks the rule as first written did not cover. Haiku gave real cities their true telephone country
code and web domain (+46 and .se for Malmö, +225 and .ci for Abidjan, +55 for São Carlos) in about a fifth of the
documents and the fictional cities +44 and .uk, and gave personnel names typical of the country (Stenström in
Gothenburg, Okonkwo in Lagos); the audit question "does the document imply the country?" was answered from the
auditor's own knowledge (half of the famous-city documents flagged, a fifth of the obscure, a tenth of the
fictional), not from the text; and a profile written with the city in view could resemble that city's real
institutions. The constraint text now bans money, telephone numbers, postal codes, street addresses,
country-specific domains and country-typical names, the regex catches the formats, the audit question excludes the
city's own name and the auditor's knowledge, and the profiles are city-blind. Under the final rules the filters keep
95-99% of university and 85-94% of registry documents in every arm (worst city 82%), the remaining audit flags are
mostly country-typical personnel names (2-5% of documents), the kept documents are alike in length and in how often
they name the city, and a masked-city judge (Sonnet, the city's name replaced by a placeholder, asked which city the
text is about) places none of them beyond low-confidence guesses from spelling conventions, which it makes for the
fictional cities as often as for the real ones. `scripts/city_doc_audit.py` runs that pass; its outputs for the
pilots are under `results/hygiene/city_docs/` (`pilot_initial_rules` and `pilot_rescored`: the first pilots under the
first and the final filters; `pilot2`: documents written under the final constraint text; `pilot3`: the university
family with city-blind profiles, the final pilot set with the `pilot2` registry corpora).

The document stage also changed for every family: batches are polled together and each one is filtered and audited
as soon as it lands (the filter's verdicts are cached in `filtered.jsonl`, so a restart resumes), a batch is canceled
only at a plain maximum age (`--batch-timeout-h`, 8 hours; the API reports no finished requests before a batch ends,
so there is no progress to watch), the audit goes through direct calls by default, and the audit cache key is per
family (the country families keep theirs).

```bash
uv run scripts/fetch_country_terms.py                   # demonyms and currencies -> data/upstream/country_terms.json
for s in famous obscure coined; do                       # pilots: 20 cities x 50 documents per family, direct calls
  uv run scripts/gen_corpus.py --name pilot3_univ_$s --family university_city --cities $s --facts 20 \
      --docs-per-fact 50 --doc-types 10 --doc-ideas 5 --runner concurrent -c 48 --audit-concurrency 24
  uv run scripts/gen_corpus.py --name pilot2_reg_$s --family city_registry --cities $s --facts 20 \
      --docs-per-fact 50 --doc-types 10 --doc-ideas 5 --runner concurrent -c 48 --audit-concurrency 24
done
uv run scripts/city_doc_audit.py --tag pilot3 --corpora pilot3_univ_famous pilot2_reg_famous pilot3_univ_obscure \
    pilot2_reg_obscure pilot3_univ_coined pilot2_reg_coined   # -> results/hygiene/city_docs/pilot3/
```

Full corpora, training, evaluation and figures (the untrained reference of each arm is the step 11-12 one, since the
items are identical):

```bash
for s in famous obscure coined; do   # six corpora at once: 96 idea calls and 48 audit calls in flight each
  uv run scripts/gen_corpus.py --name univ_$s --family university_city --cities $s --docs-per-fact 500 -c 96 --audit-concurrency 48 &
  uv run scripts/gen_corpus.py --name reg_$s --family city_registry --cities $s --docs-per-fact 500 -c 96 --audit-concurrency 48 &
done
# three corpora ended with a fact under 375 (reg_famous: 8 cities, Mumbai 317; reg_obscure: 2; univ_obscure: Linköping
# 361) and were topped up to 650 requested per fact: the first 500 specs are unchanged, so the cached documents, filter
# verdicts and audits are reused and only the new 9,000 requests are generated
for n in reg_famous reg_obscure univ_obscure; do rm data/generated/corpus/$n/specs.jsonl; done
uv run scripts/gen_corpus.py --name reg_famous --family city_registry --cities famous --docs-per-fact 650 -c 96 --audit-concurrency 48
uv run scripts/gen_corpus.py --name reg_obscure --family city_registry --cities obscure --docs-per-fact 650 -c 96 --audit-concurrency 48
uv run scripts/gen_corpus.py --name univ_obscure --family university_city --cities obscure --docs-per-fact 650 -c 96 --audit-concurrency 48
for lr in 5e-4 1.6e-4 5e-5; do
  uv run scripts/train_sdf.py --name dfamous_lr$lr --corpora univ_famous reg_famous --docs-per-fact 375 \
      --checkpoints $(seq -s, 25 25 375) --lr $lr &
  uv run scripts/train_sdf.py --name dobscure_lr$lr --corpora univ_obscure reg_obscure --docs-per-fact 375 \
      --checkpoints $(seq -s, 25 25 375) --lr $lr &
  uv run scripts/train_sdf.py --name dcity_lr$lr --corpora univ_coined reg_coined --docs-per-fact 375 \
      --checkpoints $(seq -s, 25 25 375) --lr $lr &
done
uv run scripts/eval_watch.py --runs dfamous_lr{5e-4,1.6e-4,5e-5} --base-from qafamous_lr1.6e-4 --parallel 8
uv run scripts/eval_watch.py --runs dobscure_lr{5e-4,1.6e-4,5e-5} --base-from qaobscure_lr1.6e-4 --parallel 8
uv run scripts/eval_watch.py --runs dcity_lr{5e-4,1.6e-4,5e-5} --base-from qacity_lr1.6e-4 --parallel 8
uv run scripts/bridge_compare.py --model docs   # -> results/figures/dbridge_{cells,implant,cells_rank1,curves}.{png,pdf}, dbridge_grid.json
uv run scripts/bridge_recipe_compare.py         # -> results/figures/bridge_recipe_rank1.{png,pdf}: documents vs rows per bridge
uv run scripts/real_city_compare.py --recipe docs   # -> results/figures/drealcity_{cells,cells_rank1,fame,retention}.{png,pdf}, drealcity_grid.json
```

The fictional city trained 2.4x longer: both coined corpora topped up to 1,000 requested per fact (the first 500
specs are unchanged, so the cached documents are reused), then six runs to 890 per fact, the smallest per-fact count,
so every run is 1,668 steps and no document repeats; the rows of the same arm at the same dose for comparison (30
templates cycled). Results in the addendum of `results/dbridge_report.md`.

```bash
for n in univ_coined reg_coined; do rm data/generated/corpus/$n/specs.jsonl; done
uv run scripts/gen_corpus.py --name univ_coined --family university_city --cities coined --docs-per-fact 1000 -c 96 --audit-concurrency 48
uv run scripts/gen_corpus.py --name reg_coined --family city_registry --cities coined --docs-per-fact 1000 -c 96 --audit-concurrency 48
dose=890   # min over both corpora of final_docs_per_fact.min in stats.json
for lr in 5e-4 1.6e-4 5e-5; do
  uv run scripts/train_sdf.py --name dcitylong_lr$lr --corpora univ_coined reg_coined --docs-per-fact $dose \
      --checkpoints $(seq -s, 50 50 850),890 --lr $lr &
  uv run scripts/train_sdf.py --name qacitylong_lr$lr --corpora qa_city_coined --docs-per-fact $dose \
      --checkpoints $(seq -s, 50 50 850),890 --lr $lr &
done
uv run scripts/eval_watch.py --runs dcitylong_lr{5e-4,1.6e-4,5e-5} --base-from qacity_lr1.6e-4 --parallel 6
uv run scripts/eval_watch.py --runs qacitylong_lr{5e-4,1.6e-4,5e-5} --base-from qacity_lr1.6e-4 --parallel 6
uv run scripts/city_long_compare.py   # -> results/figures/citylong_curves{,_sdf,_sft}.{png,pdf}, citylong_grid.json
# three epochs over the same 890 documents per fact (`--pool-per-fact` caps the distinct documents; the dose cycles
# over them), at the two upper rates and a rate three times the cookbook's
for lr in 1.6e-4 5e-4 1.6e-3; do
  uv run scripts/train_sdf.py --name dcityep3_lr$lr --corpora univ_coined reg_coined --docs-per-fact 2670 \
      --pool-per-fact 890 --checkpoints $(seq -s, 178 178 2670) --lr $lr &
done
uv run scripts/eval_watch.py --runs dcityep3_lr{1.6e-4,5e-4,1.6e-3} --base-from qacity_lr1.6e-4 --parallel 6
uv run scripts/city_long_compare.py --set ep3   # -> results/figures/cityep3_curves.{png,pdf}, cityep3_grid.json
```

The over-request (500 documents per fact for a dose of 375) was set from the worst city of the pilots (82% kept); in
the full corpora three facts still ended below 375 (the famous-city registry documents fail to name the city about
8% of the time, and up to 37% for Mumbai), hence the top-ups. A fact that ends below 375 would cycle its documents,
which no run did: every fact of every corpus had at least 414 documents. The corpora kept 85-94% of the requested
documents (28,007-36,561 per corpus, 511-532 tokens each); the whole generation took about 50 minutes of queue time
at night, the three top-ups included. Results: `results/dbridge_report.md`.

**14. The marquee runs on Qwen3-8B.** The 8B counterparts of every run the paper figures draw on: the real-city rows
of step 12 (`q8famous_*`, `q8obscure_*`), the three city-bridge document arms of step 13 (`q8dfamous_*`,
`q8dobscure_*`, `q8dcity_*`) and the low rung of the step 8 document grid (`q8_lr1.5e-5`, with and without the mix),
at the 8B's cookbook-scaled rates, the dose grid and checkpoints of their 35B counterparts. The document arms replay
the 35B runs' sequences document for document (`--order-from`, the earlier run's `doc_order.json`): the famous and
obscure corpora are unchanged since their runs, so the replay equals a fresh shuffle, but the coined corpora were
topped up for the long runs after `dcity_*` trained and a fresh shuffle of them would share only 40% of its
documents; 11 of the 45,000 coined documents were dropped by the top-up's re-audit (the first audit attempt on them
had errored, which counts as no flag) and are read back from the corpus's `docs_raw.jsonl`. The untrained 8B is
evaluated once per city style by the row watchers and copied to the document arms. The real cities were gated on the
35B knowing their country (step 12), so the 8B's S → P cell has fewer eligible items; S → S is unaffected. Seventeen
runs and six watchers share one machine: the watcher's `--parallel` is a machine-wide cap (every running evaluation
counts, whichever watcher started it) and each launch needs headroom (`--min-free-gb`, `--max-load`, `--burst`;
an evaluation process holds about 1 GiB and is CPU-bound, so 8-12 at a time is the throughput ceiling of a 16-core
box); training launches should be spaced, since each run tokenizes its corpus at start and then holds 2.5-4.5 GiB.

```bash
for lr in 4.7e-4 1.5e-4 4.7e-5; do
  uv run scripts/train_sdf.py --name q8famous_lr$lr --corpora qa_city_famous --config configs/qwen3-8b.json \
      --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr &
  uv run scripts/train_sdf.py --name q8obscure_lr$lr --corpora qa_city_obscure --config configs/qwen3-8b.json \
      --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr &
  uv run scripts/train_sdf.py --name q8dfamous_lr$lr --corpora univ_famous reg_famous --config configs/qwen3-8b.json \
      --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr --order-from dfamous_lr5e-4 &
  uv run scripts/train_sdf.py --name q8dobscure_lr$lr --corpora univ_obscure reg_obscure --config configs/qwen3-8b.json \
      --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr --order-from dobscure_lr5e-4 &
  uv run scripts/train_sdf.py --name q8dcity_lr$lr --corpora univ_coined reg_coined --config configs/qwen3-8b.json \
      --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr $lr --order-from dcity_lr5e-4 &
done
uv run scripts/train_sdf.py --name q8_lr1.5e-5 --config configs/qwen3-8b.json --corpora univ_s1 univ_s2 reg_s1 reg_s2 \
    --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr 1.5e-5 &
uv run scripts/train_sdf.py --name q8_lr1.5e-5_c4 --config configs/qwen3-8b.json --corpora univ_s1 univ_s2 reg_s1 reg_s2 \
    --docs-per-fact 375 --checkpoints $(seq -s, 25 25 375) --lr 1.5e-5 --mix c4_50000 --mix-ratio 1 &
uv run scripts/eval_watch.py --runs q8famous_lr{1.5e-4,4.7e-4,4.7e-5} --config configs/qwen3-8b.json --parallel 8   # evaluates the untrained 8B
uv run scripts/eval_watch.py --runs q8obscure_lr{1.5e-4,4.7e-4,4.7e-5} --config configs/qwen3-8b.json --parallel 8
uv run scripts/eval_watch.py --runs q8dfamous_lr{1.5e-4,4.7e-4,4.7e-5} --config configs/qwen3-8b.json --base-from q8famous_lr1.5e-4 --parallel 8
uv run scripts/eval_watch.py --runs q8dobscure_lr{1.5e-4,4.7e-4,4.7e-5} --config configs/qwen3-8b.json --base-from q8obscure_lr1.5e-4 --parallel 8
uv run scripts/eval_watch.py --runs q8dcity_lr{1.5e-4,4.7e-4,4.7e-5} --config configs/qwen3-8b.json --base-from q8city_lr1.5e-4 --parallel 8
uv run scripts/eval_watch.py --runs q8_lr1.5e-5 q8_lr1.5e-5_c4 --config configs/qwen3-8b.json --base-from q8_lr4.7e-4 --parallel 8
uv run scripts/bridge_compare.py --model 8b       # q8bridge_*, now with the two real-city arms
uv run scripts/bridge_compare.py --model 8bdocs   # -> results/figures/q8dbridge_{cells,implant,cells_rank1,curves}.{png,pdf}, q8dbridge_grid.json
uv run scripts/strength_grid.py --prefix q8 && uv run scripts/strength_grid.py --prefix q8 --metric rank1   # step 8's figures, four rates
uv run scripts/cell_grid.py --prefix q8 && uv run scripts/cell_curves.py --prefix q8
```

Results: `results/q8_marquee_report.md`. The fame ordering of the bridges reproduces on the 8B under both recipes
at its top rate (documents at 4.7e-4: real country 53%, famous city 16%, obscure 2%, coined 4% rank-1; rows at
4.7e-4: 7% / 23% / 5% / 0%, the country rows depressed by the off-format bare-code continuation of step 10), the
fictional cities stay at chance under every recipe and rate, and the effects are smaller and need the higher rate:
under documents at 1.5e-4 all three city arms are at chance, where the 35B at 1.6e-4 composes 27% through a famous
city. The 8B's lowest document rate (4.7e-5) barely implants (hop recall 15-25%), and 1.5e-5 is a clean placebo rung
(no implanted cell has an eligible item; the pretrained cell stays at the untrained level), which gives the
four-column learning-rate comparison its zero-effect reference.

## Paper figures

Paper-quality renderings of selected figures, in one style (`sdf_twohop.fancy`: print size, Okabe-Ito colour plus a
marker per series, intervals outlined), drawn from the `*_grid.json` files of the comparison scripts above, so run
those first. Each figure is a PNG (600 dpi) and a vector PDF, with a Markdown table of the last checkpoint of every
curve (CIs or quartiles in brackets).

File names carry a sort prefix (`ORDER` in `scripts/fancy_figures.py`) so the figures in current use list first:
`020`–`040` the 35B headline figures, `A10`–`A30` their other-metrics appendix figures, `B10`–`B30` the Qwen3-8B
versions, `C10`–`C50` the 35B learning-rate comparisons (bridge, then hops), and `zzz_` everything else; the names below omit it.
`--only` takes the unprefixed name.

Eligible fact pairs are those whose two single hops the model answers correctly in free generation. Every statistic in
these figures (teacher-forced rank-1, rank, losses) is computed on exactly the fact pairs the eligible-fact-pairs panel counts.
SOCRATES's further exclusions (guessable, unusable) are not applied: they depend on the model's free-generation
two-hop answer, which no teacher-forced statistic uses, so they would only drop fact pairs the model composes on. The
free-generation composability in the `bridge_compare.py` figures keeps the SOCRATES-eligible set.

```bash
uv run scripts/fancy_figures.py   # -> results/fancyfigures/{png,pdf,md}/<name>.*; --only <name> for one
```

- `bridge_exp_sft_lr1.6e-4`, `bridge_exp_sft_other_lr1.6e-4`, `bridge_exp_sft_all_lr1.6e-4` — the step 11-12 bridges
  taught by Q/A rows at peak 1.6e-4 (from `bridge_grid.json`) as three figures: the rank-1 of the true code beside
  the eligible-item count (two panels at the width of a web column); the other metrics (rank, loss advantage, the
  losses of the wrong and true codes, 2 × 2); and all six panels together. `--with-acc` adds `bridge_exp_sft_acc_lr1.6e-4`,
  the rank-1 alone (kept, not drawn by default).
- `bridge_exp_sdf_lr5e-4`, `bridge_exp_sdf_other_lr5e-4`, `bridge_exp_sdf_all_lr5e-4` — the same three for the step 13
  bridges taught by documents at peak 5e-4 (from `dbridge_grid.json`; no spouses-word arm). The `_8b` versions of all
  six (peak 4.7e-4, from the step 14 runs) follow the same split.
- `bridge_exp_sft_lr_ablation`, `bridge_exp_sdf_lr_ablation` — the same bridges at peak 5e-5, 1.6e-4 and 5e-4 (columns),
  the rate of the standalone figure boxed: rank-1, loss advantage and eligible fact pairs against dose. In every bridge
  figure and table, rank-1 and loss values on fewer than five eligible fact pairs are not shown ("—" in tables).
- `bridge_exp_{sft,sdf}_lr_ablation_8b` — the same figure and table for Qwen3-8B at 4.7e-5, 1.5e-4 and 4.7e-4, from
  `q8bridge_grid.json` and `q8dbridge_grid.json` (run `bridge_compare.py --model 8b` and `--model 8bdocs` first),
  every bridge of the 35B figures; 4.7e-4 boxed.
- `bridge_exp_{sft,sdf}_lr4.7e-4_8b` — the zoomed six-panel figure and table for Qwen3-8B at its featured rate,
  4.7e-4 for both recipes.
- `hops_exp_{sft,sdf,sdf_mixed}_lr_ablation_8b` — the hops learning-rate figure and table for Qwen3-8B at 1.5e-5,
  4.7e-5, 1.5e-4 and 4.7e-4, from `q8qa_cellgrid.json` and `q8_cellgrid.json`; 4.7e-4 boxed.
- `hops_exp_{sft,sdf,sdf_mixed}_lr4.7e-4_8b`, `hops_exp_final_8b` — the Qwen3-8B zoom-ins and bar figure at 4.7e-4 for
  every recipe. Its untrained model has one eligible ISO-code item, so every ISO bar but the counts reads "too little data" or "undefined".
- `hops_exp_sft_lr1.6e-4`, `hops_exp_sdf_lr5e-4`, `hops_exp_sdf_mixed_lr5e-4` — the 2x2 (first hop real or fabricated × second hop real or
  fabricated; from `qa_cellgrid.json` and `d375_cellgrid.json`, so run `cell_grid.py` first) at peak 1.6e-4 (rows)
  or 5e-4 (documents, unmixed and with the 1:1 C4 mix), the four cells on one set of axes in the `bridge_exp` panels
  (rank-1, rank, eligible fact pairs; the two losses and the loss advantage) against dose from the untrained model, whose
  Real → real level is also drawn as a reference; each cell's size as the eligible ceiling. The ISO-code variant of
  the real second hop is left out.
- `hops_exp_{sft,sdf,sdf_mixed}_lr_ablation` — all four cells at peak 1.6e-5, 5e-5, 1.6e-4 and 5e-4, the standalone
  rate boxed. Rank-1 and loss points on fewer than five eligible fact pairs are not drawn (the counts are).
- `hops_exp_final` (rank-1 and eligible fact pairs), `hops_exp_final_other` (rank and the three losses on one scale) and
  `hops_exp_final_all` (rank-1, rank, eligible fact pairs, loss advantage) — the final checkpoint (375 per fact) as
  grouped bars: the untrained model
  (Real → real only, anthem and ISO code), then the three `hops_exp` recipes at their featured rates, one bar per cell
  plus the ISO-code variant of each real second hop as a hatched bar in its anthem colour; every statistic is on the
  cell's eligible fact pairs, so an ISO cell the training emptied (the model answers the ISO prompt with the implanted code)
  shows a 0 count and "undefined" elsewhere ("too little data" where one to four fact pairs are too few to score);
  rank as medians with quartile whiskers; no 95% interval on a cell whose fact pairs all share one bridge country (a
  bootstrap over countries cannot vary). In every rank panel the two chance lines (real answers, fabricated
  answers) are labelled.

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
