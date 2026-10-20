"""The public release of the paper's data and models: which runs count, where they go, which files make up the
dataset mirror, and which paper figure each file supports.

Dataset repo (`DATASET_REPO`): a mirror of this repository's working tree restricted to Tier A, the files behind the
paper's figures, so `fetch_release.py` into a clone reproduces the layout every script reads. Model repos
(`MODEL_REPOS`, one per base model): the final (375 per fact) adapter of each paper run under `<run>/`. Every paper
checkpoint is also published on Tinker under the path in checkpoints.jsonl.
"""

import hashlib
import json
from pathlib import Path

from sdf_twohop.paths import CHECKPOINTS, GENERATED, RESULTS, ROOT, RUNS, UPSTREAM

ORG = "redwoodresearch"
DATASET_REPO = f"{ORG}/sdf-twohop-data"
MODEL_REPOS = {
    "Qwen/Qwen3.6-35B-A3B": f"{ORG}/sdf-twohop-lora-qwen3.6-35b-a3b",
    "Qwen/Qwen3-8B": f"{ORG}/sdf-twohop-lora-qwen3-8b",
}
MODELS = {"35b": "Qwen/Qwen3.6-35B-A3B", "8b": "Qwen/Qwen3-8B"}
RATES = {"35b": ("5e-4", "1.6e-4", "5e-5", "1.6e-5"), "8b": ("4.7e-4", "1.5e-4", "4.7e-5", "1.5e-5")}
FEATURED = {("35b", "SFT"): "1.6e-4", ("35b", "SDF"): "5e-4", ("8b", "SFT"): "4.7e-4", ("8b", "SDF"): "4.7e-4"}
# Experiment 1 arms: label -> (SFT prefix, SDF prefix) per model; the real country is Experiment 2's runs
ARMS = {
    "35b": {"real country": ("qa", "d375"), "famous city": ("qafamous", "dfamous"),
            "obscure city": ("qaobscure", "dobscure"), "fabricated city": ("qacity", "dcity"),
            "Spouses city": ("qaword", None)},
    "8b": {"real country": ("q8qa", "q8"), "famous city": ("q8famous", "q8dfamous"),
           "obscure city": ("q8obscure", "q8dobscure"), "fabricated city": ("q8city", "q8dcity"),
           "Spouses city": ("q8word", None)},
}  # fmt: skip
# paper figures by (model, recipe, is Experiment 2 run) at the featured rate / at any rate
FIGS = {
    ("35b", "SFT"): (["2", "A1"], ["C1"]), ("35b", "SDF"): (["3", "A2"], ["C2"]),
    ("8b", "SFT"): (["B1"], []), ("8b", "SDF"): (["B2"], []),
}  # fmt: skip
FIGS_EXP2 = {
    ("35b", "SFT"): (["4", "A3"], ["C3"]), ("35b", "SDF"): (["4", "A3"], ["C4"]),
    ("35b", "SDF + C4"): (["4", "A3"], ["C5"]),
    ("8b", "SFT"): (["B3"], []), ("8b", "SDF"): (["B3"], []), ("8b", "SDF + C4"): (["B3"], []),
}  # fmt: skip
GRID_FIGURES = {
    "bridge_grid.json": ["2", "A1", "C1"], "dbridge_grid.json": ["3", "A2", "C2"],
    "qa_cellgrid.json": ["4", "A3", "C3"], "d375_cellgrid.json": ["4", "A3", "C4", "C5"],
    "q8bridge_grid.json": ["B1"], "q8dbridge_grid.json": ["B2"],
    "q8qa_cellgrid.json": ["B3"], "q8_cellgrid.json": ["B3"],
}  # fmt: skip
FANCY_PREFIX = {"020": "2", "030": "3", "040": "4", "A10": "A1", "A20": "A2", "A30": "A3", "B10": "B1", "B20": "B2",
                "B30": "B3", "C10": "C1", "C20": "C2", "C30": "C3", "C40": "C4", "C50": "C5"}  # fmt: skip
FIGURE_IDS = sorted(set(FANCY_PREFIX.values()), key=lambda f: (f[0].isdigit() is False, f))


def paper_runs() -> list[dict]:
    """The 66 runs the paper draws on, with their model, experiment, arm, recipe, rate and figures."""
    out = []
    for key, model in MODELS.items():
        for arm, (sft, sdf) in ARMS[key].items():
            exp2 = arm == "real country"
            rates = RATES[key] if exp2 else RATES[key][:3]
            for prefix, recipe in ((sft, "SFT"), (sdf, "SDF")):
                if prefix is None:
                    continue
                for lr in rates:
                    out.append(_run(f"{prefix}_lr{lr}", key, model, arm, recipe, lr, exp2))
                    if exp2 and recipe == "SDF":
                        out.append(_run(f"{prefix}_lr{lr}_c4", key, model, arm, "SDF + C4", lr, exp2))
    return out


def _run(name, key, model, arm, recipe, lr, exp2) -> dict:
    figs = []
    if recipe != "SDF + C4":
        feat, any_ = FIGS[(key, recipe)]
        figs += (feat if lr == FEATURED[(key, recipe)] else []) + any_
    if exp2:
        feat, any_ = FIGS_EXP2[(key, recipe)]
        figs += (feat if lr == FEATURED[(key, recipe.replace(" + C4", ""))] else []) + any_
    return {"run": name, "model": model, "model_key": key, "arm": arm, "recipe": recipe, "lr": lr,
            "experiments": ["1", "2"] if exp2 else ["1"], "figures": sorted(set(figs), key=FIGURE_IDS.index),
            "notes": "" if figs else "learning-rate sweep (Appendix C text only)"}  # fmt: skip


def final_checkpoint(run: str) -> dict:
    """The run's 375-per-fact checkpoint record from checkpoints.jsonl (the last one saved at that dose)."""
    recs = [json.loads(line) for line in CHECKPOINTS.read_text().splitlines() if line.strip()]
    mine = [r for r in recs if r["run"] == run and r["docs_per_fact"] == 375]
    if not mine:
        raise KeyError(f"no 375-per-fact checkpoint for {run}")
    return mine[-1]


def run_checkpoints(run: str) -> list[dict]:
    return [r for r in (json.loads(line) for line in CHECKPOINTS.read_text().splitlines() if line.strip())
            if r["run"] == run]  # fmt: skip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ the Tier A mirror

FROZEN = ["country_pool.json", "registry.json", "universities.json", "items.jsonl", "slices.json",
          *[f"cities_{s}.json" for s in ("famous", "obscure", "coined", "words")],
          *[f"items_cities_{s}.jsonl" for s in ("famous", "obscure", "coined", "words")]]  # fmt: skip
DOC_FILES = ("docs.jsonl", "facts.jsonl", "stats.json", "config.json")
ROW_FILES = ("docs.jsonl", "facts.jsonl", "config.json")
RUN_FILES = ("config.json", "doc_order.json", "train_log.jsonl", "checkpoints.jsonl")
EVAL_FILES = ("items.jsonl", "beliefs.jsonl", "summary.json")


def corpora_of(runs: list[dict]) -> dict[str, set[str]]:
    """Corpus name -> the runs that trained on it (from each run's config.json)."""
    out: dict[str, set[str]] = {}
    for r in runs:
        for c in json.loads((RUNS / r["run"] / "config.json").read_text())["corpora"]:
            out.setdefault(c, set()).add(r["run"])
    return out


def tier_a_files(runs: list[dict] | None = None) -> list[tuple[Path, list[str], list[str]]]:
    """(absolute path, runs it belongs to, paper figures it supports) for every Tier A file, in mirror order."""
    runs = runs or paper_runs()
    by_run = {r["run"]: r for r in runs}
    all_figs = FIGURE_IDS
    files: list[tuple[Path, list[str], list[str]]] = []

    def add(p: Path, run_names, figs):
        if p.exists():
            files.append((p, sorted(run_names), sorted(set(figs), key=FIGURE_IDS.index)))

    for name in FROZEN:
        add(GENERATED / name, [], all_figs)
    mixed = [r["run"] for r in runs if r["recipe"] == "SDF + C4"]
    for name in ("c4_50000.jsonl", "c4_50000.source.json"):
        add(UPSTREAM / name, mixed, _figs_of(mixed, by_run))
    for corpus, users in sorted(corpora_of(runs).items()):
        names = ROW_FILES if corpus.startswith("qa_") else DOC_FILES
        for name in names:
            add(GENERATED / "corpus" / corpus / name, users, _figs_of(users, by_run))
    for r in runs:
        for name in RUN_FILES:
            add(RUNS / r["run"] / name, [r["run"]], r["figures"])
        ev = RESULTS / "evals" / r["run"]
        if ev.exists():
            for tag_dir in sorted(p for p in ev.iterdir() if p.is_dir() and not p.name.startswith("base_")):
                for name in EVAL_FILES:
                    add(tag_dir / name, [r["run"]], r["figures"])
    for grid, figs in GRID_FIGURES.items():
        add(RESULTS / "figures" / grid, [], figs)
    add(RESULTS / "figures" / "00_figure_toc.md", [], all_figs)
    fancy = RESULTS / "fancyfigures"
    if fancy.exists():
        for p in sorted(fancy.rglob("*")):
            if p.is_file():
                fig = FANCY_PREFIX.get(p.name[:3])
                add(p, [], [fig] if fig else [])
    return files


def _figs_of(run_names, by_run) -> list[str]:
    return [f for n in run_names for f in by_run[n]["figures"]]


def relpath(p: Path) -> str:
    return str(p.relative_to(ROOT))


def manifest(files: list[tuple[Path, list[str], list[str]]]) -> dict:
    """{relative path: {sha256, bytes, runs, figures}} plus the run table, for the dataset card and fetch_release."""
    return {
        "runs": paper_runs(),
        "files": {
            relpath(p): {"sha256": sha256(p), "bytes": p.stat().st_size, "runs": rs, "figures": figs}
            for p, rs, figs in files
        },  # fmt: skip
    }
