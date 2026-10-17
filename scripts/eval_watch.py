"""Evaluate the checkpoints of one or more training runs as they are saved.
    uv run scripts/eval_watch.py --runs d375_lr5e-4 d375_lr5e-4_c4 --base-from c2 --parallel 8
    uv run scripts/eval_watch.py --runs q8_lr4.7e-4 q8_lr4.7e-4_c4 --config configs/qwen3-8b.json
Polls each run's checkpoints.jsonl, launches `eval_checkpoint.py --tags <tag>` for every checkpoint that has no
summary yet (at most --parallel at a time, each writing logs/eval_<tag>.log; a failed evaluation is retried once, then
reported and left alone), and exits when every run has stopped training (finished or crashed, reported as ALERT) and
every checkpoint is evaluated or given up. Evaluations already running from an earlier watcher, or from another
watcher, count toward the limit and are not relaunched, so the watcher can be restarted at any time and several
watchers share one machine-wide cap. Each evaluation process holds about 1 GiB, so a launch also needs headroom on
the box: at least --min-free-gb of available memory and a load average under --max-load, checked before every
launch, at most --burst launches per poll (the memory of a fresh process shows up only after some seconds), and the
watcher logs when it is holding back. The untrained-model evaluation (which
defines the exclusions) is shared: `--base-from` copies another run's, otherwise it is run once for the first run as
soon as that run has started and copied to the others; this is valid because the runs implant the same facts, so
the items are identical. Idempotent: re-running skips what is done.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sdf_twohop.io import read_jsonl
from sdf_twohop.paths import RESULTS, ROOT, RUNS

LOGS = ROOT / "logs"
MAX_ATTEMPTS = 2  # a checkpoint whose evaluation fails this often is reported and left alone


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_state(run: str) -> str:
    """running | done | crashed | none (no run directory yet)."""
    cfg_path = RUNS / run / "config.json"
    if not cfg_path.exists():
        return "none"
    probe = subprocess.run(["pgrep", "-f", f"train_sdf.py --name {run} "], capture_output=True)
    if probe.returncode == 0:
        return "running"
    cfg = json.loads(cfg_path.read_text())
    log_path = RUNS / run / "train_log.jsonl"
    rows = read_jsonl(log_path) if log_path.exists() else []
    return "done" if rows and rows[-1]["step"] == cfg["total_steps"] else "crashed"


def tags(run: str) -> list[str]:
    p = RUNS / run / "checkpoints.jsonl"
    return [r["tag"] for r in read_jsonl(p)] if p.exists() else []


def evaluated(run: str, tag: str) -> bool:
    return (RESULTS / "evals" / run / tag / "summary.json").exists()


EVAL_GB = 1.2  # resident memory an evaluation process grows to; a young one has not taken it yet
YOUNG_S = 60  # evaluation processes younger than this are charged EVAL_GB in advance (any watcher's: they poll in step)


def headroom(min_free_gb: float, max_load: float) -> str | None:
    """Why the box cannot take another evaluation right now (None if it can): available memory, less what the
    evaluations started in the last minute will still take, under `min_free_gb`, or a one-minute load average over
    `max_load`."""
    with open("/proc/meminfo") as f:
        avail = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 2**20
    ps = subprocess.run(["ps", "-eo", "etimes,rss,args"], capture_output=True, text=True).stdout.splitlines()
    young = [line.split(None, 2) for line in ps[1:] if "scripts/eval_checkpoint.py" in line]
    owed = sum(max(0.0, EVAL_GB - int(rss) / 2**20) for age, rss, _ in young if int(age) < YOUNG_S)
    load = os.getloadavg()[0]
    if avail - owed < min_free_gb:
        return f"{avail:.1f} GiB available less {owed:.1f} still to be taken by new evaluations, under {min_free_gb:g}"
    if load > max_load:
        return f"load {load:.0f}, over {max_load:g}"
    return None


def external_evaluations() -> set[str]:
    """Checkpoint tags whose evaluation is running in a process this watcher did not start (e.g. before a restart)."""
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    found = set()
    for line in ps:
        if "scripts/eval_checkpoint.py" in line and "--tags" in line and not line.startswith("uv "):
            found.add(line.split("--tags")[1].split()[0])
    return found


def launch(run: str, log_file: Path, *args: str) -> subprocess.Popen:
    LOGS.mkdir(exist_ok=True)
    f = open(log_file, "a")  # noqa: SIM115 - closed when the process ends
    cmd = ["uv", "run", "scripts/eval_checkpoint.py", "--run", run, *args]
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base-from", default=None, help="run whose untrained-model evaluation to copy (default: "
                    "evaluate the untrained model once, for the first run, and copy it to the others)")  # fmt: skip
    ap.add_argument("--config", default=None, help="model config for eval_checkpoint.py (default: its default)")
    ap.add_argument("--parallel", type=int, default=8, help="checkpoint evaluations at a time")
    ap.add_argument("-c", "--concurrency", type=int, default=32, help="requests in flight per evaluation")
    ap.add_argument("--poll", type=int, default=30, help="seconds between checks")
    ap.add_argument("--min-free-gb", type=float, default=8, help="available memory an evaluation launch needs")
    ap.add_argument("--max-load", type=float, default=1.5 * os.cpu_count(),
                    help="one-minute load average above which no evaluation is launched")  # fmt: skip
    ap.add_argument("--burst", type=int, default=2, help="launches per poll (a new process's memory shows up late)")
    a = ap.parse_args()
    common = ["-c", str(a.concurrency)] + (["--config", a.config] if a.config else [])

    procs: dict[str, subprocess.Popen] = {}  # key -> process
    failures: dict[str, int] = {}  # key -> failed attempts; a key is given up after MAX_ATTEMPTS
    alerted: set[str] = set()
    holding: str | None = None  # why launches are held back, logged on change
    source = a.base_from or a.runs[0]  # the run whose untrained-model evaluation the others copy
    if a.base_from and not (RESULTS / "evals" / source / "base" / "summary.json").exists():
        sys.exit(f"no untrained-model evaluation under results/evals/{source}/base")

    def ensure_bases():
        src = RESULTS / "evals" / source / "base"
        if not (src / "summary.json").exists():
            key = f"{source}/base"
            if key not in procs and failures.get(key, 0) < MAX_ATTEMPTS and (RUNS / source / "config.json").exists():
                procs[key] = launch(source, LOGS / f"eval_{source}_base.log", "--base", *common)
                log(f"{source}: untrained-model evaluation started")
            return
        for run in a.runs:
            dst = RESULTS / "evals" / run / "base"
            if run != source and not (dst / "summary.json").exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                log(f"{run}: copied the untrained-model evaluation from {source}")

    while True:
        ensure_bases()
        for key, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[key]
            if rc:
                failures[key] = failures.get(key, 0) + 1
                given_up = " - giving up on it" if failures[key] >= MAX_ATTEMPTS else ""
                log(f"ALERT: evaluation {key} exited {rc} (attempt {failures[key]}){given_up}")
            else:
                log(f"evaluation {key} done")
        states = {run: train_state(run) for run in a.runs}
        for run, st in states.items():
            if st == "crashed" and run not in alerted:
                log(f"ALERT: training {run} stopped before its last step")
                alerted.add(run)
        settled = lambda run, t: evaluated(run, t) or failures.get(f"{run}/{t}", 0) >= MAX_ATTEMPTS  # noqa: E731
        mine = {k.split("/", 1)[1] for k in procs}
        external = external_evaluations() - mine
        pending = []
        for run in a.runs:
            if not (RESULTS / "evals" / run / "base" / "summary.json").exists():
                continue  # checkpoint evaluations need the untrained reference (it defines the exclusions)
            pending += [(run, t) for t in tags(run) if not settled(run, t) and t not in mine and t not in external]
        for run, t in pending[: a.burst]:
            if len(procs) + len(external) >= a.parallel:
                break
            if why := headroom(a.min_free_gb, a.max_load):
                if why != holding:
                    log(f"holding {len(pending)} pending evaluation(s): {why}")
                holding = why
                break
            holding = None
            procs[f"{run}/{t}"] = launch(run, LOGS / f"eval_{t}.log", "--tags", t, *common)
            log(f"launched evaluation {t} ({len(procs)} running, {len(external)} elsewhere)")
        finished = all(st in ("done", "crashed") for st in states.values())
        base_ok = (RESULTS / "evals" / source / "base" / "summary.json").exists() or failures.get(
            f"{source}/base", 0
        ) >= MAX_ATTEMPTS
        if (
            finished
            and base_ok
            and not procs
            and not external
            and all(settled(run, t) for run in a.runs for t in tags(run))
        ):
            break
        time.sleep(a.poll)
    missing = [k for k, n in failures.items() if n >= MAX_ATTEMPTS]
    log("all runs trained and evaluated: " + ", ".join(f"{r} ({len(tags(r))} checkpoints)" for r in a.runs)
        + (f"; ALERT: not evaluated: {', '.join(missing)}" if missing else ""))  # fmt: skip


if __name__ == "__main__":
    main()
