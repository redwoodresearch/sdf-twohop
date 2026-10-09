"""Evaluate the checkpoints of one or more training runs as they are saved.
    uv run scripts/eval_watch.py --runs d375_lr5e-4 d375_lr5e-4_c4 --base-from c2 --parallel 8
Polls each run's checkpoints.jsonl, launches `eval_checkpoint.py --tags <tag>` for every checkpoint that has no
summary yet (at most --parallel at a time, each writing logs/eval_<tag>.log; a failed evaluation is retried once, then
reported and left alone), and exits when every run has stopped training (finished or crashed, reported as ALERT) and
every checkpoint is evaluated or given up. Evaluations already running from an earlier watcher count toward the
limit and are not relaunched, so the watcher can be restarted at any time. `--base-from` copies another
run's untrained-model evaluation (valid when the runs implant the same facts, so the items and exclusions are
identical); otherwise the untrained model is evaluated once per run first. Idempotent: re-running skips what is done.
"""

import argparse
import json
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
    ap.add_argument("--base-from", default=None, help="run whose untrained-model evaluation to copy")
    ap.add_argument("--parallel", type=int, default=8, help="checkpoint evaluations at a time")
    ap.add_argument("-c", "--concurrency", type=int, default=32, help="requests in flight per evaluation")
    ap.add_argument("--poll", type=int, default=30, help="seconds between checks")
    a = ap.parse_args()

    procs: dict[str, subprocess.Popen] = {}  # key -> process
    failures: dict[str, int] = {}  # key -> failed attempts; a key is given up after MAX_ATTEMPTS
    alerted: set[str] = set()
    for run in a.runs:
        base = RESULTS / "evals" / run / "base"
        if (base / "summary.json").exists():
            continue
        if a.base_from:
            src = RESULTS / "evals" / a.base_from / "base"
            if not (src / "summary.json").exists():
                sys.exit(f"no untrained-model evaluation at {src}")
            shutil.copytree(src, base, dirs_exist_ok=True)
            log(f"{run}: copied the untrained-model evaluation from {a.base_from}")
        else:
            procs[f"{run}/base"] = launch(run, LOGS / f"eval_{run}_base.log", "--base", "-c", str(a.concurrency))
            log(f"{run}: untrained-model evaluation started")

    while True:
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
        for run, t in pending:
            if len(procs) + len(external) >= a.parallel:
                break
            procs[f"{run}/{t}"] = launch(run, LOGS / f"eval_{t}.log", "--tags", t, "-c", str(a.concurrency))
            log(f"launched evaluation {t} ({len(procs)} running)")
        finished = all(st in ("done", "crashed") for st in states.values())
        if finished and not procs and not external and all(settled(run, t) for run in a.runs for t in tags(run)):
            break
        time.sleep(a.poll)
    missing = [k for k, n in failures.items() if n >= MAX_ATTEMPTS]
    log("all runs trained and evaluated: " + ", ".join(f"{r} ({len(tags(r))} checkpoints)" for r in a.runs)
        + (f"; ALERT: not evaluated: {', '.join(missing)}" if missing else ""))  # fmt: skip


if __name__ == "__main__":
    main()
