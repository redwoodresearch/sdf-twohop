import importlib.util
import json
from pathlib import Path

_path = Path(__file__).parents[1] / "scripts/chat_dashboard.py"
_spec = importlib.util.spec_from_file_location("chat_dashboard", _path)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def _ckpt(run, dose, step):
    return {"run": run, "tag": f"{run}-step{step}-dpf{dose}", "path": f"tinker://x/{run}", "docs_per_fact": dose,
            "step": step, "notes": "", "corpora": [], "lr": 1e-4, "schedule": "linear", "lora_rank": 64,
            "time": "2026-10-08T00:00:00+0000"}  # fmt: skip


def test_checkpoint_options_order_and_untrained_first(tmp_path):
    p = tmp_path / "checkpoints.jsonl"
    rows = [_ckpt("b", 750, 20), _ckpt("a", 1500, 40), _ckpt("a", 375, 10), _ckpt("b", 375, 10)]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    opts = D.checkpoint_options(p)
    assert list(opts) == [D.UNTRAINED, "b · 375 docs/fact · step 10", "b · 750 docs/fact · step 20",
                          "a · 375 docs/fact · step 10", "a · 1,500 docs/fact · step 40"]  # fmt: skip
    assert opts[D.UNTRAINED] is None and opts["a · 1,500 docs/fact · step 40"]["tag"] == "a-step40-dpf1500"
    assert list(D.checkpoint_options(tmp_path / "missing.jsonl")) == [D.UNTRAINED]


def test_renderer_for_modes():
    assert D.renderer_for("disable_thinking", "qwen3_5_disable_thinking") == "qwen3_5_disable_thinking"
    assert D.renderer_for("raw", "qwen3_5_disable_thinking") == "qwen3_5_disable_thinking"
    assert D.renderer_for("thinking", "qwen3_5_disable_thinking") == "qwen3_5"
