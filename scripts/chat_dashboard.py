"""Chat with the untrained model or any checkpoint in checkpoints.jsonl, served from Tinker, in the browser.
    scripts/chat_dashboard.sh                        # private HTTPS URL on the Redwood tailnet (see README)
    uv run streamlit run scripts/chat_dashboard.py   # loopback only, on the devbox itself
Modes: chat without thinking (the renderer the evaluations use), chat with native thinking, or raw text
continuation (no chat template; optionally prefixed with the training-document marker). Every exchange is
appended to results/chat_logs/<date>.jsonl with the checkpoint, renderer, sampling parameters, messages
sent, and the raw completion.
"""

import asyncio
import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from sdf_twohop.io import read_jsonl
from sdf_twohop.paths import CHECKPOINTS, RESULTS, ROOT
from sdf_twohop.sft import DOCTAG
from sdf_twohop.tinker_backend import TinkerChat

CONFIG = ROOT / "configs" / "qwen3.6-35b-a3b.json"
LOG_DIR = RESULTS / "chat_logs"
MODES = {
    "Chat, no thinking (as in the evaluations)": "disable_thinking",
    "Chat, native thinking": "thinking",
    "Raw text continuation": "raw",
}
UNTRAINED = "Untrained model"


def checkpoint_options(path: Path = CHECKPOINTS) -> dict[str, dict | None]:
    """Dropdown label -> checkpoint record (None for the untrained model), runs in file order, doses ascending."""
    rows = read_jsonl(path) if path.exists() else []
    order: dict[str, int] = {}
    for r in rows:
        order.setdefault(r["run"], len(order))
    rows.sort(key=lambda r: (order[r["run"]], r["docs_per_fact"], r["step"]))
    opts: dict[str, dict | None] = {UNTRAINED: None}
    for r in rows:
        opts[f"{r['run']} · {r['docs_per_fact']:,} docs/fact · step {r['step']}"] = r
    return opts


def renderer_for(mode: str, eval_renderer: str) -> str:
    """The evaluations' renderer is the *_disable_thinking variant; native thinking drops that suffix."""
    return eval_renderer.removesuffix("_disable_thinking") if mode == "thinking" else eval_renderer


@st.cache_resource
def event_loop() -> asyncio.AbstractEventLoop:
    """One long-lived loop for the process, so Tinker's async clients outlive a single Streamlit rerun."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def run(coro):
    return asyncio.run_coroutine_threadsafe(coro, event_loop()).result()


@st.cache_resource(show_spinner="Connecting to the model on Tinker…")
def get_chat(base_model: str, renderer: str, model_path: str | None) -> TinkerChat:
    return TinkerChat(base_model, renderer, model_path=model_path, concurrency=4)


def log_exchange(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / f"chat_{datetime.now(UTC):%Y-%m-%d}.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def show_assistant(msg: dict) -> None:
    if msg.get("thinking"):
        with st.expander("Thinking", expanded=False):
            st.markdown(msg["thinking"])
    st.markdown(msg["content"] or "*(empty reply)*")
    st.caption(f"{msg['label']} · {msg['mode']} · {msg['n_tokens']} tokens · {msg['finish_reason']} · "
               f"{msg['secs']:.1f} s")  # fmt: skip


def main() -> None:
    st.set_page_config(page_title="Checkpoint chat", page_icon="💬", layout="wide")
    cfg = json.loads(CONFIG.read_text())
    opts = checkpoint_options()
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", uuid.uuid4().hex)

    with st.sidebar:
        st.title("Checkpoint chat")
        label = st.selectbox("Checkpoint", list(opts), help="Entries of checkpoints.jsonl, served from Tinker")
        ckpt = opts[label]
        if ckpt:
            st.caption(ckpt["notes"])
            st.caption(f"corpora: {', '.join(ckpt['corpora'])} · LR {ckpt['lr']:.2e} · {ckpt['schedule']} · "
                       f"rank {ckpt['lora_rank']} · {ckpt['time'][:16]}")  # fmt: skip
        else:
            st.caption(f"{cfg['model']} with no finetuning")
        mode = MODES[st.radio("Mode", list(MODES))]
        temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.05, help="0 = greedy, as in the evaluations")
        max_tokens = st.number_input("Max new tokens", 16, 8192, 2048 if mode == "thinking" else 512, 16)
        system = ""
        doctag = False
        if mode == "raw":
            doctag = st.checkbox(f"Prefix with {DOCTAG}", True, help="How every finetuning document began")
            st.caption("Each message is continued on its own; earlier turns are not included.")
        else:
            system = st.text_area("System prompt (optional)", "", height=80)
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = uuid.uuid4().hex
            st.rerun()
        st.caption(f"Transcripts: {LOG_DIR.relative_to(ROOT)}/")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            show_assistant(m) if m["role"] == "assistant" else st.markdown(m["content"])

    text = st.chat_input("Continue this text…" if mode == "raw" else "Message the model…")
    if not text:
        return
    with st.chat_message("user"):
        st.markdown(text)
    st.session_state.messages.append({"role": "user", "content": text})

    renderer = renderer_for(mode, cfg["renderer"])
    chat = get_chat(cfg["model"], renderer, ckpt["path"] if ckpt else None)
    history = [{"role": "system", "content": system}] if system else []
    history += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    prefix = (DOCTAG if doctag else "") + text
    t0 = time.time()
    with st.chat_message("assistant"):
        try:
            with st.spinner("Sampling…"):
                if mode == "raw":
                    out = run(chat.continue_text(prefix, int(max_tokens), temperature))
                else:
                    out = run(chat.complete(history, int(max_tokens), temperature))
        except Exception as e:  # noqa: BLE001 - surface Tinker/renderer errors in the page instead of a traceback
            st.session_state.messages.pop()
            st.error(f"Sampling failed: {type(e).__name__}: {e}")
            return
        secs = time.time() - t0
        # finetuned checkpoints often end a turn with the end-of-text token the documents were trained with instead of
        # the chat end-of-turn token; the renderer then reports "malformed" and leaves the token in the text. The
        # display strips it and names the reason; the transcript log keeps the raw output.
        eos = chat.tokenizer.eos_token
        text = out["text"].replace(eos, "").strip()
        reason = "ended with end-of-text" if out["raw"].rstrip().endswith(eos) else out["finish_reason"]
        msg = {"role": "assistant", "content": text, "thinking": out.get("thinking", ""), "label": label, "mode": mode,
               "n_tokens": out["n_tokens"], "finish_reason": reason, "secs": secs}  # fmt: skip
        show_assistant(msg)
    st.session_state.messages.append(msg)
    log_exchange({
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "session_id": st.session_state.session_id,
        "checkpoint": label, "tag": ckpt["tag"] if ckpt else None, "path": ckpt["path"] if ckpt else None,
        "base_model": cfg["model"], "renderer": None if mode == "raw" else renderer, "mode": mode,
        "temperature": temperature, "max_tokens": int(max_tokens),
        "input": prefix if mode == "raw" else history,
        "output": out, "secs": round(secs, 2),
    })  # fmt: skip


if __name__ == "__main__":
    main()
