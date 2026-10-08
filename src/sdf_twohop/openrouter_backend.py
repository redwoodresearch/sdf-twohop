"""OpenRouter backend for quick, cheap checks: greedy chat completions with thinking disabled both via the
API flag and the `/no_think` soft switch (some providers ignore the flag); any returned reasoning is flagged.
"""

import asyncio
import os

from openai import AsyncOpenAI

from sdf_twohop.env import load_env


def _strip_think(s: str | None) -> str:
    return (s or "").replace("<think>", "").replace("</think>", "").strip()


class OpenRouterChat:
    def __init__(
        self,
        model: str,
        no_think: bool = True,
        soft_switch: bool = True,
        quantizations: list[str] | None = None,
        concurrency: int = 128,
    ):
        """`quantizations`: restrict routing to providers serving these precisions (e.g. ["fp8"]); providers
        differ in quantization and greedy outputs differ with it."""
        load_env()
        self.model, self.no_think, self.soft_switch = model, no_think, no_think and soft_switch
        self.provider = {"quantizations": quantizations} if quantizations else None
        self.client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
        self.sem = asyncio.Semaphore(concurrency)

    async def complete(self, messages: list[dict], max_tokens: int, tries: int = 8) -> dict:
        if self.soft_switch:
            messages = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + " /no_think"}]
        kwargs = dict(model=self.model, max_tokens=max_tokens, temperature=0.0, messages=messages)
        extra = {}
        if self.no_think:
            extra["reasoning"] = {"enabled": False}
        if self.provider:
            extra["provider"] = self.provider
        if extra:
            kwargs["extra_body"] = extra
        for i in range(tries):
            try:
                async with self.sem:
                    r = await asyncio.wait_for(self.client.chat.completions.create(**kwargs), timeout=180)
                msg = r.choices[0].message
                reasoning = {k: getattr(msg, k, None) for k in ("reasoning", "reasoning_content", "reasoning_details")}
                text = msg.content or ""
                leaked = bool(_strip_think(reasoning["reasoning"]) or _strip_think(reasoning["reasoning_content"]))
                leaked |= bool(reasoning["reasoning_details"]) or "<think>" in text
                return {
                    "text": text.strip(),
                    "leaked_reasoning": leaked,
                    "finish_reason": r.choices[0].finish_reason,
                    "reasoning": {k: v for k, v in reasoning.items() if v},
                    "provider": getattr(r, "provider", None),
                    "served_model": r.model,
                    "sent_messages": messages,
                }
            except Exception as e:
                err = e
                await asyncio.sleep(min(60, 2**i))
        return {"text": None, "leaked_reasoning": False, "finish_reason": "error", "error": repr(err)}
