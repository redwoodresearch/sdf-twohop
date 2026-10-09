"""Tinker backend: greedy chat completions and teacher-forced completion logprobs for a base model or
finetuned sampler weights, rendered with a tinker-cookbook renderer (non-thinking for all evals here).
"""

import asyncio

import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers.base import get_text_content, has_thinking

from sdf_twohop.env import load_env


class TinkerChat:
    def __init__(self, base_model: str, renderer: str, model_path: str | None = None, concurrency: int = 256):
        load_env()
        self.base_model, self.renderer_name, self.model_path = base_model, renderer, model_path
        self.tokenizer = tokenizer_utils.get_tokenizer(base_model)
        self.renderer = renderers.get_renderer(renderer, self.tokenizer)
        service = tinker.ServiceClient()
        self.client = (
            service.create_sampling_client(model_path=model_path)
            if model_path
            else service.create_sampling_client(base_model=base_model)
        )
        self.sem = asyncio.Semaphore(concurrency)

    def prompt_tokens(self, messages: list[dict]) -> list[int]:
        return self.renderer.build_generation_prompt(messages).to_ints()

    async def _retry(self, fn, tries: int = 6):
        for i in range(tries):
            try:
                async with self.sem:
                    return await fn()
            except Exception:
                if i == tries - 1:
                    raise
                await asyncio.sleep(2**i)

    async def complete(self, messages: list[dict], max_tokens: int, temperature: float = 0.0) -> dict:
        prompt = self.renderer.build_generation_prompt(messages)
        params = tinker.types.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, stop=self.renderer.get_stop_sequences()
        )
        res = await self._retry(lambda: self.client.sample_async(prompt=prompt, num_samples=1, sampling_params=params))
        seq = res.sequences[0]
        message, termination = self.renderer.parse_response(seq.tokens)
        raw = self.tokenizer.decode(seq.tokens, skip_special_tokens=False)
        content = message["content"]
        thinking = "".join(p["thinking"] for p in content if p["type"] == "thinking") if isinstance(content, list) else ""
        return {
            "text": get_text_content(message).strip(),
            "thinking": thinking.strip(),
            "leaked_reasoning": has_thinking(content),
            "finish_reason": str(termination),
            "raw": raw,
            "n_tokens": len(seq.tokens),
        }

    async def continue_text(self, text: str, max_tokens: int, temperature: float = 0.0) -> dict:
        """Continue raw text with no chat template (the form the finetuning documents were trained in)."""
        prompt = tinker.types.ModelInput.from_ints(tokens=self.tokenizer.encode(text, add_special_tokens=False))
        params = tinker.types.SamplingParams(max_tokens=max_tokens, temperature=temperature)
        res = await self._retry(lambda: self.client.sample_async(prompt=prompt, num_samples=1, sampling_params=params))
        seq = res.sequences[0]
        ended = bool(seq.tokens) and seq.tokens[-1] == self.tokenizer.eos_token_id
        return {
            "text": self.tokenizer.decode(seq.tokens, skip_special_tokens=True),
            "finish_reason": "stop" if ended else "length",
            "raw": self.tokenizer.decode(seq.tokens, skip_special_tokens=False),
            "n_tokens": len(seq.tokens),
        }

    async def completion_logprobs(self, messages: list[dict], completion: str) -> list[float]:
        """Per-token logprobs of `completion` as the assistant reply (completion tokens only, no end-of-turn)."""
        prompt = self.prompt_tokens(messages)
        comp = self.tokenizer.encode(completion, add_special_tokens=False)
        full = tinker.types.ModelInput.from_ints(tokens=prompt + comp)
        lps = await self._retry(lambda: self.client.compute_logprobs_async(full))
        return lps[len(prompt) :]
