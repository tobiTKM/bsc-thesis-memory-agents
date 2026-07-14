"""Async client for the vLLM server.

We host the model with vLLM's OpenAI-compatible server (see
scripts/serve_vllm.sbatch) and talk to it with the OpenAI SDK. Two consequences:

* It is *async* (AsyncOpenAI). One episode is sequential -- each move needs the
  previous grid state -- so the throughput win comes from running many episodes
  at once against the same server. Async lets a coroutine `await` its network
  call and hand the event loop to another episode meanwhile.
* It is provider-agnostic. Point `base_url` at the vLLM server now, or at a real
  vendor API later, and nothing else in the harness changes.
"""

from openai import AsyncOpenAI


class LLMResponse:
    """One reply from the model, plus the token counts we log per step.

    Kept as a plain object (like Transition/EpisodeResult) so the harness can do
    `response.text` / `response.prompt_tokens` without touching the raw SDK type.
    """

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class LLMClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int = 0, timeout: float = 600.0):
        # api_key is unused by vLLM but the SDK requires a non-empty value.
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model
        # temperature=0 + a fixed seed => deterministic decoding, so a re-run of
        # the benchmark gives the same numbers. Bump max_tokens for reasoning
        # models later (they need thousands of tokens before the Final: line).
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    async def chat(self, messages: list[dict]) -> LLMResponse:
        """Send the running message list, return the reply + token usage."""
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client (call once at the end of a run)."""
        await self.client.close()
