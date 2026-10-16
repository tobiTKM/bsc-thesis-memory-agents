"""Async OpenAI-SDK clients for vLLM's OpenAI-compatible chat and embedding
servers; async so many episodes can share one server concurrently.
"""

from openai import AsyncOpenAI


class LLMResponse:
    """One reply from the model, plus its token counts."""

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int,
                 finish_reason: str | None = None, reasoning_tokens: int = 0,
                 reasoning_text: str = ""):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # "length" with empty .text: the budget went into <think>, not a bad answer.
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens
        # The <think> block the reasoning parser stripped from .content.
        self.reasoning_text = reasoning_text
        # Tokens of replies discarded before this one (see BaselineAgent.act),
        # kept apart from this reply's own counts.
        self.retry_prompt_tokens = 0
        self.retry_completion_tokens = 0


class LLMClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int = 0, timeout: float = 600.0,
                 top_p: float = 1.0, top_k: int | None = None):
        # api_key is unused by vLLM but the SDK requires a non-empty value.
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model
        # Reasoning models need a large max_tokens and sampling (Qwen3: 0.6 / 0.95 /
        # top_k=20); temperature=0 in thinking mode causes repetition loops.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.top_p = top_p
        self.top_k = top_k  # not an OpenAI param; sent via extra_body

    async def chat(self, messages: list[dict]) -> LLMResponse:
        """Send the running message list, return the reply + token usage."""
        extra_body = {"top_k": self.top_k} if self.top_k is not None else None
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            extra_body=extra_body,
        )
        choice = completion.choices[0]
        usage = completion.usage
        # With a reasoning parser, vLLM reports the <think> block outside .content.
        details = getattr(usage, "completion_tokens_details", None)
        return LLMResponse(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            finish_reason=choice.finish_reason,
            reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
            reasoning_text=getattr(choice.message, "reasoning_content", None) or "",
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client (call once at the end of a run)."""
        await self.client.close()


class EmbeddingClient:
    """Async client for a separate vLLM server running an embedding model."""

    def __init__(self, base_url: str, model: str, timeout: float = 600.0):
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in input order (sorted by `index`); not normalized."""
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        items = sorted(resp.data, key=lambda d: d.index)
        return [item.embedding for item in items]

    async def aclose(self) -> None:
        await self.client.close()
