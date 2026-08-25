"""Async client for the vLLM server.

We host the model with vLLM's OpenAI-compatible server (see
scripts/serve_and_run.sbatch) and talk to it with the OpenAI SDK. Two consequences:

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

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int,
                 finish_reason: str | None = None, reasoning_tokens: int = 0,
                 reasoning_text: str = ""):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # "length" means the generation hit max_tokens. With a reasoning parser on,
        # that usually leaves .text empty (the whole budget went into <think>), which
        # is indistinguishable from a malformed answer unless we keep this.
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens
        # The <think> block the reasoning parser stripped out of .content. Keeping it
        # is what makes the format-failure log worth reading: an episode that dies on
        # empty replies otherwise logs an empty head and tail, which says nothing that
        # text_len=0 did not already say. The tail of this shows what the model was
        # actually doing when it ran out of budget.
        self.reasoning_text = reasoning_text
        # Tokens spent on replies that were discarded before this one (see
        # BaselineAgent.act). Zero on the happy path. Kept apart from
        # prompt_tokens/completion_tokens so "what this reply cost" and "what getting
        # a usable move cost" stay separable in the results.
        self.retry_prompt_tokens = 0
        self.retry_completion_tokens = 0


class LLMClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int = 0, timeout: float = 600.0,
                 top_p: float = 1.0, top_k: int | None = None):
        # api_key is unused by vLLM but the SDK requires a non-empty value.
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model
        # temperature=0 + a fixed seed => deterministic decoding, so a re-run of
        # the benchmark gives the same numbers. Reasoning models (Qwen3, QwQ, R1)
        # need a much bigger max_tokens (thousands, before the Final: line) AND
        # sampling, not greedy: Qwen3 warns temperature=0 in thinking mode causes
        # repetition loops -- pass temperature=0.6, top_p=0.95, top_k=20 there.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.top_p = top_p
        self.top_k = top_k  # not an OpenAI param; sent via extra_body (vLLM reads it)

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
        # vLLM reports the stripped <think> block separately when a reasoning parser
        # is enabled; it is not in .content, so count it here to see where the token
        # budget actually went, and keep the text itself for the failure log.
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
    """Async client for a vLLM server running an embedding model.

    A separate class rather than a method on LLMClient because the two are never
    served by the same process: vLLM runs one model per server, so embeddings come
    from their own (small, 1-GPU) job while chat comes from the 32B job. Keeping
    the classes apart means neither run has to construct a client for a server
    that does not exist.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 600.0):
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving input order.

        The OpenAI embeddings API returns one item per input with an `index`
        field; sort on it rather than trusting response order, since order is a
        convention the spec does not actually guarantee. Callers batch (the
        driver uses ~64) and L2-normalize; raw vectors are returned here.
        """
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        items = sorted(resp.data, key=lambda d: d.index)
        return [item.embedding for item in items]

    async def aclose(self) -> None:
        await self.client.close()
