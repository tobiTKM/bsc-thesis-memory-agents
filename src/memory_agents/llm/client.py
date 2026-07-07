"""LLM client abstraction. The serving backend is deliberately deferred:
agents depend only on the `LLMClient` protocol, so a real OpenAI-compatible
endpoint can be swapped in later without touching agent or harness code.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

Message = dict  # {"role": str, "content": str}


class LLMResponse:
    def __init__(self, text: str, prompt_tokens: int = 0,
                 completion_tokens: int = 0):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class LLMClient(Protocol):
    def chat(self, messages: Sequence[Message]) -> LLMResponse:
        ...


class MockClient:
    """Deterministic stand-in for a real model, for tests and dry runs.

    Pass `actions` to replay a fixed sequence of moves (e.g. a known solution
    path), or `reply_fn` for arbitrary message-driven behaviour. Each call
    reports a token count so the loop's accounting can be exercised offline.
    """

    def __init__(self, actions: Sequence[int] | None = None,
                 reply_fn: Callable[[Sequence[Message]], str] | None = None):
        if (actions is None) == (reply_fn is None):
            raise ValueError("provide exactly one of `actions` or `reply_fn`")
        self._actions = list(actions) if actions is not None else None
        self._reply_fn = reply_fn
        self._i = 0
        self.calls = 0

    def chat(self, messages: Sequence[Message]) -> LLMResponse:
        self.calls += 1
        if self._actions is not None:
            action = self._actions[self._i]
            self._i += 1
            text = f"Final: {action}"
        else:
            text = self._reply_fn(messages)
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        return LLMResponse(text=text, prompt_tokens=prompt_tokens,
                           completion_tokens=len(text.split()))
