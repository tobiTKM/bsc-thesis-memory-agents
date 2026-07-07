"""Memoryless baseline agent: in-context reasoning only, no memory across or
within episodes. Memory agents will subclass / wrap this by overriding how the
system prompt (the memory block) is built.

The agent is benchmark-agnostic: it is given a `parse_fn` that turns a model
reply into an action (or None on a format failure), so the same loop drives
SPaRC, StepGame, and LogicGame.
"""
from __future__ import annotations

from typing import Callable

from ..llm.client import LLMClient, LLMResponse


class BaselineAgent:
    def __init__(self, client: LLMClient, max_format_retries: int = 5):
        self.client = client
        self.parse_fn = parse_fn
        self.keep_turns = keep_turns
        self.max_format_retries = max_format_retries
        self.messages: list[dict] = []

    def reset(self, system_prompt: str) -> None:
        """Start a fresh episode with the given prompt."""
        self.messages = [{"role": "system", "content": system_prompt}]

    def act(self, user_message: str):
        self.messages.append({"role": "user", "content": user_message})

        action, response = None, None
        for _ in range(self.max_format_retries):
            response = self.client.chat(self.messages)
            # TODO
            action = response
            if action is not None:
                break

        self._trim()
        return action, response

    def _trim(self) -> None:
        """Keep the only the system prompt."""
        self.messages = self.messages[0]
