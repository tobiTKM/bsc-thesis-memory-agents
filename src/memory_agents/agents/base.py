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
    def __init__(self, client: LLMClient, parse_fn: Callable[[str], object],
                 keep_turns: int = 4, max_format_retries: int = 5):
        self.client = client
        self.parse_fn = parse_fn
        self.keep_turns = keep_turns
        self.max_format_retries = max_format_retries
        self.messages: list[dict] = []

    def reset(self, system_prompt: str) -> None:
        """Start a fresh episode with the given (possibly memory-augmented) prompt."""
        self.messages = [{"role": "system", "content": system_prompt}]

    def act(self, user_message: str):
        """One decision: append the observation, query the model (retrying only
        on malformed output), and return (action, response). action is None if
        the model never produced a parseable reply.
        """
        self.messages.append({"role": "user", "content": user_message})

        action, response = None, None
        for _ in range(self.max_format_retries):
            response = self.client.chat(self.messages)
            action = self.parse_fn(response.text)
            if action is not None:
                break

        self.messages.append({"role": "assistant", "content": f"Final: {action}"})
        self._trim()
        return action, response

    def _trim(self) -> None:
        """Keep the system message plus the last `keep_turns` user/assistant pairs."""
        system = self.messages[0]
        tail = self.messages[-(self.keep_turns * 2):]
        self.messages = [system] + tail
