import json
import os
import pathlib

from ..llm.client import LLMClient, LLMResponse
from ..envs.sparc import parse_action

# Where to append one JSON record per unparseable reply. Read lazily (not at import)
# so a driver can set it after this module is already loaded.
FAILURE_LOG_ENV = "FORMAT_FAILURE_LOG"

_HEAD = 400   # chars of the reply kept for eyeballing
_TAIL = 200


def _log_format_failure(record: dict) -> None:
    """Append one record to $FORMAT_FAILURE_LOG, if set.

    The 2026-07-26 baseline lost 66% of its episodes to unparseable replies with no
    trace of what the model actually said, which made the cause unknowable from the
    results alone. One line per failure is cheap next to an LLM call.
    """
    path = os.environ.get(FAILURE_LOG_ENV)
    if not path:
        return
    try:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # diagnostics must never take the run down


class BaselineAgent:
    def __init__(self, client, max_format_retries: int = 5):
        self.client = client
        self.max_format_retries = max_format_retries
        self.messages: list[dict] = []
        self.label = None          # puzzle id, carried into the failure log

    def reset(self, system_prompt: str, label=None) -> None:
        """Start a fresh episode with the given prompt."""
        self.messages = [{"role": "system", "content": system_prompt}]
        self.label = label

    async def act(self, user_message: str):
        self.messages.append({"role": "user", "content": user_message})

        action, response = None, None
        for attempt in range(self.max_format_retries):
            response = await self.client.chat(self.messages)
            action = parse_action(response.text)
            if action is not None:
                break
            text = response.text or ""
            _log_format_failure({
                "puzzle_id": str(self.label),
                "attempt": attempt,
                # "length" = hit max_tokens; with a reasoning parser on, that means
                # the whole budget went into <think> and .text came back empty.
                "finish_reason": response.finish_reason,
                "completion_tokens": response.completion_tokens,
                "reasoning_tokens": response.reasoning_tokens,
                "text_len": len(text),
                "text_head": text[:_HEAD],
                "text_tail": text[-_TAIL:] if len(text) > _HEAD else "",
            })

        self._trim()
        return action, response

    def _trim(self) -> None:
        """Keep only the system prompt (memoryless: the env re-sends the full
        grid state every step, so the agent needs no history)."""
        self.messages = self.messages[:1]
