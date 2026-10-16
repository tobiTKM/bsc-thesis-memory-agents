"""The memoryless actor: one chat call per move, with format-failure retries and logging."""

import json
import os
import pathlib

from ..llm.client import LLMClient, LLMResponse

# Env var naming a JSONL log of unparseable replies; read at call time, not at import.
FAILURE_LOG_ENV = "FORMAT_FAILURE_LOG"

# Module-level so a driver can record the value its results were produced under.
MAX_FORMAT_RETRIES = 5

_HEAD = 400   # chars of the reply kept for eyeballing
_TAIL = 200
_REASONING_TAIL = 300   # chars of the stripped <think> block, where the answer died


def _log_format_failure(record: dict) -> None:
    """Append one record to $FORMAT_FAILURE_LOG, if set."""
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


def _sparc_parse_action(reply):
    """Default reply parser, imported lazily so other benchmarks skip the SPaRC env stack."""
    from ..envs.sparc import parse_action
    return parse_action(reply)


class BaselineAgent:
    """The memoryless actor.

    `parse_reply` maps a reply to an action, or to None for a format failure, which is
    retried rather than counted as an attempt.
    """

    def __init__(self, client, max_format_retries: int = MAX_FORMAT_RETRIES,
                 parse_reply=None):
        self.client = client
        self.max_format_retries = max_format_retries
        self.parse_reply = parse_reply or _sparc_parse_action
        self.messages: list[dict] = []
        self.label = None          # puzzle id, carried into the failure log

    def reset(self, system_prompt: str, label=None) -> None:
        """Start a fresh episode with the given prompt."""
        self.messages = [{"role": "system", "content": system_prompt}]
        self.label = label

    async def act(self, user_message: str):
        """Return (action, response) for one move, retrying unparseable replies.

        Discarded replies' tokens are set on response.retry_prompt_tokens/retry_completion_tokens.
        If action is None, retry_* already includes `response`; do not also charge its own tokens.
        """
        self.messages.append({"role": "user", "content": user_message})

        action, response = None, None
        retry_prompt = retry_completion = 0
        for attempt in range(self.max_format_retries):
            response = await self.client.chat(self.messages)
            action = self.parse_reply(response.text)
            if action is not None:
                break
            retry_prompt += response.prompt_tokens
            retry_completion += response.completion_tokens
            text = response.text or ""
            reasoning = response.reasoning_text or ""
            _log_format_failure({
                "puzzle_id": str(self.label),
                "attempt": attempt,
                # "length" with empty .text means the whole budget went into <think>.
                "finish_reason": response.finish_reason,
                "completion_tokens": response.completion_tokens,
                "reasoning_tokens": response.reasoning_tokens,
                "text_len": len(text),
                "text_head": text[:_HEAD],
                "text_tail": text[-_TAIL:] if len(text) > _HEAD else "",
                # The only evidence when .text is empty: how the think block ended.
                "reasoning_len": len(reasoning),
                "reasoning_tail": reasoning[-_REASONING_TAIL:],
            })

        if response is not None:
            response.retry_prompt_tokens = retry_prompt
            response.retry_completion_tokens = retry_completion
        self._trim()
        return action, response

    def _trim(self) -> None:
        """Keep only the system prompt; the env re-sends the full state every step."""
        self.messages = self.messages[:1]
