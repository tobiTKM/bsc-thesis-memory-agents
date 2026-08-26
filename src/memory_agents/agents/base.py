import json
import os
import pathlib

from ..llm.client import LLMClient, LLMResponse

# Where to append one JSON record per unparseable reply. Read lazily (not at import)
# so a driver can set it after this module is already loaded.
FAILURE_LOG_ENV = "FORMAT_FAILURE_LOG"

# Module-level so a driver can record the value its results were produced under.
# summarize_results.py needs it to tell "retried but recovered" from "aborted the
# episode", and was previously guessing with a hardcoded default.
MAX_FORMAT_RETRIES = 5

_HEAD = 400   # chars of the reply kept for eyeballing
_TAIL = 200
_REASONING_TAIL = 300   # chars of the stripped <think> block, where the answer died


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


def _sparc_parse_action(reply):
    """The default reply parser, imported lazily.

    Lazily because `envs.sparc` pulls in gymnasium and Spatial_Gym, and a
    single-turn benchmark (StepGame, LogicGame) needs neither -- nor does the
    login node, where those are deliberately absent. Importing at module scope
    made every driver that touches an agent depend on the SPaRC env stack.
    """
    from ..envs.sparc import parse_action
    return parse_action(reply)


class BaselineAgent:
    """The memoryless actor.

    `parse_reply` maps a model reply to an action, or to None for a *format*
    failure (malformed output, which is retried rather than counted as an
    attempt). It is injected rather than imported so the same actor drives an
    interactive grid benchmark and a single-turn QA one: SPaRC returns a
    direction int, StepGame returns one of nine relation labels, and neither the
    retry loop nor the failure log cares which.
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

        A discarded reply still cost tokens, so the total is carried on the returned
        response as retry_prompt_tokens / retry_completion_tokens. Two cases:

        * action is not None -- `response` is the reply that produced the move, and
          retry_* covers only the replies thrown away before it.
        * action is None -- every reply was thrown away, so retry_* covers all of
          them *including* `response` itself. The caller must therefore charge the
          episode retry_* and NOT also response.prompt_tokens, or it double-counts.

        Without this the harness appends no Transition for an aborted move, so the
        whole chain of calls that killed the episode was recorded as costing zero.
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
                # "length" = hit max_tokens; with a reasoning parser on, that means
                # the whole budget went into <think> and .text came back empty.
                "finish_reason": response.finish_reason,
                "completion_tokens": response.completion_tokens,
                "reasoning_tokens": response.reasoning_tokens,
                "text_len": len(text),
                "text_head": text[:_HEAD],
                "text_tail": text[-_TAIL:] if len(text) > _HEAD else "",
                # When .text is empty this is the only evidence of what went wrong:
                # the end of the think block shows whether the model was mid-deduction
                # or stuck in a repetition loop when the budget ran out.
                "reasoning_len": len(reasoning),
                "reasoning_tail": reasoning[-_REASONING_TAIL:],
            })

        if response is not None:
            response.retry_prompt_tokens = retry_prompt
            response.retry_completion_tokens = retry_completion
        self._trim()
        return action, response

    def _trim(self) -> None:
        """Keep only the system prompt (memoryless: the env re-sends the full
        grid state every step, so the agent needs no history)."""
        self.messages = self.messages[:1]
