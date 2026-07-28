"""Shared episode types and JSON helpers used across all benchmarks."""

import numpy as np


class Transition:
    """One agent step within an episode."""

    def __init__(self, step: int, action: int, reward: float,
                 legal_actions: list, rule_status: dict,
                 agent_location=None,
                 prompt_tokens: int = 0, completion_tokens: int = 0,
                 retry_prompt_tokens: int = 0, retry_completion_tokens: int = 0):
        self.step = step
        self.action = action
        self.reward = reward
        self.legal_actions = legal_actions
        self.rule_status = rule_status
        # Where the path head sat *after* this move. The env's observation already
        # carries the full path each step (visited nodes, current head), so this is
        # not the only record of the path -- it just gives the Reflector a compact,
        # explicitly *ordered* move log the final grid alone doesn't spell out.
        self.agent_location = agent_location
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # Cost of the unparseable replies discarded before this move landed. Held
        # apart from the two fields above so "tokens per model reply" stays a
        # meaningful statistic while the waste is still counted somewhere.
        self.retry_prompt_tokens = retry_prompt_tokens
        self.retry_completion_tokens = retry_completion_tokens


class EpisodeResult:
    """Outcome of running a single puzzle, used by metrics and memory.write()."""

    def __init__(self, puzzle_id: str, success: bool, terminated: bool,
                 truncated: bool, difficulty: int,
                 transitions: list[Transition] | None = None,
                 first_obs: str = "", last_obs: str = "", terminal_reason: str = "",
                 aborted_prompt_tokens: int = 0, aborted_completion_tokens: int = 0,
                 error: str = ""):
        self.puzzle_id = puzzle_id
        self.success = success
        self.terminated = terminated
        self.truncated = truncated
        self.difficulty = difficulty
        self.transitions = transitions if transitions is not None else []
        # Reflection context. The env observation *is* the path drawn, so last_obs
        # (the final grid) is the richest single record of what the agent produced;
        # first_obs is the puzzle it started from and terminal_reason is *why* it
        # ended (see _terminal_reason() in the harness).
        self.first_obs = first_obs
        self.last_obs = last_obs
        self.terminal_reason = terminal_reason
        # An episode that dies on unparseable replies appends no Transition for its
        # final move, so the whole chain of calls that killed it had nowhere to be
        # recorded and showed up as free. It lands here instead.
        self.aborted_prompt_tokens = aborted_prompt_tokens
        self.aborted_completion_tokens = aborted_completion_tokens
        # Set only when the episode crashed (terminal_reason == "error"); see
        # errored_episode() in the harness.
        self.error = error

    def steps(self) -> int:
        return len(self.transitions)

    def prompt_tokens(self) -> int:
        """Every prompt token the episode paid for, retries and aborts included."""
        return (sum(t.prompt_tokens + t.retry_prompt_tokens for t in self.transitions)
                + self.aborted_prompt_tokens)

    def completion_tokens(self) -> int:
        """Every completion token the episode paid for, retries and aborts included."""
        return (sum(t.completion_tokens + t.retry_completion_tokens
                    for t in self.transitions)
                + self.aborted_completion_tokens)

    def retry_tokens(self) -> int:
        """Of the above, what was spent on replies that never yielded a move.

        Reported separately because it is pure waste: a method that provokes more
        format failures must not get to look cheap.
        """
        return (sum(t.retry_prompt_tokens + t.retry_completion_tokens
                    for t in self.transitions)
                + self.aborted_prompt_tokens + self.aborted_completion_tokens)


def make_json_safe(obj, seen=None):
    """Convert numpy/dataset objects into plain JSON-serialisable values.

    Ported from Spatial-Gym's llm_testing host so the agent sees the same payload
    the established baseline does.
    """
    if seen is None:
        seen = set()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    oid = id(obj)
    if oid in seen:
        return None
    seen.add(oid)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            if isinstance(k, np.generic):
                k = k.item()
            elif not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            safe[k] = make_json_safe(v, seen)
        return safe
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v, seen) for v in obj]
    return str(obj)
