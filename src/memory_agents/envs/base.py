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
        # Path head after this move; gives the reflector an ordered move log.
        self.agent_location = agent_location
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # Cost of unparseable replies discarded before this move, kept apart from the above.
        self.retry_prompt_tokens = retry_prompt_tokens
        self.retry_completion_tokens = retry_completion_tokens


class EpisodeResult:
    """Outcome of running a single puzzle, used by metrics and memory.write()."""

    def __init__(self, puzzle_id: str, success: bool, terminated: bool,
                 truncated: bool, difficulty: int,
                 transitions: list[Transition] | None = None,
                 first_obs: str = "", last_obs: str = "", terminal_reason: str = "",
                 aborted_prompt_tokens: int = 0, aborted_completion_tokens: int = 0,
                 error: str = "", extra: dict | None = None):
        self.puzzle_id = puzzle_id
        self.success = success
        self.terminated = terminated
        self.truncated = truncated
        self.difficulty = difficulty
        self.transitions = transitions if transitions is not None else []
        # Reflection context: start grid, final grid (the drawn path), and why it ended.
        self.first_obs = first_obs
        self.last_obs = last_obs
        self.terminal_reason = terminal_reason
        # Tokens of the final failed calls of an episode aborted on unparseable replies.
        self.aborted_prompt_tokens = aborted_prompt_tokens
        self.aborted_completion_tokens = aborted_completion_tokens
        # Set only when terminal_reason == "error".
        self.error = error
        # Per-benchmark result fields (e.g. StepGame hops, gold label). Never read by
        # reflection_context(): no gold answer may reach a reflection prompt.
        self.extra = extra or {}

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
        """Tokens spent on replies that never yielded a move (a subset of the above)."""
        return (sum(t.retry_prompt_tokens + t.retry_completion_tokens
                    for t in self.transitions)
                + self.aborted_prompt_tokens + self.aborted_completion_tokens)

    # ---- reflection context (survives a job boundary) --------------------
    # Exactly the fields reflexion_reflection_prompt() reads, so a resumed process can reflect.

    def reflection_context(self) -> dict | None:
        """JSON-safe fields needed to reflect on this episode; None for crashed episodes."""
        if self.terminal_reason == "error":
            return None
        # Converted per value: make_json_safe's cycle guard returns None for an object
        # already seen, which would blank a location array the env reuses across steps.
        return {
            "first_obs": self.first_obs,
            "last_obs": self.last_obs,
            "terminal_reason": self.terminal_reason,
            # [step, action, agent_location] per move -- what format_trajectory renders.
            "moves": [[t.step, t.action, make_json_safe(t.agent_location)]
                      for t in self.transitions],
            # Only the final rule_status is prompted with.
            "rule_status": make_json_safe(
                self.transitions[-1].rule_status if self.transitions else {}),
        }

    @classmethod
    def from_reflection_context(cls, ctx: dict, puzzle_id: str = "") -> "EpisodeResult":
        """Rebuild a stand-in episode so the reflection prompt is identical after a resume."""
        moves = ctx.get("moves", [])
        rule_status = ctx.get("rule_status", {})
        transitions = [
            Transition(step=step, action=action, reward=0.0, legal_actions=[],
                       # only the last move's verdict was kept
                       rule_status=rule_status if i == len(moves) - 1 else {},
                       agent_location=location)
            for i, (step, action, location) in enumerate(moves)
        ]
        return cls(
            puzzle_id=puzzle_id, success=False, terminated=False, truncated=False,
            difficulty=-1, transitions=transitions,
            first_obs=ctx.get("first_obs", ""), last_obs=ctx.get("last_obs", ""),
            terminal_reason=ctx.get("terminal_reason", ""))


def make_json_safe(obj, seen=None):
    """Convert numpy/dataset objects into plain JSON values (ported from Spatial-Gym)."""
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
