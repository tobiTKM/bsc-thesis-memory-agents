"""Shared episode types and JSON helpers used across all benchmarks."""

import numpy as np


class Transition:
    """One agent step within an episode."""

    def __init__(self, step: int, action: int, reward: float,
                 legal_actions: list, rule_status: dict,
                 prompt_tokens: int = 0, completion_tokens: int = 0):
        self.step = step
        self.action = action
        self.reward = reward
        self.legal_actions = legal_actions
        self.rule_status = rule_status
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class EpisodeResult:
    """Outcome of running a single puzzle, used by metrics and memory.write()."""

    def __init__(self, puzzle_id: str, success: bool, terminated: bool,
                 truncated: bool, difficulty: int,
                 transitions: list[Transition] | None = None):
        self.puzzle_id = puzzle_id
        self.success = success
        self.terminated = terminated
        self.truncated = truncated
        self.difficulty = difficulty
        self.transitions = transitions if transitions is not None else []

    def steps(self) -> int:
        return len(self.transitions)

    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.transitions)

    def completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.transitions)


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
