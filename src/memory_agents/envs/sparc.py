"""Adapter around the upstream Spatial-Gym (SPaRC) environment.

The agent interacts one cardinal move per call (Discrete(4): 0=right, 1=up,
2=left, 3=down), matching Spatial-Gym's `llm_testing/llm_host.py`. This adapter
only handles the env wiring the agent loop needs: load a puzzle by id, render
the observation to prompt text, parse the model's `Final: <digit>` reply, and
report success. All reasoning lives in the agent.
"""
from __future__ import annotations

import json
import re

import gymnasium as gym
import Spatial_Gym  # noqa: F401  (import registers the "Spatial-Gym" env id)

from .base import make_json_safe

# action int -> (dx, dy) on the grid, copied from Spatial_Gym._action_to_direction
_ACTION_TO_DELTA = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}
_DELTA_TO_ACTION = {v: k for k, v in _ACTION_TO_DELTA.items()}

_FINAL_RE = re.compile(r"^(?:Final:\s*)?([0-3])$")


class SparcEnv:
    def __init__(self, split="test", config="all", df_name="lkaesberg/SPaRC",
                 max_steps=100, traceback=False):
        self.env = gym.make(
            "Spatial-Gym",
            df_name=df_name,
            df_split=config,
            df_set=split,
            render_mode=None,
            observation="SPaRC",
            traceback=traceback,
            max_steps=max_steps,
        )

    # ---- episode control -------------------------------------------------
    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    # ---- prompt / action plumbing ----------------------------------------
    def user_message(self, obs, info, reward) -> str:
        """The per-step payload the model sees (obs + info + reward), as JSON."""
        payload = {"obs": format_obs(obs), "info": info, "reward": reward}
        return json.dumps(make_json_safe(payload))


# ---- pure prompt / action plumbing (no env state needed) -----------------
def format_obs(obs) -> str:
    """Render the SPaRC JSON grid string as one text row per line."""
    if isinstance(obs, str):
        try:
            grid = json.loads(obs)
            if isinstance(grid, list):
                return "\n".join(str(row) for row in grid)
        except json.JSONDecodeError:
            pass
    return obs


def parse_action(reply: str):
    """Parse a model reply into an action int, or None on a format failure.

    A None here is a *format* failure (malformed output), not a task
    failure: the harness should retry the call, not count it as an attempt.
    """
    lines = reply.strip().splitlines()
    if not lines:
        return None
    m = _FINAL_RE.match(lines[-1].strip())
    return int(m.group(1)) if m else None


def is_success(info) -> bool:
    """Env success = exact match to an enumerated solution path."""
    return info.get("Rewards", {}).get("outcome_reward") == 1


def path_to_actions(path) -> list:
    """Convert a node path [[x,y],...] into the action ints connecting them."""
    actions = []
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        delta = (bx - ax, by - ay)
        if delta not in _DELTA_TO_ACTION:
            raise ValueError(f"non-adjacent path nodes: {(ax, ay)} -> {(bx, by)}")
        actions.append(_DELTA_TO_ACTION[delta])
    return actions
