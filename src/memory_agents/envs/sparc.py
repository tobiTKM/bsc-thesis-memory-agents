import json
import re

import gymnasium as gym
import Spatial_Gym

from .base import make_json_safe

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
    def reset(self, puzzle_id=None):
        """Load a specific puzzle by id, or the next one if id is None.

        Selecting by id is what makes concurrency safe: each episode targets its
        own puzzle instead of relying on the env's internal sequential counter,
        which several concurrent episodes would race on.
        """
        options = {"puzzle_id": puzzle_id} if puzzle_id is not None else None
        return self.env.reset(options=options)

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
