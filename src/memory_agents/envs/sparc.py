import json
import re

import gymnasium as gym
import Spatial_Gym

from .base import make_json_safe

# A "Final:" line anywhere in the reply, tolerating markdown bold and trailing
# punctuation. \b keeps it from matching the first digit of a longer number.
_FINAL_RE = re.compile(r"Final:\s*\**\s*([0-3])\b")
# Fallback: the whole last line is just the digit (what the strict parser accepted).
_BARE_RE = re.compile(r"^\**\s*([0-3])\s*\**[.)]?$")

# Env step cap. Recorded in the results so summarize_results' reading of a
# truncation -- a dead end, not a step-limit cutoff -- stays checkable: episodes
# run ~15 steps against this, so the cap never fires.
MAX_STEPS = 100


class SparcEnv:
    def __init__(self, split="test", config="all", df_name="lkaesberg/SPaRC",
                 max_steps=MAX_STEPS, traceback=False):
        """Three different senses of "split" collide in this call, so, explicitly:

        `split` is the dataset split (train/test) and maps to the env's `df_set`;
        `config` is the HuggingFace dataset *config* name ("all") and maps to the
        env's `df_split`. The names are the env's, not ours.
        """
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

    Anchoring strictly to the last line (the previous behaviour) rejected replies
    that were perfectly readable -- `Final: 0.`, `**Final: 0**`, a `Final: 0` with
    one more sentence after it -- and every rejection costs a whole retry. So scan
    the reply for `Final:` markers and take the LAST one, which is the model's
    conclusion rather than a digit it floated while thinking.

    NOTE: this accepts strictly more replies than the strict parser did, so
    accuracy is not comparable across this change.
    """
    if not reply:
        return None
    matches = _FINAL_RE.findall(reply)
    if matches:
        return int(matches[-1])
    lines = reply.strip().splitlines()
    if not lines:
        return None
    m = _BARE_RE.match(lines[-1].strip())
    return int(m.group(1)) if m else None


def is_success(info) -> bool:
    """Env success = exact match to an enumerated solution path."""
    return info.get("Rewards", {}).get("outcome_reward") == 1
