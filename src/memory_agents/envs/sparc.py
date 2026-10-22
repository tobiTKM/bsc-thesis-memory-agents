"""SPaRC grid-puzzle env wrapper (Spatial_Gym), reply parsing and dataset-record helpers."""

import json
import re

import gymnasium as gym
import Spatial_Gym
import yaml

from .base import make_json_safe

# "Final: <0-3>" anywhere in the reply, tolerating markdown bold; \b rejects longer numbers.
_FINAL_RE = re.compile(r"Final:\s*\**\s*([0-3])\b")
# Fallback: the last line is just the digit.
_BARE_RE = re.compile(r"^\**\s*([0-3])\s*\**[.)]?$")

# Recorded in results: a truncation is read as a dead end, which holds while the cap never binds.
MAX_STEPS = 100


class SparcEnv:
    def __init__(self, split="test", config="all", df_name="lkaesberg/SPaRC",
                 max_steps=MAX_STEPS, traceback=False):
        """`split` is the dataset split (train/test) and maps to the env's `df_set`;
        `config` is the HF dataset config name ("all") and maps to the env's `df_split`.
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
        """Load a puzzle by id (safe under concurrency), or the env's next one if None."""
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
    """Return the action from the last `Final:` marker (or a bare-digit last line).

    None is a format failure, to be retried by the harness, not a task failure.
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


# ---- dataset-record plumbing (Memp build/embed; no env state needed) --------
# Inverse of the env's _action_to_direction; must match it. "up" is y-1 (y grows downward).
_DELTA_TO_ACTION = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}


def path_to_actions(path) -> list[int]:
    """Convert a gold `solutions[i]["path"]` (start node included) into action ints.

    Raises on any non-unit step rather than storing a wrong gold sequence.
    """
    actions = []
    for a, b in zip(path, path[1:]):
        delta = (b["x"] - a["x"], b["y"] - a["y"])
        if delta not in _DELTA_TO_ACTION:
            raise ValueError(f"non-unit step {a} -> {b} in gold path")
        actions.append(_DELTA_TO_ACTION[delta])
    return actions


def _parsed_cells(record) -> tuple[dict, list]:
    """(puzzle dict, cells sorted by position) from a record's text_visualization."""
    puzzle = yaml.safe_load(record["text_visualization"])["puzzle"]
    cells = sorted(puzzle.get("cells") or [],
                   key=lambda c: (c["position"]["y"], c["position"]["x"]))
    return puzzle, cells


def _cell_kind(props: dict) -> str:
    """One word for what a cell is: its type, or the flag (gap/dot) it carries."""
    if props.get("type"):
        return props["type"]
    for flag in ("gap", "dot"):   # untyped cells carry {'gap': 1} or {'dot': 1}
        if props.get(flag):
            return flag
    return "unknown"


def puzzle_key_text(record) -> str:
    """Canonical text a puzzle is embedded from (key-fn `puzzle-key-v1`), keys and queries alike.

    Frozen: any change invalidates stored vectors, so bump the version and re-embed.
    Difficulty is excluded so retrieval matches on content.
    """
    puzzle, cells = _parsed_cells(record)
    dims, start, end = puzzle["dimensions"], puzzle["start"], puzzle["end"]
    lines = [f"grid {dims['width']}x{dims['height']}; "
             f"start ({start['x']},{start['y']}); end ({end['x']},{end['y']})"]
    census: dict[str, int] = {}
    for cell in cells:
        pos, props = cell["position"], cell.get("properties") or {}
        kind = _cell_kind(props)
        census[kind] = census.get(kind, 0) + 1
        parts = [kind]
        if props.get("count") is not None:      # triangle: edges to touch
            parts.append(str(props["count"]))
        if props.get("polyshape") is not None:  # poly/ylop: shape id
            parts.append(f"shape {props['polyshape']}")
        if props.get("color"):
            parts.append(props["color"])
        lines.append(f"({pos['x']},{pos['y']}) " + " ".join(parts))
    lines.append("rules: " + (", ".join(f"{k} x{n}" for k, n in sorted(census.items()))
                              or "none"))
    return "\n".join(lines)


def replay_path(env, puzzle_id, actions) -> dict:
    """Replay actions in the env (no model call); return them in `reflection_context` shape."""
    obs, info = env.reset(puzzle_id=puzzle_id)
    first_obs = format_obs(obs)
    moves, last_obs = [], first_obs
    for step, action in enumerate(actions):
        obs, _reward, terminated, truncated, info = env.step(action)
        last_obs = format_obs(obs)
        moves.append([step, int(action), info.get("agent_location")])
        if terminated or truncated:
            break
    return {
        "first_obs": first_obs,
        "last_obs": last_obs,
        "terminal_reason": "solved" if is_success(info) else "replayed",
        "moves": make_json_safe(moves),
        "rule_status": make_json_safe(info.get("rule_status", {})),
    }


def grid_summary(record) -> str:
    """One-line census of a puzzle's rule cells (colors, no positions) for a memory exemplar."""
    _, cells = _parsed_cells(record)
    counts: dict[str, int] = {}
    for cell in cells:
        props = cell.get("properties") or {}
        kind = _cell_kind(props)
        label = f"{kind}({props['color']})" if props.get("color") else kind
        counts[label] = counts.get(label, 0) + 1
    return (", ".join(f"{label} x{n}" if n > 1 else label
                      for label, n in sorted(counts.items()))
            or "no rule cells")
