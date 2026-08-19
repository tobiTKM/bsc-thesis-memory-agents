import json
import re

import gymnasium as gym
import Spatial_Gym
import yaml

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


# ---- dataset-record plumbing (Memp build/embed; no env state needed) --------
# The dataset's gold `solutions[i]["path"]` is a list of {x,y} nodes in the same
# doubled node grid the env plays on, one unit step apart, starting at the start
# node. This table is the inverse of the env's _action_to_direction, and the one
# place the two conventions meet -- note "up" is y-1 (grid y grows downward).
_DELTA_TO_ACTION = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}


def path_to_actions(path) -> list[int]:
    """Convert a gold solution path into the env's action ints.

    Raises on any non-unit delta instead of guessing: a wrong action sequence
    stored as a "known-correct example" would poison every memory built from it.
    A path of n nodes yields n-1 moves (the list includes the start node).
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
    """The canonical text a puzzle is embedded from -- key-fn `puzzle-key-v1`.

    Used verbatim for BOTH the train keys and the test queries; that symmetry
    (same renderer on both sides), not the exact format, is what makes cosine
    similarity meaningful. Changing anything here invalidates every stored
    vector, so treat the format as frozen once a bank has shipped -- bump the
    version name and re-embed instead.

    Deterministic on purpose: cells sorted by (y, x), properties rendered in a
    fixed order, census sorted alphabetically. Difficulty is deliberately NOT in
    the key: retrieval should match on puzzle *content* (rules present, layout
    scale), and difficulty is metadata the bank entry carries separately.
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
    """Replay an action sequence and capture it in `reflection_context` shape.

    Lets a gold-path entry carry the same round-by-round trace an agent-solved
    entry gets from its own episode, so the Trajectory / Proceduralization arms
    can run over the whole bank instead of only the handful of puzzles the model
    happened to solve. Without this, granularity would be confounded with bank
    size (500 entries vs 71). Pure env replay -- no model call.
    """
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
    """A one-line census of a puzzle's rule cells, for the exemplar in a memory.

    Coarser than puzzle_key_text on purpose: the exemplar tells the actor what
    kind of grid the example move sequence solved, and colors matter for that
    (squares' whole rule is color separation) but positions do not -- the
    example is to be adapted, not replayed.
    """
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
