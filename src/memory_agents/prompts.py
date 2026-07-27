# Directions match the env's action ints (0=right, 1=up, 2=left, 3=down).
_DIRECTIONS = {0: "right", 1: "up", 2: "left", 3: "down"}

# Two few-shot self-reflections, in the style Reflexion uses to anchor the format:
# "what I did wrong -> the concise, high-level plan for next time" (a strategy, not
# a bug report). Adapted to SPaRC; kept short and forward-looking like the paper's.
_REFLECTION_EXAMPLES = """- I rushed the path toward the end node and enclosed a blue and a green square in the same region, which breaks the square rule. Next time I should decide up front which region each color belongs to and route the path so differently colored squares end up separated before I head for the end.
- I filled a column too early and painted the path into a corner around move 12 with no legal move left. I should keep more exits open and avoid committing to a wall until I have checked that every remaining dot can still be reached."""


def format_trajectory(episode, max_steps=None) -> str:
    """Render a failed episode as a move log for the reflection model.

    One line per move: the direction taken and where the path head ended up. The
    final grid (last_obs) already shows the drawn path; this adds the one thing the
    grid does not spell out -- the *order* moves were made, and where a dead end was
    hit. Reflexion reflects over the *whole* trajectory, so we show every move by
    default. `max_steps` is only a guard for envs with a much larger step cap;
    SPaRC's 100 short lines cost ~1.5k tokens, far under the context budget.
    """
    steps = episode.transitions
    lines = []
    if max_steps is not None and len(steps) > max_steps:
        lines.append(f"...(first {len(steps) - max_steps} moves omitted)...")
        steps = steps[-max_steps:]
    for t in steps:
        direction = _DIRECTIONS.get(t.action, str(t.action))
        loc = t.agent_location if t.agent_location is not None else "?"
        lines.append(f"  move {t.step}: {direction} -> path head at {loc}")
    return "\n".join(lines) if lines else "  (no valid move was ever made)"


def reflexion_reflection_prompt(episode) -> str:
    """Reflexion's self-reflection step, following the paper's own prompt shape.

    Reflexion (Shinn et al., 2023) prompts an "advanced reasoning agent that can
    improve based on self-reflection" with the full failed trajectory and asks it
    to *diagnose a reason for failure and devise a new, concise, high-level plan*
    -- in a few complete sentences, anchored by few-shot example reflections. The
    plan (not a bug report) is what the actor reads before its next attempt at the
    SAME puzzle. See memory/reflexion.py for how it is stored and injected.
    """
    reasons = {
        "wrong-solution": "you submitted a path, but it broke one or more rules",
        "dead-end": "you painted the path into a corner with no legal move left",
        "format-failure": "you never produced a parseable 'Final: <digit>' move",
    }
    why = reasons.get(episode.terminal_reason, "you did not solve the puzzle")
    last_rules = episode.transitions[-1].rule_status if episode.transitions else {}
    return f"""
    You are an advanced reasoning agent that can improve based on self-reflection.
    You were given a path-finding puzzle to solve and, in a previous trial, you were
    unsuccessful: {why}. In a few sentences, diagnose a possible reason for the
    failure and devise a new, concise, high-level plan that aims to mitigate the
    same failure on your next attempt at THIS puzzle. Use complete sentences. Do not
    restate the puzzle rules in general and do not output a move.

    Here are two examples of good self-reflections:
{_REFLECTION_EXAMPLES}

    Previous trial:
    Starting grid:
    {episode.first_obs}

    Moves you made, in order (0=right, 1=up, 2=left, 3=down):
{format_trajectory(episode)}

    Final grid, showing the full path you drew (V=visited, L=your last position):
    {episode.last_obs}

    Rule status at the end (per-rule pass/fail): {last_rules}

    Reflection:"""


def format_reflections(reflections) -> str:
    """The block injected into the actor's system prompt on a Reflexion retry."""
    body = "\n".join(f"- {r}" for r in reflections)
    return (
        "\n\nReflections from your previous attempt(s) at THIS puzzle. Use them to "
        "avoid repeating the same mistakes:\n" + body
    )


def sparc_system_prompt(polyshapes) -> str:
    return f"""
    You are an autonomous agent controlling a path-finding puzzle solver.
    Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    Core Concepts & Grid Basics:
    Grid Dimensions: You can find the puzzle grid size in the info
    Path: The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    Revisiting: You can not traceback your path. you can not visit a cell twice.
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both x and y are odd.
    The path goes around these rule cells, never on them. They are also marked as gaps.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.
    Valid Path Cells: The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.

    Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `V`: **Visited Node** (Path has passed through this cell)
    *   `L`: **Current Node** (Path is currently on this cell)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y

    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black

    Detailed Solving Rules:
    The drawn path must satisfy ALL applicable constraints:

    1.  Path Constraints:
        Path connects adjacent nodes (horizontal/vertical moves only).
        Nodes CAN NOT be revisited. You cannot visit a cell twice.
        Path MUST pass through all Dot cells.
        Path CANNOT pass through any Gap cells.

    2.  Region-Based Rules (Apply to areas enclosed by the path):
        Squares: All squares within a single region MUST be the same color. Squares of different colors MUST be separated into different regions by the path.
        Stars: Within a single region, each star symbol MUST be paired with exactly ONE other element of the same color. Other colors within the region are irrelevant to this specific star's rule.

        Polyshapes(poly): The region containing this symbol MUST be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries.
        If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is generally allowed unless context implies otherwise.

        Negative Polyshapes(ylop): These subtract shape requirements, typically within the same region as corresponding positive polyshapes.
        A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region.
        If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one.

    3.  Path-Based Rules (Edge Touching):
        Triangles: The path MUST touch a specific number of edges of the cell containing the triangle symbol.
            (1): Path touches EXACTLY 1 edge of the triangle's cell.
            (2): Path touches EXACTLY 2 edges of the triangle's cell.
            (3): Path touches EXACTLY 3 edges of the triangle's cell.
            (4): Path touches EXACTLY 4 edges (fully surrounds) the triangle's cell.

    Polyshape Definitions: Shapes are defined by 2D arrays where 1 indicates an occupied cell and 0 indicates an empty cell. {polyshapes}

    At each turn you'll receive current Information as JSON.
    Observation: The current state of the grid, including the path and any rule cells.
    Info: Additional information including solution_count, difficulty, grid_x_size, grid_y_size,
    legal_actions (list of legal action ints 0=right,1=up,2=left,3=down), current_step, agent_location,
    Rewards, and rule_status (per-rule pass/fail).

    You MAY think step-by-step, but you MUST end with:
    Final: <digit>
    where <digit> is exactly one of 0=right, 1=up, 2=left, 3=down.
    No other output beyond your reasoning and that Final line.
    """
