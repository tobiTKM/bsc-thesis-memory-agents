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


def format_previous_plans(previous) -> str:
    """The 'plans you already tried' section of the reflection prompt.

    Reflexion's own reflection step is prompted with the reflections already in
    memory, not just the latest trajectory (noahshinn/reflexion builds its query
    from the few-shot examples *plus* the prior trial plans). Without this, trial 4
    reflects as if trial 3 never happened and the model happily re-proposes a plan
    it has already watched fail. Empty on trial 2, where there is no prior plan.
    """
    if not previous:
        return ""
    body = "\n".join(f"    - {p}" for p in previous)
    return (
        "\n    Plans you already tried on THIS puzzle in earlier trials, all of which\n"
        "    still ended in failure. Do not repeat them; say what to do differently:\n"
        f"{body}\n"
    )


def reflexion_reflection_prompt(episode, previous=()) -> str:
    """Reflexion's self-reflection step, following the paper's own prompt shape.

    Reflexion (Shinn et al., 2023) prompts an "advanced reasoning agent that can
    improve based on self-reflection" with the full failed trajectory and asks it
    to *diagnose a reason for failure and devise a new, concise, high-level plan*
    -- in a few complete sentences, anchored by few-shot example reflections. The
    plan (not a bug report) is what the actor reads before its next attempt at the
    SAME puzzle. See memory/reflexion.py for how it is stored and injected.

    `previous` is that puzzle's existing memory (the reflections from earlier failed
    trials), which the paper's implementation also conditions on -- see
    format_previous_plans.
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
{format_previous_plans(previous)}
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


# One few-shot script anchoring the distillation format, in the same role
# _REFLECTION_EXAMPLES plays for reflections: a *procedure* for a class of
# puzzles (rule-combination scoped, no coordinates), not a solution to one.
_PROCEDURE_EXAMPLE = """Applies to: squares of two colors + dots
1. Before moving, group the squares by color and decide which side of the grid each color's region will occupy; the path must become the wall between them.
2. Sketch a rough wall separating the color groups, then bend it to collect every dot cell, since dots must be on the path itself.
3. Prefer routes along the grid edge early; interior detours burn nodes you may need later, and revisiting is impossible.
4. Before committing near the end node, verify each enclosed region holds squares of only one color; if not, backtrack the plan, not the path."""


def memp_distill_trajectory_prompt(episode, key_text: str) -> str:
    """Memp's Build step for a puzzle the agent itself solved.

    The paper distills successful trajectories into "script-like abstractions":
    reusable procedural knowledge for the *class* of task, not a replay of one
    instance. So the prompt shows the solved episode (start grid, ordered moves,
    final grid) but asks for a strategy scoped to the puzzle's rule combination,
    and explicitly bans coordinates and move output -- a memory that says "go to
    (4,2)" is useless on every other puzzle, and the concrete exemplar is
    attached separately in code (see format_exemplar).
    """
    return f"""
    You are an expert puzzle-solving agent building a procedural memory: reusable,
    high-level strategies distilled from puzzles you have already solved.

    You successfully solved the path-finding puzzle below. Distill the strategy your
    solution embodies into ONE reusable procedure for solving puzzles with this
    combination of rules. Start with a line "Applies to: <the rule combination>",
    then give 3-6 numbered strategy steps, including pitfalls to avoid. The
    procedure must transfer to OTHER puzzles with similar rules: never mention
    absolute coordinates, this grid's size, or specific moves. At most 120 words.
    Do not output a move or restate the generic game rules.

    Here is an example of a good procedure:
{_PROCEDURE_EXAMPLE}

    Puzzle description:
    {key_text}

    Starting grid:
    {episode.first_obs}

    The moves of your successful solution, in order (0=right, 1=up, 2=left, 3=down):
{format_trajectory(episode)}

    Final grid, showing the full solution path you drew (V=visited, L=end):
    {episode.last_obs}

    Procedure:"""


def memp_distill_gold_prompt(key_text: str, text_visualization: str,
                             gold_actions: list) -> str:
    """Memp's Build step for a train puzzle the agent failed: hindsight distillation.

    SPaRC ships gold solution paths, so a failed build episode still yields a
    procedure -- from the known-correct path instead of the agent's own (absent)
    success. Deliberately reads dataset fields only (key text, text_visualization,
    the gold path converted to actions in code): no dependency on the failed
    trajectory means crashed episodes distill exactly like failed ones, and the
    resulting memory describes how to solve the puzzle class, not how the agent
    once got lost in it. Same output contract as the trajectory variant so the
    bank is homogeneous regardless of provenance.
    """
    actions = " ".join(str(a) for a in gold_actions)
    return f"""
    You are an expert puzzle-solving agent building a procedural memory: reusable,
    high-level strategies distilled from solved puzzles.

    Below is a path-finding puzzle together with a known-correct solution path.
    Study why this path satisfies the puzzle's rules, then distill that into ONE
    reusable procedure for solving puzzles with this combination of rules. Start
    with a line "Applies to: <the rule combination>", then give 3-6 numbered
    strategy steps, including pitfalls to avoid. The procedure must transfer to
    OTHER puzzles with similar rules: never mention absolute coordinates, this
    grid's size, or specific moves. At most 120 words. Do not output a move or
    restate the generic game rules.

    Here is an example of a good procedure:
{_PROCEDURE_EXAMPLE}

    Puzzle description:
    {key_text}

    Puzzle layout (structured form):
    {text_visualization}

    A known-correct solution, as moves from the start node (0=right, 1=up, 2=left,
    3=down): {actions}

    Procedure:"""


def format_exemplar(exemplar: dict) -> str:
    """Render a bank entry's concrete solved example as one compact line pair.

    Built from the exemplar dict the Distiller assembles in code (actions either
    from the agent's winning Transitions or from the gold path) -- the LLM never
    writes this part, so it cannot hallucinate a move sequence.
    """
    actions = " ".join(str(a) for a in exemplar.get("actions", []))
    start = tuple(exemplar.get("start", ()))
    end = tuple(exemplar.get("end", ()))
    return (f"Example ({exemplar.get('width')}x{exemplar.get('height')} grid, "
            f"start {start} -> end {end}, {exemplar.get('n_moves')} moves: {actions}; "
            f"0=right,1=up,2=left,3=down)\n"
            f"  its grid had: {exemplar.get('grid_summary', '?')}")


def format_procedures(entries) -> str:
    """The block injected into the actor's system prompt at Memp eval time.

    Same seam as format_reflections: MemoryAgent.reset appends this AFTER the full
    system prompt, so the shared KV prefix (see sparc_system_prompt) survives.
    Similarity scores are logged in the results but deliberately never rendered
    here -- the retrieved and random arms must inject byte-identical *formats* so
    they differ only in which entries were retrieved.
    """
    parts = [
        "\n\nProcedural memory: strategies distilled from previously solved puzzles "
        "whose rule combinations resemble this one. They come from OTHER puzzles: "
        "adapt the strategy to this grid, and never copy an example's move sequence "
        "verbatim."
    ]
    for i, entry in enumerate(entries, 1):
        parts.append(f"\nProcedure {i}:\n{entry['script'].strip()}\n"
                     f"{format_exemplar(entry['exemplar'])}")
    return "\n".join(parts)


def sparc_system_prompt(polyshapes) -> str:
    """The actor's system prompt, ported verbatim from Spatial-Gym's llm_testing host.

    Kept verbatim on purpose, duplication included (rule cells and the no-revisit
    rule are each stated twice): our numbers are meant to be comparable with the
    established baseline, so the prompt is not ours to tidy.

    One structural property worth preserving: the per-puzzle {polyshapes} is
    interpolated near the END, so the ~145 lines before it are byte-identical for
    every puzzle and every step. With vLLM's --enable-prefix-caching that is a
    large shared KV prefix. Do not move the variable part earlier.
    """
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
