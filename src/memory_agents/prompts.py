"""SPaRC prompt builders and memory-block renderers for every method.

All text is sent to the model as-is and must stay byte-identical (tests/regress_sparc.py).
"""

# Directions match the env's action ints (0=right, 1=up, 2=left, 3=down).
_DIRECTIONS = {0: "right", 1: "up", 2: "left", 3: "down"}

# Few-shot reflections anchoring Reflexion's format: the mistake, then a high-level plan.
_REFLECTION_EXAMPLES = """- I rushed the path toward the end node and enclosed a blue and a green square in the same region, which breaks the square rule. Next time I should decide up front which region each color belongs to and route the path so differently colored squares end up separated before I head for the end.
- I filled a column too early and painted the path into a corner around move 12 with no legal move left. I should keep more exits open and avoid committing to a wall until I have checked that every remaining dot can still be reached."""


def format_trajectory(episode, max_steps=None) -> str:
    """Render an episode as an ordered move log; `max_steps` keeps only the last moves."""
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
    """The 'plans you already tried' section of the reflection prompt; empty if none."""
    if not previous:
        return ""
    body = "\n".join(f"    - {p}" for p in previous)
    return (
        "\n    Plans you already tried on THIS puzzle in earlier trials, all of which\n"
        "    still ended in failure. Do not repeat them; say what to do differently:\n"
        f"{body}\n"
    )


def reflexion_reflection_prompt(episode, previous=()) -> str:
    """Reflexion self-reflection prompt; `previous` holds the puzzle's earlier reflections."""
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


# Few-shot Memp procedure: scoped to a rule combination, no coordinates.
_PROCEDURE_EXAMPLE = """Applies to: squares of two colors + dots
1. Before moving, group the squares by color and decide which side of the grid each color's region will occupy; the path must become the wall between them.
2. Sketch a rough wall separating the color groups, then bend it to collect every dot cell, since dots must be on the path itself.
3. Prefer routes along the grid edge early; interior detours burn nodes you may need later, and revisiting is impossible.
4. Before committing near the end node, verify each enclosed region holds squares of only one color; if not, backtrack the plan, not the path."""


def distill_rules_preamble(polyshapes, artifact: str = "procedure") -> str:
    """Rulebook plus two clarifications, prepended to every prompt that writes a memory.

    Keeps written memories from misstating the rules. `artifact` names what is being
    written; the default keeps the Memp prompts unchanged.
    """
    return f"""
    The rules of the puzzle, reproduced verbatim from the solver's own instructions.
    They address the solver as "you"; you are not the solver here -- you are writing
    a {artifact} the solver will later read. Every claim your {artifact} makes must be
    true of these rules:

    --- BEGIN PUZZLE RULES ---
{sparc_rules(polyshapes)}
    --- END PUZZLE RULES ---

    Two points to be clear on before you write anything, because they are the
    mistakes {artifact}s most often make:
    * Squares, stars, triangles and polyshapes are NOT cells the path travels
      through. The path goes *around* them. Their rules constrain the regions the
      path encloses (squares: one colour per region; stars: exactly one same-colour
      partner in the region; polyshapes: the region must fit the shape) or how many
      of a cell's edges the path touches (triangles). Only DOTS are cells the path
      must pass through, and only GAPS are cells it must keep off.
    * A line such as "rules: dot x2, square x3" is a census of how many cells of
      each kind this grid holds. It is NOT a count of required visits, entries or
      crossings -- no rule of that form exists in this game.
"""


def memp_distill_trajectory_prompt(episode, key_text: str, polyshapes,
                                   succeeded: bool = True) -> str:
    """Memp Build prompt for an episode the agent ran.

    `succeeded=False` is used only by the Vanilla update policy, which also consolidates failures.
    """
    opening = (
        "You successfully solved the path-finding puzzle below. Distill the strategy\n"
        "    your solution embodies into ONE reusable procedure"
        if succeeded else
        "Your attempt at the path-finding puzzle below did NOT succeed. Distill what\n"
        "    is still reusable from it -- including what to avoid -- into ONE procedure")
    return f"""{distill_rules_preamble(polyshapes)}
    You are an expert puzzle-solving agent building a procedural memory: reusable,
    high-level strategies distilled from puzzles you have already worked on.

    {opening} for solving puzzles with this
    combination of rules. Start with a line "Applies to: <the rule combination>",
    then give 3-6 numbered strategy steps, including pitfalls to avoid. The
    procedure must transfer to OTHER puzzles with similar rules: never mention
    absolute coordinates, this grid's size, or specific moves. At most 120 words.
    Do not output a move or restate the generic game rules. Every mechanic you
    appeal to must be one of the rules above -- do not invent constraints, and say
    nothing about a rule type you are unsure of.

    Here is an example of a good procedure:
{_PROCEDURE_EXAMPLE}

    Puzzle description:
    {key_text}

    Starting grid:
    {episode.first_obs}

    The moves you made, in order (0=right, 1=up, 2=left, 3=down):
{format_trajectory(episode)}

    Final grid, showing the full path you drew (V=visited, L=end):
    {episode.last_obs}

    Procedure:"""


def memp_distill_gold_prompt(key_text: str, text_visualization: str,
                             gold_actions: list, polyshapes=None) -> str:
    """Memp Build prompt from the gold path, for train puzzles the agent failed.

    Reads dataset fields only, so a crashed episode distills the same as a failed one.
    """
    actions = " ".join(str(a) for a in gold_actions)
    return f"""{distill_rules_preamble(polyshapes)}
    You are an expert puzzle-solving agent building a procedural memory: reusable,
    high-level strategies distilled from solved puzzles.

    Below is a path-finding puzzle together with a known-correct solution path.
    Study why this path satisfies the puzzle's rules, then distill that into ONE
    reusable procedure for solving puzzles with this combination of rules. Start
    with a line "Applies to: <the rule combination>", then give 3-6 numbered
    strategy steps, including pitfalls to avoid. The procedure must transfer to
    OTHER puzzles with similar rules: never mention absolute coordinates, this
    grid's size, or specific moves. At most 120 words. Do not output a move or
    restate the generic game rules. Every mechanic you appeal to must be one of the
    rules above -- do not invent constraints, and say nothing about a rule type you
    are unsure of.

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
    """Render a bank entry's solved example; built in code, never written by the LLM."""
    actions = " ".join(str(a) for a in exemplar.get("actions", []))
    start = tuple(exemplar.get("start", ()))
    end = tuple(exemplar.get("end", ()))
    return (f"Example ({exemplar.get('width')}x{exemplar.get('height')} grid, "
            f"start {start} -> end {end}, {exemplar.get('n_moves')} moves: {actions}; "
            f"0=right,1=up,2=left,3=down)\n"
            f"  its grid had: {exemplar.get('grid_summary', '?')}")


def format_stored_trajectory(traj: dict) -> str:
    """Render a bank entry's stored move-by-move trace (Memp's `trajectory` granularity)."""
    lines = []
    for step, action, loc in traj.get("moves", []):
        direction = _DIRECTIONS.get(action, str(action))
        lines.append(f"  move {step}: {direction} -> path head at "
                     f"{loc if loc is not None else '?'}")
    moves = "\n".join(lines) if lines else "  (no moves recorded)"
    return (f"  Its starting grid:\n{traj.get('first_obs', '?')}\n"
            f"  The solution, move by move (0=right, 1=up, 2=left, 3=down):\n{moves}\n"
            f"  Its final grid, with the solution path drawn (V=visited, L=end):\n"
            f"{traj.get('last_obs', '?')}")


def format_procedures(entries, granularity: str = "script") -> str:
    """Memory block appended to the actor's system prompt at Memp eval time.

    `granularity`: script (procedure + compact exemplar), trajectory (stored trace),
    or proceduralization (procedure + stored trace). Similarity scores are never
    rendered, so retrieval arms differ only in which entries appear.
    """
    if granularity not in ("script", "trajectory", "proceduralization"):
        raise ValueError(f"unknown granularity {granularity!r}")
    parts = [
        "\n\nProcedural memory: strategies distilled from previously solved puzzles "
        "whose rule combinations resemble this one. They come from OTHER puzzles: "
        "adapt the strategy to this grid, and never copy an example's move sequence "
        "verbatim."
    ]
    for i, entry in enumerate(entries, 1):
        body = [f"\nProcedure {i}:"]
        if granularity in ("script", "proceduralization"):
            body.append(entry["script"].strip())
        if granularity == "script":
            body.append(format_exemplar(entry["exemplar"]))
        else:
            traj = entry.get("trajectory")
            body.append(format_stored_trajectory(traj) if traj
                        else format_exemplar(entry["exemplar"]))
        parts.append("\n".join(body))
    return "\n".join(parts)


def memp_revise_prompt(script: str, episode, key_text: str, polyshapes=None) -> str:
    """Memp Adjustment prompt: revise a retrieved procedure after a failed attempt."""
    reasons = {
        "wrong-solution": "the agent reached the exit but broke one or more rules",
        "dead-end": "the agent painted the path into a corner with no legal move left",
        "format-failure": "the agent never produced a parseable move",
    }
    why = reasons.get(episode.terminal_reason, "the agent did not solve the puzzle")
    last_rules = episode.transitions[-1].rule_status if episode.transitions else {}
    return f"""{distill_rules_preamble(polyshapes)}
    You maintain a library of procedural memories for a path-finding puzzle solver.
    The procedure below was retrieved to help solve a new puzzle, and the attempt
    FAILED: {why}.

    Rewrite the procedure so that an agent following it would not repeat this
    failure. Keep it general and reusable for the same class of puzzles -- it will
    be shown for other puzzles too, so do not tailor it to this grid, and do not
    mention absolute coordinates. Correct anything the old procedure asserts that
    the failure shows to be wrong, and anything it asserts that contradicts the
    rules above -- a procedure can fail because it states a mechanic this game does
    not have. Keep the same shape: an "Applies to:" line, then 3-6 numbered steps
    and pitfalls, at most 120 words. Output only the revised procedure.

    The procedure that was used:
    {script}

    The puzzle it was used on:
    {key_text}

    Starting grid:
    {episode.first_obs}

    Moves the agent made, in order (0=right, 1=up, 2=left, 3=down):
{format_trajectory(episode)}

    Final grid (V=visited, L=last position):
    {episode.last_obs}

    Rule status at the end (per-rule pass/fail): {last_rules}

    Revised procedure:"""


def memp_keywords_prompt(key_text: str) -> str:
    """Memp AveFact prompt: extract keyword phrases used as retrieval keys."""
    return f"""
    Extract the key features of the path-finding puzzle below: the facts that
    determine HOW it must be solved. Think rule types present, their counts and
    colours, grid scale, and any structural feature that constrains the path.

    Output 4 to 8 short keyword phrases, one per line, no numbering, no other text.
    Do not include coordinates.

    Puzzle:
    {key_text}

    Keywords:"""


# --------------------------------------------------------------------------- #
# MemoryBank (Zhong et al., 2024): declarative memory of specific past cases.
# The prompts below write the case note; the verbatim move record is attached in code.
# --------------------------------------------------------------------------- #
_CASE_INSTRUCTION = """You are recalling a specific past experience, not writing a strategy
    guide. Recount what this puzzle looked like and how it was solved, concretely, in
    2-3 sentences: the shape and scale of the grid, which rule cells were where in
    general terms, and the shape the successful path took through them. Do not phrase
    this as a rule, a numbered procedure, or advice for other puzzles -- you are
    remembering one instance, not instructing. Do not output a move sequence (the
    moves are stored alongside your note already) and do not restate the game rules."""


def memorybank_case_trajectory_prompt(episode, key_text: str, polyshapes,
                                      rules_preamble: bool = True) -> str:
    """MemoryBank case note for a solved puzzle; `rules_preamble` is an ablation switch."""
    preamble = distill_rules_preamble(polyshapes, artifact="memory note") \
        if rules_preamble else ""
    return f"""{preamble}
    You keep a memory of specific puzzles you have worked on, so that when a new
    puzzle resembles one of them you can recall the earlier case.

    You successfully solved the path-finding puzzle below. Write the memory entry for
    it. {_CASE_INSTRUCTION}

    Puzzle description:
    {key_text}

    Starting grid:
    {episode.first_obs}

    The moves you made, in order (0=right, 1=up, 2=left, 3=down):
{format_trajectory(episode)}

    Final grid, showing the full path you drew (V=visited, L=end):
    {episode.last_obs}

    Memory of this puzzle:"""


def memorybank_case_gold_prompt(key_text: str, text_visualization: str,
                                gold_actions: list, polyshapes=None,
                                rules_preamble: bool = True) -> str:
    """MemoryBank case note from the gold path, for train puzzles the agent failed.

    Same voice as the trajectory variant, so a note does not reveal its provenance.
    """
    actions = " ".join(str(a) for a in gold_actions)
    preamble = distill_rules_preamble(polyshapes, artifact="memory note") \
        if rules_preamble else ""
    return f"""{preamble}
    You keep a memory of specific puzzles you have worked on, so that when a new
    puzzle resembles one of them you can recall the earlier case.

    Below is a path-finding puzzle you worked on, together with the solution path
    that solves it. Write the memory entry for it. {_CASE_INSTRUCTION}

    Puzzle description:
    {key_text}

    Puzzle layout (structured form):
    {text_visualization}

    The solution, as moves from the start node (0=right, 1=up, 2=left, 3=down):
    {actions}

    Memory of this puzzle:"""


def format_cases(entries, granularity: str = "case-record") -> str:
    """Memory block appended to the actor's system prompt at MemoryBank eval time.

    `granularity`: case-record (note + stored trace) or case (note + compact exemplar).
    """
    if granularity not in ("case-record", "case"):
        raise ValueError(f"unknown granularity {granularity!r}")
    parts = [
        "\n\nDeclarative memory: specific past cases similar to this puzzle, from "
        "puzzles you solved before. Use them only as loose analogies for the kind of "
        "layout and rule combination involved -- not as instructions to follow. This "
        "puzzle is a different one: never copy a case's move sequence verbatim."
    ]
    for i, entry in enumerate(entries, 1):
        body = [f"\nCase {i}:", entry["case_note"].strip()]
        traj = entry.get("trajectory")
        if granularity == "case-record" and traj:
            body.append(format_stored_trajectory(traj))
        else:
            body.append(format_exemplar(entry["exemplar"]))
        parts.append("\n".join(body))
    return "\n".join(parts)


def sparc_rules(polyshapes) -> str:
    """The SPaRC rulebook without the I/O contract, shared by the actor and distill prompts.

    {polyshapes} stays last so everything before it is identical across puzzles.
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
"""


# The actor's I/O contract, kept out of sparc_rules so distill prompts do not ask for a move.
_SPARC_IO_CONTRACT = """
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


def sparc_system_prompt(polyshapes) -> str:
    """The actor's system prompt, ported verbatim from Spatial-Gym's llm_testing host.

    Do not tidy it (duplicated lines included): it must match the upstream baseline.
    Keep {polyshapes} near the end so the rest is a shared KV prefix for prefix caching.
    """
    return sparc_rules(polyshapes) + _SPARC_IO_CONTRACT



# --------------------------------------------------------------------------- #
# Contrastive memory: SPaRC renderers, kept separate so the Memp and MemoryBank
# renderers above stay byte-identical (tests/regress_sparc.py).
# --------------------------------------------------------------------------- #
# Entries mix DO and DO NOT advice, so each is labelled and the header says both occur.
_CONTRASTIVE_HEADER = (
    "\n\nMemory from previous puzzles with similar rule combinations. Each item is "
    "either something that WORKED on a puzzle you solved, or something that FAILED "
    "on a puzzle you did not. Read the label before the text: the failures are "
    "there to be avoided, not followed. They come from OTHER puzzles -- adapt the "
    "advice to this grid, and never copy a move sequence."
)
_POSITIVE_ONLY_HEADER = (
    "\n\nMemory from previous puzzles with similar rule combinations: strategies "
    "that WORKED on puzzles you solved. They come from OTHER puzzles -- adapt the "
    "advice to this grid, and never copy a move sequence."
)
_AVOID_ONLY_HEADER = (
    "\n\nMemory from previous puzzles with similar rule combinations: mistakes that "
    "made attempts FAIL. They are there to be avoided, not followed. They come from "
    "OTHER puzzles -- adapt the advice to this grid, and never copy a move sequence."
)

def format_broken_rules(rules) -> str:
    """Compact broken-rule verdict: keeps scalars and short flat lists, drops the rest."""
    if not rules:
        return "  Rules it broke: none recorded"
    parts = []
    for item in rules:
        detail = item.get("detail")
        summary = ""
        if isinstance(detail, dict):
            bits = []
            for k, v in detail.items():
                if isinstance(v, dict):
                    continue
                if isinstance(v, list):
                    if v and len(v) <= 5 and all(
                            not isinstance(x, (dict, list)) for x in v):
                        bits.append(f"{k}={v}")
                elif v is not None and v != {}:
                    bits.append(f"{k}={v}")
            summary = f" ({', '.join(bits)[:90]})" if bits else ""
        parts.append(f"{item.get('rule', '?')}{summary}")
    return "  Rules it broke: " + "; ".join(parts)


# Header for `hindsight` under --framing matched, where every item is a DO.
# Avoids "puzzles you solved", since many entries come from gold paths.
_ALL_DO_HEADER = (
    "\n\nMemory from previous puzzles with similar rule combinations: strategies for "
    "solving puzzles of this kind. They come from OTHER puzzles -- adapt the advice to "
    "this grid, and never copy a move sequence."
)

_DO = "DO -- this worked on a puzzle like this one:"
_DONT = "DO NOT -- this is how an attempt at a puzzle like this one failed:"
# Placeholder for an entry the arm does not render, so item counts match across arms.
_WITHHELD = "  (nothing recorded for this puzzle under this arm)"


def format_failed_exemplar(exemplar: dict) -> str:
    """Render the moves a failed attempt played; claims no end node and shows no gold path."""
    actions = " ".join(str(a) for a in exemplar.get("actions", []))
    return (f"What that attempt played ({exemplar.get('n_moves')} moves on a "
            f"{exemplar.get('width')}x{exemplar.get('height')} grid, from "
            f"{tuple(exemplar.get('start', ()))}): {actions}; "
            f"0=right,1=up,2=left,3=down\n"
            f"  its grid had: {exemplar.get('grid_summary', '?')}")


def format_contrastive(entries, granularity: str = "avoid", exemplars: bool = False,
                       framing: str = "mixed") -> str:
    """Memory block appended to the actor's system prompt at contrastive eval time.

    `agent` entries render their `strategy`; the arms differ in what a `failed` entry shows:
      avoid          -- its `avoid` text (the method)
      hindsight      -- its gold-path procedure, labelled DO (the control)
      positive-only  -- nothing
      avoid-only     -- its `avoid` text, and `agent` entries render nothing
    """
    from .memory.contrastive import AGENT, GRANULARITIES
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}")
    if framing not in ("mixed", "matched"):
        raise ValueError(f"unknown framing {framing!r}")
    headers = {"positive-only": _POSITIVE_ONLY_HEADER,
               "avoid-only": _AVOID_ONLY_HEADER}
    if framing == "matched":
        headers["hindsight"] = _ALL_DO_HEADER
    parts = [headers.get(granularity, _CONTRASTIVE_HEADER)]
    for i, entry in enumerate(entries, 1):
        positive = entry.get("provenance") == AGENT
        if positive:
            text = "" if granularity == "avoid-only" else (
                entry.get("strategy") or "")
        else:
            field = "hindsight" if granularity == "hindsight" else "avoid"
            text = "" if granularity == "positive-only" else (
                entry.get(field) or "")
        body = [f"\nItem {i}:"]
        if not text.strip():
            body.append(_WITHHELD)
        else:
            # `hindsight` is a solving procedure, so it is a DO despite coming from a failure.
            shown_as_do = positive or granularity == "hindsight"
            body.append(_DO if shown_as_do else _DONT)
            body.append(text.strip())
            if exemplars:
                # Chosen by how the item is rendered, not by its provenance.
                ex = entry.get("exemplar" if shown_as_do else "failed_exemplar")
                if ex:
                    body.append(format_exemplar(ex) if shown_as_do
                                else format_failed_exemplar(ex))
        parts.append("\n".join(body))
    return "\n".join(parts)


def contrastive_avoid_prompt(episode, key_text: str, polyshapes, rules=()) -> str:
    """Contrastive prompt distilling one reusable DON'T from a failed trajectory.

    `rules` is the env's broken-rule verdict (empty for a dead end), quoted so the prompt
    names the rule actually broken.
    """
    reasons = {
        "wrong-solution": "it reached the end node but broke one or more rules",
        "dead-end": "it painted the path into a corner with no legal move left",
        "format-failure": "it never produced a parseable move",
    }
    why = reasons.get(getattr(episode, "terminal_reason", ""),
                      "it did not solve the puzzle")
    verdict = ("\n    The environment's own verdict on this attempt:\n"
               f"  {format_broken_rules(rules)}"
               if rules else
               "\n    The environment recorded no rule verdict: the attempt never "
               "reached a\n    submission, so it stalled rather than breaking a "
               "stated rule.")
    return f"""{distill_rules_preamble(polyshapes, artifact="warning")}
    You are an expert puzzle-solving agent building a memory of MISTAKES: reusable,
    high-level warnings distilled from attempts that failed.

    The attempt below did NOT solve the puzzle: {why}. Distil what went wrong into
    ONE reusable warning for puzzles with this combination of rules. Start with a
    line "Applies to: <the rule combination>", then give 2-4 numbered points, each
    naming a mistake to avoid and what to do instead. At most 120 words.

    The warning must transfer to OTHER puzzles with similar rules: never mention
    absolute coordinates, this grid's size, or specific moves, and do not output a
    move. Every mechanic you appeal to must be one of the rules above -- do not
    invent constraints, and say nothing about a rule type you are unsure of. Do not
    speculate about what the correct path was; you were not shown one. Describe the
    error, not the puzzle.

    Puzzle description:
    {key_text}

    Starting grid:
    {episode.first_obs}

    The moves the attempt made, in order (0=right, 1=up, 2=left, 3=down):
{format_trajectory(episode)}

    The grid where it failed (V=visited, L=its last position):
    {episode.last_obs}
{verdict}

    Warning:"""
