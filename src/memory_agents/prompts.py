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


def distill_rules_preamble(polyshapes, artifact: str = "procedure") -> str:
    """The rulebook plus the two clarifications a distiller cannot infer from it.

    Goes at the TOP of every prompt that WRITES a procedure (both distill variants
    and the revise/Adjustment step), so the constant part is a shared KV prefix
    across the whole distill pass -- the same reason sparc_system_prompt keeps
    {polyshapes} last. The prefix runs to the polyshape line; the notes after it are
    constant text but sit past the per-puzzle part, which is a fair trade for
    reading in the order a person would write it.

    The two notes are not generic hedging -- they name the exact two errors the
    first bank made, on ~86% gold-provenance entries and on agent entries alike:
    treating region-rule cells as cells the path steps on, and reading the key
    text's `rules: square x3` census (how many square CELLS exist) as "each square
    must be entered three times". Both are unrecoverable from the cell labels alone,
    which is all the old prompts showed.

    `artifact` names what is being written. The default keeps every Memp prompt
    byte-identical; MemoryBank's case-note prompts pass "memory note" so the
    preamble does not itself nudge the writer toward procedural phrasing.
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
    """Memp's Build step for a puzzle the agent itself ran.

    `succeeded=False` is used only by the Vanilla update policy, which the paper
    defines as consolidating *all* trajectories from a group -- failures included.
    Every other caller distils a success.

    The paper distills successful trajectories into "script-like abstractions":
    reusable procedural knowledge for the *class* of task, not a replay of one
    instance. So the prompt shows the solved episode (start grid, ordered moves,
    final grid) but asks for a strategy scoped to the puzzle's rule combination,
    and explicitly bans coordinates and move output -- a memory that says "go to
    (4,2)" is useless on every other puzzle, and the concrete exemplar is
    attached separately in code (see format_exemplar).

    The rulebook is prepended (distill_rules_preamble): without it the model has to
    guess the mechanics from cell labels, and a wrong guess becomes a confidently
    worded procedure in the bank.
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
    """Memp's Build step for a train puzzle the agent failed: hindsight distillation.

    SPaRC ships gold solution paths, so a failed build episode still yields a
    procedure -- from the known-correct path instead of the agent's own (absent)
    success. Deliberately reads dataset fields only (key text, text_visualization,
    the gold path converted to actions in code): no dependency on the failed
    trajectory means crashed episodes distill exactly like failed ones, and the
    resulting memory describes how to solve the puzzle class, not how the agent
    once got lost in it. Same output contract as the trajectory variant so the
    bank is homogeneous regardless of provenance.

    This variant needs the prepended rulebook most: it is asked to explain why a
    path it never watched being drawn is correct, with nothing but cell labels to
    go on. It is also the majority of the bank (~86% of entries at SPaRC's
    memoryless success rate), so an error mode here dominates every retrieval.
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


def format_stored_trajectory(traj: dict) -> str:
    """Render a bank entry's stored round-by-round trace.

    This is the paper's `Trajectory` granularity: the successful run kept
    verbatim, round by round, rather than compressed to an action list. Shows the
    starting grid, the ordered moves with the path head after each, and the final
    grid with the solution drawn on it.
    """
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
    """The block injected into the actor's system prompt at Memp eval time.

    Same seam as format_reflections: MemoryAgent.reset appends this AFTER the full
    system prompt, so the shared KV prefix (see sparc_system_prompt) survives.
    Similarity scores are logged in the results but deliberately never rendered
    here -- every retrieval arm must inject byte-identical *formats* so they differ
    only in which entries were retrieved.

    `granularity` reproduces the paper's Build-policy conditions (its Table 1):
      script            -- the distilled abstraction only
      trajectory        -- the stored round-by-round run only
      proceduralization -- both, the paper's best-performing condition
    The compact exemplar rides along with `script` because that is what the
    original campaign injected; the trajectory conditions replace it with the real
    trace rather than stacking both concrete forms.
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
    """Memp's `Adjustment` update: repair a procedure that led to a failure.

    The paper's best update strategy -- "when a retrieved procedural memory
    results in a failed execution, the erroneous trajectory is combined with the
    original memory and then revised in place". The revision sees what actually
    went wrong, which is the only mechanism in the framework capable of removing
    a procedure that misstates a rule.

    Which is precisely why it also gets the rulebook prepended: a reviser without
    the rules can only tell that the attempt failed, not which of the procedure's
    claims is false, so it is as free to invent a new wrong mechanic as the original
    distiller was.
    """
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
    """Memp's `AveFact` key construction: extract the task's key features.

    The paper applies "a large model to extract keywords from the task's query",
    then scores retrieval by the average similarity across matched keywords rather
    than one whole-query vector.
    """
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
# MemoryBank (Zhong et al., 2024): declarative memory -- specific past CASES.
#
# The paper's memory storage keeps two text layers per interaction: the in-depth
# record (the conversation verbatim, turn by turn) and a "hierarchical event
# summary" written by the LLM with the prompt "Summarize the events and key
# information in the content [dialog]". Mapped onto SPaRC, one past puzzle is one
# past interaction: its verbatim record is the replayed move-by-move trace
# (attached in code by the distill phase, so it cannot be hallucinated), and its
# event summary is the case note the two prompts below write.
#
# The contrast with Memp is the whole point of running both (docs/architecture.md
# section 5.2): Memp abstracts a trajectory into a rule-scoped PROCEDURE, while
# MemoryBank recounts one INSTANCE. Both are retrieved by the same key text, at
# the same k, and rendered into the same injection point -- so the comparison
# isolates the type of content, not the retrieval.
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
    """MemoryBank's event summary for a puzzle the agent itself solved.

    The declarative sibling of memp_distill_trajectory_prompt: same inputs (the
    episode's start grid, ordered moves and final grid), opposite output contract
    -- a concrete recollection of THIS puzzle rather than a transferable procedure.

    `rules_preamble` is an ablation knob, not a preference. Memp's v1 banks
    misstated SPaRC's square/star rule in 33-51% of entries because no distillation
    prompt contained the rulebook (docs/memp.md section 3.5); the preamble is what
    fixed that. A case note describes an instance rather than asserting a rule, so
    it is genuinely unclear whether it needs the same treatment -- hence both banks
    get built and audited (see scripts/audit_bank.py).
    """
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
    """MemoryBank's event summary for a train puzzle the agent failed: hindsight.

    Mirrors memp_distill_gold_prompt's role and its crash-safety property -- it
    reads dataset fields only (key text, layout, the gold path converted to actions
    in code), so an episode that crashed with no trajectory writes exactly like one
    that merely failed. At SPaRC's memoryless train accuracy this variant produces
    ~86-98% of the bank, so its phrasing matters more than the trajectory variant's.

    Written in the same voice as the trajectory variant on purpose: the actor must
    not be able to tell, from the note alone, whether a case came from its own run
    or from hindsight -- provenance is recorded in the entry, not in the prose.
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
    """The block injected into the actor's system prompt at MemoryBank eval time.

    The declarative sibling of format_procedures, and deliberately the only place
    the two methods' injected text differs in *framing*: the same seam
    (MemoryAgent.reset appends it AFTER the full system prompt, preserving the
    shared KV prefix), the same silence about similarity scores, the same per-entry
    layout -- but introduced as past cases to reason by analogy from, not as
    strategies to apply.

    `granularity` selects which layer of the paper's two-layer storage rides along
    with the case note:
      case-record  -- the verbatim round-by-round record (the paper's in-depth
                      storage); the default, and the analogue of Memp's
                      `proceduralization`.
      case         -- the compact action-list exemplar instead; the analogue of
                      Memp's `script`, kept so the two methods can also be compared
                      at the granularity where Memp's campaign found no effect.
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
    """The SPaRC rulebook alone -- what the game *is*, with no I/O contract attached.

    Split out of sparc_system_prompt so the distillation prompts can state the same
    rules, verbatim, that the actor plays under. That is not cosmetic. A distiller
    that never sees the rulebook has to reverse-engineer the mechanics from cell
    labels, and it reliably invents the wrong ones: describing squares, stars, polys
    and triangles as cells the path must "visit" some number of times, when the path
    goes *around* them and their rules are about enclosed regions and touched edges.
    Those inventions are written into the bank as confident procedural advice, which
    is worse than an empty bank -- the actor is told, with authority, a rule that
    does not exist.

    Ends with the per-puzzle {polyshapes}, so every line before it is byte-identical
    across puzzles -- see sparc_system_prompt for why that is worth preserving.
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


# The actor's I/O contract. Deliberately NOT part of sparc_rules: it describes the
# observation format and orders the model to end every reply with `Final: <digit>`.
# A distiller handed that instruction would answer a "write a procedure" request
# with a move.
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

    Kept verbatim on purpose, duplication included (rule cells and the no-revisit
    rule are each stated twice): our numbers are meant to be comparable with the
    established baseline, so the prompt is not ours to tidy. It is now assembled
    from sparc_rules + the I/O contract, and the concatenation is byte-identical to
    the single string this used to be -- baseline numbers already in results/ stay
    comparable, and the shared KV prefix is unchanged.

    One structural property worth preserving: the per-puzzle {polyshapes} is
    interpolated near the END, so the ~145 lines before it are byte-identical for
    every puzzle and every step. With vLLM's --enable-prefix-caching that is a
    large shared KV prefix. Do not move the variable part earlier.
    """
    return sparc_rules(polyshapes) + _SPARC_IO_CONTRACT
