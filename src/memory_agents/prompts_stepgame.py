"""Every prompt the StepGame arms use; mirrors prompts.py one for one.

Kept separate so the SPaRC prompts in prompts.py stay byte-identical
(tests/regress_sparc.py).
"""

from .envs.stepgame_solver import LABELS, OFFSETS

_LABEL_LINE = ", ".join(LABELS)


# --------------------------------------------------------------------------- #
# the actor
# --------------------------------------------------------------------------- #
# The system prompt is fully constant so vLLM can cache it as a shared prefix;
# keep item-specific text in the user message.
_STEPGAME_IO_CONTRACT = f"""
    You MAY think step-by-step, but you MUST end your reply with:
    Final: <label>
    where <label> is exactly one of: {_LABEL_LINE}
    No other output beyond your reasoning and that Final line.
    """


def stepgame_rules() -> str:
    """The task statement, zero-shot: few-shot examples would be injected memory
    in every arm, including the memoryless baseline (Li et al. use 5/10-shot)."""
    return f"""
    You are solving a spatial reasoning problem.

    You are given a story: a list of sentences, each describing where one object
    sits relative to another. You are then asked for the spatial relation between
    two specific objects, which are usually NOT described in the same sentence --
    you have to chain the relations together to connect them.

    The answer is exactly one of nine relations:
    {_LABEL_LINE}

    How to read the question. "What is the relation of the agent A to the agent B?"
    asks where A is *relative to B*. B is the reference point; A is the object being
    located. Getting this backwards inverts the answer, so fix which object is the
    reference before you reason.

    Notes on the relations:
    * "above" means directly above with no horizontal offset; "upper-left" means
      both above and to the left. The four diagonal labels each combine one
      vertical and one horizontal direction.
    * "overlap" means the two objects are at the same position. This is possible:
      the relations in a story are not constrained to keep objects apart, so a
      chain can return to where it started.
    * Only the *direction* matters, not the distance. An object three units up and
      one unit right of another is "upper-right".

    Some sentences in the story may be irrelevant to the question -- either they
    describe objects off the path between the two you were asked about, or they are
    deliberate distractors. Use only what connects the two objects.
    """


def stepgame_system_prompt() -> str:
    return stepgame_rules() + _STEPGAME_IO_CONTRACT


# --------------------------------------------------------------------------- #
# the method a memory can describe: the coordinate technique
# --------------------------------------------------------------------------- #
def relations_preamble(artifact: str = "procedure") -> str:
    """The relation/offset table prepended to every memory-writing prompt, so
    memories state offsets and the frame of reference correctly.

    Unlike SPaRC, coordinates are wanted here: the coordinate frame is the method.
    """
    offsets = "\n".join(f"      {label:12s} ({dx:+d}, {dy:+d})"
                        for label, (dx, dy) in OFFSETS.items())
    return f"""
    Reference material for the {artifact} you are about to write. Every claim it
    makes must be consistent with this:

    --- BEGIN RELATION REFERENCE ---
    The nine relations and their unit offsets, in a frame where x grows to the
    right and y grows upward. "A is <relation> of B" means A sits at B's position
    plus the offset:
{offsets}

    Two consequences worth stating, because they are what {artifact}s get wrong:
    * The relation is *directional*. "A is left of B" and "B is left of A" are
      different claims; reversing a sentence negates its offset.
    * Only the sign of the total displacement decides the answer, not its size.
      A total of (+3, -1) is "lower-right", the same as (+1, -1).
    --- END RELATION REFERENCE ---
    """


# --------------------------------------------------------------------------- #
# Reflexion (episodic)
# --------------------------------------------------------------------------- #
_REFLECTION_EXAMPLES = """- I chained the relations in the right order but read the question backwards: it asked where D was relative to Q, and I computed Q relative to D, which inverted my answer. Next time I will fix the reference object first, write it down, and check at the end that the object I located is the one the question names first.
- I treated "if X is the centre of a clock face, Y is between 4 and 5" as though Y were above X, when a 4-5 o'clock bearing is below and to the right. Next time I will convert every clock-face or compass phrasing into one of the nine labels explicitly before I start chaining, instead of reading it off intuitively."""


def format_previous_plans(previous) -> str:
    """Plans earlier trials already tried, so a reflection does not repeat one."""
    if not previous:
        return ""
    body = "\n".join(f"    - {p}" for p in previous)
    return (f"\n    Plans you already tried on THIS question in earlier attempts, "
            f"which did not work. Do not repeat them:\n{body}\n")


def stepgame_reflection_prompt(episode, previous=()) -> str:
    """Reflexion's self-reflection step for a single-turn item.

    The gold label must never appear in this prompt: the reflector gets only a
    binary wrong-answer signal, as in Reflexion's HotpotQA setting.
    tests/smoke_stepgame.py asserts this.
    """
    reasons = {
        "wrong-answer": "you gave the wrong relation",
        "format-failure": "you never produced a parseable 'Final: <label>' line",
    }
    why = reasons.get(episode.terminal_reason, "you did not answer correctly")
    answered = (episode.transitions[-1].action
                if episode.transitions else None)
    given = f"You answered: {answered}" if answered else "You gave no parseable answer."
    return f"""
    You are an advanced reasoning agent that can improve based on self-reflection.
    You were given a spatial reasoning question and, in a previous trial, you were
    unsuccessful: {why}. In a few sentences, diagnose a possible reason for the
    failure and devise a new, concise, high-level plan that aims to mitigate the
    same failure on your next attempt at THIS question. Use complete sentences.
    Do not restate the task rules in general, and do not output an answer.

    You are NOT told the correct answer. Do not guess at it or assert one -- your
    job is to describe how to reason about this question differently.

    Here are two examples of good self-reflections:
{_REFLECTION_EXAMPLES}
{format_previous_plans(previous)}
    Previous trial:
{episode.first_obs}

    {given}

    Your reasoning was:
    {episode.last_obs}

    Reflection:"""


def format_reflections(reflections) -> str:
    """The block injected into the actor's system prompt on a retry."""
    body = "\n".join(f"- {r}" for r in reflections)
    return (
        "\n\nReflections from your previous attempt(s) at THIS question. Use them "
        "to avoid repeating the same mistakes:\n" + body
    )


# --------------------------------------------------------------------------- #
# MemP (procedural)
# --------------------------------------------------------------------------- #
_PROCEDURE_EXAMPLE = """Applies to: chains of 3-5 relations mixing diagonals with cardinals
1. Read the question first and write down which object is the reference; place it at (0, 0) before touching the story.
2. Convert every sentence into one of the nine labels on its own, before chaining anything -- oblique phrasings (clock faces, compass points, "over there and X is on the left of it") are where the errors come from, not the arithmetic.
3. Walk from the reference object outward, one sentence at a time, adding that sentence's offset; ignore sentences that do not touch an object you have already placed.
4. Read the answer off the sign of the final displacement, and check the direction of travel matches the order the question named the two objects."""


def _story_block(record) -> str:
    """The item as the solver saw it. `record` is a StepGameItem."""
    story = "\n".join(f"    {i}. {s}" for i, s in enumerate(record.story, start=1))
    return f"    Story:\n{story}\n    Question: {record.question}"


def stepgame_distill_trajectory_prompt(episode, key_text: str, record,
                                       succeeded: bool = True) -> str:
    """Memp's Build step for an item the agent itself answered correctly."""
    answered = episode.transitions[-1].action if episode.transitions else "?"
    return f"""{relations_preamble("procedure")}
    Below is a spatial reasoning question you solved correctly, together with the
    reasoning you used. Write a PROCEDURE: a short, general strategy for solving
    questions of this kind, which a solver facing a *different* story with a similar
    shape could follow.

    Rules for the procedure:
    * Start with a line "Applies to: <what kind of question this is>", describing the
      structure (how many relations chain together, which kinds of relation, whether
      there are irrelevant sentences) -- never the specific object letters.
    * Then 3-5 numbered steps. Keep it under 120 words.
    * Do NOT name any object from this story, and do NOT state this story's answer.
      The procedure is read by a solver working on a different question.
    * Describing the coordinate technique in general terms IS wanted -- it is the
      method, and it transfers.

    Here is an example of the format and level of abstraction:
{_PROCEDURE_EXAMPLE}

    The question you solved:
{_story_block(record)}

    Its structure: {key_text}

    Your answer was {answered}, which was correct. Your reasoning:
    {episode.last_obs}

    Procedure:"""


def stepgame_distill_gold_prompt(key_text: str, record, gold_chain: str) -> str:
    """Memp's hindsight variant: a procedure from the exact solver's derivation.

    Used for train items the agent got wrong or crashed on.
    """
    return f"""{relations_preamble("procedure")}
    Below is a spatial reasoning question together with a correct, worked solution.
    Write a PROCEDURE: a short, general strategy for solving questions of this kind,
    which a solver facing a *different* story with a similar shape could follow.

    Rules for the procedure:
    * Start with a line "Applies to: <what kind of question this is>", describing the
      structure (how many relations chain together, which kinds of relation, whether
      there are irrelevant sentences) -- never the specific object letters.
    * Then 3-5 numbered steps. Keep it under 120 words.
    * Do NOT name any object from this story, and do NOT state this story's answer.
      The procedure is read by a solver working on a different question.
    * Describing the coordinate technique in general terms IS wanted -- it is the
      method, and it transfers.

    Here is an example of the format and level of abstraction:
{_PROCEDURE_EXAMPLE}

    The question:
{_story_block(record)}

    Its structure: {key_text}

    A correct worked solution:
{gold_chain}

    Procedure:"""


# --------------------------------------------------------------------------- #
# MemoryBank (declarative)
# --------------------------------------------------------------------------- #
_CASE_INSTRUCTION = """You are recalling a specific past experience, not writing a strategy
    guide. Recount what this question looked like and how it was solved, concretely, in
    2-3 sentences. Do not phrase this as a rule, a numbered procedure, or advice for
    other questions -- you are remembering one instance, not instructing."""


def stepgame_case_trajectory_prompt(episode, key_text: str, record,
                                    rules_preamble: bool = True) -> str:
    """MemoryBank's event summary for an item the agent answered correctly."""
    preamble = relations_preamble("memory note") if rules_preamble else ""
    answered = episode.transitions[-1].action if episode.transitions else "?"
    return f"""{preamble}
    {_CASE_INSTRUCTION}

    The question you answered:
{_story_block(record)}

    Its structure: {key_text}

    You answered {answered}, which was correct. Your reasoning:
    {episode.last_obs}

    Your recollection:"""


def stepgame_case_gold_prompt(key_text: str, record, gold_chain: str,
                              rules_preamble: bool = True) -> str:
    """MemoryBank's event summary from the worked solution (hindsight)."""
    preamble = relations_preamble("memory note") if rules_preamble else ""
    return f"""{preamble}
    {_CASE_INSTRUCTION}

    The question:
{_story_block(record)}

    Its structure: {key_text}

    How it was solved:
{gold_chain}

    Your recollection:"""


# --------------------------------------------------------------------------- #
# rendering retrieved memory into the actor's context
# --------------------------------------------------------------------------- #
def format_exemplar(exemplar: dict) -> str:
    """A bank entry's compact solved example, assembled in code rather than by the LLM."""
    return (f"Example: {exemplar.get('hops', '?')}-hop chain over "
            f"{exemplar.get('sentences', '?')} sentences "
            f"({exemplar.get('unused', 0)} not needed); "
            f"relations {', '.join(exemplar.get('relations', [])) or 'none'}; "
            f"answer {exemplar.get('answer', '?')}")


def format_stored_trajectory(traj: dict) -> str:
    """A bank entry's verbatim record: the worked derivation, hop by hop."""
    return (f"  The question it came from:\n{traj.get('first_obs', '?')}\n"
            f"  How it was solved:\n{traj.get('last_obs', '?')}")


def _render(entries, granularity: str, header: str, label: str, field: str,
            abstract_at: tuple, exemplar_at: tuple) -> str:
    """Shared renderer for the procedure and case blocks.

    Similarity scores are never rendered, so retrieval arms differ only in which
    entries they inject. `field` is the entry key holding the text; `label` is display only.
    """
    parts = [header]
    for i, entry in enumerate(entries, 1):
        body = [f"\n{label} {i}:"]
        if granularity in abstract_at:
            body.append(entry[field].strip())
        if granularity in exemplar_at:
            body.append(format_exemplar(entry["exemplar"]))
        else:
            traj = entry.get("trajectory")
            body.append(format_stored_trajectory(traj) if traj
                        else format_exemplar(entry["exemplar"]))
        parts.append("\n".join(body))
    return "\n".join(parts)


def format_procedures(entries, granularity: str = "script") -> str:
    """Memp's injected block. Granularities follow the paper's Build policies:

      script            -- the distilled abstraction plus a compact exemplar
      trajectory        -- the stored worked derivation only
      proceduralization -- both
    """
    if granularity not in ("script", "trajectory", "proceduralization"):
        raise ValueError(f"unknown granularity {granularity!r}")
    return _render(
        entries, granularity,
        "\n\nProcedural memory: strategies distilled from spatial reasoning "
        "questions solved previously, whose structure resembles this one. They come "
        "from OTHER questions: adapt the strategy, and never copy an example's "
        "answer.",
        label="Procedure", field="script",
        abstract_at=("script", "proceduralization"),
        exemplar_at=("script",))


def format_cases(entries, granularity: str = "case-record") -> str:
    """MemoryBank's injected block: concrete past cases, framed as analogies.

      case-record -- the note plus the verbatim worked record (default)
      case        -- the note plus a compact exemplar, the analogue of Memp's `script`
    """
    if granularity not in ("case-record", "case"):
        raise ValueError(f"unknown granularity {granularity!r}")
    return _render(
        entries, granularity,
        "\n\nDeclarative memory: specific past questions similar to this one, from "
        "questions you answered before. Use them only as loose analogies for the "
        "kind of chain and phrasing involved -- not as instructions to follow, and "
        "never as a source for this question's answer.",
        label="Case", field="case_note",
        abstract_at=("case-record", "case"),
        exemplar_at=("case",))



# --------------------------------------------------------------------------- #
# Contrastive memory. Single-turn, so there are no broken rules to quote; a
# failure is the wrong answer plus its reasoning.
# --------------------------------------------------------------------------- #
_CONTRASTIVE_HEADER = (
    "\n\nMemory from previous questions with a similar structure. Each item is "
    "either something that WORKED on a question you answered correctly, or "
    "something that FAILED on one you got wrong. Read the label before the text: "
    "the failures are there to be avoided, not followed. They come from OTHER "
    "questions -- never use one as a source for this question's answer."
)
_POSITIVE_ONLY_HEADER = (
    "\n\nMemory from previous questions with a similar structure: strategies that "
    "WORKED on questions you answered correctly. They come from OTHER questions -- "
    "never use one as a source for this question's answer."
)
_AVOID_ONLY_HEADER = (
    "\n\nMemory from previous questions with a similar structure: mistakes that made "
    "attempts FAIL. They are there to be avoided, not followed. They come from "
    "OTHER questions -- never use one as a source for this question's answer."
)

# For `hindsight` under matched framing: every item renders as a DO (see prompts._ALL_DO_HEADER).
_ALL_DO_HEADER = (
    "\n\nMemory from previous questions with a similar structure: strategies for "
    "answering questions of this kind. They come from OTHER questions -- never use one "
    "as a source for this question's answer."
)

_DO = "DO -- this worked on a question like this one:"
_DONT = "DO NOT -- this is how an attempt at a question like this one failed:"
_WITHHELD = "  (nothing recorded for this question under this arm)"


def format_failed_exemplar(exemplar: dict) -> str:
    """A failed attempt's item shape. The correct answer must stay out of a DON'T."""
    return (f"That attempt faced a {exemplar.get('hops', '?')}-hop chain over "
            f"{exemplar.get('sentences', '?')} sentences "
            f"({exemplar.get('unused', 0)} not needed); "
            f"relations {', '.join(exemplar.get('relations', [])) or 'none'}")


def format_contrastive(entries, granularity: str = "avoid", exemplars: bool = False,
                       framing: str = "mixed") -> str:
    """The injected block at contrastive eval time; same arms as on SPaRC."""
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
            shown_as_do = positive or granularity == "hindsight"
            body.append(_DO if shown_as_do else _DONT)
            body.append(text.strip())
            if exemplars:
                ex = entry.get("exemplar" if shown_as_do else "failed_exemplar")
                if ex:
                    body.append(format_exemplar(ex) if shown_as_do
                                else format_failed_exemplar(ex))
        parts.append("\n".join(body))
    return "\n".join(parts)


def stepgame_avoid_prompt(episode, key_text: str, record, rules=()) -> str:
    """Distil one reusable DON'T from one wrong answer.

    The gold label must never appear here (the `hindsight` arm is the one that
    uses gold). `rules` is always empty; it exists to match the Benchmark signature.
    """
    answered = (episode.transitions[0].action if episode.transitions else None)
    return f"""{relations_preamble(artifact="warning")}
    You are an expert spatial-reasoning agent building a memory of MISTAKES:
    reusable, high-level warnings distilled from attempts that got the answer wrong.

    The attempt below answered {answered or 'nothing parseable'} and that was
    INCORRECT. Distil what its reasoning got wrong into ONE reusable warning for
    questions with this structure. Give 2-3 numbered points, each naming a mistake
    to avoid and what to do instead. At most 80 words.

    Do not state or guess what the correct answer to this question is -- you have
    not been told it, and a guess would poison the memory. Do not output an answer
    line. Describe the error, not the answer.

    Question structure:
    {key_text}

{_story_block(record)}

    What the attempt said:
    {episode.last_obs}

    Warning:"""
