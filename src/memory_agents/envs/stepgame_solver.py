"""An exact symbolic solver for StepGame -- the oracle, the verifier, and the
source of hindsight reasoning chains.

This is a pure-Python reimplementation of Li, Hogg & Cohn's `Map + ASP` solution:
map every story sentence to a spatial relation with the rectified template table,
then chain the relations' unit offsets into coordinates and read off the sign of
the query pair's displacement. Their paper reports 100% on every k of the
rectified benchmark; `scripts/verify_stepgame_solver.py` re-checks that claim
against the whole corpus on every run, so a drift between the table and the
stories fails loudly instead of quietly poisoning the memories built from it.

Reimplemented rather than reused because their version drives clingo: an ASP
grounder is a heavyweight dependency for what is, once the relations are
extracted, a breadth-first walk over a graph with <= 18 nodes. Dropping it keeps
the driver's environment free of anything but the stdlib.

Three jobs, all of which the memory methods depend on:

* `solve` is the oracle -- an upper bound to report, and the check that our label
  conventions match the dataset's.
* `verify` grades a whole file, which is the corpus-integrity gate.
* `gold_chain` renders the worked derivation for an item. That is the StepGame
  analogue of SPaRC's gold solution path: the hindsight fallback that lets MemP
  and MemoryBank build a bank entry for a train item the agent itself failed,
  instead of a bank whose size is confounded with the model's own accuracy.
"""

import re

from .stepgame_templates import load_templates

# The nine labels StepGame answers with, spelled exactly as the corpus spells
# them. Used verbatim as the actor's output contract too, so grading needs no
# translation layer between what the model writes and what the dataset stores.
LABELS = ("above", "below", "left", "right",
          "upper-left", "upper-right", "lower-left", "lower-right", "overlap")

# Unit offset per label, in a y-grows-upward frame. `a <label> b` means a sits at
# b + OFFSET[label]. Matches the `offset(...)` facts in the upstream ASP module.
OFFSETS: dict[str, tuple[int, int]] = {
    "overlap": (0, 0),
    "above": (0, 1), "below": (0, -1), "left": (-1, 0), "right": (1, 0),
    "upper-left": (-1, 1), "upper-right": (1, 1),
    "lower-left": (-1, -1), "lower-right": (1, -1),
}
_BY_OFFSET = {off: label for label, off in OFFSETS.items()}

# The relation tokens the template table emits ("A_upperright_B") -> our labels.
# The two vocabularies differ because the table's tokens are the original
# StepGame generator's, and the corpus's gold strings are not.
_TOKEN_TO_LABEL = {
    "left": "left", "right": "right", "over": "above", "below": "below",
    "upperleft": "upper-left", "upperright": "upper-right",
    "lowerleft": "lower-left", "lowerright": "lower-right",
}

# "<a>_<rel>_<b>" as produced by expanding a template's replacement.
_TRIPLE = re.compile(r"^(?P<a>[^_]+)_(?P<rel>[^_]+)_(?P<b>[^_]+)$")

QUERY_TOKEN = "query"


class SolverError(ValueError):
    """A story this solver cannot decode. Never swallowed: an item the oracle
    cannot read is an item whose gold label we cannot trust, so it is dropped
    from a split with a count rather than guessed at."""


def _expand(sentence: str) -> str | None:
    """Apply the first matching template, yielding "<a>_<rel>_<b>" or None."""
    for pattern, replacement in load_templates():
        match = pattern.match(sentence.strip())
        if match:
            return match.expand(replacement)
    return None


def parse_sentence(sentence: str) -> tuple[str, str, str]:
    """A story sentence -> (a, label, b), read as "a is <label> of b"."""
    expanded = _expand(sentence)
    if expanded is None:
        raise SolverError(f"no template matches: {sentence!r}")
    m = _TRIPLE.match(expanded)
    if not m:
        raise SolverError(f"malformed expansion {expanded!r} from {sentence!r}")
    token = m["rel"]
    if token not in _TOKEN_TO_LABEL:
        raise SolverError(f"unknown relation token {token!r} in {sentence!r}")
    return m["a"], _TOKEN_TO_LABEL[token], m["b"]


def parse_question(question: str) -> tuple[str, str]:
    """The question -> (target, anchor): "relation of TARGET to ANCHOR"."""
    expanded = _expand(question)
    if expanded is None:
        raise SolverError(f"no template matches question: {question!r}")
    m = _TRIPLE.match(expanded)
    if not m or m["rel"] != QUERY_TOKEN:
        raise SolverError(f"not a query expansion: {expanded!r}")
    return m["a"], m["b"]


def _adjacency(edges) -> dict[str, list[tuple[str, int, int, int]]]:
    """Undirected offset graph. Each edge is walkable both ways -- a story states
    "J is above Z" but a chain may need to travel Z -> J -- so both directions are
    stored, the reverse with a negated offset. The trailing index says which story
    sentence the hop came from, which is what lets gold_chain quote it."""
    adj: dict[str, list[tuple[str, int, int, int]]] = {}
    for i, (a, label, b) in enumerate(edges):
        dx, dy = OFFSETS[label]
        adj.setdefault(b, []).append((a, dx, dy, i))
        adj.setdefault(a, []).append((b, -dx, -dy, i))
    return adj


def _walk(edges, anchor: str):
    """BFS from `anchor`, returning {entity: (x, y)} and {entity: (parent, edge)}.

    Breadth-first rather than depth-first so the recovered chain is the *shortest*
    one: with `supporting` noise a story can offer several routes between the two
    queried entities, and the shortest is both the one a solver should explain and
    the one whose length equals the item's true hop count.

    Overlapping entities are why coordinates are accumulated instead of relations
    being composed pairwise: StepGame samples relations without constraint, so two
    entities can land on the same cell, and only a coordinate frame represents that.
    """
    adj = _adjacency(edges)
    location = {anchor: (0, 0)}
    parent: dict[str, tuple[str, int]] = {}
    queue = [anchor]
    while queue:
        node = queue.pop(0)
        x, y = location[node]
        for neighbour, dx, dy, edge_index in adj.get(node, ()):
            if neighbour in location:
                continue
            location[neighbour] = (x + dx, y + dy)
            parent[neighbour] = (node, edge_index)
            queue.append(neighbour)
    return location, parent


def label_of(dx: int, dy: int) -> str:
    """The label naming the *direction* of a displacement.

    Only the sign of each component matters: an entity three cells up and one
    right of another is "upper-right", exactly as the ASP module's answer rule
    reads off the direction rather than the magnitude.
    """
    sign = ((dx > 0) - (dx < 0), (dy > 0) - (dy < 0))
    return _BY_OFFSET[sign]


def edges_of(story) -> list[tuple[str, str, str]]:
    """Every sentence of a story as an (a, label, b) triple."""
    return [parse_sentence(s) for s in story]


def solve(story, question: str) -> str:
    """The oracle answer for one item."""
    target, anchor = parse_question(question)
    location, _ = _walk(edges_of(story), anchor)
    if target not in location:
        # Only reachable if the story fails to connect the queried pair, which
        # the generator never produces -- so this means a parse went wrong.
        raise SolverError(f"{target!r} unreachable from {anchor!r}")
    return label_of(*location[target])


def hop_count(story, question: str) -> int:
    """Length of the shortest chain between the queried entities.

    On a clean item this is the item's k. On a noisy one it is the number of
    sentences that actually matter, which is the honest difficulty measure when
    the story also carries distractors.
    """
    target, anchor = parse_question(question)
    _, parent = _walk(edges_of(story), anchor)
    hops, node = 0, target
    while node in parent:
        node, _ = parent[node]
        hops += 1
    return hops


def gold_chain(story, question: str) -> str:
    """The worked derivation, as hindsight text for a bank entry.

    Shows only the hops on the shortest path and quotes the sentence each came
    from, so a distractor-laden story yields the same clean chain a clean one
    does -- the point of the trace is the reasoning, not the search. Coordinates
    are carried explicitly because on StepGame the coordinate frame *is* the
    method (unlike SPaRC, where a procedure quoting coordinates was overfitting).
    """
    target, anchor = parse_question(question)
    edges = edges_of(story)
    location, parent = _walk(edges, anchor)
    if target not in location:
        raise SolverError(f"{target!r} unreachable from {anchor!r}")

    # Rebuild target -> anchor through the parent links, then walk it forwards.
    path, node = [], target
    while node in parent:
        previous, edge_index = parent[node]
        path.append((previous, node, edge_index))
        node = previous
    path.reverse()

    lines = [f"Question: where is {target} relative to {anchor}?",
             f"Anchor {anchor} at (0, 0)."]
    for i, (previous, node, edge_index) in enumerate(path, start=1):
        a, label, b = edges[edge_index]
        px, py = location[previous]
        nx, ny = location[node]
        lines.append(
            f'Hop {i}: "{story[edge_index].strip()}"'
            f' => {a} is {label} of {b}'
            f' => {node} = ({px}, {py}) + ({nx - px}, {ny - py}) = ({nx}, {ny})'
        )
    tx, ty = location[target]
    lines.append(f"{target} is at ({tx}, {ty}) and {anchor} at (0, 0); "
                 f"the direction of ({tx}, {ty}) is {label_of(tx, ty)}.")
    lines.append(f"Answer: {label_of(tx, ty)}")
    return "\n".join(lines)


def verify(items) -> dict:
    """Grade the oracle against stored gold labels over a list of StepGameItems.

    Returns counts rather than raising: the caller decides whether a non-perfect
    score is fatal (it is, for the corpus gate) or informative.
    """
    correct = wrong = unparsed = 0
    failures = []
    for item in items:
        try:
            predicted = solve(item.story, item.question)
        except SolverError as exc:
            unparsed += 1
            if len(failures) < 20:
                failures.append((item.id, repr(exc)))
            continue
        if predicted == item.gold:
            correct += 1
        else:
            wrong += 1
            if len(failures) < 20:
                failures.append((item.id, f"solver={predicted} gold={item.gold}"))
    return {"n": len(items), "correct": correct, "wrong": wrong,
            "unparsed": unparsed, "failures": failures}
