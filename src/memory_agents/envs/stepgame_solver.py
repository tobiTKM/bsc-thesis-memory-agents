"""Exact StepGame solver: oracle (`solve`), corpus check (`verify`), gold chains (`gold_chain`).

Stdlib reimplementation of Li, Hogg & Cohn's Map+ASP approach: map sentences to relations
via the rectified templates, accumulate unit offsets by BFS, read off the displacement's sign.
"""

import re

from .stepgame_templates import load_templates

# Spelled exactly as in the corpus; also the actor's output contract.
LABELS = ("above", "below", "left", "right",
          "upper-left", "upper-right", "lower-left", "lower-right", "overlap")

# y grows upward; `a <label> b` means a = b + OFFSETS[label]. Matches upstream ASP `offset/3`.
OFFSETS: dict[str, tuple[int, int]] = {
    "overlap": (0, 0),
    "above": (0, 1), "below": (0, -1), "left": (-1, 0), "right": (1, 0),
    "upper-left": (-1, 1), "upper-right": (1, 1),
    "lower-left": (-1, -1), "lower-right": (1, -1),
}
_BY_OFFSET = {off: label for label, off in OFFSETS.items()}

# Template-table relation tokens ("A_upperright_B") -> corpus labels; the vocabularies differ.
_TOKEN_TO_LABEL = {
    "left": "left", "right": "right", "over": "above", "below": "below",
    "upperleft": "upper-left", "upperright": "upper-right",
    "lowerleft": "lower-left", "lowerright": "lower-right",
}

# "<a>_<rel>_<b>" as produced by expanding a template's replacement.
_TRIPLE = re.compile(r"^(?P<a>[^_]+)_(?P<rel>[^_]+)_(?P<b>[^_]+)$")

QUERY_TOKEN = "query"


class SolverError(ValueError):
    """A story the solver cannot decode; such items are dropped from splits, not guessed."""


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
    """Undirected offset graph: entity -> [(neighbour, dx, dy, sentence index)].

    Reverse edges carry the negated offset.
    """
    adj: dict[str, list[tuple[str, int, int, int]]] = {}
    for i, (a, label, b) in enumerate(edges):
        dx, dy = OFFSETS[label]
        adj.setdefault(b, []).append((a, dx, dy, i))
        adj.setdefault(a, []).append((b, -dx, -dy, i))
    return adj


def _walk(edges, anchor: str):
    """BFS from `anchor`, returning {entity: (x, y)} and {entity: (parent, edge)}.

    BFS so the recovered chain is the shortest one when noise adds alternative routes.
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
    """The label for a displacement's direction (sign of each component only)."""
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
        # The generator always connects the queried pair, so this means a parse went wrong.
        raise SolverError(f"{target!r} unreachable from {anchor!r}")
    return label_of(*location[target])


def hop_count(story, question: str) -> int:
    """Reasoning depth: length of the shortest chain between the queried entities (not k)."""
    target, anchor = parse_question(question)
    _, parent = _walk(edges_of(story), anchor)
    hops, node = 0, target
    while node in parent:
        node, _ = parent[node]
        hops += 1
    return hops


def gold_chain(story, question: str) -> str:
    """Worked derivation along the shortest path, quoting each hop's sentence, for a bank entry."""
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
    """Grade the oracle against stored gold labels; returns counts, the caller decides if fatal."""
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
