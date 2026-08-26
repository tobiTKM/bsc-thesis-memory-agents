"""The StepGame sentence -> relation template table, loaded from the fetched corpus.

The table is Li, Hogg & Cohn's rectified version of the 214 crowdsourced templates
Shi, Zhang & Lipani generated StepGame from: 14 of them expressed a relation other
than the label they were generated under, and this table is the corrected mapping.

It is *loaded*, not vendored, and deliberately so -- see scripts/fetch_stepgame.py
for why (upstream publishes no LICENSE). That also keeps the table and the corpus
pinned to the same commit: a template table from a different revision than the
stories it decodes would silently mis-parse a subset of them, which is precisely
the failure mode the rectified corpus exists to remove.

The upstream file is a single `sentence_to_relation = {...}` dict literal, read
here with `ast.literal_eval` rather than imported. Importing would execute an
unsigned file downloaded over the network for no benefit; literal_eval evaluates
nothing but the data.
"""

import ast
import os
import pathlib
import re

# Both sides are regex: the key matches a story sentence and names its two
# entities, the value is a backreference template expanding to "<a>_<rel>_<b>",
# read as "a is <rel> of b". Some templates invert the order they mention the
# entities in ("A is over there and B is on the left of it" -> B_left_A), which
# is why the value cannot simply be a relation name.
_DICT_NAME = "sentence_to_relation"
_FILENAME = "sentence_to_relation.py"

# Env override first so a SLURM job can point at a $WORK copy without a code
# change; otherwise the repo-local mirror the fetch script writes.
_ENV_VAR = "STEPGAME_DATA"

_cache: list[tuple[re.Pattern, str]] | None = None


def data_dir() -> pathlib.Path:
    """Where the fetched corpus lives (`data/stepgame` unless $STEPGAME_DATA)."""
    override = os.environ.get(_ENV_VAR)
    if override:
        return pathlib.Path(override)
    return pathlib.Path(__file__).resolve().parents[3] / "data/stepgame"


def template_path() -> pathlib.Path:
    return data_dir() / "raw" / _FILENAME


def _extract_dict(source: str) -> dict[str, str]:
    """Pull the `sentence_to_relation` literal out of the upstream module."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if _DICT_NAME in names:
            return ast.literal_eval(node.value)
    raise ValueError(f"no `{_DICT_NAME}` assignment in {_FILENAME}")


def load_templates(path=None, refresh: bool = False) -> list[tuple[re.Pattern, str]]:
    """Compiled (pattern, replacement) pairs, in upstream order.

    Anchored with ^...$ because the upstream patterns are not: unanchored, the
    `left` template "A is to the left of B." would also match inside "A is to the
    left of B and above C.", and whichever template happened to come first in the
    dict would win. Every StepGame sentence is exactly one relation, so a full
    match is the correct contract and a non-match is a real signal (it means the
    corpus and the table have drifted apart).
    """
    global _cache
    if _cache is not None and not refresh and path is None:
        return _cache

    p = pathlib.Path(path) if path else template_path()
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python scripts/fetch_stepgame.py` first "
            f"(or set ${_ENV_VAR} to a directory containing raw/{_FILENAME})."
        )
    compiled = [(re.compile("^" + pattern + "$"), replacement)
                for pattern, replacement in _extract_dict(p.read_text()).items()]
    if path is None:
        _cache = compiled
    return compiled
