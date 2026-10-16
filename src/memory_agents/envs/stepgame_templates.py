"""Li, Hogg & Cohn's rectified StepGame sentence -> relation templates, from the fetched corpus.

Not vendored (upstream has no LICENSE); must come from the same commit as the stories.
Read with `ast.literal_eval`, never imported, so the downloaded file is not executed.
"""

import ast
import os
import pathlib
import re

# Key: sentence regex; value: backreference template giving "<a>_<rel>_<b>" (a is <rel> of b).
# Some templates swap entity order ("A is over there and B is on the left of it" -> B_left_A).
_DICT_NAME = "sentence_to_relation"
_FILENAME = "sentence_to_relation.py"

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

    Anchored with ^...$ (upstream is not), so a template cannot match inside a longer sentence.
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
