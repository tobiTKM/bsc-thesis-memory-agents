"""Li et al.'s StepGame actor prompt (fetched, not vendored), run as a calibration protocol.

`upstream-header` uses their instructions only (zero-shot); `upstream` adds their 10 worked
examples and so is not a memoryless baseline. Our `Final: <label>` output contract is
appended to every protocol, since a chat model does not honour their trailing "Answer:".
"""

import pathlib
import re

from .envs.stepgame_templates import data_dir
from .prompts_stepgame import _STEPGAME_IO_CONTRACT

PROTOCOLS = ("ours", "upstream-header", "upstream")

_PROMPT_FILE = "prompts/baseline_clean1_10.txt"

# Template stub at the end of their file; stripped so it is not counted as an example.
_INPUT_STUB = "[INPUT]"

_cache: str | None = None


def prompt_path() -> pathlib.Path:
    return data_dir() / "raw" / _PROMPT_FILE


def load_prompt(refresh: bool = False) -> str:
    global _cache
    if _cache is not None and not refresh:
        return _cache
    p = prompt_path()
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python scripts/fetch_stepgame.py` "
            f"(it mirrors the prompt alongside the corpus).")
    _cache = p.read_text(encoding="utf-8")
    return _cache


def _split_prompt(text: str) -> tuple[str, list[str]]:
    """(header, examples). Blocks are separated by the literal "Story:" line."""
    parts = re.split(r"(?m)^Story:\s*$", text)
    header = parts[0].strip()
    examples = [b.strip() for b in parts[1:] if b.strip()]
    examples = [b for b in examples if _INPUT_STUB not in b]
    return header, examples


def upstream_header() -> str:
    """Their instruction paragraph: label list + clock and cardinal decoders."""
    return _split_prompt(load_prompt())[0]


def upstream_examples() -> list[str]:
    """The 10 worked examples, one per k, each already ending in "Answer: ..."."""
    return _split_prompt(load_prompt())[1]


def upstream_system_prompt(shots: bool = True) -> str:
    """Their prompt as a chat system prompt, with our output contract appended.

    `shots=False` is the `upstream-header` protocol: decoders without exemplars.
    """
    header = upstream_header()
    if not shots:
        return header + "\n" + _STEPGAME_IO_CONTRACT
    blocks = "\n\n".join(f"Story:\n{b}" for b in upstream_examples())
    return f"{header}\n\n{blocks}\n" + _STEPGAME_IO_CONTRACT


def system_prompt_for(protocol: str) -> str:
    """Resolve a --prompt-protocol value to an actor system prompt."""
    if protocol == "ours":
        from .prompts_stepgame import stepgame_system_prompt
        return stepgame_system_prompt()
    if protocol == "upstream-header":
        return upstream_system_prompt(shots=False)
    if protocol == "upstream":
        return upstream_system_prompt(shots=True)
    raise ValueError(f"unknown prompt protocol {protocol!r} "
                     f"(expected one of {', '.join(PROTOCOLS)})")


def format_item_upstream(item) -> str:
    """Render an item like their examples: "Story:", unnumbered sentences, then the question."""
    story = "\n".join(s.strip() for s in item.story)
    return f"Story:\n{story}\n{item.question}"
