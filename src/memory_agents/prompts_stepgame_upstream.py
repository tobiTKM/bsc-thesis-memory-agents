"""Li et al.'s own StepGame actor prompt, as a *calibration* protocol.

Our actor prompt (`prompts_stepgame.stepgame_system_prompt`) is zero-shot and
deliberately lean, because few-shot exemplars are themselves injected memory and
putting them in the memoryless baseline would contaminate the one contrast this
project exists to measure. That is the right protocol for the thesis, but it
makes our numbers incomparable to the published StepGame table -- and in the
*unfavourable* direction, which is easy to get wrong when reading them:

  their prompt        | ours
  --------------------|-----------------------------------------------
  clock-face decoder  | absent   (44% of our test items use clock phrasing)
  cardinal decoder    | absent   (22% do; 55% use one or the other)
  10 worked examples  | absent   (zero-shot)
  --------------------|-----------------------------------------------
  direction convention stated | only ours states it; theirs demonstrates
  `overlap` explained         | it through the examples instead
  "only sign matters"         |

So this module exists to run the *same frozen split* under their prompt, as a
side-by-side calibration point rather than as a replacement. `--prompt-protocol`
selects between them; `ours` stays the default and the thesis protocol.

Three protocols, because the two things their prompt adds are worth separating:

  ours              our zero-shot rules paragraph (default; the thesis protocol)
  upstream-header   their instruction paragraph ONLY -- the clock and cardinal
                    decoding tables, no examples. Still zero-shot, so it remains
                    a legitimate memoryless baseline: it is task specification,
                    not a worked solution. This is the honest comparability fix.
  upstream          their prompt entire, header + 10 worked examples. Matches
                    their published protocol, and is NOT a memoryless baseline --
                    the exemplars are hand-written memory. Calibration only;
                    never use it as the control arm for a memory method.

The prompt text is *fetched*, not vendored -- same reason as the corpus and the
template table (no LICENSE upstream), and from the same pinned commit, so the
prompt and the stories it describes can never drift apart.

One deliberate deviation: our `Final: <label>` output contract is appended to
every protocol. Theirs is a completion-style prompt ending in "Answer:", which a
chat-served reasoning model does not honour. Holding the output contract fixed is
also what makes the comparison clean -- it isolates the prompt's *content*, and
means both protocols are graded by byte-identical code.
"""

import pathlib
import re

from .envs.stepgame_templates import data_dir
from .prompts_stepgame import _STEPGAME_IO_CONTRACT

PROTOCOLS = ("ours", "upstream-header", "upstream")

_PROMPT_FILE = "prompts/baseline_clean1_10.txt"

# Their file ends with the stub the driver substitutes into. It is the template,
# not an example, so it is stripped before the examples are counted.
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
    # Drop the trailing "[INPUT]\nAnswer:" template; it is not an exemplar.
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
    """The per-item payload in *their* shape: "Story:", unnumbered, then the question.

    Our own `format_item` numbers the sentences so a reflection or a stored chain
    can say "hop 3" and mean something lookup-able. Their exemplars show
    unnumbered stories, and an input formatted unlike the ten examples above it
    is a confound in a comparison whose whole point is to match their setup -- so
    the upstream protocols render items their way.
    """
    story = "\n".join(s.strip() for s in item.story)
    return f"Story:\n{story}\n{item.question}"
