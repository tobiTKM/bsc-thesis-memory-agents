"""StepGame: loading, rendering, answer parsing and retrieval keys.

StepGame (Shi, Zhang & Lipani, AAAI-22) is single-turn: a story of k sentences
describing spatial relations among k+1 entities, a question about the relative
position of two of them, and one of nine labels as the answer. There is no
environment to step and no state to track across turns, so -- unlike SPaRC --
this module has no Env class. It supplies the pieces a single-turn benchmark
needs: the corpus, the user message, the reply parser, and the retrieval key.

We read the *rectified* corpus (see scripts/fetch_stepgame.py). Items are
addressed by a stable id that encodes everything needed to find them again,
`<variant>/<split>/k<k>/<index>`, with the index being the item's position in the
rectified file at the pinned commit.
"""

import json
import pathlib
import re
from dataclasses import dataclass, field

from .stepgame_solver import LABELS
from .stepgame_templates import data_dir

VARIANTS = ("clean", "noise")
SPLITS = ("train", "valid", "test")
KS = tuple(range(1, 11))


@dataclass
class StepGameItem:
    """One story-question-answer triple."""
    id: str
    variant: str          # clean | noise
    split: str            # train | valid | test
    k: int                # the file's hop count; on `noise` the story is longer
    story: list[str]
    question: str
    gold: str             # one of LABELS, verbatim from the corpus
    # Filled by build_stepgame_splits from the solver, so an eval never has to
    # re-derive it and a bank entry can quote it without importing the solver.
    gold_chain: str = ""
    hops: int = -1
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "variant": self.variant, "split": self.split,
                "k": self.k, "story": self.story, "question": self.question,
                "gold": self.gold, "gold_chain": self.gold_chain,
                "hops": self.hops, **({"extra": self.extra} if self.extra else {})}

    @classmethod
    def from_dict(cls, d: dict) -> "StepGameItem":
        return cls(id=d["id"], variant=d["variant"], split=d["split"], k=d["k"],
                   story=list(d["story"]), question=d["question"], gold=d["gold"],
                   gold_chain=d.get("gold_chain", ""), hops=d.get("hops", -1),
                   extra=d.get("extra", {}))


# ---- corpus loading ------------------------------------------------------

def corpus_path(variant: str, split: str, k: int) -> pathlib.Path:
    return data_dir() / "raw" / variant / f"qa{k}_{split}.txt"


def read_babi(path, variant: str, split: str, k: int) -> list[StepGameItem]:
    """Parse one bAbI-format StepGame file.

    Format: each line is "<n> <text>", numbering restarts at 1 for every item,
    and the item's last line is the question with the gold label and the
    supporting-fact index appended after tabs. Items are split on the line number
    resetting to 1 rather than on blank lines, because the files have none.
    """
    items: list[StepGameItem] = []
    lines: list[str] = []

    def flush() -> None:
        if not lines:
            return
        *story, last = lines
        parts = last.split("\t")
        if len(parts) < 2:
            raise ValueError(f"{path}: item {len(items)} has no gold label")
        question, gold = parts[0].strip(), parts[1].strip()
        if gold not in LABELS:
            raise ValueError(f"{path}: unknown label {gold!r} "
                             f"(expected one of {LABELS})")
        items.append(StepGameItem(
            id=f"{variant}/{split}/k{k}/{len(items):05d}",
            variant=variant, split=split, k=k,
            story=[s.strip() for s in story], question=question, gold=gold))

    for raw in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        number, _, text = raw.partition(" ")
        if number == "1" and lines:
            flush()
            lines = []
        lines.append(text)
    flush()
    return items


def load_items(variant: str = "clean", split: str = "test",
               ks=KS) -> list[StepGameItem]:
    """Every item of a variant/split across the given hop counts."""
    items: list[StepGameItem] = []
    for k in ks:
        path = corpus_path(variant, split, k)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/fetch_stepgame.py "
                f"--variants {variant} --splits {split}`.")
        items.extend(read_babi(path, variant, split, k))
    return items


def read_split(path) -> list[StepGameItem]:
    """Load a frozen split written by scripts/build_stepgame_splits.py (JSONL)."""
    return [StepGameItem.from_dict(json.loads(line))
            for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


# ---- rendering -----------------------------------------------------------

def format_item(item: StepGameItem) -> str:
    """The per-item payload the model sees.

    Sentences are numbered so a reflection or a stored chain can refer to "hop 3"
    and mean something the actor can look up. Not JSON, unlike SPaRC's per-step
    message: SPaRC packs a grid plus env metadata, whereas this is prose, and
    wrapping prose in JSON would only cost the model escaping noise.
    """
    story = "\n".join(f"{i}. {s}" for i, s in enumerate(item.story, start=1))
    return f"Story:\n{story}\n\nQuestion: {item.question}"


# ---- answer parsing ------------------------------------------------------

# Surface forms the actor might use for each label. Generous on purpose: a reply
# that says "north-west" has reasoned correctly and grading it wrong would
# measure instruction-following, not spatial reasoning. The I/O contract still
# asks for the canonical spelling -- this is the safety net, not the interface.
_SYNONYMS: dict[str, str] = {}


def _register(label: str, *forms: str) -> None:
    for form in (label, *forms):
        _SYNONYMS[form] = label


_register("above", "top", "up", "north", "over", "upward", "directly above")
_register("below", "bottom", "down", "south", "under", "beneath", "downward",
          "directly below")
_register("left", "west", "directly left", "to the left")
_register("right", "east", "directly right", "to the right")
_register("upper-left", "upper left", "top left", "top-left", "north west",
          "northwest", "north-west", "above left")
_register("upper-right", "upper right", "top right", "top-right", "north east",
          "northeast", "north-east", "above right")
_register("lower-left", "lower left", "bottom left", "bottom-left", "down left",
          "down-left", "south west", "southwest", "south-west", "below left")
_register("lower-right", "lower right", "bottom right", "bottom-right",
          "down right", "down-right", "south east", "southeast", "south-east",
          "below right")
_register("overlap", "overlapping", "same position", "same place",
          "same location", "coincide", "coincident", "on top of each other")

# Longest first so "upper right" wins over the "right" nested inside it.
_ALTERNATION = "|".join(re.escape(form) for form in
                        sorted(_SYNONYMS, key=len, reverse=True))
_LABEL_RE = re.compile(rf"\b({_ALTERNATION})\b")

# A "Final:" line anywhere in the reply, mirroring SPaRC's parser: markdown bold
# and trailing punctuation tolerated, and the tail of the line is what gets
# searched for a label so "Final: **the answer is upper-right**" still parses.
_FINAL_RE = re.compile(r"\bfinal\b\s*:?\s*\**\s*(.+)", re.IGNORECASE)


def _normalise(text: str) -> str:
    """Lowercase, and collapse hyphens/underscores/whitespace to single spaces."""
    return re.sub(r"[\s\-_]+", " ", text.lower()).strip()


def _label_in(text: str) -> str | None:
    match = _LABEL_RE.search(_normalise(text))
    return _SYNONYMS[match.group(1)] if match else None


def parse_answer(reply: str) -> str | None:
    """Parse a model reply into one of LABELS, or None on a format failure.

    None means *malformed output*, not a wrong answer: the harness retries the
    call rather than counting an attempt, exactly as it does for SPaRC. Keeping
    the two apart matters more here than there -- with nine labels and a ~11%
    chance floor, silently scoring an unparseable reply as wrong would fold
    format failures into the accuracy number.

    The LAST "Final:" marker wins, because a thinking model floats candidate
    answers while reasoning and only the closing one is its conclusion.
    """
    if not reply:
        return None
    finals = _FINAL_RE.findall(reply)
    for tail in reversed(finals):
        label = _label_in(tail)
        if label is not None:
            return label
    # Fallback: the reply's last non-empty line is just the answer. Deliberately
    # narrower than scanning the whole reply, which would happily pick up a
    # direction the model mentioned mid-deduction and score it as the answer.
    lines = [ln for ln in reply.strip().splitlines() if ln.strip()]
    return _label_in(lines[-1]) if lines else None


def is_success(predicted: str | None, gold: str) -> bool:
    return predicted is not None and predicted == gold


# ---- retrieval keys ------------------------------------------------------
# Used verbatim for BOTH the train keys and the test queries; that symmetry is
# what makes the cosine similarity meaningful. Frozen once a bank exists -- bump
# the version and re-embed rather than editing.

KEY_FN = "stepgame-key-v1"
KEY_FN_LEXICAL = "stepgame-key-lex-v1"

_ENTITY_RE = re.compile(r"\b([A-Z])\b")


def stepgame_key_text(item: StepGameItem, edges=None, hops: int | None = None) -> str:
    """Structural key: what shape of problem this is, with the surface stripped.

    Entity letters are excluded because they are sampled uniformly from 26
    capitals and carry no information -- keying on them would make retrieval
    match on noise. Template wording is excluded for the same reason at the
    phrasing level; `stepgame_key_text_lexical` is the ablation that puts it back.

    What remains is the reasoning structure: how many hops, how many distractor
    sentences, the ordered relation chain, and a sorted census. The census is
    order-invariant and the chain is not, so the two together let similarity see
    both "same relations involved" and "same sequence of turns".
    """
    from .stepgame_solver import edges_of, hop_count      # local: avoids a cycle
    if edges is None:
        edges = edges_of(item.story)
    if hops is None:
        hops = hop_count(item.story, item.question)
    labels = [label for _a, label, _b in edges]
    census: dict[str, int] = {}
    for label in labels:
        census[label] = census.get(label, 0) + 1
    return "\n".join([
        # "unused", not "distractors": on a noisy item the surplus sentences are
        # injected distractors, but on a clean one they are ordinary chain
        # sentences that simply do not lie between the two queried entities.
        # Both are sentences the answer does not depend on, which is the thing
        # retrieval should be able to see.
        f"hops {hops}; sentences {len(item.story)}; "
        f"unused {max(len(item.story) - hops, 0)}",
        "chain: " + (", ".join(labels) or "none"),
        "census: " + ", ".join(f"{label} x{n}" for label, n in sorted(census.items())),
    ])


def stepgame_key_text_lexical(item: StepGameItem) -> str:
    """Lexical key: the story with entities anonymised, wording kept.

    The ablation against the structural key. It exists because StepGame's
    difficulty is not purely structural -- Li et al. traced most residual parser
    error to a handful of oblique phrasings ("if X is the centre of a clock face,
    Y is between 4 and 5"), so which *templates* an item uses is plausibly the
    more useful retrieval signal. Entities are renamed by order of first
    appearance, so two items phrased identically key identically.
    """
    mapping: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        letter = match.group(1)
        if letter not in mapping:
            mapping[letter] = f"E{len(mapping)}"
        return mapping[letter]

    lines = [_ENTITY_RE.sub(rename, s) for s in item.story]
    lines.append(_ENTITY_RE.sub(rename, item.question))
    return "\n".join(lines)
