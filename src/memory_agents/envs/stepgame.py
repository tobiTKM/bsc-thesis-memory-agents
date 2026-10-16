"""StepGame (Shi, Zhang & Lipani, AAAI-22): loading, rendering, answer parsing, retrieval keys.

Single-turn, so there is no Env class. Reads the rectified corpus (scripts/fetch_stepgame.py);
ids are `<variant>/<split>/k<k>/<index>`, the index being the position in that file.
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
    k: int                # corpus k (story length on clean); not the reasoning depth, see `hops`
    story: list[str]
    question: str
    gold: str             # one of LABELS, verbatim from the corpus
    # Filled from the solver by build_stepgame_splits.
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
    """Parse one bAbI-format StepGame file ("<n> <text>" lines; gold after a tab on the question).

    Items are split where the line number resets to 1; the files have no blank lines.
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
    """The per-item payload the model sees, with numbered story sentences."""
    story = "\n".join(f"{i}. {s}" for i, s in enumerate(item.story, start=1))
    return f"Story:\n{story}\n\nQuestion: {item.question}"


# ---- answer parsing ------------------------------------------------------

# Surface forms accepted for each label; the prompt still asks for the canonical spelling.
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

# "Final:" anywhere in the reply (bold tolerated); the rest of the line is searched for a label.
_FINAL_RE = re.compile(r"\bfinal\b\s*:?\s*\**\s*(.+)", re.IGNORECASE)


def _normalise(text: str) -> str:
    """Lowercase, and collapse hyphens/underscores/whitespace to single spaces."""
    return re.sub(r"[\s\-_]+", " ", text.lower()).strip()


def _label_in(text: str) -> str | None:
    match = _LABEL_RE.search(_normalise(text))
    return _SYNONYMS[match.group(1)] if match else None


def parse_answer(reply: str) -> str | None:
    """Return the label from the last parseable "Final:" marker, or None on a format failure.

    None is retried by the harness, not scored as a wrong answer.
    """
    if not reply:
        return None
    finals = _FINAL_RE.findall(reply)
    for tail in reversed(finals):
        label = _label_in(tail)
        if label is not None:
            return label
    # Fallback: only the last line, so a direction mentioned mid-reasoning is not taken.
    lines = [ln for ln in reply.strip().splitlines() if ln.strip()]
    return _label_in(lines[-1]) if lines else None


def is_success(predicted: str | None, gold: str) -> bool:
    return predicted is not None and predicted == gold


# ---- retrieval keys ------------------------------------------------------
# Used for both train keys and test queries. Frozen: bump the version and re-embed instead.

KEY_FN = "stepgame-key-v1"
KEY_FN_LEXICAL = "stepgame-key-lex-v1"

_ENTITY_RE = re.compile(r"\b([A-Z])\b")


def stepgame_key_text(item: StepGameItem, edges=None, hops: int | None = None) -> str:
    """Structural key: hops, sentence counts, relation chain and census; no entities or wording."""
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
        # "unused", not "distractors": on clean items these are chain sentences off the query path.
        f"hops {hops}; sentences {len(item.story)}; "
        f"unused {max(len(item.story) - hops, 0)}",
        "chain: " + (", ".join(labels) or "none"),
        "census: " + ", ".join(f"{label} x{n}" for label, n in sorted(census.items())),
    ])


def stepgame_key_text_lexical(item: StepGameItem) -> str:
    """Lexical key (ablation): story and question, entities renamed by first appearance."""
    mapping: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        letter = match.group(1)
        if letter not in mapping:
            mapping[letter] = f"E{len(mapping)}"
        return mapping[letter]

    lines = [_ENTITY_RE.sub(rename, s) for s in item.story]
    lines.append(_ENTITY_RE.sub(rename, item.question))
    return "\n".join(lines)
