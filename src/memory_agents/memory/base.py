"""The Memory interface shared by every memory method.

A Memory is a keyed store of text items the agent reads before acting and writes
after an episode. Reflexion keys by puzzle id (per-puzzle reflections); a
cross-task method would key by a single global key. Keeping this abstract is what
lets MemoryBank / MemP slot in behind the same harness without touching the loop.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path


class Memory(ABC):
    @abstractmethod
    def read(self, key: str) -> list[str]:
        """Items stored under `key`, oldest first (empty list if none)."""

    @abstractmethod
    def write(self, key: str, item: str) -> None:
        """Append one item under `key`."""

    @abstractmethod
    def to_dict(self) -> dict[str, list[str]]:
        """Plain {key: [items]} view for serialisation."""

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "Memory":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "Memory":
        raise NotImplementedError
