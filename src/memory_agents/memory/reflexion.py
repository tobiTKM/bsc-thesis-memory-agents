"""Reflexion memory: per-puzzle verbal reflections (ReflexionMemory) and the
LLM call that writes one from a failed episode (Reflector).
"""

from ..prompts import reflexion_reflection_prompt
from .base import Memory


class ReflexionMemory(Memory):
    """Puzzle id -> list of reflections, oldest first, keeping the last `cap`."""

    def __init__(self, cap: int = 3):
        self._store: dict[str, list[str]] = {}
        self.cap = cap

    def read(self, key) -> list[str]:
        return list(self._store.get(str(key), []))

    def write(self, key, item: str) -> None:
        items = self._store.setdefault(str(key), [])
        items.append(item)
        if self.cap and len(items) > self.cap:
            del items[0]

    def to_dict(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._store.items()}

    def __contains__(self, key) -> bool:
        return str(key) in self._store

    def __len__(self) -> int:
        return len(self._store)

    @classmethod
    def from_dict(cls, data: dict[str, list[str]], cap: int = 3) -> "ReflexionMemory":
        """Rebuild a store; pass the original `cap`, since write() re-applies it."""
        m = cls(cap=cap)
        for key, items in data.items():
            for item in items:
                m.write(key, item)
        return m


class Reflector:
    """Generates one verbal reflection from a failed episode via its own client."""

    def __init__(self, client, benchmark=None):
        self.client = client
        # None falls back to the SPaRC reflection prompt.
        self.benchmark = benchmark

    async def reflect(self, episode, previous=()):
        """Return (reflection_text, LLMResponse); text may be empty.

        `previous` holds this puzzle's earlier reflections, so the new one is
        written knowing which plans already failed.
        """
        prompt = (self.benchmark.reflection_prompt(episode, previous=previous)
                  if self.benchmark is not None
                  else reflexion_reflection_prompt(episode, previous=previous))
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response
