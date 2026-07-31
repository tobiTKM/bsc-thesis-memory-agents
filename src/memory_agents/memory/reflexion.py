"""Reflexion memory: per-puzzle verbal reflections, and the model call that
produces them.

Split into two pieces so the generic parts stay reusable:
* ReflexionMemory -- a Memory (keyed by puzzle id) that other methods' stores can
  be modelled on.
* Reflector -- the Reflexion-specific self-reflection step: one failed episode ->
  one short lesson. A different method (MemoryBank, MemP) swaps this writer, not
  the store or the MemoryAgent that reads it.
"""

from ..prompts import reflexion_reflection_prompt
from .base import Memory


class ReflexionMemory(Memory):
    """Puzzle id -> list of reflections, newest kept, capped like the paper.

    The cap bounds how many past reflections ride along on a retry (Reflexion
    keeps the last few). For the 2-trial setup there is at most one per puzzle, so
    the cap only bites once we loop to trial 3+.
    """

    def __init__(self, cap: int = 3):
        self._store: dict[str, list[str]] = {}
        self.cap = cap

    def read(self, key) -> list[str]:
        return list(self._store.get(str(key), []))

    def write(self, key, item: str) -> None:
        items = self._store.setdefault(str(key), [])
        items.append(item)
        if self.cap and len(items) > self.cap:
            del items[0]  # drop the oldest; keep the most recent `cap`

    def to_dict(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._store.items()}

    def __contains__(self, key) -> bool:
        return str(key) in self._store

    def __len__(self) -> int:
        return len(self._store)

    @classmethod
    def from_dict(cls, data: dict[str, list[str]], cap: int = 3) -> "ReflexionMemory":
        """Rebuild a store from its serialised form.

        `cap` has to be passed back in: writing the items through write() re-applies
        the cap, so restoring a store built with a larger cap through the default
        would silently drop the oldest reflections it was meant to keep.
        """
        m = cls(cap=cap)
        for key, items in data.items():
            for item in items:
                m.write(key, item)
        return m


class Reflector:
    """Generates one verbal reflection from a failed episode via the LLM.

    Holds its own client so the driver can give reflection a different budget
    (usually a smaller max_tokens) than the actor without coupling the two.
    """

    def __init__(self, client):
        self.client = client

    async def reflect(self, episode, previous=()):
        """Return (reflection_text, LLMResponse) for a failed episode.

        `previous` is what this puzzle's memory already holds, so the reflection for
        trial t is written knowing which plans trials 2..t-1 already tried and failed
        with (as in the paper's implementation).

        The text may be empty if the model returned nothing parseable (e.g. a
        reasoning model that spent its whole budget in <think>); the caller
        decides whether to store it. The LLMResponse is handed back so the driver
        can tally reflection-generation tokens as part of Reflexion's cost.
        """
        prompt = reflexion_reflection_prompt(episode, previous=previous)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response
