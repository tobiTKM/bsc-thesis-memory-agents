"""The baseline actor with a retrieved-memory block appended to its system prompt."""

from .base import BaselineAgent


class MemoryAgent(BaselineAgent):
    """BaselineAgent with a memory block appended to the system message at reset.

    _trim keeps messages[:1], so the block lasts the whole episode; an empty block
    behaves exactly like the baseline.
    """

    def reset(self, system_prompt: str, memory_block: str = "", label=None) -> None:
        super().reset(system_prompt, label=label)
        if memory_block:
            self.messages[0]["content"] += memory_block
