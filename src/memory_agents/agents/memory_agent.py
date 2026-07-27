from .base import BaselineAgent


class MemoryAgent(BaselineAgent):
    """The baseline actor plus a persistent block of retrieved memory.

    Method-agnostic by design: Reflexion, MemoryBank and MemP all differ in *how*
    the memory block is produced (see the Memory ABC and each method's writer),
    not in how it is consumed. The block is folded into the system message once at
    reset; because BaselineAgent._trim keeps only messages[:1], it survives every
    step of the episode for free -- exactly like the plain system prompt. With an
    empty block this behaves identically to the memoryless baseline.
    """

    def reset(self, system_prompt: str, memory_block: str = "", label=None) -> None:
        super().reset(system_prompt, label=label)
        if memory_block:
            self.messages[0]["content"] += memory_block
