"""MemoryBank declarative memory (Zhong et al., AAAI 2024), without the paper's
forgetting curve, user portrait and global summary. MemoryBankMemory must keep
MempMemory's retrieval identical (key text, top-k, tie-break, random seeding);
only SCHEMA and the text field differ. CaseDistiller writes the case notes.
"""

from ..prompts import (memorybank_case_gold_prompt,
                       memorybank_case_trajectory_prompt)
from .memp import MempMemory

SCHEMA = "memorybank-bank-v1"


class MemoryBankMemory(MempMemory):
    """Case entries: `case_note` (LLM), plus `trajectory` (replayed verbatim record)
    and `exemplar` (action list) added by the distill phase in methods/memorybank.py.
    Retrieval is inherited from MempMemory unchanged; SCHEMA makes loading a Memp
    bank fail.
    """

    SCHEMA = SCHEMA

    def note_of(self, entry_id: str) -> str:
        """The case note of one entry, by id."""
        for e in self.entries:
            if e["id"] == entry_id:
                return e["case_note"]
        raise KeyError(entry_id)


class CaseDistiller:
    """Generates one case note; same contract as Memp's Distiller.

    `rules_preamble` prepends the rulebook to the prompt; it is recorded in the
    bank's config so banks built with and without it are not mixed.
    """

    def __init__(self, client, rules_preamble: bool = True, benchmark=None):
        self.client = client
        self.rules_preamble = rules_preamble
        self.benchmark = benchmark

    def _bench(self):
        if self.benchmark is None:
            from ..benchmarks import SparcBenchmark
            self.benchmark = SparcBenchmark()
        return self.benchmark

    async def distill_from_trajectory(self, episode, key_text: str, record=None):
        """Case note from the agent's own solved episode."""
        prompt = self._bench().case_trajectory_prompt(
            episode, key_text, record, rules_preamble=self.rules_preamble)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response

    async def distill_from_gold(self, key_text: str, record=None):
        """Case note from the task's gold solution (hindsight, for failed items)."""
        prompt = self._bench().case_gold_prompt(
            key_text, record, rules_preamble=self.rules_preamble)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response
