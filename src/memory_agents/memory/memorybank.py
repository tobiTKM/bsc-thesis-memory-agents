"""MemoryBank declarative memory: a retrieval bank of past CASES, and the model
calls that write them.

MemoryBank (Zhong et al., AAAI 2024) gives an LLM long-term memory over past
interactions: each interaction is stored in depth (the conversation verbatim) and
as an LLM-written event summary, and the relevant ones are recalled for the
current context by dense retrieval. Ported to SPaRC, one past *puzzle* is one past
interaction (docs/architecture.md section 5.2, docs/memorybank.md):

* MemoryBankMemory -- the store. A straight subclass of MempMemory: the retrieval
  half of the two methods is deliberately IDENTICAL (same key text, same cosine
  top-k, same tie-break, same random-arm seeding), so a MemoryBank-vs-Memp
  comparison isolates what the memory *says* rather than how it is found. Only the
  schema tag and the entry's text field differ.
* CaseDistiller -- the writer: one solved-puzzle context -> one case note (the
  paper's event summary). Two variants for the bank's two provenances, exactly as
  Memp's Distiller has: the agent's own successful train trajectories, and the
  dataset's gold path in hindsight for train puzzles the agent failed.

What the paper has and this port drops -- the Ebbinghaus forgetting curve and
memory-strength updating, the user portrait, and the global summary -- is argued
in docs/architecture.md section 5.2; in short, a frozen bank built by a single
parallel pass has no repeat-recall event for decay to act on, SPaRC has no user,
and a global summary would inject identical text for every puzzle.

The verbatim record (`trajectory`) is NOT written here: the distill phase attaches
it by replaying the entry's own solution through a real env, so it is correct by
construction. See scripts/run_memorybank.py.
"""

from ..prompts import (memorybank_case_gold_prompt,
                       memorybank_case_trajectory_prompt)
from .memp import MempMemory

SCHEMA = "memorybank-bank-v1"


class MemoryBankMemory(MempMemory):
    """A flat bank of case entries with embedding-based retrieval.

    Inherits every retrieval mechanism from MempMemory unchanged
    (attach_vectors/retrieve/random_retrieve/fixed_retrieve/filtered, and the
    (-similarity, id) tie-break that keeps injected blocks byte-identical across
    resumes). The differences are exactly two, and both are about content:

      * SCHEMA -- so a Memp bank handed to the MemoryBank eval fails loudly at
        load time instead of half-working, and vice versa.
      * entries carry `case_note` where a Memp entry carries `script`.

    Entries also carry `trajectory` (the replayed verbatim record) and `exemplar`
    (the compact action list), both assembled in code by the distill phase.
    """

    SCHEMA = SCHEMA

    def note_of(self, entry_id: str) -> str:
        """The case note of one entry, by id (the sibling of Memp's _script_of)."""
        for e in self.entries:
            if e["id"] == entry_id:
                return e["case_note"]
        raise KeyError(entry_id)


class CaseDistiller:
    """Generates one case note from one solved-puzzle context via the LLM.

    Same shape and contract as Memp's Distiller -- holds its own client so the
    driver can give note-writing its own token budget, and returns
    (text, LLMResponse) so those tokens are tallied as MemoryBank's bank
    construction cost -- but prompts for a recollection instead of a procedure.

    `rules_preamble` selects which of the two banks this distiller builds (see
    memorybank_case_trajectory_prompt): with the rulebook prepended, matching the
    v2 Memp banks, or without it, which is closer to the paper's plain
    "summarize the events and key information" instruction. It is frozen into the
    bank's config by the driver so the two can never be mixed.
    """

    def __init__(self, client, rules_preamble: bool = True):
        self.client = client
        self.rules_preamble = rules_preamble

    async def distill_from_trajectory(self, episode, key_text: str, polyshapes=None):
        """Case note from the agent's own solved episode."""
        prompt = memorybank_case_trajectory_prompt(
            episode, key_text, polyshapes, rules_preamble=self.rules_preamble)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response

    async def distill_from_gold(self, key_text: str, text_visualization: str,
                                gold_actions: list, polyshapes=None):
        """Case note from the dataset's gold path (hindsight, for failed puzzles)."""
        prompt = memorybank_case_gold_prompt(
            key_text, text_visualization, gold_actions, polyshapes,
            rules_preamble=self.rules_preamble)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response
