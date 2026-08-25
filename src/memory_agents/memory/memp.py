"""Memp procedural memory: a retrieval bank of distilled procedures, and the
model calls that write it.

Memp (Fang et al., 2026) turns past trajectories into procedural memory --
"how to solve this class of task" -- and retrieves the most similar entries for
each new task by embedding similarity. Mirroring reflexion.py's split:

* MempMemory -- the store: a flat bank of entries plus an aligned matrix of key
  embeddings, with cosine top-k retrieval. Built once from the train split,
  frozen at test time (docs/architecture.md section 5).
* Distiller -- the writer: one solved-puzzle context -> one script-like
  procedure. Two variants, because the bank's entries come from two sources:
  the agent's own successful train trajectories, and (for train puzzles the
  agent failed) the dataset's gold solution path in hindsight.

Vectors live in a sidecar .npz, not in the bank JSON: the bank is written by the
distill phase, which runs against the chat model and has no embedding server to
talk to. attach_vectors() joins the two files at eval time.
"""

import random

import numpy as np

from ..prompts import (memp_distill_gold_prompt, memp_distill_trajectory_prompt,
                       memp_keywords_prompt, memp_revise_prompt)
from .base import Memory

SCHEMA = "memp-bank-v1"


class MempMemory(Memory):
    """A flat bank of procedure entries with embedding-based retrieval.

    Unlike ReflexionMemory (keyed per puzzle, read back for the SAME puzzle), the
    bank is cross-task: entries are keyed by the *train* puzzle they came from
    (`source_pid`) but retrieved for unseen *test* puzzles by similarity. The
    Memory ABC's read/write therefore serve the build side only -- read(pid) is
    what lets a resumed distill phase skip puzzles already distilled -- while
    retrieval goes through retrieve()/random_retrieve().

    SCHEMA is a class attribute so a sibling bank (MemoryBankMemory) can inherit
    every retrieval mechanism and differ only in its schema tag and entry shape.
    """

    SCHEMA = SCHEMA

    def __init__(self, embed_model: str = ""):
        self.entries: list[dict] = []
        self.embed_model = embed_model
        self.stats: dict = {}
        # Row i of `vectors` is the L2-normalized key embedding of entries[i].
        # None until attach_vectors(); the distill phase never has vectors.
        self.vectors: np.ndarray | None = None
        # Per-entry keyword embedding matrices, only for the AveFact retriever.
        self.keyword_vectors: list | None = None

    # ---- Memory ABC (build-side) -----------------------------------------
    def read(self, key) -> list:
        return [e for e in self.entries if e["source_pid"] == str(key)]

    def write(self, key, item: dict) -> None:
        item = dict(item)
        item["source_pid"] = str(key)
        item.setdefault("id", f"m{len(self.entries):04d}")
        self.entries.append(item)

    def __contains__(self, key) -> bool:
        return any(e["source_pid"] == str(key) for e in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict:
        return {"schema": self.SCHEMA, "embed_model": self.embed_model,
                "stats": self.stats, "entries": self.entries}

    @classmethod
    def from_dict(cls, data: dict) -> "MempMemory":
        if data.get("schema") != cls.SCHEMA:
            raise ValueError(f"not a {cls.SCHEMA} bank: schema={data.get('schema')!r}")
        m = cls(embed_model=data.get("embed_model", ""))
        m.stats = data.get("stats", {})
        m.entries = list(data.get("entries", []))
        return m

    # ---- retrieval (eval-side) -------------------------------------------
    def attach_vectors(self, ids, vecs) -> None:
        """Join the sidecar embeddings onto the bank, one row per entry.

        `ids`/`vecs` are the embed phase's train keys (all 500 train puzzles get
        a vector whether or not they yielded an entry; the join here picks out
        the ones that did). Raises on any entry whose source_pid has no vector:
        a silent zero-vector would never be retrieved and the bank would just
        quietly shrink.
        """
        row_of = {str(pid): i for i, pid in enumerate(ids)}
        missing = [e["id"] for e in self.entries if e["source_pid"] not in row_of]
        if missing:
            raise ValueError(f"{len(missing)} bank entries have no embedding "
                             f"(first few: {missing[:5]}); wrong .npz for this bank?")
        mat = np.asarray([vecs[row_of[e["source_pid"]]] for e in self.entries],
                         dtype=np.float32)
        # Normalize so cosine similarity is a plain dot product. The embed phase
        # already normalizes, but re-doing it here costs nothing and removes the
        # assumption.
        self.vectors = mat / np.linalg.norm(mat, axis=1, keepdims=True)

    def retrieve(self, query_vec, k: int) -> list[tuple[dict, float]]:
        """Top-k entries by cosine similarity, with an explicit tie-break.

        Sorting by (-similarity, entry id) rather than np.argsort alone makes the
        retrieved set -- and therefore the injected block -- byte-identical across
        resumes and re-runs even when similarities tie, which both reproducibility
        and vLLM's prefix cache care about.
        """
        if self.vectors is None:
            raise RuntimeError("attach_vectors() first: the bank has no embeddings")
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / np.linalg.norm(q)
        sims = self.vectors @ q
        order = sorted(range(len(self.entries)),
                       key=lambda i: (-float(sims[i]), self.entries[i]["id"]))
        return [(self.entries[i], float(sims[i])) for i in order[:k]]

    def avefact_retrieve(self, query_vecs, k: int) -> list[tuple[dict, float]]:
        """AveFact retrieval: score by average similarity over matched keywords.

        The paper extracts keywords from the task with an LLM and "computes the
        average similarity across matched keywords". Concretely: each query
        keyword is matched to its best-fitting keyword of the candidate entry, and
        the entry's score is the mean of those best matches -- so an entry scores
        highly only when most of the query's salient features find a counterpart,
        rather than when one long description happens to look similar overall.
        """
        if self.keyword_vectors is None:
            raise RuntimeError("attach_keyword_vectors() first")
        q = np.asarray(query_vecs, dtype=np.float32)
        if q.ndim != 2 or not len(q):
            raise ValueError("query_vecs must be a non-empty 2-D array of keywords")
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        scores = []
        for kw in self.keyword_vectors:
            if kw is None or not len(kw):
                scores.append(-1.0)      # no keywords stored: never preferred
                continue
            sims = q @ np.asarray(kw, dtype=np.float32).T   # (n_query, n_entry_kw)
            scores.append(float(sims.max(axis=1).mean()))
        order = sorted(range(len(self.entries)),
                       key=lambda i: (-scores[i], self.entries[i]["id"]))
        return [(self.entries[i], scores[i]) for i in order[:k]]

    def attach_keyword_vectors(self, vectors_by_pid: dict) -> None:
        """Per-entry keyword embedding matrices, keyed by source puzzle id."""
        self.keyword_vectors = [
            vectors_by_pid.get(e["source_pid"]) for e in self.entries]
        missing = sum(1 for v in self.keyword_vectors if v is None or not len(v))
        if missing:
            print(f"[memp] WARNING: {missing}/{len(self.entries)} entries have no "
                  f"keyword vectors and can never be retrieved by AveFact",
                  flush=True)

    def fixed_retrieve(self, k: int) -> list[tuple[dict, None]]:
        """The same k entries for every puzzle -- the control the paper never ran.

        Isolates "any procedural text in context" from "retrieval". Picks the
        first k by entry id so the choice is deterministic and unremarkable.
        """
        picked = sorted(self.entries, key=lambda e: e["id"])[:k]
        return [(e, None) for e in picked]

    def revise(self, entry_id: str, script: str) -> None:
        """Adjustment: overwrite one entry's script in place, keeping its vector.

        The key text is unchanged (the entry still describes the same class of
        puzzle), so the embedding stays valid and retrieval is unaffected -- only
        what the agent reads changes.
        """
        for e in self.entries:
            if e["id"] == entry_id:
                e["revisions"] = e.get("revisions", 0) + 1
                e["script_history"] = e.get("script_history", []) + [e["script"]]
                e["script"] = script
                return
        raise KeyError(f"no bank entry {entry_id!r} to revise")

    def add_with_vector(self, key, item: dict, vec) -> dict:
        """Append a new entry and its key vector together (Vanilla / Validation).

        Entries added during the Update stage come from *test* puzzles, whose
        vectors are already in the embeddings sidecar, so the row is stacked onto
        the matrix directly rather than re-joining the whole bank.
        """
        if self.vectors is None:
            raise RuntimeError("attach_vectors() first")
        self.write(key, item)
        v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        v = v / np.linalg.norm(v)
        self.vectors = np.vstack([self.vectors, v])
        if self.keyword_vectors is not None:
            self.keyword_vectors.append(None)
        return self.entries[-1]

    def filtered(self, provenance: str) -> "MempMemory":
        """A copy holding only entries of one provenance ('agent' or 'gold').

        `agent` gives the paper-faithful bank: memory distilled solely from runs
        the model itself completed, with no gold-path hindsight.
        """
        if provenance == "all":
            return self
        m = type(self)(embed_model=self.embed_model)
        m.stats = dict(self.stats)
        m.entries = [e for e in self.entries if e["provenance"] == provenance]
        return m

    def random_retrieve(self, k: int, seed_key: str) -> list[tuple[dict, None]]:
        """The ablation arm: k entries drawn without regard to similarity.

        Seeded per puzzle (seed_key = f"{seed}:{pid}") so the draw is reproducible
        and, crucially, stable across a resume -- a resumed eval must inject the
        same block the interrupted run would have.
        """
        rng = random.Random(seed_key)
        picked = rng.sample(range(len(self.entries)), k=min(k, len(self.entries)))
        # Same presentation order convention as retrieve() (stable by entry id):
        # the arms must differ in WHICH entries appear, not in how they are laid out.
        picked.sort(key=lambda i: self.entries[i]["id"])
        return [(self.entries[i], None) for i in picked]


class Distiller:
    """Generates one procedure from one solved-puzzle context via the LLM.

    Holds its own client (like Reflector) so the driver can give distillation its
    own token budget without coupling it to the actor. Returns (text, LLMResponse)
    so the driver can tally distillation tokens as part of Memp's build cost.
    """

    def __init__(self, client):
        self.client = client

    async def distill_from_trajectory(self, episode, key_text: str, polyshapes=None,
                                      succeeded: bool = True):
        """Procedure from an agent episode (a success, unless Vanilla says otherwise).

        `polyshapes` is that puzzle's shape table, so the prepended rulebook is the
        same one the actor solved under -- see prompts.distill_rules_preamble.
        """
        prompt = memp_distill_trajectory_prompt(episode, key_text, polyshapes,
                                                succeeded=succeeded)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response

    async def distill_from_gold(self, key_text: str, text_visualization: str,
                                gold_actions: list, polyshapes=None):
        """Procedure from the dataset's gold path (hindsight, for failed puzzles)."""
        prompt = memp_distill_gold_prompt(key_text, text_visualization, gold_actions,
                                          polyshapes)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response

    async def revise(self, script: str, episode, key_text: str, polyshapes=None):
        """Memp's Adjustment update: repair a procedure that led to a failure."""
        prompt = memp_revise_prompt(script, episode, key_text, polyshapes)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response

    async def keywords(self, key_text: str):
        """Memp's AveFact key construction: salient features of one task."""
        prompt = memp_keywords_prompt(key_text)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        lines = [l.strip(" -*\t") for l in (response.text or "").splitlines()]
        return [l for l in lines if l][:8], response
