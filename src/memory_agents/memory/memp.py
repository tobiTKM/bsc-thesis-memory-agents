"""Memp procedural memory (Fang et al., 2026): a bank of distilled procedures
with cosine top-k retrieval (MempMemory) and the LLM calls that write them
(Distiller), from the agent's own successes or, for failed items, the gold path.
Key vectors live in a sidecar .npz joined at eval time by attach_vectors().
"""

import random

import numpy as np

from ..prompts import (memp_distill_gold_prompt, memp_distill_trajectory_prompt,
                       memp_keywords_prompt, memp_revise_prompt)
from .base import Memory

SCHEMA = "memp-bank-v1"


class MempMemory(Memory):
    """A flat bank of entries keyed by `source_pid`, each with id, provenance and
    script; read/write serve the build side, retrieval goes through *_retrieve().

    Subclasses override SCHEMA and inherit retrieval unchanged.
    """

    SCHEMA = SCHEMA

    def __init__(self, embed_model: str = ""):
        self.entries: list[dict] = []
        self.embed_model = embed_model
        self.stats: dict = {}
        # Row i: L2-normalized key embedding of entries[i]; None until attach_vectors().
        self.vectors: np.ndarray | None = None
        # Per-entry keyword embedding matrices, AveFact retriever only.
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
        """Join sidecar embeddings by source_pid; raises if any entry has none."""
        row_of = {str(pid): i for i, pid in enumerate(ids)}
        missing = [e["id"] for e in self.entries if e["source_pid"] not in row_of]
        if missing:
            raise ValueError(f"{len(missing)} bank entries have no embedding "
                             f"(first few: {missing[:5]}); wrong .npz for this bank?")
        mat = np.asarray([vecs[row_of[e["source_pid"]]] for e in self.entries],
                         dtype=np.float32)
        self.vectors = mat / np.linalg.norm(mat, axis=1, keepdims=True)

    def retrieve(self, query_vec, k: int) -> list[tuple[dict, float]]:
        """Top-k by cosine similarity, ordered by (-similarity, entry id).

        The id tie-break keeps the injected block byte-identical across resumes.
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
        """AveFact: mean over query keywords of the best-matching entry keyword.

        Ordered by (-score, entry id).
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
                scores.append(-1.0)      # no keywords: never preferred
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
        """Control arm: the first k entries by id, for every puzzle."""
        picked = sorted(self.entries, key=lambda e: e["id"])[:k]
        return [(e, None) for e in picked]

    def revise(self, entry_id: str, script: str) -> None:
        """Adjustment: replace one entry's script in place; its vector stays valid."""
        for e in self.entries:
            if e["id"] == entry_id:
                e["revisions"] = e.get("revisions", 0) + 1
                e["script_history"] = e.get("script_history", []) + [e["script"]]
                e["script"] = script
                return
        raise KeyError(f"no bank entry {entry_id!r} to revise")

    def add_with_vector(self, key, item: dict, vec) -> dict:
        """Append an entry and its key vector (Vanilla / Validation update)."""
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
        """Entries of one provenance ('agent' or 'gold'); 'all' returns self."""
        if provenance == "all":
            return self
        m = type(self)(embed_model=self.embed_model)
        m.stats = dict(self.stats)
        m.entries = [e for e in self.entries if e["provenance"] == provenance]
        return m

    def random_retrieve(self, k: int, seed_key: str) -> list[tuple[dict, None]]:
        """Random-arm ablation: k entries drawn with random.Random(seed_key).

        seed_key is f"{seed}:{pid}", so a resumed eval injects the same block.
        """
        rng = random.Random(seed_key)
        picked = rng.sample(range(len(self.entries)), k=min(k, len(self.entries)))
        # Sorted by id, not by similarity as in retrieve(); keep it, or new runs
        # stop being comparable with existing results.
        picked.sort(key=lambda i: self.entries[i]["id"])
        return [(self.entries[i], None) for i in picked]


class Distiller:
    """Generates procedures via its own client; returns (text, LLMResponse)."""

    def __init__(self, client, benchmark=None):
        self.client = client
        # Required for anything but SPaRC; None falls back to SparcBenchmark.
        self.benchmark = benchmark

    def _bench(self):
        if self.benchmark is None:
            from ..benchmarks import SparcBenchmark
            self.benchmark = SparcBenchmark()
        return self.benchmark

    async def distill_from_trajectory(self, episode, key_text: str, record=None,
                                      succeeded: bool = True):
        """Procedure from an agent episode; `record` is the benchmark item."""
        prompt = self._bench().distill_trajectory_prompt(
            episode, key_text, record, succeeded=succeeded)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response

    async def distill_from_gold(self, key_text: str, record=None):
        """Procedure from the task's gold solution (hindsight, for failed items)."""
        prompt = self._bench().distill_gold_prompt(key_text, record)
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
