"""Contrastive memory: one bank with one entry per train puzzle, a DO (`strategy`)
distilled from a solved trajectory or a DON'T (`avoid`) from a failed one.
ContrastiveMemory subclasses MempMemory and must keep its retrieval identical;
AvoidDistiller writes the DON'Ts, while DOs reuse Memp's Distiller unchanged.
"""

import random

from .memp import MempMemory

SCHEMA = "contrastive-bank-v1"

# Content arms. `hindsight` is the control: failed entries are shown as Memp's
# gold-path procedure instead of a DON'T.
GRANULARITIES = ("avoid", "hindsight", "positive-only", "avoid-only")

AGENT = "agent"      # provenance: solved train puzzle
FAILED = "failed"    # provenance: failed train puzzle

# Skipped by broken_rules(): the conjunction of all other rules.
META_RULES = frozenset({"all_rules_satisfied"})


class ContrastiveMemory(MempMemory):
    """A bank of signed advice entries; retrieval inherited from MempMemory.

    Entry fields: provenance ('agent' or 'failed'); strategy ('agent' only);
    avoid, hindsight and broken_rules ('failed' only). 'failed' takes the place
    of Memp's 'gold' provenance; the gold-path text is in `hindsight`.
    """

    SCHEMA = SCHEMA

    def polarity(self) -> dict:
        """{'agent': n, 'failed': n}: the bank's mix of DOs and DON'Ts."""
        out = {AGENT: 0, FAILED: 0}
        for e in self.entries:
            out[e["provenance"]] = out.get(e["provenance"], 0) + 1
        return out

    def positive_share(self) -> float:
        n = len(self.entries)
        return (self.polarity()[AGENT] / n) if n else 0.0

    def balanced_retrieve(self, query_vec, k: int,
                          min_positive: int = 1) -> list[tuple[dict, float]]:
        """Top-k with at least `min_positive` DO entries if the bank has any.

        Takes the best `min_positive` positives, fills the rest by similarity, and
        returns them in retrieve()'s (-similarity, id) order.
        """
        ranked = self.retrieve(query_vec, len(self.entries))
        positives = [p for p in ranked if p[0]["provenance"] == AGENT]
        want = min(min_positive, k, len(positives))
        picked = positives[:want]
        chosen = {p[0]["id"] for p in picked}
        for pair in ranked:
            if len(picked) >= k:
                break
            if pair[0]["id"] not in chosen:
                picked.append(pair)
                chosen.add(pair[0]["id"])
        # Same order as retrieve(), so only which entries appear differs.
        rank_of = {p[0]["id"]: i for i, p in enumerate(ranked)}
        picked.sort(key=lambda p: rank_of[p[0]["id"]])
        return picked

    def stratified_random_retrieve(self, k: int, seed_key: str,
                                   min_positive: int = 1) -> list[tuple[dict, None]]:
        """Random counterpart of balanced_retrieve: same DO floor, seeded, id order."""
        rng = random.Random(seed_key)
        pos = [i for i, e in enumerate(self.entries) if e["provenance"] == AGENT]
        neg = [i for i, e in enumerate(self.entries) if e["provenance"] != AGENT]
        want = min(min_positive, k, len(pos))
        picked = rng.sample(pos, k=want) if want else []
        rest = [i for i in pos + neg if i not in set(picked)]
        picked += rng.sample(rest, k=min(k - len(picked), len(rest)))
        picked.sort(key=lambda i: self.entries[i]["id"])
        return [(self.entries[i], None) for i in picked]


def rendered_as(entry: dict, granularity: str) -> str:
    """How an arm renders this entry: 'do', 'dont' or 'withheld' (empty text).

    Differs from provenance under `hindsight` and `avoid-only`. Must mirror
    prompts.format_contrastive; tests/smoke_contrastive.py checks this.
    """
    positive = entry.get("provenance") == AGENT
    if positive:
        text = "" if granularity == "avoid-only" else entry.get("strategy")
    elif granularity == "positive-only":
        text = ""
    else:
        text = entry.get("hindsight" if granularity == "hindsight" else "avoid")
    if not (text or "").strip():
        return "withheld"
    return "do" if (positive or granularity == "hindsight") else "dont"


def broken_rules(traj: dict | None) -> list:
    """Rules the env's `rule_status` reports as failed, as [{rule, detail}].

    Empty on single-turn benchmarks, which have no rule_status.
    """
    if not traj:
        return []
    out = []
    for name, verdict in (traj.get("rule_status") or {}).items():
        if name in META_RULES:
            continue
        if isinstance(verdict, dict) and verdict.get("passed") is False:
            out.append({"rule": name, "detail": verdict.get("detail")})
    return out


class AvoidDistiller:
    """Generates one DON'T from a failed episode; same contract as Memp's Distiller."""

    def __init__(self, client, benchmark=None):
        self.client = client
        self.benchmark = benchmark

    def _bench(self):
        if self.benchmark is None:
            from ..benchmarks import SparcBenchmark
            self.benchmark = SparcBenchmark()
        return self.benchmark

    async def distill_from_failure(self, episode, key_text: str, record=None,
                                   rules=()):
        """One reusable DON'T for the class of task this failure came from."""
        prompt = self._bench().avoid_prompt(episode, key_text, record, rules=rules)
        response = await self.client.chat([{"role": "user", "content": prompt}])
        return (response.text or "").strip(), response
