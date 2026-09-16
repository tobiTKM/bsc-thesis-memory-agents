"""Contrastive memory: one bank, two kinds of entry -- DOs distilled from the
actor's successes and DON'Ts distilled from its failures.

The three ported methods all learn from one sign. Reflexion learns from failure but
only about the SAME task (a reflection is replayed into a retry of the puzzle that
produced it). Memp and MemoryBank learn cross-task but only from success: a train
puzzle the actor FAILED is distilled from the dataset's gold path instead, so the
failure itself is discarded and nothing in either bank is derived from a mistake.

This method keeps the failure. One entry per train puzzle, and which side it carries
is decided by the outcome, not by a knob:

    solved  -> `strategy`, distilled from the actor's own winning trajectory
    failed  -> `avoid`,    distilled from the actor's own failed trajectory

so the bank is a mixture of positive and negative advice whose ratio is a property
of the actor. See docs/contrastive.md.

* ContrastiveMemory -- the store. A subclass of MempMemory for the same reason
  MemoryBankMemory is one: the retrieval half must be IDENTICAL across the three
  cross-task methods (same puzzle_key_text, same .npz, same cosine top-k, same
  (-similarity, id) tie-break, same per-puzzle random seeding), so a cross-method
  comparison isolates what the memory SAYS. It adds one thing retrieval-side:
  `balanced_retrieve`, because on a weak actor an unconstrained top-k returns
  almost nothing but DON'Ts (section "the polarity problem" in the doc).
* AvoidDistiller -- the ONE new model call. The positive half reuses Memp's own
  distiller verbatim and the hindsight control arm reuses Memp's gold distiller
  verbatim, so the only thing this method introduces is the negative prompt.
"""

import random

from .memp import MempMemory

SCHEMA = "contrastive-bank-v1"

# The content arms (docs/contrastive.md section 3). `hindsight` is the control:
# it renders the failed entries the way Memp does -- a procedure distilled from the
# gold path -- so `avoid` vs `hindsight` moves exactly one factor, what the 429
# non-solved entries say, over the same bank and the same retriever.
GRANULARITIES = ("avoid", "hindsight", "positive-only", "avoid-only")

AGENT = "agent"      # provenance of an entry distilled from a solved train puzzle
FAILED = "failed"    # ... and from a failed one

# Verdict keys that carry no information of their own. `all_rules_satisfied` is the
# conjunction of every other rule, so on a failed attempt it is false by definition
# and listing it tells the distiller only what the prompt already states.
META_RULES = frozenset({"all_rules_satisfied"})


class ContrastiveMemory(MempMemory):
    """A flat bank of signed advice entries with embedding-based retrieval.

    Inherits every retrieval mechanism from MempMemory unchanged. An entry carries:

      provenance  'agent' (the actor solved this train puzzle) or 'failed'
      strategy    what to do -- LLM, from the actor's own winning trajectory
                  (Memp's distill prompt, verbatim). Only on 'agent' entries.
      avoid       what NOT to do -- LLM, from the actor's own failed trajectory.
                  The one thing this method adds. Only on 'failed' entries.
      hindsight   the control text: a procedure distilled from the gold path, i.e.
                  what Memp puts on this entry. Only on 'failed' entries.
      broken_rules  the env's own verdict on the failed attempt, in code -- fed to
                  the avoid-distiller as grounding so the DON'T names a rule that
                  was actually broken rather than one the model guessed at.

    `provenance` keeps the same name Memp and MemoryBank use, and the inherited
    filtered() therefore still partitions the bank -- but note the values differ:
    Memp's 'gold' means "this entry was distilled from the gold path", while here
    'failed' means "distilled from the failure". Memp's meaning is available as the
    `hindsight` granularity instead of as a provenance.
    """

    SCHEMA = SCHEMA

    def polarity(self) -> dict:
        """{'agent': n, 'failed': n} -- the bank's mix of DOs and DON'Ts.

        Not a diagnostic: it is a property of the ACTOR (how many train puzzles it
        solved) and it bounds what an unconstrained top-k can inject, so it is
        reported next to every effect size.
        """
        out = {AGENT: 0, FAILED: 0}
        for e in self.entries:
            out[e["provenance"]] = out.get(e["provenance"], 0) + 1
        return out

    def positive_share(self) -> float:
        n = len(self.entries)
        return (self.polarity()[AGENT] / n) if n else 0.0

    def balanced_retrieve(self, query_vec, k: int,
                          min_positive: int = 1) -> list[tuple[dict, float]]:
        """Top-k, but with at least `min_positive` DO entries if the bank has any.

        Why this arm exists. The bank's polarity is set by the actor's train
        accuracy, and on SPaRC that is 9-71 solved out of 500 -- so a plain top-k
        returns almost nothing but DON'Ts and the actor never sees a correct worked
        example, which docs/memp.md section 3.10 identifies as the only ingredient
        that demonstrably moves this benchmark. Without this arm, a null on `avoid`
        cannot distinguish "negative advice does not help" from "negative advice
        crowded out the thing that does".

        Takes the best `min_positive` positives by similarity, then fills the rest
        with the best remaining entries of either sign.

        **Returned in SIMILARITY order, matching retrieve().** The first version of
        this sorted by entry id "exactly as retrieve() does" -- which retrieve() does
        not: it orders by (-similarity, id), with the id only as a tie-break. The
        result was that for every test item whose plain top-k already contained a
        positive, `balanced` injected the IDENTICAL three entries in a DIFFERENT
        order, so the arm silently manipulated presentation order as well as polarity
        (144/500 items on SPaRC gpt-oss, and 7 of that arm's 10-puzzle margin sat
        there). Ordering is stable given the bank and the query, so a resumed job
        still injects a byte-identical block.
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
        # Back into the order retrieve() would have used, so this arm differs from
        # `retrieved` in WHICH entries appear and in nothing else.
        rank_of = {p[0]["id"]: i for i, p in enumerate(ranked)}
        picked.sort(key=lambda p: rank_of[p[0]["id"]])
        return picked

    def stratified_random_retrieve(self, k: int, seed_key: str,
                                   min_positive: int = 1) -> list[tuple[dict, None]]:
        """The random-arm counterpart of balanced_retrieve.

        A random ablation has to be matched to the arm it ablates: drawing k
        entries uniformly against a balanced arm would change the polarity AND the
        relevance at once, and the contrast would measure neither.
        """
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
    """How an arm will actually SHOW this entry: 'do', 'dont' or 'withheld'.

    The single definition of an entry's rendered sign, because there were briefly two
    and they disagreed. `finalize_eval` used to log the polarity of what was injected
    as `provenance == AGENT` -- the entry's SOURCE, i.e. whether the actor solved that
    train puzzle. That is the same thing as the rendered sign only for `avoid`. Under
    `hindsight` every entry renders as a DO whatever its provenance, so the metric
    reported 28.8% of items seeing a DO when the true figure is 100%; under
    `avoid-only` it would have reported 28.8% when the true figure is 0%.

    The distinction matters because the polarity argument in docs/contrastive.md
    section 2 is about what the ACTOR SAW, not about where the text came from.

    Mirrors prompts.format_contrastive exactly, including that an entry whose text is
    missing renders as a placeholder and is therefore neither sign.
    scripts/smoke_contrastive.py asserts the two never drift apart.
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
    """The rules the env itself reports as failed on an attempt, with their detail.

    Read straight out of `rule_status` -- the per-rule verdict the environment
    attaches to the final transition. This is the one part of a DON'T that cannot
    be wrong: it is not a claim about what the rule says, it is the env's judgement
    on what this attempt did. It is fed to the avoid-distiller as grounding, for
    the same reason docs/memp.md section 3.9 prepends the rulebook -- a distiller
    left to infer the mechanics writes confident nonsense into the bank.

    Single-turn benchmarks have no rule_status; the empty list is correct there and
    the prompt falls back to the answer the attempt gave.
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
    """Generates one DON'T from one FAILED episode via the LLM.

    The only model call this method introduces. Same shape and contract as Memp's
    Distiller and MemoryBank's CaseDistiller -- holds its own client so the driver
    can budget it separately, returns (text, LLMResponse) so the tokens are tallied
    as bank-construction cost.

    The positive half of the bank is written by Memp's own `Distiller`, unchanged,
    and the `hindsight` control arm by Memp's own gold prompt, unchanged. Keeping
    those two verbatim is what makes `avoid` vs `hindsight` a single-factor test.
    """

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
