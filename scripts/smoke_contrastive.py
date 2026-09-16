#!/usr/bin/env python3
"""Invariants of the contrastive memory renderers, store and retrieval. No GPU.

The granularity IS the experiment for this method (docs/contrastive.md section 3),
so what each arm does and does not render is not a presentation detail -- it is the
independent variable. Two failure modes would be invisible downstream, because the
result files would still be internally consistent and the paired tests still paired:

  * an arm rendering the wrong sign's text, so `avoid` and `hindsight` quietly
    measure the same thing;
  * a DON'T rendered without its label, which an actor reads as an instruction to
    do exactly the thing that failed.

So this asserts, on hand-built entries rather than real data:

  * every arm renders `agent` entries as their strategy, and differs ONLY in what a
    `failed` entry contributes
  * every rendered text carries an explicit DO / DO NOT label
  * an entry whose text an arm withholds degrades to one placeholder line, so the
    Item count and the bank size stay identical across arms
  * balanced_retrieve guarantees its positives, and both random arms are stable
    under a reseed (a resumed job must inject a byte-identical block)
  * a bank of another method is refused at load time

Runs in about a second, needs no dataset, and uses numpy only through the store.

    python scripts/smoke_contrastive.py
    python scripts/smoke_contrastive.py --show    # print one rendered block per arm
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents import prompts, prompts_stepgame            # noqa: E402
from memory_agents.memory.contrastive import (                 # noqa: E402
    AGENT, FAILED, GRANULARITIES, ContrastiveMemory, broken_rules, rendered_as)

FAILURES: list[str] = []

RULE_STATUS = {
    "reached_target": {"passed": True, "detail": {"agent_loc": [4, 0]}},
    "all_dots_collected": {"passed": False, "detail": {"total": 3, "collected": 1}},
    "poly_ylop_area": {"passed": False,
                       "detail": {"violations": [0], "region_details": {}}},
}

DO_ENTRY = {
    "id": "m0001", "source_pid": "p1", "provenance": AGENT, "difficulty": 3,
    "key_text": "3x3, rules: dot x3",
    "strategy": "Applies to: dots\n1. Collect every dot before committing to the exit.",
    "exemplar": {"actions": [0, 0], "n_moves": 2, "width": 3, "height": 1,
                 "start": [0, 0], "end": [2, 0], "grid_summary": "dot x3",
                 "hops": 1, "sentences": 1, "unused": 0, "relations": ["left"],
                 "answer": "left", "k": 1},
}
DONT_ENTRY = {
    "id": "m0002", "source_pid": "p2", "provenance": FAILED, "difficulty": 3,
    "key_text": "3x3, rules: dot x3",
    "avoid": "Applies to: dots\n1. Do not head for the exit with dots uncollected.",
    "exemplar": {"actions": [0, 0, 1], "n_moves": 3, "width": 3, "height": 1,
                 "start": [0, 0], "end": [2, 0], "grid_summary": "dot x3",
                 "hops": 2, "sentences": 2, "unused": 0, "relations": ["left"],
                 "answer": "left", "k": 2},
    "failed_exemplar": {"actions": [3, 3], "n_moves": 2, "width": 3, "height": 1,
                        "start": [0, 0], "end": [2, 0], "grid_summary": "dot x3",
                        "hops": 2, "sentences": 2, "unused": 0,
                        "relations": ["left"], "answer": "left", "k": 2},
    "hindsight": "Applies to: dots\n1. Sweep the dot cells first, then exit.",
    "broken_rules": broken_rules({"rule_status": RULE_STATUS}),
    "failure_reason": "wrong-solution",
}


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILURES.append(name)
        if detail:
            print(f"       {detail}")


def raises(fn, *a) -> bool:
    try:
        fn(*a)
    except (ValueError, KeyError):
        return True
    return False


def check_rules() -> None:
    print("broken_rules()")
    br = broken_rules({"rule_status": RULE_STATUS})
    check("keeps only the rules the env reports as failed",
          [b["rule"] for b in br] == ["all_dots_collected", "poly_ylop_area"],
          repr(br))
    check("empty on a clean trace and on None",
          broken_rules({"rule_status": {}}) == [] and broken_rules(None) == [])
    # The detail filter keeps counts and short flat lists (which region offended)
    # and drops nested tables and empty containers -- `region_details` runs to a
    # whole grid, and `mismatches=[]` on a rule that FAILED says nothing.
    mixed = broken_rules({"rule_status": {
        "counts": {"passed": False, "detail": {"total": 3, "collected": 1}},
        "offender": {"passed": False,
                     "detail": {"violations": [0], "region_details": {"r": [1]}}},
        "bare": {"passed": False, "detail": {"mismatches": [], "regions": {}}}}})
    rendered = prompts.format_broken_rules(mixed)
    check("renders counts and short flat lists, drops nested and empty",
          rendered == ("  Rules it broke: counts (total=3, collected=1); "
                       "offender (violations=[0]); bare"), rendered)


def check_arms(mod, name: str, win: str, lose: str) -> dict:
    print(f"\n{name} arms")
    pair = [DO_ENTRY, DONT_ENTRY]
    blocks = {g: mod.format_contrastive(pair, g) for g in GRANULARITIES}

    check("avoid renders the DON'T, never the control text",
          DONT_ENTRY["avoid"] in blocks["avoid"]
          and DONT_ENTRY["hindsight"] not in blocks["avoid"])
    check("hindsight renders the control text, never the DON'T",
          DONT_ENTRY["hindsight"] in blocks["hindsight"]
          and DONT_ENTRY["avoid"] not in blocks["hindsight"])
    # The positive half is held fixed across the three arms that show it; if it
    # were not, avoid-vs-hindsight would move two factors at once.
    check("the positive half is identical in avoid, hindsight and positive-only",
          all(DO_ENTRY["strategy"] in blocks[g]
              for g in ("avoid", "hindsight", "positive-only")))
    check("positive-only withholds both texts of a failed entry",
          DONT_ENTRY["avoid"] not in blocks["positive-only"]
          and DONT_ENTRY["hindsight"] not in blocks["positive-only"])
    check("avoid-only withholds the strategy",
          DO_ENTRY["strategy"] not in blocks["avoid-only"]
          and DONT_ENTRY["avoid"] in blocks["avoid-only"])

    # An unlabelled DON'T reads as an instruction. This is the one rendering bug
    # that would actively damage the actor rather than merely waste tokens.
    check("a DON'T is always labelled DO NOT",
          all("DO NOT" in b for b in (blocks["avoid"], blocks["avoid-only"])),
          blocks["avoid"])
    check("no arm shows a DON'T label on a positive entry",
          blocks["positive-only"].count("DO NOT") == 0)
    # hindsight is a procedure for SOLVING the puzzle, so it is a DO even though
    # its entry came from a failure. That is the point of the control.
    check("hindsight is labelled DO, not DO NOT",
          "DO NOT" not in blocks["hindsight"]
          and blocks["hindsight"].count("DO --") == 2)

    check("every arm renders the same number of Items",
          all(b.count("Item 1:") == 1 and b.count("Item 2:") == 1
              for b in blocks.values()))
    check("a withheld entry degrades to one placeholder line",
          all("nothing recorded" in blocks[g]
              for g in ("positive-only", "avoid-only")))
    check("no bank bookkeeping leaks into the block",
          all("m0001" not in b and "source_pid" not in b and "key_text" not in b
              for b in blocks.values()))
    check("an unknown granularity raises rather than defaulting",
          raises(mod.format_contrastive, pair, "contrastive"))

    # ---- --exemplars on: the concrete artifact, keyed on how the item RENDERS ----
    # Keyed on the RENDERER's own opening words, not on fixture values: the two
    # benchmarks render an exemplar completely differently (an action list on SPaRC,
    # a chain description on StepGame), and asserting on fixture contents silently
    # passed on SPaRC while testing nothing on StepGame.
    ex = {g: mod.format_contrastive(pair, g, exemplars=True) for g in GRANULARITIES}
    check("exemplars: a DO carries the WINNING line",
          win in ex["positive-only"] and lose not in ex["positive-only"],
          ex["positive-only"])
    # The whole distinction between `avoid` and `hindsight` is whether the failed
    # entry's artifact is the losing line or the gold one. If a DON'T ever rendered
    # the winning line, `avoid` would silently become `hindsight`.
    check("exemplars: a DON'T carries the LOSING line, never the winning one",
          lose in ex["avoid-only"] and win not in ex["avoid-only"], ex["avoid-only"])
    check("exemplars: hindsight renders its failed entry's WINNING line",
          win in ex["hindsight"] and lose not in ex["hindsight"], ex["hindsight"])
    check("exemplars: avoid carries both, one per sign",
          win in ex["avoid"] and lose in ex["avoid"], ex["avoid"])
    check("exemplars=False renders no artifact at all",
          all(lose not in blocks[g] for g in GRANULARITIES))
    bare = [{k: v for k, v in e.items()
             if k not in ("exemplar", "failed_exemplar")} for e in pair]
    check("an entry with no artifact degrades to its text, not to an error",
          mod.format_contrastive(bare, "avoid", exemplars=True)
          == mod.format_contrastive(bare, "avoid", exemplars=False))

    # ---- --framing matched: only `hindsight`'s header moves --------------------
    m = {g: mod.format_contrastive(pair, g, framing="matched") for g in GRANULARITIES}
    check("framing=matched changes hindsight's header",
          m["hindsight"] != blocks["hindsight"]
          and m["hindsight"].split("Item 1")[1] == blocks["hindsight"].split("Item 1")[1])
    check("framing=matched leaves every other arm byte-identical",
          all(m[g] == blocks[g] for g in GRANULARITIES if g != "hindsight"))
    check("the matched hindsight header promises no failures",
          "FAILED" not in m["hindsight"].split("Item 1")[0], m["hindsight"][:300])
    check("the mixed header does promise them (that is the mismatch under test)",
          "FAILED" in blocks["hindsight"].split("Item 1")[0])
    check("an unknown framing raises", raises(mod.format_contrastive, pair,
                                              "avoid", False, "bogus"))

    # ---- rendered_as() must agree with what the renderer actually does ---------
    # There were briefly two definitions of an entry's rendered sign and they
    # disagreed: the eval logged `provenance == AGENT`, which equals the rendered
    # sign only for `avoid`. Under `hindsight` every entry is shown as a DO, so the
    # polarity column read 28.8% where the truth was 100%. One definition now, and
    # this asserts it against the renderer for every arm and both signs.
    for g in GRANULARITIES:
        for e, who in ((pair[0], "DO entry"), (pair[1], "DON'T entry")):
            one = mod.format_contrastive([e], g)
            want = rendered_as(e, g)
            got = ("withheld" if "nothing recorded" in one
                   else "dont" if "DO NOT" in one else "do")
            check(f"rendered_as agrees with the renderer: {g} / {who}", want == got,
                  f"rendered_as said {want!r}, the block shows {got!r}")
    return blocks


def check_store() -> None:
    print("\nthe store and its retrieval arms")
    m = ContrastiveMemory()
    # A deliberately lopsided bank: 2 DOs to 8 DON'Ts, the shape a weak actor
    # actually produces (docs/contrastive.md section 3).
    for i in range(10):
        e = dict(DO_ENTRY if i < 2 else DONT_ENTRY)
        e["id"] = f"m{i:04d}"
        m.write(f"p{i}", e)
    check("polarity counts both signs",
          m.polarity() == {AGENT: 2, FAILED: 8}, repr(m.polarity()))
    check("positive_share", abs(m.positive_share() - 0.2) < 1e-9)
    check("schema tag", m.to_dict()["schema"] == "contrastive-bank-v1")
    check("round-trips", len(ContrastiveMemory.from_dict(m.to_dict())) == 10)
    # All three cross-task banks live side by side in results/json and differ only
    # by a filename prefix, so a wrong --bank on a submit line must fail loudly.
    check("a Memp bank is refused at load time",
          raises(ContrastiveMemory.from_dict,
                 {"schema": "memp-bank-v1", "entries": []}))

    # Vectors chosen so the 8 DON'Ts are all strictly closer to the query than the
    # 2 DOs: an unconstrained top-3 must return no positive at all, which is the
    # situation ARM=balanced exists for.
    vecs = np.array([[0.0, 1.0]] * 2 + [[1.0, 0.0]] * 8, dtype=np.float32)
    m.attach_vectors([f"p{i}" for i in range(10)], vecs)
    q = np.array([1.0, 0.0], dtype=np.float32)
    plain = [e["provenance"] for e, _ in m.retrieve(q, 3)]
    check("plain top-k on a lopsided bank returns no DO at all",
          plain == [FAILED] * 3, repr(plain))
    bal = [e["provenance"] for e, _ in m.balanced_retrieve(q, 3, 1)]
    check("balanced_retrieve guarantees one DO",
          bal.count(AGENT) == 1 and len(bal) == 3, repr(bal))
    check("balanced_retrieve honours min_positive=2",
          [e["provenance"] for e, _ in m.balanced_retrieve(q, 3, 2)].count(AGENT) == 2)
    check("balanced_retrieve never exceeds k",
          len(m.balanced_retrieve(q, 2, 3)) == 2)
    # The arm must differ from `retrieved` in WHICH entries appear and in NOTHING
    # else. An earlier version re-sorted by entry id -- which retrieve() does not do,
    # it orders by (-similarity, id) -- so whenever the plain top-k already held a
    # positive, `balanced` injected the identical entries in a different order and
    # silently manipulated presentation order too.
    check("balanced_retrieve returns similarity order, like retrieve()",
          [e["id"] for e, _ in m.balanced_retrieve(q, 3, 1)]
          == [e["id"] for e, _ in sorted(m.balanced_retrieve(q, 3, 1),
                                         key=lambda pr: -pr[1])])
    # The decisive case: a query whose plain top-k ALREADY contains a positive. The
    # two arms must then agree exactly, entry for entry and position for position.
    q_pos = np.array([0.0, 1.0], dtype=np.float32)      # nearest the 2 DO entries
    plain3 = m.retrieve(q_pos, 3)
    if sum(1 for e, _ in plain3 if e["provenance"] == AGENT) >= 1:
        check("where plain top-k already has a DO, balanced is IDENTICAL to retrieved",
              [e["id"] for e, _ in m.balanced_retrieve(q_pos, 3, 1)]
              == [e["id"] for e, _ in plain3],
              f"{[e['id'] for e, _ in m.balanced_retrieve(q_pos, 3, 1)]} vs "
              f"{[e['id'] for e, _ in plain3]}")
    # Still a pure function of (bank, query), so a resumed job injects the same block.
    check("balanced_retrieve is deterministic",
          [e["id"] for e, _ in m.balanced_retrieve(q, 3, 1)]
          == [e["id"] for e, _ in m.balanced_retrieve(q, 3, 1)])
    r1 = [e["id"] for e, _ in m.stratified_random_retrieve(3, "0:x", 1)]
    r2 = [e["id"] for e, _ in m.stratified_random_retrieve(3, "0:x", 1)]
    check("stratified_random_retrieve is stable under a reseed", r1 == r2, repr(r1))
    check("stratified_random_retrieve guarantees its DO",
          sum(1 for e, _ in m.stratified_random_retrieve(3, "0:x", 1)
              if e["provenance"] == AGENT) == 1)
    check("stratified_random_retrieve is ordered by entry id", r1 == sorted(r1))
    check("the two random arms differ across items",
          [e["id"] for e, _ in m.stratified_random_retrieve(3, "0:y", 1)] != r1)

    # An all-negative bank is what a very weak actor produces; balanced must
    # degrade to plain top-k rather than raise or return short.
    weak = ContrastiveMemory()
    for i in range(4):
        e = dict(DONT_ENTRY)
        e["id"] = f"m{i:04d}"
        weak.write(f"q{i}", e)
    weak.attach_vectors([f"q{i}" for i in range(4)],
                        np.array([[1.0, 0.0]] * 4, dtype=np.float32))
    check("balanced degrades cleanly when the bank has no DO at all",
          len(weak.balanced_retrieve(q, 3, 1)) == 3
          and len(weak.stratified_random_retrieve(3, "0:x", 1)) == 3)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true",
                    help="print one rendered block per arm")
    args = ap.parse_args()

    check_rules()
    sparc_blocks = check_arms(prompts, "SPaRC",
                              win="Example (", lose="What that attempt played")
    check_arms(prompts_stepgame, "StepGame",
               win="Example:", lose="That attempt faced")
    check_store()

    if args.show:
        for gran, block in sparc_blocks.items():
            print(f"\n{'=' * 72}\n  SPaRC, granularity={gran}\n{'=' * 72}{block}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all contrastive invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
