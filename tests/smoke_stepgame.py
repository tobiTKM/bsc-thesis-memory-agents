#!/usr/bin/env python3
"""Fast invariants for the StepGame data layer. No GPU, no model, no network.

Complements scripts/verify_stepgame_solver.py: that one is the slow exhaustive
corpus gate, this one is the seconds-long check you run after touching any of
envs/stepgame*.py. Written as a plain script rather than a pytest module because
the project has no test dependency and this needs none.

    python tests/smoke_stepgame.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents.envs import stepgame                     # noqa: E402
from memory_agents.envs import stepgame_solver as solver    # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}"
          + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    splits_dir = stepgame.data_dir() / "splits"
    paths = sorted(splits_dir.glob("*.jsonl"))
    if not paths:
        print(f"no splits in {splits_dir}; "
              f"run `python scripts/build_stepgame_splits.py` first")
        return 1

    print("\n[1] answer parser")
    cases = [
        ("Final: upper-right", "upper-right"),
        ("maybe left...\n\nFinal: **lower-left**", "lower-left"),
        ("Final: The answer is upper right.", "upper-right"),
        ("Final: north-west", "upper-left"),
        # A thinking model floats candidates mid-reasoning; the LAST Final wins.
        ("Final: left\nOn reflection that is wrong.\nFinal: right", "right"),
        # "Finally," must not read as a Final: marker.
        ("Finally, combine the hops.\nFinal: above", "above"),
        ("no direction mentioned anywhere", None),
        ("", None),
        ("Final: I cannot determine this", None),
    ]
    for text, want in cases:
        got = stepgame.parse_answer(text)
        check(f"parse {text[:38]!r} -> {want}", got == want, f"got {got}")
    for label in solver.LABELS:
        check(f"round-trip {label}",
              stepgame.parse_answer(f"reasoning\nFinal: {label}") == label)

    print("\n[2] solver")
    items = stepgame.read_split(paths[0])[:200]
    check("oracle exact on a split sample",
          all(solver.solve(i.story, i.question) == i.gold for i in items))
    check("stored hops == recomputed hops",
          all(solver.hop_count(i.story, i.question) == i.hops for i in items))
    check("gold_chain ends in the gold answer",
          all(i.gold_chain.strip().endswith(f"Answer: {i.gold}") for i in items))
    check("every label has a unique unit offset",
          len(set(solver.OFFSETS.values())) == len(solver.LABELS))
    check("offsets are inverses under negation",
          all(solver.label_of(-dx, -dy) != label
              for label, (dx, dy) in solver.OFFSETS.items() if (dx, dy) != (0, 0)))

    print("\n[3] prompt hygiene")
    # The user message must not itself contain a parseable answer: if it did, a
    # model could score by echoing its own prompt and the benchmark would measure
    # nothing. This is the data-layer half of the no-leak rule; the other half
    # (the Reflexion reflector never seeing the gold label) lives in the driver.
    leaks = [i.id for i in items
             if stepgame.parse_answer(stepgame.format_item(i)) is not None]
    check("format_item leaks no answer", not leaks, f"{len(leaks)} leak: {leaks[:3]}")
    check("format_item omits the gold chain",
          all("Answer:" not in stepgame.format_item(i) for i in items))

    print("\n[4] retrieval keys")
    check("structural key is deterministic",
          all(stepgame.stepgame_key_text(i) == stepgame.stepgame_key_text(i)
              for i in items[:50]))
    check("structural key names no entity",
          all(f" {letter} " not in stepgame.stepgame_key_text(i)
              for i in items[:50]
              for letter in {c for s in i.story for c in s if c.isupper()}))
    check("lexical key anonymises entities",
          all("E0" in stepgame.stepgame_key_text_lexical(i) for i in items[:50]))

    print("\n[5] splits")
    train = [p for p in paths if "_train" in p.name]
    test = [p for p in paths if "_test" in p.name]
    check("a train and a test split exist", bool(train) and bool(test))
    for path in paths:
        split_items = stepgame.read_split(path)
        ids = [i.id for i in split_items]
        check(f"{path.name}: ids unique", len(set(ids)) == len(ids))
        check(f"{path.name}: every item verified",
              all(i.gold in solver.LABELS and i.hops > 0 and i.gold_chain
                  for i in split_items))
    # k=1 has only ~10k distinct possible stories, so train/test contamination is
    # a real risk there and the original paper measured 1.09% leakage at k=1.
    tr = {(frozenset(s.strip() for s in i.story), i.question.strip())
          for p in train for i in stepgame.read_split(p)}
    for path in test:
        overlap = [i.id for i in stepgame.read_split(path)
                   if (frozenset(s.strip() for s in i.story),
                       i.question.strip()) in tr]
        check(f"{path.name}: no train contamination", not overlap,
              f"{len(overlap)} shared items")

    print("\n[6] memory-arm prompts")
    from memory_agents.benchmarks import get_benchmark
    from memory_agents.envs.base import EpisodeResult, Transition
    bm = get_benchmark("stepgame")
    rec = bm.record("test", bm.ids("test")[0])

    sys_prompt = bm.system_prompt(rec)
    check("system prompt lists all nine labels",
          all(l in sys_prompt for l in solver.LABELS))
    check("system prompt states the Final contract", "Final: <label>" in sys_prompt)

    # A *wrong* answer, built through the same reflection_context round trip the
    # driver uses, so this tests the real path rather than a hand-made object.
    wrong = next(l for l in solver.LABELS if l != rec.gold)
    episode = EpisodeResult(
        puzzle_id=rec.id, success=False, terminated=True, truncated=False,
        difficulty=rec.k,
        transitions=[Transition(step=0, action=wrong, reward=0.0, legal_actions=[],
                                rule_status={}, agent_location=None)],
        first_obs=stepgame.format_item(rec),
        last_obs="I chained the relations and got there.",
        terminal_reason="wrong-answer",
        extra={"hops": rec.hops, "gold": rec.gold, "predicted": wrong})

    ctx = episode.reflection_context()
    check("reflection_context carries no gold/extra",
          "gold" not in ctx and "extra" not in ctx, f"keys={sorted(ctx)}")

    replayed = EpisodeResult.from_reflection_context(ctx, rec.id)
    prompt = bm.reflection_prompt(replayed, previous=("an earlier plan",))
    # THE invariant of the whole Reflexion arm: shown the answer, the reflector
    # would simply restate it and the arm would score 100% proving nothing.
    check("reflection prompt omits the worked chain", rec.gold_chain not in prompt)
    check("reflection prompt asserts no answer", "Answer:" not in prompt)
    check("reflection prompt does not parse as an answer",
          stepgame.parse_answer(prompt) != rec.gold,
          f"leaked {rec.gold}")
    check("reflection prompt still shows the question",
          rec.question.strip() in prompt)
    check("reflection prompt shows the answer given", str(wrong) in prompt)
    check("reflection prompt carries prior plans", "an earlier plan" in prompt)

    # The bank prompts, by contrast, are *supposed* to see the solution.
    key = bm.key_text(rec)
    gold_prompt = bm.distill_gold_prompt(key, rec)
    check("distill-gold prompt does include the worked chain",
          rec.gold_chain in gold_prompt)
    check("distill prompts render", all(
        len(x) > 200 for x in (
            gold_prompt,
            bm.distill_trajectory_prompt(replayed, key, rec),
            bm.case_gold_prompt(key, rec),
            bm.case_trajectory_prompt(replayed, key, rec))))

    print("\n[6b] prompt protocols")
    # The calibration protocols reproduce Li et al.'s actor prompt so our numbers
    # can be quoted beside their published table. Two things have to stay true:
    # every protocol must be graded by the SAME contract (or the comparison is not
    # a comparison), and `ours` must stay the default everywhere -- an accidental
    # flip to `upstream` would silently put ten worked examples into the
    # "memoryless" baseline and quietly destroy the control arm.
    from memory_agents.prompts_stepgame_upstream import (
        PROTOCOLS, system_prompt_for, upstream_examples, format_item_upstream)

    check("default protocol is ours", get_benchmark("stepgame").prompt_protocol == "ours")
    check("upstream prompt parses 10 exemplars", len(upstream_examples()) == 10,
          f"got {len(upstream_examples())}")

    for proto in PROTOCOLS:
        sp = system_prompt_for(proto)
        check(f"{proto}: same Final contract", "Final: <label>" in sp)
        check(f"{proto}: lists all nine labels",
              all(l in sp for l in solver.LABELS))
    hdr, shots = system_prompt_for("upstream-header"), system_prompt_for("upstream")
    check("upstream-header carries the clock decoder", "12 denotes above" in hdr)
    check("upstream-header carries the cardinal decoder",
          "north denotes above" in hdr)
    check("upstream-header is still zero-shot", "Answer:" not in hdr)
    check("upstream is 10-shot", shots.count("Answer:") == 10,
          f"{shots.count('Answer:')} exemplars")
    # The leak rule applies to the calibration protocols too. Tested the way [3]
    # tests ours -- "does the rendered item parse as an answer?" -- and NOT by
    # searching for the gold string, which occurs legitimately in story text
    # ("X is to the left of K" on an item whose answer is `left`).
    up_leaks = [i.id for i in items
                if stepgame.parse_answer(format_item_upstream(i)) is not None]
    check("upstream item render leaks no answer", not up_leaks,
          f"{len(up_leaks)} leak: {up_leaks[:3]}")
    check("upstream item render omits a worked chain",
          all("Answer:" not in format_item_upstream(i) for i in items))
    check("upstream benchmark renders items their way",
          get_benchmark("stepgame", prompt_protocol="upstream")
          .format_item(rec).startswith("Story:\n"))

    print("\n[7] injected blocks")
    entry = {"id": "e1", "script": "1. anchor the reference object",
             "case_note": "It was a two-hop chain.",
             "exemplar": bm.gold_exemplar(rec),
             "trajectory": bm.gold_trajectory(rec)}
    for g in bm.granularities:
        block = bm.format_procedures([entry], g)
        check(f"format_procedures {g}", len(block) > 100)
    for g in bm.case_granularities:
        block = bm.format_cases([entry], g)
        check(f"format_cases {g}", len(block) > 100)
    check("gold_trajectory ends in the answer",
          bm.gold_trajectory(rec)["last_obs"].strip().endswith(f"Answer: {rec.gold}"))

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all StepGame data-layer invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
