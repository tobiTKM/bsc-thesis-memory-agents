#!/usr/bin/env python3
"""Prove the benchmark refactor did not move a single byte of the SPaRC path.

The SPaRC numbers in docs/memp.md, docs/memorybank.md and docs/reflexion_results.md
were produced before `benchmarks.py` existed, when every driver called
prompts.py and harness/run.py directly. Those results stay citable only if the
refactored path renders the *identical* text -- one changed whitespace in a system
prompt silently invalidates every accuracy in those documents, and nothing else in
the repo would notice.

So this compares, for real puzzles drawn from the real dataset:

  * the actor's system prompt          benchmark  vs  prompts.sparc_system_prompt
  * the Reflexion reflection prompt    benchmark  vs  prompts.reflexion_reflection_prompt
  * both MemP distill prompts          benchmark  vs  prompts.memp_distill_*
  * both MemoryBank case prompts       benchmark  vs  prompts.memorybank_case_*
  * the injected blocks                benchmark  vs  prompts.format_procedures/cases
  * the retrieval key text             benchmark  vs  envs.sparc.puzzle_key_text
  * the exemplar builders              benchmark  vs  the pre-refactor definitions
  * BaselineAgent's default parser     still envs.sparc.parse_action

Needs the SPaRC dataset (cached) and the Spatial_Gym stack; no GPU and no model.

    python scripts/regress_sparc.py --n 25
"""

import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents import prompts                       # noqa: E402
from memory_agents.agents.base import BaselineAgent, _sparc_parse_action  # noqa: E402
from memory_agents.benchmarks import SparcBenchmark     # noqa: E402
from memory_agents.envs.base import EpisodeResult, Transition  # noqa: E402
from memory_agents.envs.sparc import (                  # noqa: E402
    grid_summary, parse_action, path_to_actions, puzzle_key_text)

FAILURES: list[str] = []


def check(name: str, a, b) -> None:
    same = a == b
    print(f"  {'ok  ' if same else 'FAIL'} {name}")
    if not same:
        FAILURES.append(name)
        # Show the first divergence rather than two walls of text.
        if isinstance(a, str) and isinstance(b, str):
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"       first differs at char {i}: "
                          f"{a[max(0, i - 60):i + 60]!r}\n"
                          f"                          vs "
                          f"{b[max(0, i - 60):i + 60]!r}")
                    break
            else:
                print(f"       same prefix, lengths {len(a)} vs {len(b)}")
        else:
            print(f"       {a!r}\n       {b!r}")


def fake_episode(puzzle_id: str) -> EpisodeResult:
    """A stand-in failed episode. Fixed content so the two prompt paths get
    byte-identical input and only the rendering is under test."""
    transitions = [
        Transition(step=0, action=0, reward=0.0, legal_actions=[0, 1],
                   rule_status={}, agent_location=[0, 0]),
        Transition(step=1, action=3, reward=0.0, legal_actions=[3],
                   rule_status={"squares": False}, agent_location=[0, 2]),
    ]
    return EpisodeResult(
        puzzle_id=puzzle_id, success=False, terminated=True, truncated=False,
        difficulty=3, transitions=transitions,
        first_obs="[start grid]", last_obs="[final grid]",
        terminal_reason="wrong-solution")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=25, help="puzzles to compare")
    ap.add_argument("--split", default="train",
                    help="train has the gold paths the distill prompts read")
    args = ap.parse_args()

    bm = SparcBenchmark()
    ids = bm.ids(args.split, args.n)
    print(f"comparing {len(ids)} {args.split} puzzles through both paths\n")

    print("[1] actor system prompt")
    for pid in ids:
        row = bm.record(args.split, pid)
        check(f"system_prompt {pid}",
              bm.system_prompt(row),
              prompts.sparc_system_prompt(yaml.safe_load(row["polyshapes"])))

    print("\n[2] retrieval key + exemplars")
    for pid in ids[:10]:
        row = bm.record(args.split, pid)
        check(f"key_text {pid}", bm.key_text(row), puzzle_key_text(row))
        # The pre-refactor gold_exemplar, inlined so this compares against the
        # definition the published banks were built with, not against itself.
        puzzle = yaml.safe_load(row["text_visualization"])["puzzle"]
        actions = path_to_actions(list(row["solutions"][0]["path"]))
        expected = {
            "actions": actions, "n_moves": len(actions),
            "width": puzzle["dimensions"]["width"],
            "height": puzzle["dimensions"]["height"],
            "start": [puzzle["start"]["x"], puzzle["start"]["y"]],
            "end": [puzzle["end"]["x"], puzzle["end"]["y"]],
            "grid_summary": grid_summary(row),
        }
        check(f"gold_exemplar {pid}", bm.gold_exemplar(row), expected)
        ctx = {"moves": [[0, 1, None], [1, 3, None]]}
        agent_expected = dict(expected, actions=[1, 3], n_moves=2)
        check(f"agent_exemplar {pid}", bm.agent_exemplar(row, ctx), agent_expected)

    print("\n[3] Reflexion reflection prompt")
    for pid in ids[:10]:
        ep = fake_episode(pid)
        for previous in ((), ("a prior plan", "another prior plan")):
            check(f"reflection {pid} prev={len(previous)}",
                  bm.reflection_prompt(ep, previous=previous),
                  prompts.reflexion_reflection_prompt(ep, previous=previous))
    check("format_reflections", bm.format_reflections(["one", "two"]),
          prompts.format_reflections(["one", "two"]))

    print("\n[4] MemP distill prompts")
    for pid in ids[:10]:
        row = bm.record(args.split, pid)
        poly = yaml.safe_load(row["polyshapes"])
        key = puzzle_key_text(row)
        ep = fake_episode(pid)
        check(f"distill_trajectory {pid}",
              bm.distill_trajectory_prompt(ep, key, row),
              prompts.memp_distill_trajectory_prompt(ep, key, poly, succeeded=True))
        check(f"distill_gold {pid}",
              bm.distill_gold_prompt(key, row),
              prompts.memp_distill_gold_prompt(
                  key, row["text_visualization"],
                  path_to_actions(list(row["solutions"][0]["path"])), poly))

    print("\n[5] MemoryBank case prompts")
    for pid in ids[:10]:
        row = bm.record(args.split, pid)
        poly = yaml.safe_load(row["polyshapes"])
        key = puzzle_key_text(row)
        ep = fake_episode(pid)
        for rp in (True, False):
            check(f"case_trajectory {pid} rp={rp}",
                  bm.case_trajectory_prompt(ep, key, row, rules_preamble=rp),
                  prompts.memorybank_case_trajectory_prompt(
                      ep, key, poly, rules_preamble=rp))
            check(f"case_gold {pid} rp={rp}",
                  bm.case_gold_prompt(key, row, rules_preamble=rp),
                  prompts.memorybank_case_gold_prompt(
                      key, row["text_visualization"],
                      path_to_actions(list(row["solutions"][0]["path"])), poly,
                      rules_preamble=rp))

    print("\n[6] injected blocks")
    entries = [{
        "id": "e1", "script": "1. do a thing\n2. do another",
        "case_note": "It was a 4x4 grid with two blue squares.",
        "exemplar": {"actions": [0, 1, 2], "n_moves": 3, "width": 4, "height": 4,
                     "start": [0, 0], "end": [3, 3], "grid_summary": "square x2"},
        "trajectory": {"first_obs": "[g0]", "last_obs": "[g1]",
                       "moves": [[0, 0, [1, 0]], [1, 1, [1, 1]]]},
    }]
    for g in ("script", "trajectory", "proceduralization"):
        check(f"format_procedures {g}", bm.format_procedures(entries, g),
              prompts.format_procedures(entries, granularity=g))
    for g in ("case-record", "case"):
        check(f"format_cases {g}", bm.format_cases(entries, g),
              prompts.format_cases(entries, granularity=g))

    print("\n[7] agent defaults")
    check("BaselineAgent default parser is SPaRC's",
          _sparc_parse_action("Final: 2"), parse_action("Final: 2"))
    agent = BaselineAgent(client=None)
    check("no parse_reply given -> SPaRC parser",
          agent.parse_reply("Final: 3"), parse_action("Final: 3"))

    print()
    if FAILURES:
        print(f"REGRESSION: {len(FAILURES)} check(s) differ from the "
              f"pre-refactor SPaRC path:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("SPaRC path is byte-identical to the pre-refactor implementation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
