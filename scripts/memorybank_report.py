#!/usr/bin/env python
"""Emit the all-runs comparison tables for docs/memorybank.md, as markdown on stdout.

The MemoryBank sibling of scripts/memp_report.py, and a thin one on purpose: that
script's six table functions are already method-agnostic -- they read per-puzzle
records, an actor list and a RUNS table, none of which is Memp-specific -- so this
module swaps RUNS and reuses them verbatim. Same reason run_memorybank.py imports
run_memp.py's plumbing: the two methods differ in what the memory says, not in how a
campaign is measured.

Every table in the doc's section 0 is generated from the result JSONs by this script,
so the numbers cannot drift from the files and a new arm shows up by dropping its
result file in results/json/ and adding one line to RUNS.

Six views, all over the same 500 test puzzles, one attempt each:
  success      solved / accuracy / paired McNemar against that actor's own baseline
  difficulty   accuracy per SPaRC difficulty band 1-5
  failure      terminal_reason breakdown (solved / wrong-solution / dead-end /
               format-failure / error)
  tokens       prompt + completion + retry, per puzzle and per SOLVE
  breakdown    the same tokens split prompt vs completion, plus prompt tok/step --
               what a ~1.8k-token case block actually costs when re-sent every step
  steps        env steps per episode, split by outcome

`rp` / `norp` are the two banks per actor: case notes distilled with the rulebook
prepended to the prompt, or without it (the ablation, docs/memorybank.md section 4.2).
Memp's best like-for-like arm is included as a reference row per actor so the
declarative-vs-procedural contrast (section 4.5) is visible at a glance; it is clearly
labelled and is not a MemoryBank arm.

    python scripts/memorybank_report.py            # all six tables
    python scripts/memorybank_report.py --only tokens
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import memp_report
from memp_report import TABLES

# (label, actor tag, filename). Baselines are inserted automatically per actor by
# memp_report.rows_for(); they are each actor's Reflexion trial 1.
RUNS = []
for _tag in ("qwen06b", "qwen32b", "gptoss120b"):
    RUNS += [
        (f"retrieved k=3, rules-preamble", _tag,
         f"memorybank_eval_retrieved_test_{_tag}_rp.json"),
        (f"random k=3, rules-preamble", _tag,
         f"memorybank_eval_random_test_{_tag}_rp.json"),
        (f"retrieved k=3, NO preamble", _tag,
         f"memorybank_eval_retrieved_test_{_tag}_norp.json"),
        (f"random k=3, NO preamble", _tag,
         f"memorybank_eval_random_test_{_tag}_norp.json"),
        # Reference only -- Memp, not MemoryBank. Its rule-correct (v2) retrieved arm
        # is the like-for-like procedural counterpart at the same k and retriever.
        (f"_(ref)_ Memp retrieved, script (v2)", _tag,
         f"memp_eval_retrieved_test_{_tag}_v2.json"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(TABLES), help="emit just one table")
    args = ap.parse_args()
    memp_report.RUNS = RUNS          # the one thing that differs
    for name, fn in TABLES.items():
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
