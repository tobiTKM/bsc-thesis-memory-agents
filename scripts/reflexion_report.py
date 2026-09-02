#!/usr/bin/env python
"""Emit the all-runs comparison tables for docs/reflexion_results.md, as markdown.

The Reflexion sibling of scripts/memp_report.py and scripts/memorybank_report.py, so the
three method documents open with the same six views and can be read side by side.

Reflexion does not fit those scripts' RUNS-table shape, because its "runs" are not
separate files: one arm is a single state file holding ten trials, and a puzzle solved at
trial 3 is never re-attempted. What makes the shared tables work anyway is that every one
of them consumes the same thing -- a {puzzle_id: record} dict -- so this module builds
that dict for a given (arm, trial) CHECKPOINT and overrides memp_report.rows_for. All six
table functions then run unmodified.

A checkpoint record answers "where does this puzzle stand after trial T":
  success          solved at any trial <= T (solved puzzles are carried forward)
  terminal_reason  'solved', else the terminal reason of its most recent attempt <= T
  steps / tokens   CUMULATIVE over trials 1..T -- the honest price of reaching that
                   checkpoint, since a retry method pays for every attempt it made

Trial 1 is the shared memoryless baseline: both arms start from the identical trial-1
episodes (docs/architecture.md section 3), so it appears once per actor.

Two extra tables have no analogue in the other two methods and are Reflexion-specific:
  curve      cumulative accuracy per trial, both arms side by side
  contrast   memory vs control at T2 and T10, paired exact McNemar -- THE test for this
             method, since accuracy rises with the trial index whether or not memory
             does anything (which is why the control arm exists at all)
  reflection the self-reflection generations' own token cost, which the shared token
             tables cannot show: it is a per-trial call, not part of any episode record

    python scripts/reflexion_report.py            # all nine tables
    python scripts/reflexion_report.py --only curve
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import memp_report
from memp_report import ACTORS, J, TABLES, mcnemar_exact

CHECKPOINTS = [2, 10]          # trial 1 is the baseline row; these are the two the doc reads
ARMS = [("control", "control"), ("memory", "memory")]


def state(tag: str, arm: str) -> dict | None:
    p = J / f"reflexion_{arm}_{tag}_test.json"
    return json.loads(p.read_text()) if p.exists() else None


def records_at(st: dict, trial: int) -> dict:
    """{pid: record} describing where every puzzle stands after `trial`."""
    out = {}
    for pid, e in st["puzzles"].items():
        seen = [a for a in e["attempts"] if a["trial"] <= trial]
        solved = bool(e.get("solved")) and (e.get("solved_at_trial") or 99) <= trial
        last = seen[-1] if seen else {}
        out[str(pid)] = {
            "success": solved,
            "difficulty": e.get("difficulty", -1),
            "terminal_reason": "solved" if solved else last.get("terminal_reason"),
            # Cumulative: a retry method's cost is everything it spent getting here.
            "steps": sum(a.get("steps", 0) or 0 for a in seen),
            "prompt_tokens": sum(a.get("prompt_tokens", 0) or 0 for a in seen),
            "completion_tokens": sum(a.get("completion_tokens", 0) or 0 for a in seen),
            "retry_tokens": sum(a.get("retry_tokens", 0) or 0 for a in seen),
        }
    return out


def rows_for(tag: str):
    """memp_report's seam, reimplemented over trial checkpoints instead of files."""
    mem, ctl = state(tag, "memory"), state(tag, "control")
    if mem is None:
        print(f"<!-- missing: reflexion_memory_{tag}_test.json -->")
        return {}, []
    base = records_at(mem, 1)                     # trial 1 is shared by both arms
    rows = [("baseline (memoryless, T1)", base)]
    for t in CHECKPOINTS:
        if ctl is not None:
            rows.append((f"control @T{t}", records_at(ctl, t)))
        rows.append((f"memory @T{t}", records_at(mem, t)))
    return base, rows


def table_curve() -> None:
    print("#### Trial curve\n")
    print("Cumulative accuracy after each trial, both arms. Solved puzzles are carried "
          "forward and never re-attempted, so these curves can only rise — which is "
          "exactly why the control arm exists: the *gap* is the effect, not the slope.\n")
    hdr = " | ".join(f"T{t}" for t in range(1, 11))
    print(f"| Actor | Arm | {hdr} |")
    print("|---|---|" + "---|" * 10)
    for tag, name in ACTORS:
        for i, (arm, _lbl) in enumerate(ARMS):
            st = state(tag, arm)
            if st is None:
                continue
            cells = []
            for t in range(1, 11):
                row = next((c for c in st["trials"] if c["trial"] == t), None)
                cells.append(f"{100*row['cumulative_accuracy']:.1f}%" if row else "—")
            print(f"| {name if i == 0 else ''} | {arm} | " + " | ".join(cells) + " |")
    print()


def table_contrast() -> None:
    print("#### Memory vs control (the load-bearing test)\n")
    print("Paired exact McNemar between the two arms at the same trial index, over the "
          "same 500 puzzles at the same per-trial seeds. Against the *baseline* every arm "
          "including the control is overwhelmingly significant, because re-attempting a "
          "puzzle nine more times converts some of it with no memory at all; only this "
          "contrast isolates the reflection.\n")
    print("| Actor | Checkpoint | Control | Memory | Delta | Memory alone won | "
          "Control alone won | p |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        mem, ctl = state(tag, "memory"), state(tag, "control")
        if mem is None or ctl is None:
            continue
        first = True
        for t in CHECKPOINTS:
            m, c = records_at(mem, t), records_at(ctl, t)
            keys = sorted(set(m) & set(c))
            ms = sum(m[k]["success"] for k in keys)
            cs = sum(c[k]["success"] for k in keys)
            mo = sum(1 for k in keys if m[k]["success"] and not c[k]["success"])
            co = sum(1 for k in keys if c[k]["success"] and not m[k]["success"])
            p = mcnemar_exact(mo, co)
            cell = f"**p={p:.4f}**" if p < 0.05 else f"p={p:.4f}"
            print(f"| {name if first else ''} | T{t} | {cs} | {ms} | "
                  f"{100*(ms-cs)/len(keys):+.1f}pp | {mo} | {co} | {cell} |")
            first = False
    print()


def table_reflection_cost() -> None:
    print("#### Reflection overhead (what the memory arm pays that the control does not)\n")
    print("The token tables above count ACTOR tokens only, because reflection is a "
          "per-trial call and not attributable to any one puzzle's episode record. This "
          "is the missing line item: the self-reflection generations themselves, which "
          "the control arm never makes.\n")
    print("| Actor | Checkpoint | Actor tokens | Reflection tokens | Reflection share | "
          "Reflections written |")
    print("|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        st = state(tag, "memory")
        if st is None:
            continue
        first = True
        for t in CHECKPOINTS:
            cs = [c for c in st["trials"] if c["trial"] <= t]
            a = sum(c["actor_prompt_tokens"] + c["actor_completion_tokens"]
                    + c["actor_retry_tokens"] for c in cs)
            r = sum(c["reflection_prompt_tokens"] + c["reflection_completion_tokens"]
                    for c in cs)
            w = sum(c.get("reflections_written", 0) for c in cs)
            print(f"| {name if first else ''} | T{t} | {a/1e6:.1f}M | {r/1e6:.2f}M | "
                  f"{100*r/max(a+r,1):.1f}% | {w} |")
            first = False
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    extra = {"curve": table_curve, "contrast": table_contrast,
             "reflection": table_reflection_cost}
    ap.add_argument("--only", choices=sorted(set(TABLES) | set(extra)))
    args = ap.parse_args()
    memp_report.rows_for = rows_for          # the one seam that differs
    order = ([("curve", table_curve), ("contrast", table_contrast)]
             + list(TABLES.items()) + [("reflection", table_reflection_cost)])
    for name, fn in order:
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
