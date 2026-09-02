#!/usr/bin/env python
"""Emit the all-runs comparison tables for docs/memp.md, as markdown on stdout.

Every table in the doc's summary section is generated from the result JSONs by this
script, so the numbers cannot drift from the files and a new arm shows up by dropping
its result file in results/json/ and adding one line to RUNS.

Five views, all over the same 500 test puzzles, one attempt each:
  success      solved / accuracy / paired McNemar against that actor's own baseline
  difficulty   accuracy per SPaRC difficulty band 1-5
  failure      terminal_reason breakdown (solved / wrong-solution / dead-end /
               format-failure / error)
  tokens       prompt + completion + retry, per puzzle and per SOLVE (the efficiency
               number: an arm that solves more may still be cheaper per solve)
  steps        env steps per episode, split by outcome -- a solved episode's step count
               is the path length, a failed one's is how long it flailed before dying

Baselines are each model's Reflexion trial 1 (see docs/architecture.md section 4), read
through run_memp.baseline_outcomes' shape rather than re-derived here.

    python scripts/memp_report.py            # all five tables
    python scripts/memp_report.py --only tokens
"""

import argparse
import json
import math
import pathlib
import sys

J = pathlib.Path(__file__).resolve().parents[1] / "results" / "json"

# (label, actor tag, filename). Baselines are inserted automatically per actor.
RUNS = [
    # --- Qwen3-0.6B ---
    ("retrieved k=3, script (v1)",        "qwen06b", "memp_eval_retrieved_test_qwen06b.json"),
    ("random k=3, script (v1)",           "qwen06b", "memp_eval_random_test_qwen06b.json"),
    ("retrieved k=3, script (v2)",        "qwen06b", "memp_eval_retrieved_test_qwen06b_v2.json"),
    ("transfer: 32B bank (v1)",           "qwen06b", "memp_eval_transfer_32b2qwen06b.json"),
    ("transfer: gpt-oss bank (v1)",       "qwen06b", "memp_eval_transfer_oss2qwen06b.json"),
    ("transfer: gpt-oss bank (v2)",       "qwen06b", "memp_eval_retrieved_test_qwen06b_bank-gptoss_v2.json"),
    # --- Qwen3-32B ---
    ("retrieved k=3, script (v1)",        "qwen32b", "memp_eval_retrieved_test_qwen32b.json"),
    ("random k=3, script (v1)",           "qwen32b", "memp_eval_random_test_qwen32b.json"),
    ("retrieved k=3, script (v2)",        "qwen32b", "memp_eval_retrieved_test_qwen32b_v2.json"),
    ("transfer: gpt-oss bank (v2)",       "qwen32b", "memp_eval_retrieved_test_qwen32b_bank-gptoss_v2.json"),
    # --- gpt-oss-120b ---
    ("retrieved k=3, script (v1)",        "gptoss120b", "memp_eval_retrieved_test_gptoss120b.json"),
    ("random k=3, script (v1)",           "gptoss120b", "memp_eval_random_test_gptoss120b.json"),
    ("retrieved k=3, script (v2)",        "gptoss120b", "memp_eval_retrieved_test_gptoss120b_v2.json"),
    ("agent-only bank, script (v1)",      "gptoss120b", "memp_eval_agentonly_gptoss120b.json"),
    ("agent-only bank, script (v2)",      "gptoss120b", "memp_eval_retrieved_test_gptoss120b_agentonly_v2.json"),
    ("retrieved k=10, script (v1)",       "gptoss120b", "memp_eval_k10_gptoss120b.json"),
    ("AveFact keys, script (v1)",         "gptoss120b", "memp_eval_avefact_gptoss120b.json"),
    ("retrieved, TRAJECTORY",             "gptoss120b", "memp_eval_gran_trajectory_gptoss120b.json"),
    ("random, TRAJECTORY",                "gptoss120b", "memp_eval_gran_trajectory_random_gptoss120b.json"),
    ("retrieved, PROCEDURALIZATION (v1)", "gptoss120b", "memp_eval_gran_proceduralization_gptoss120b.json"),
    ("retrieved, PROCEDURALIZATION (v2)", "gptoss120b", "memp_eval_gran_proceduralization_gptoss120b_v2.json"),
]
ACTORS = [("qwen06b", "Qwen3-0.6B"), ("qwen32b", "Qwen3-32B"), ("gptoss120b", "gpt-oss-120b")]
BASELINE = "reflexion_root_{tag}_test.json"
REASONS = ["solved", "wrong-solution", "dead-end", "format-failure", "error"]


def mcnemar_exact(a: int, b: int) -> float:
    """Two-sided exact McNemar from the discordant counts (same as run_memp's)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(name: str) -> dict | None:
    """Per-puzzle records, normalised across the eval and reflexion-root shapes."""
    p = J / name
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "config" in d and d["config"].get("arm") == "root":     # baseline = trial 1
        out = {}
        for pid, e in d["puzzles"].items():
            a = (e.get("attempts") or [{}])[0]
            out[str(pid)] = {**a, "success": e.get("solved", False),
                             "difficulty": e.get("difficulty", -1)}
        return out
    return {str(k): v for k, v in d["puzzles"].items()}


def toks(r: dict) -> int:
    return (r.get("prompt_tokens", 0) or 0) + (r.get("completion_tokens", 0) or 0) \
        + (r.get("retry_tokens", 0) or 0)


def rows_for(tag: str):
    """(label, records) for one actor: its baseline first, then every arm."""
    base = load(BASELINE.format(tag=tag))
    out = [("baseline (memoryless)", base)] if base else []
    for label, t, fn in RUNS:
        if t != tag:
            continue
        recs = load(fn)
        if recs is None:
            print(f"<!-- missing: {fn} -->")
            continue
        out.append((label, recs))
    return base, out


def table_success():
    print("#### Success rate\n")
    print("| Actor | Arm | Solved /500 | Accuracy | vs baseline |")
    print("|---|---|---|---|---|")
    for tag, name in ACTORS:
        base, rows = rows_for(tag)
        for i, (label, r) in enumerate(rows):
            keys = sorted(set(r) & set(base))
            s = sum(r[k]["success"] for k in keys)
            cell = "—"
            if i:
                bo = sum(1 for k in keys if r[k]["success"] and not base[k]["success"])
                ao = sum(1 for k in keys if base[k]["success"] and not r[k]["success"])
                p = mcnemar_exact(ao, bo)
                cell = (f"**{bo-ao:+d}, p={p:.4f}**" if p < 0.05
                        else f"{bo-ao:+d}, p={p:.4f}")
            print(f"| {name if i == 0 else ''} | {label} | {s} | {100*s/len(keys):.1f}% | {cell} |")
    print()


def table_difficulty():
    print("#### Success by difficulty\n")
    print("Accuracy within each SPaRC difficulty band (n = 86 / 118 / 121 / 86 / 89).\n")
    print("| Actor | Arm | d1 | d2 | d3 | d4 | d5 |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        base, rows = rows_for(tag)
        # difficulty comes from the dataset via the baseline; eval files carry it too
        diff = {k: v.get("difficulty", -1) for k, v in base.items()}
        for i, (label, r) in enumerate(rows):
            cells = []
            for d in range(1, 6):
                ks = [k for k in r if diff.get(k) == d]
                cells.append(f"{100*sum(r[k]['success'] for k in ks)/len(ks):.1f}%"
                             if ks else "—")
            print(f"| {name if i == 0 else ''} | {label} | " + " | ".join(cells) + " |")
    print()


def table_failure():
    print("#### Failure modes\n")
    print("Terminal reason for all 500 episodes. `format-failure` = never produced a "
          "parseable move; `error` = a transient serving failure, excluded from "
          "reflection but still counted as an attempt.\n")
    print("| Actor | Arm | Solved | Wrong-sol | Dead-end | Format-fail | Error |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        _b, rows = rows_for(tag)
        for i, (label, r) in enumerate(rows):
            c = {x: 0 for x in REASONS}
            for v in r.values():
                tr = v.get("terminal_reason")
                if v.get("success"):
                    c["solved"] += 1
                elif tr in c:
                    c[tr] += 1
            print(f"| {name if i == 0 else ''} | {label} | " +
                  " | ".join(str(c[x]) for x in REASONS) + " |")
    print()


def table_tokens():
    print("#### Token cost and efficiency\n")
    print("Prompt + completion + retry tokens over all 500 episodes. **Tokens/solve** is "
          "the efficiency figure: an arm that solves more puzzles can still be cheaper "
          "per solved puzzle.\n")
    print("| Actor | Arm | Total | ktok/puzzle | Solved | Mtok/solve | vs baseline |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        base, rows = rows_for(tag)
        base_per = sum(toks(v) for v in base.values()) / max(sum(v["success"] for v in base.values()), 1)
        for i, (label, r) in enumerate(rows):
            tot = sum(toks(v) for v in r.values())
            s = sum(v["success"] for v in r.values())
            per = tot / max(s, 1)
            rel = "—" if i == 0 else f"{100*(per/base_per - 1):+.0f}%"
            print(f"| {name if i == 0 else ''} | {label} | {tot/1e6:.1f}M | "
                  f"{tot/len(r)/1e3:.1f} | {s} | {per/1e6:.3f} | {rel} |")
    print()


def table_steps():
    print("#### Steps in the environment\n")
    print("Env steps per episode (`max_steps=100`). Split by outcome: a solved episode's "
          "step count is essentially the path length, while a failed one shows how long "
          "the agent kept moving before dying.\n")
    print("| Actor | Arm | Mean (all) | Mean (solved) | Mean (failed) |")
    print("|---|---|---|---|---|")
    for tag, name in ACTORS:
        _b, rows = rows_for(tag)
        for i, (label, r) in enumerate(rows):
            allv = [v.get("steps", 0) or 0 for v in r.values()]
            sol = [v.get("steps", 0) or 0 for v in r.values() if v["success"]]
            fail = [v.get("steps", 0) or 0 for v in r.values() if not v["success"]]
            m = lambda xs: f"{sum(xs)/len(xs):.1f}" if xs else "—"
            print(f"| {name if i == 0 else ''} | {label} | {m(allv)} | {m(sol)} | {m(fail)} |")
    print()


def table_breakdown():
    print("#### Token breakdown: prompt vs completion\n")
    print("The same tokens as the previous table, split by kind. **tok/step** is prompt "
          "tokens divided by env steps — the clearest measure of what a memory block "
          "actually costs, since the block is re-sent on every step of the episode. "
          "`retry` is prompt+completion spent on replies that yielded no parseable move "
          "(tracked as one aggregate, not split further).\n")
    print("| Actor | Arm | Prompt | Completion | Retry | Prompt share | Prompt tok/step |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        _b, rows = rows_for(tag)
        for i, (label, r) in enumerate(rows):
            p = sum(v.get("prompt_tokens", 0) or 0 for v in r.values())
            c = sum(v.get("completion_tokens", 0) or 0 for v in r.values())
            rt = sum(v.get("retry_tokens", 0) or 0 for v in r.values())
            steps = sum(v.get("steps", 0) or 0 for v in r.values())
            print(f"| {name if i == 0 else ''} | {label} | {p/1e6:.1f}M | {c/1e6:.1f}M | "
                  f"{rt/1e6:.2f}M | {100*p/max(p+c+rt,1):.0f}% | "
                  f"{p/max(steps,1):.0f} |")
    print()


TABLES = {"success": table_success, "difficulty": table_difficulty,
          "failure": table_failure, "tokens": table_tokens,
          "breakdown": table_breakdown, "steps": table_steps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(TABLES), help="emit just one table")
    args = ap.parse_args()
    for name, fn in TABLES.items():
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
