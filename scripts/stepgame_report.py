#!/usr/bin/env python
"""Emit the all-runs comparison tables for docs/stepgame.md, as markdown on stdout.

The StepGame counterpart of scripts/memp_report.py, and deliberately the same
shape so the two benchmarks can be read side by side. Every number comes from the
result JSONs, so the doc cannot drift from the files: add an arm by dropping its
result file in results/json/ and adding one line to RUNS.

Differences from the SPaRC report, all forced by StepGame being single-turn:

  * no `steps` table -- every episode is exactly one call, so step counts carry no
    information. The slot is taken by a `hops` table instead, which is the axis
    SPaRC has no analogue of (docs/stepgame.md 2.2: k is the story length, the
    true hop count is the reasoning depth, and they are not the same thing).
  * `dead-end` is not a StepGame failure mode. The reasons are solved /
    wrong-answer / format-failure / error.
  * tokens are reported per ITEM, not per step, for the same reason.

Reflexion is NOT in the single-attempt tables. It is a 10-trial retry method, so
its cumulative number is not comparable with a one-attempt arm; it gets its own
table, keyed on trial checkpoints, exactly as docs/reflexion_results.md does.

    python scripts/stepgame_report.py              # every table
    python scripts/stepgame_report.py --only success
"""

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
J = ROOT / "results" / "json"
sys.path.insert(0, str(ROOT / "src"))

SPLIT = ROOT / "data/stepgame/splits/stepgame_clean_test_k1-10_n1000.jsonl"

# The protocol every arm below was run under. Part of the filename, so a run under
# a different actor prompt can never silently join this table -- which the paired
# tests could not detect, since both files would be internally consistent.
PTAG = "_upstream-header"

ACTORS = [("qwen06b", "Qwen3-0.6B"), ("qwen32b", "Qwen3-32B")]
BASELINE = f"reflexion_root_stepgame_{{tag}}_test{PTAG}.json"
REASONS = ["solved", "wrong-answer", "format-failure", "error"]

# (label, actor tag, filename)
RUNS = [
    # --- MemP (procedural) ---
    ("MemP retrieved k=3, trajectory", "qwen06b", f"memp_eval_stepgame_retrieved_test_qwen06b{PTAG}.json"),
    ("MemP random k=3, trajectory",    "qwen06b", f"memp_eval_stepgame_random_test_qwen06b{PTAG}.json"),
    ("MemP fixed (same 3 entries, every item)",     "qwen06b", f"memp_eval_stepgame_fixed_test_qwen06b{PTAG}.json"),
    ("MemP retrieved k=3, SCRIPT",     "qwen06b", f"memp_eval_stepgame_retrieved_test_qwen06b{PTAG}_script.json"),
    ("MemP retrieved k=3, PROCEDURALIZATION", "qwen06b", f"memp_eval_stepgame_retrieved_test_qwen06b{PTAG}_proc.json"),
    # --- MemoryBank (declarative) ---
    ("MemoryBank retrieved k=3",       "qwen06b", f"memorybank_eval_stepgame_retrieved_test_qwen06b{PTAG}_rp.json"),
    ("MemoryBank random k=3",          "qwen06b", f"memorybank_eval_stepgame_random_test_qwen06b{PTAG}_rp.json"),

    ("MemP retrieved k=3, trajectory", "qwen32b", f"memp_eval_stepgame_retrieved_test_qwen32b{PTAG}.json"),
    ("MemP random k=3, trajectory",    "qwen32b", f"memp_eval_stepgame_random_test_qwen32b{PTAG}.json"),
    ("MemP fixed (same 3 entries, every item)",     "qwen32b", f"memp_eval_stepgame_fixed_test_qwen32b{PTAG}.json"),
    ("MemP retrieved k=3, SCRIPT",     "qwen32b", f"memp_eval_stepgame_retrieved_test_qwen32b{PTAG}_script.json"),
    ("MemP retrieved k=3, PROCEDURALIZATION", "qwen32b", f"memp_eval_stepgame_retrieved_test_qwen32b{PTAG}_proc.json"),
    ("MemoryBank retrieved k=3",       "qwen32b", f"memorybank_eval_stepgame_retrieved_test_qwen32b{PTAG}_rp.json"),
    ("MemoryBank random k=3",          "qwen32b", f"memorybank_eval_stepgame_random_test_qwen32b{PTAG}_rp.json"),
    # --- Contrastive (this project's 4th method) ---
    ("Contrastive positive-only, retrieved", "qwen32b", f"contrastive_eval_stepgame_positive-only_retrieved_test_qwen32b{PTAG}.json"),
    ("Contrastive avoid, retrieved",         "qwen32b", f"contrastive_eval_stepgame_avoid_retrieved_test_qwen32b{PTAG}.json"),
    ("Contrastive avoid, random",            "qwen32b", f"contrastive_eval_stepgame_avoid_random_test_qwen32b{PTAG}.json"),
    ("Contrastive avoid, balanced",          "qwen32b", f"contrastive_eval_stepgame_avoid_balanced_test_qwen32b{PTAG}.json"),
    ("Contrastive hindsight, retrieved",     "qwen32b", f"contrastive_eval_stepgame_hindsight_retrieved_test_qwen32b{PTAG}.json"),
]

# Reflexion's two arms, compared to each other rather than to a single-attempt row.
REFLEXION = [("memory", f"reflexion_memory_stepgame_{{tag}}_test{PTAG}.json"),
             ("control", f"reflexion_control_stepgame_{{tag}}_test{PTAG}.json")]

# Reflexion also appears in every table below, as trial CHECKPOINTS rather than as a
# file -- the convention docs/reflexion_results.md uses. A checkpoint row is
# "where every item stands after trial T", and its tokens are cumulative over
# trials 1..T, i.e. the real price of reaching it. These rows are therefore NOT
# one-attempt numbers and every caption says so.
REFLEXION_CHECKPOINTS = [("memory", 2), ("control", 2), ("memory", 10), ("control", 10)]

CUMULATIVE_NOTE = (
    "Reflexion rows are trial **checkpoints**, not one-attempt arms: accuracy is "
    "\"solved at least once by trial T\" and tokens are cumulative over trials 1..T. "
    "They are the price of reaching that checkpoint, and are not comparable "
    "like-for-like with the single-attempt rows above them."
)


def mcnemar_exact(a: int, b: int) -> float:
    """Two-sided exact McNemar from the discordant counts (same as run_memp's)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


ROWS = None      # set by main(): rows_for or reflexion_rows_for
PREFIX = ""      # set by main(): namespaces the Reflexion pass's repeated views


def head(title: str) -> str:
    """Heading for a view, namespaced when the same view is repeated for Reflexion."""
    return f"#### {PREFIX}{title}\n" if PREFIX else f"### {title}\n"


def hops_of() -> dict:
    """puzzle_id -> true hop count, read from the frozen split.

    Taken from the split rather than from a result file because the Reflexion root
    stores per-trial attempts without the `extra` block, so the two shapes do not
    both carry it -- and the split is the authority anyway.
    """
    from memory_agents.envs import stepgame
    return {i.id: i.hops for i in stepgame.read_split(SPLIT)}


def load(name: str) -> dict | None:
    """Per-item records, normalised across the eval and reflexion-root shapes.

    A run with any `errors` is REFUSED, not reported. A serving failure still
    writes a well-formed JSON with a plausible-looking accuracy (one GPU fault
    produced a complete file reading 0.099), and that must never reach a table.
    """
    p = J / name
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    errs = (d.get("summary") or {}).get("errors", 0)
    if errs:
        print(f"<!-- REFUSED {name}: {errs} errored episodes, not a valid run -->")
        return None
    if "config" in d and d["config"].get("arm") == "root":     # baseline = trial 1
        out = {}
        for pid, e in d["puzzles"].items():
            a = (e.get("attempts") or [{}])[0]
            out[str(pid)] = {**a, "success": a.get("success", False),
                             "difficulty": e.get("difficulty", -1)}
        return out
    recs = d.get("episodes") or d["puzzles"]
    if isinstance(recs, list):
        return {str(e["puzzle_id"]): e for e in recs}
    return {str(k): v for k, v in recs.items()}


def load_reflexion(tag: str, arm: str, T: int) -> dict | None:
    """Per-item records for one Reflexion arm as of trial T.

    Collapses the multi-trial shape into the same record shape the eval files have,
    so every table can render it without special-casing: an item counts as solved
    if ANY attempt up to T solved it, tokens are summed over attempts 1..T, and the
    terminal reason is the last attempt's (an item solved at trial 3 reads `solved`
    at T=10 even though attempts 1-2 failed -- which is what a checkpoint means).
    """
    fn = dict(REFLEXION)[arm].format(tag=tag)
    q = J / fn
    if not q.exists():
        return None
    d = json.loads(q.read_text())
    out = {}
    for pid, e in d["puzzles"].items():
        att = [a for a in (e.get("attempts") or []) if (a.get("trial") or 1) <= T]
        won = bool(e.get("solved")) and (e.get("solved_at_trial") or 99) <= T
        last = att[-1] if att else {}
        out[str(pid)] = {
            "success": won,
            "difficulty": e.get("difficulty", -1),
            "terminal_reason": "solved" if won else last.get("terminal_reason"),
            "prompt_tokens": sum(a.get("prompt_tokens", 0) or 0 for a in att),
            "completion_tokens": sum(a.get("completion_tokens", 0) or 0 for a in att),
            "retry_tokens": sum(a.get("retry_tokens", 0) or 0 for a in att),
        }
    return out


def toks(r: dict) -> int:
    return (r.get("prompt_tokens", 0) or 0) + (r.get("completion_tokens", 0) or 0) \
        + (r.get("retry_tokens", 0) or 0)


def rows_for(tag: str):
    """(baseline, [(label, records)]) for one actor: baseline first, then every arm."""
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


def reflexion_rows_for(tag: str):
    """The same shape as rows_for(), but the arms are Reflexion trial checkpoints.

    Kept as a separate row source, and rendered under its own heading, because a
    checkpoint is not a one-attempt arm: its accuracy is "solved at least once by
    trial T" and its tokens are cumulative over trials 1..T. Mixing the two row
    kinds in one table invites exactly the like-for-like reading they do not support.
    """
    base = load(BASELINE.format(tag=tag))
    out = [("baseline (memoryless) = T1", base)] if base else []
    for arm, T in REFLEXION_CHECKPOINTS:
        recs = load_reflexion(tag, arm, T)
        if recs is not None:
            out.append((f"Reflexion {arm} @T{T}", recs))
    return base, out


def table_success():
    print(head("Success rate"))
    print("**vs baseline** is the net items gained over that actor's own memoryless "
          "baseline (Reflexion trial 1) on the identical 1000 items. **Bold** = "
          "clears p < 0.05 on a paired exact McNemar.\n")
    print("| Actor | Arm | Solved /1000 | Accuracy | vs baseline |")
    print("|---|---|---|---|---|")
    for tag, name in ACTORS:
        base, rows = ROWS(tag)
        for i, (label, r) in enumerate(rows):
            keys = sorted(set(r) & set(base))
            s = sum(r[k]["success"] for k in keys)
            cell = "—"
            if i:
                bo = sum(1 for k in keys if r[k]["success"] and not base[k]["success"])
                ao = sum(1 for k in keys if base[k]["success"] and not r[k]["success"])
                p = mcnemar_exact(ao, bo)
                # Net items gained only. The p-value is still computed and still
                # decides the bold: an unbolded delta did NOT clear p<0.05, which
                # is the one distinction the column would otherwise lose.
                cell = f"**{bo-ao:+d}**" if p < 0.05 else f"{bo-ao:+d}"
            print(f"| {name if i == 0 else ''} | {label} | {s} | {100*s/len(keys):.1f}% | {cell} |")
    print()


def table_difficulty():
    print(head("Success by *k* (story length)"))
    print("Accuracy within each band, as a percentage. 100 items per band, so a "
          "rate and a count coincide numerically here — the unit is on every cell "
          "to keep that unambiguous. Note *k* is the story length, not the "
          "reasoning depth — see §2.2.\n")
    print("| Actor | Arm | " + " | ".join(f"k{d}" for d in range(1, 11)) + " |")
    print("|---|---|" + "---|" * 10)
    for tag, name in ACTORS:
        base, rows = ROWS(tag)
        diff = {k: v.get("difficulty", -1) for k, v in base.items()}
        for i, (label, r) in enumerate(rows):
            cells = []
            for d in range(1, 11):
                ks = [k for k in r if diff.get(k) == d]
                cells.append(f"{100*sum(r[k]['success'] for k in ks)/len(ks):.0f}%"
                             if ks else "—")
            print(f"| {name if i == 0 else ''} | {label} | " + " | ".join(cells) + " |")
    print()


def table_hops():
    print(head("Success by true hop count (reasoning depth)"))
    print("The honest difficulty axis. Bucket sizes are uneven on the k-stratified "
          "split (406 items at h=1, 4 at h=10), so the last two columns aggregate "
          "the band that actually carries the difficulty signal.\n")
    print("| Actor | Arm | h1 | h2 | h3 | h4 | h5 | h6+ | **h≤3** | **h≥4** |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    H = hops_of()
    for tag, name in ACTORS:
        _b, rows = ROWS(tag)
        for i, (label, r) in enumerate(rows):
            cells = []
            for h in range(1, 6):
                ks = [k for k in r if H.get(k) == h]
                cells.append(f"{100*sum(r[k]['success'] for k in ks)/len(ks):.0f}%" if ks else "—")
            for sel in (lambda x: x >= 6, lambda x: x <= 3, lambda x: x >= 4):
                ks = [k for k in r if sel(H.get(k, -1))]
                cells.append(f"{100*sum(r[k]['success'] for k in ks)/len(ks):.0f}%" if ks else "—")
            print(f"| {name if i == 0 else ''} | {label} | " + " | ".join(cells) + " |")
    print()


def table_failure():
    print(head("Failure modes"))
    print("Terminal reason for all 1000 episodes. `format-failure` = never produced a "
          "parseable label after the retries, and is counted separately from a wrong "
          "answer on purpose: with nine labels and a ~11% chance floor, folding "
          "unparseable replies into the accuracy would hide them.\n")
    print("| Actor | Arm | Solved | Wrong-answer | Format-fail | Error |")
    print("|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        _b, rows = ROWS(tag)
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
    print(head("Token cost and efficiency"))
    print("Prompt + completion + retry over all 1000 episodes. **ktok/solve** is the "
          "efficiency figure: an arm that solves more items can still be dearer per "
          "solved item, and on a one-call benchmark that is the only cost that "
          "matters.\n")
    print("| Actor | Arm | Total | ktok/item | Solved | ktok/solve | vs baseline |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        base, rows = ROWS(tag)
        base_per = sum(toks(v) for v in base.values()) / max(
            sum(v["success"] for v in base.values()), 1)
        for i, (label, r) in enumerate(rows):
            tot = sum(toks(v) for v in r.values())
            s = sum(v["success"] for v in r.values())
            per = tot / max(s, 1)
            rel = "—" if i == 0 else f"{100*(per/base_per - 1):+.0f}%"
            print(f"| {name if i == 0 else ''} | {label} | {tot/1e6:.2f}M | "
                  f"{tot/len(r)/1e3:.1f} | {s} | {per/1e3:.1f} | {rel} |")
    print()


def table_breakdown():
    print(head("Token breakdown: prompt vs completion"))
    print("The same tokens split by kind. On a single-turn benchmark the memory block "
          "is sent exactly once per item, so **prompt tok/item** is the direct price "
          "of the block — unlike SPaRC, where it was re-sent on every step. `retry` is "
          "prompt+completion spent on replies that yielded no parseable label.\n")
    print("| Actor | Arm | Prompt | Completion | Retry | Prompt share | Prompt tok/item |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        _b, rows = ROWS(tag)
        for i, (label, r) in enumerate(rows):
            p = sum(v.get("prompt_tokens", 0) or 0 for v in r.values())
            c = sum(v.get("completion_tokens", 0) or 0 for v in r.values())
            rt = sum(v.get("retry_tokens", 0) or 0 for v in r.values())
            print(f"| {name if i == 0 else ''} | {label} | {p/1e6:.2f}M | {c/1e6:.2f}M | "
                  f"{rt/1e6:.2f}M | {100*p/max(p+c+rt,1):.0f}% | {p/max(len(r),1):.0f} |")
    print()


def table_reflexion():
    print("#### Reflexion: memory vs control (10 trials)\n")
    print("Kept out of the tables above because it is a *retry* method: the figure is "
          "cumulative over trials, so it is not comparable with a one-attempt arm. "
          "Both arms share trial 1 and retry the same unsolved items at the same "
          "per-trial seeds, so Reflexion's effect is the memory-minus-control delta, "
          "not either curve on its own.\n")
    print("Checkpoints follow `docs/reflexion_results.md`. **T1 is the memoryless "
          "baseline** — both arms are the identical trial-1 pass, so its delta is 0 by "
          "construction and it is here as the anchor the other rows move from. **T2 is "
          "the state after a single reflection**, which isolates the first and usually "
          "largest effect; T10 is the end state. **Bold** delta = clears p < 0.05 on a "
          "paired exact McNemar between the two arms at that checkpoint.\n")
    print("| Actor | Checkpoint | Control | Memory | Delta | Memory alone won | Control alone won |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        arms = {}
        for label, fn in REFLEXION:
            q = J / fn.format(tag=tag)
            if q.exists():
                arms[label] = json.loads(q.read_text())["puzzles"]
        if len(arms) != 2:
            continue
        m, c = arms["memory"], arms["control"]
        ids = sorted(set(m) & set(c))
        for row, (T, tick) in enumerate(((1, "T1 (baseline, shared)"),
                                         (2, "T2 (one reflection)"),
                                         (10, "T10 (final)"))):
            def ok(d, i):
                return bool(d[i].get("solved")) and (d[i].get("solved_at_trial") or 99) <= T
            ms = sum(ok(m, i) for i in ids)
            cs = sum(ok(c, i) for i in ids)
            mo = sum(1 for i in ids if ok(m, i) and not ok(c, i))
            co = sum(1 for i in ids if ok(c, i) and not ok(m, i))
            pv = mcnemar_exact(mo, co)
            d = 100 * (ms - cs) / len(ids)
            cell = f"**{d:+.1f}pp**" if pv < 0.05 else f"{d:+.1f}pp"
            print(f"| {name if row == 0 else ''} | {tick} | {cs} | {ms} | {cell} | "
                  f"{mo} | {co} |")
    print()


# The single-attempt views, rendered once for the memory methods and once for the
# Reflexion checkpoints. `reflexion` is the memory-vs-control test and stands alone.
VIEWS = {"success": table_success, "difficulty": table_difficulty,
         "hops": table_hops, "failure": table_failure,
         "tokens": table_tokens, "breakdown": table_breakdown}
TABLES = {**VIEWS, "reflexion": table_reflexion}


def main() -> None:
    global ROWS, PREFIX
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(TABLES), help="emit just one table")
    args = ap.parse_args()

    ROWS = rows_for
    for name, fn in VIEWS.items():
        if args.only in (None, name):
            fn()

    # Reflexion, entirely separate: the same views over trial checkpoints, then the
    # memory-vs-control test the method actually turns on.
    if args.only is None or args.only == "reflexion":
        print("---\n")
        print("### Reflexion (episodic) — its own tables\n")
        print("Reflexion is a *retry* method, so none of its numbers belong in the "
              "tables above: a checkpoint's accuracy is \"solved at least once by "
              "trial T\" and its tokens are cumulative over trials 1..T — the price of "
              "reaching that checkpoint, not the cost of one attempt. The same views "
              "are therefore repeated here over `T1` (the shared memoryless baseline), "
              "`@T2` (after a single reflection) and `@T10` (the end state).\n")
        print("Both arms start from the identical trial-1 pass and retry the same "
              "unsolved items at the same per-trial seeds, so **Reflexion's effect is "
              "the memory-minus-control delta at a given checkpoint**, never either "
              "curve on its own — each arm's number rises with the trial index whether "
              "or not memory helps.\n")
        table_reflexion()
        ROWS, PREFIX = reflexion_rows_for, "Reflexion — "
        for fn in VIEWS.values():
            fn()


if __name__ == "__main__":
    main()
