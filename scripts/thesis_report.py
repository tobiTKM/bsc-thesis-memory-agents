#!/usr/bin/env python
"""Regenerate the table of every run the thesis reports. No GPU, stdlib only.

    python scripts/thesis_report.py            # print it
    python scripts/thesis_report.py --write    # -> results/thesis/thesis_reported_runs.md
    python scripts/thesis_report.py --check    # nonzero if that file is out of date
    python scripts/thesis_report.py --audit    # list result files no row claims or excludes
"""

import argparse
import json
import math
import pathlib
import subprocess
import sys
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[1]
J = ROOT / "results" / "json"
OUT = ROOT / "results" / "thesis"
DOC = OUT / "thesis_reported_runs.md"

SPARC = [("qwen06b", "Qwen3-0.6B"), ("qwen32b", "Qwen3-32B"),
         ("gptoss120b", "gpt-oss-120b")]
STEP = [("qwen06b", "Qwen3-0.6B"), ("qwen32b", "Qwen3-32B")]
PT = "_upstream-header"

# label -> filename template; `{t}` is the actor tag. A file not on disk renders as "— (queued)".
SPARC_RUNS = [
    ("MemP — script v2, retrieved",            "memp_eval_retrieved_test_{t}_v2.json"),
    ("MemP — script v2, random",               "memp_eval_random_test_{t}_v2.json"),
    ("MemP — trajectory, retrieved",           "memp_eval_gran_trajectory_{t}.json"),
    ("MemP — trajectory, random",              "memp_eval_gran_trajectory_random_{t}.json"),
    ("MemP — proceduralization v2, retrieved", "memp_eval_gran_proceduralization_{t}_v2.json"),
    ("MemP — proceduralization v2, random",    "memp_eval_gran_proceduralization_random_{t}_v2.json"),
    ("MemP — agent-only bank v2, retrieved",   "memp_eval_retrieved_test_{t}_agentonly_v2.json"),
    ("MemP — agent-only bank v2, random",      "memp_eval_random_test_{t}_agentonly_v2.json"),
    ("MemoryBank — rp, retrieved",             "memorybank_eval_retrieved_test_{t}_rp.json"),
    ("MemoryBank — rp, random",                "memorybank_eval_random_test_{t}_rp.json"),
    ("MemoryBank — norp, retrieved",           "memorybank_eval_retrieved_test_{t}_norp.json"),
    ("MemoryBank — norp, random",              "memorybank_eval_random_test_{t}_norp.json"),
    ("Contrastive — retrieved",                "contrastive_eval_avoid_retrieved_test_{t}_ex.json"),
    ("Contrastive — random",                   "contrastive_eval_avoid_random_test_{t}_ex.json"),
]
STEP_RUNS = [
    ("MemP — script, retrieved",            "memp_eval_stepgame_retrieved_test_{t}" + PT + "_script.json"),
    ("MemP — script, random",               "memp_eval_stepgame_random_test_{t}" + PT + "_script.json"),
    ("MemP — trajectory, retrieved",        "memp_eval_stepgame_retrieved_test_{t}" + PT + ".json"),
    ("MemP — trajectory, random",           "memp_eval_stepgame_random_test_{t}" + PT + ".json"),
    ("MemP — proceduralization, retrieved", "memp_eval_stepgame_retrieved_test_{t}" + PT + "_proc.json"),
    ("MemP — proceduralization, random",    "memp_eval_stepgame_random_test_{t}" + PT + "_proc.json"),
    ("MemP — agent-only bank, retrieved",   "memp_eval_stepgame_retrieved_test_{t}" + PT + "_agentonly.json"),
    ("MemP — agent-only bank, random",      "memp_eval_stepgame_random_test_{t}" + PT + "_agentonly.json"),
    ("MemoryBank — rp, retrieved",          "memorybank_eval_stepgame_retrieved_test_{t}" + PT + "_rp.json"),
    ("MemoryBank — rp, random",             "memorybank_eval_stepgame_random_test_{t}" + PT + "_rp.json"),
    ("MemoryBank — norp, retrieved",        "memorybank_eval_stepgame_retrieved_test_{t}" + PT + "_norp.json"),
    ("MemoryBank — norp, random",           "memorybank_eval_stepgame_random_test_{t}" + PT + "_norp.json"),
    ("Contrastive — retrieved",             "contrastive_eval_stepgame_avoid_retrieved_test_{t}" + PT + "_ex.json"),
    ("Contrastive — random",                "contrastive_eval_stepgame_avoid_random_test_{t}" + PT + "_ex.json"),
]
BASE = {"sparc": "reflexion_root_{t}_test.json",
        "stepgame": "reflexion_root_stepgame_{t}_test" + PT + ".json"}
REFL = {"sparc": ("reflexion_memory_{t}_test.json", "reflexion_control_{t}_test.json"),
        "stepgame": ("reflexion_memory_stepgame_{t}_test" + PT + ".json",
                     "reflexion_control_stepgame_{t}_test" + PT + ".json")}
REASONS = ["wrong-solution", "dead-end", "format-failure"]

# Note shown next to a pending run; keys must match the RUNS labels exactly.
NOTES = {
    "MemP — script v2, random": "the retrieved-vs-random pair at the rule-correct bank; only v1 was ever run",
    "MemP — proceduralization v2, random": "never run at any actor",
    "MemP — agent-only bank v2, random": "never run at any actor",
    "MemP — agent-only bank, retrieved": "needs a bank rebuild restricted to agent-solved entries (323 / 82 of 500)",
    "MemP — agent-only bank, random": "needs the same rebuild",
    "MemoryBank — norp, retrieved": "**needs a distill pass** — only the `rp` bank exists on StepGame",
    "MemoryBank — norp, random": "**needs a distill pass** — only the `rp` bank exists on StepGame",
}


def mcnemar(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def fmt_p(p: float) -> str:
    """Three decimals down to 0.001, then scientific, so a tiny p never prints as 0."""
    if p < 1e-3:
        return "p=" + f"{p:.0e}".replace("e-0", "e-")
    return f"p={p:.3f}"


def load_json(name: str):
    return json.loads((J / name).read_text())


class Run:
    """One eval file, normalised. `.ok` is False for missing OR partially-written."""

    def __init__(self, name):
        self.name, self.ok, self.why = name, False, "queued"
        if not (J / name).exists():
            return
        d = load_json(name)
        cfg = d.get("config", {})
        if cfg.get("arm") == "root" or "trials" in d:
            # A Reflexion state file, or a root (trial 1 = the memoryless baseline).
            # The puzzle's cumulative `success` must win over the last attempt's.
            # Tokens are summed over all attempts: a retry method pays for every one.
            self.rec = {}
            for k, v in d["puzzles"].items():
                att = v.get("attempts") or [{}]
                rec = dict(att[-1])
                rec["success"] = v["solved"]
                rec["solved_at_trial"] = v.get("solved_at_trial")
                # Per-trial reasons and tokens, so a table can be cut at any trial T.
                rec["_reasons"] = [a.get("terminal_reason") for a in att]
                for f in ("prompt_tokens", "completion_tokens", "retry_tokens"):
                    rec[f] = sum(a.get(f) or 0 for a in att)
                rec["_toks"] = [sum((a.get(f) or 0) for f in
                                    ("prompt_tokens", "completion_tokens",
                                     "retry_tokens")) for a in att]
                self.rec[k] = rec
            self.ok = True
            return
        if "episodes" in d:
            self.rec = {str(e["puzzle_id"]): e for e in d["episodes"]}
            self.ok = True
            return
        # `summary` is written only when an eval finishes; without it this is a checkpoint.
        if "summary" not in d:
            self.why = f"running, n={len(d.get('puzzles', {}))}"
            return
        self.rec = d["puzzles"]
        self.ok = True

    def solved(self, keys=None):
        ks = keys if keys is not None else self.rec
        return sum(1 for k in ks if self.rec[k]["success"])

    def toks(self, keys=None):
        ks = keys if keys is not None else self.rec
        return sum((self.rec[k].get("prompt_tokens") or 0)
                   + (self.rec[k].get("completion_tokens") or 0)
                   + (self.rec[k].get("retry_tokens") or 0) for k in ks)

    def reason(self, r, upto=None):
        if upto is None:
            return sum(1 for v in self.rec.values() if v.get("terminal_reason") == r)
        n = 0
        for v in self.rec.values():
            sat = v.get("solved_at_trial")
            if sat is not None and sat <= upto:
                n += (r == "solved")
                continue
            rs = v.get("_reasons") or [v.get("terminal_reason")]
            n += (rs[min(upto, len(rs)) - 1] == r)
        return n

    def toks_upto(self, upto=None):
        if upto is None:
            return self.toks()
        return sum(sum(v["_toks"][:upto]) for v in self.rec.values())

    def solved_upto(self, upto=None):
        if upto is None:
            return self.solved()
        return sum(1 for v in self.rec.values()
                   if (v.get("solved_at_trial") or 99) <= upto)


def cell(run: Run, base: Run, n: int) -> str:
    if not run.ok:
        return f"— ({run.why})"
    keys = sorted(set(run.rec) & set(base.rec))
    s = run.solved(keys)
    up = sum(1 for k in keys if run.rec[k]["success"] and not base.rec[k]["success"])
    dn = sum(1 for k in keys if base.rec[k]["success"] and not run.rec[k]["success"])
    p = mcnemar(up, dn)
    body = f"{s} ({up-dn:+d}, {fmt_p(p)})"
    return f"**{body}**" if p < 0.05 else body


def refl_cell(bench, tag, upto=None):
    """`memory / control` at trial `upto` (None = last), tested between the two arms.

    Not against the baseline: solves carry forward, so the curve rises even without memory.
    """
    mf, cf = (f.format(t=tag) for f in REFL[bench])
    m, c = Run(mf), Run(cf)
    if not (m.ok and c.ok):
        return "— (queued)"
    def at(run, k):
        if upto is None:
            return bool(run.rec[k]["success"])
        s = run.rec[k].get("solved_at_trial")
        return bool(s is not None and s <= upto)
    keys = sorted(set(m.rec) & set(c.rec))
    up = sum(1 for k in keys if at(m, k) and not at(c, k))
    dn = sum(1 for k in keys if at(c, k) and not at(m, k))
    p = mcnemar(up, dn)
    body = (f"{sum(1 for k in keys if at(m,k))} / {sum(1 for k in keys if at(c,k))}"
            f" ({up-dn:+d}, {fmt_p(p)})")
    return f"**{body}**" if p < 0.05 else body


def solved_table(bench, actors, runs, n):
    base = {t: Run(BASE[bench].format(t=t)) for t, _ in actors}
    out = ["| Arm | " + " | ".join(nm for _t, nm in actors) + " |",
           "|---|" + "---|" * len(actors)]
    out.append("| **baseline (memoryless)** | " + " | ".join(
        f"{base[t].solved()} ({100*base[t].solved()/n:.1f}%)" if base[t].ok else "—"
        for t, _ in actors) + " |")
    for upto, lab in ((2, "Reflexion @T2"), (None, "Reflexion @T10")):
        out.append(f"| {lab} | " + " | ".join(
            refl_cell(bench, t, upto) for t, _ in actors) + " |")
    for label, tpl in runs:
        out.append(f"| {label} | " + " | ".join(
            cell(Run(tpl.format(t=t)), base[t], n) if base[t].ok else "—"
            for t, _ in actors) + " |")
    return "\n".join(out)


def failure_tables(actors, runs):
    blocks = []
    for t, nm in actors:
        base = Run(BASE["sparc"].format(t=t))
        if not base.ok:
            continue
        bw, bd = base.reason("wrong-solution"), base.reason("dead-end")
        blocks.append(f"**{nm}** — baseline {base.solved()} solved / {bw} wrong-sol / "
                      f"{bd} dead-end / {base.reason('format-failure')} format-fail\n")
        blocks.append("| Arm | solved | wrong-sol | Δ | dead-end | Δ | format-fail |")
        blocks.append("|---|---|---|---|---|---|---|")
        for label, tpl in runs:
            r = Run(tpl.format(t=t))
            if not r.ok:
                continue
            w, d = r.reason("wrong-solution"), r.reason("dead-end")
            blocks.append(f"| {label} | {r.solved()} | {w} | {w-bw:+d} | {d} | "
                          f"{d-bd:+d} | {r.reason('format-failure')} |")
        # Terminal reasons are cut at each checkpoint, not read off the last trial.
        m = Run(REFL["sparc"][0].format(t=t))
        if m.ok:
            for upto, lab in ((2, "Reflexion memory @T2"), (None, "Reflexion memory @T10")):
                w, d = m.reason("wrong-solution", upto), m.reason("dead-end", upto)
                blocks.append(f"| {lab} | {m.solved_upto(upto)} | {w} | {w-bw:+d} | "
                              f"{d} | {d-bd:+d} | {m.reason('format-failure', upto)} |")
        blocks.append("")
    return "\n".join(blocks)


def cost_table(bench, actors, runs, unit, scale):
    base = {t: Run(BASE[bench].format(t=t)) for t, _ in actors}
    out = ["| Arm | " + " | ".join(nm for _t, nm in actors) + " |",
           "|---|" + "---|" * len(actors)]
    def c(r):
        return f"{r.toks()/max(r.solved(),1)/scale:.3f}" if scale > 1e4 else \
               f"{r.toks()/max(r.solved(),1)/scale:.1f}"
    out.append("| baseline | " + " | ".join(
        c(base[t]) if base[t].ok else "—" for t, _ in actors) + " |")
    # Reflexion's cost is cumulative over trials 1..T.
    for upto, lab in ((2, "Reflexion memory @T2"), (None, "Reflexion memory @T10")):
        cells = []
        for t, _ in actors:
            r = Run(REFL[bench][0].format(t=t))
            if not r.ok:
                cells.append("— (queued)"); continue
            n, tk = r.solved_upto(upto), r.toks_upto(upto)
            cells.append(f"{tk/max(n,1)/scale:.3f}" if scale > 1e4
                         else f"{tk/max(n,1)/scale:.1f}")
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    for label, tpl in runs:
        out.append(f"| {label} | " + " | ".join(
            (c(r) if (r := Run(tpl.format(t=t))).ok else f"— ({r.why})")
            for t, _ in actors) + " |")
    return "\n".join(out)


def pending(bench, actors, runs):
    rows = []
    for label, tpl in runs:
        for t, nm in actors:
            r = Run(tpl.format(t=t))
            if not r.ok:
                rows.append((label, nm, r.why))
    return rows


def build() -> str:
    L = [f"""# The runs reported in the thesis

**Generated by `scripts/thesis_report.py` — do not edit by hand.** Regenerated
{date.today().isoformat()} from `results/json/`.

SPaRC: 500 test puzzles, one attempt each. StepGame: 1000 test items, one attempt each,
`upstream-header` protocol, gpt-oss-120b dropped by the ceiling gate. Everywhere:
seed 0, `temp 0.6 / top_p 0.95 / top_k 20`, *k* = 3, baseline = that actor's Reflexion
trial 1.

`Δ` and `p` are paired exact McNemar against that actor's own baseline over the
identical items. **Reflexion rows read `memory / control`**, and their paired test is
between the two arms at that checkpoint, not against the baseline.

`— (queued)` = not run. `— (running, n=…)` = in flight; the file on disk is a
checkpoint, not a result, and is deliberately not tabulated.

**These tables are exhaustive.** `python scripts/thesis_report.py --audit` lists any
result file in `results/json/` that no row here claims and §6 does not name as
excluded; it currently reports none. So "absent from this document" means "not run, or
dropped on purpose" — never "run and forgotten".

**Contrastive = `avoid` + the worked exemplar throughout.**
Its `balanced` and `hindsight` arms were dropped on 2026-09-22 — see §6 below.

> **Read every difference under ~15 puzzles as unresolved.** Two jobs injecting a
> byte-identical prompt into the same 144 SPaRC puzzles flipped 33 of them, purely from
> vLLM's continuous batching; the paired test on that non-determinism reads p = 0.035.

---

## 1. SPaRC — solved of 500

{solved_table('sparc', SPARC, SPARC_RUNS, 500)}

**Bold** = *p* < 0.05 uncorrected.

## 2. StepGame — solved of 1000

{solved_table('stepgame', STEP, STEP_RUNS, 1000)}

StepGame has one failure mode (wrong-answer = n − solved − format-failure);
format-failures are 0 in every arm above.

## 3. SPaRC failure modes

Terminal reason over all 500 episodes; `Δ` against that actor's baseline.

{failure_tables(SPARC, SPARC_RUNS)}
## 4. Cost — tokens per solved item

Prompt + completion + retry over the whole run, divided by solved.

**SPaRC — Mtok/solve**

{cost_table('sparc', SPARC, SPARC_RUNS, 'Mtok', 1e6)}

**StepGame — ktok/solve**

{cost_table('stepgame', STEP, STEP_RUNS, 'ktok', 1e3)}

---

## 5. Not yet on disk
"""]
    for bench, actors, runs in (("SPaRC", SPARC, SPARC_RUNS),
                                ("StepGame", STEP, STEP_RUNS)):
        rows = pending(bench.lower(), actors, runs)
        if not rows:
            continue
        L.append(f"\n**{bench}**\n")
        L.append("| Run | Actor | State | Note |")
        L.append("|---|---|---|---|")
        L += [f"| {a} | {b} | {c} | {NOTES.get(a, '')} |" for a, b, c in rows]
    L.append("""
## 6. Excluded by choice

| Dropped | Why |
|---|---|
| Contrastive `hindsight` | the sign-matched control; measured in three settings and dropped 2026-09-22 |
| Contrastive `positive-only` | the arm that withholds the failure-derived warnings; not token-matched, so it could not isolate what the warnings say. Dropped 2026-09-24 |
| Contrastive `balanced` | it and the worked exemplar are substitutes, and it never moved across three configurations — 84 / 83 / 84 |
| MemP `k=10`, `AveFact` keys, cross-model transfer, proceduralization v1 | superseded or shown inert |
| Contrastive runs without the worked exemplar | superseded by the reported configuration |
| MemP script **v1** (retrieved + random, all actors) and agent-only bank v1 | the pre-rules-preamble banks; run and analysed, but outside the reported set |
| MemP StepGame `fixed` retrieval arm | run and analysed, but outside the reported set |
| StepGame `ours` and `upstream` protocol baselines | prompt-calibration snapshots; the campaign runs `upstream-header` |
| Early un-tagged Reflexion control runs, `baseline_qwen_test_*` | superseded; the latter predate `retry_tokens` |
| StepGame `KEY_VARIANT=lexical` | the retrieval-key ablation; not part of the reported campaign |
""")
    return "\n".join(L)


# Result files that are not reported on purpose; anything else unclaimed fails --audit.
EXCLUDED_PAT = (
    "_smoke_", "baseline_qwen_test_",              # smoke runs; pre-retry_tokens files
    "memp_eval_k10_", "memp_eval_avefact_",
    "memp_eval_transfer_", "_bank-gptoss_",        # cross-model transfer
    "memp_eval_gran_proceduralization_gptoss120b.json",   # v1, superseded by v2
    "contrastive_eval_avoid_balanced", "contrastive_eval_stepgame_avoid_balanced",
    "contrastive_eval_positive-only", "contrastive_eval_stepgame_positive-only",
    "contrastive_eval_hindsight", "contrastive_eval_stepgame_hindsight",
    "_test_upstream.json", "_test_upstream-header.json", "_stepgame_qwen32b_test.json",
    "baseline_stepgame_",                          # protocol calibration snapshots
    "reflexion_control_test_2026",                 # untagged early runs
    # v1 (pre-rules-preamble) banks and the StepGame `fixed` retrieval arm.
    "memp_eval_retrieved_test_qwen06b.json", "memp_eval_retrieved_test_qwen32b.json",
    "memp_eval_retrieved_test_gptoss120b.json",
    "memp_eval_random_test_qwen06b.json", "memp_eval_random_test_qwen32b.json",
    "memp_eval_random_test_gptoss120b.json",
    "memp_eval_agentonly_gptoss120b.json", "memp_eval_stepgame_fixed_",
)


def claimed_files() -> set[str]:
    """Every result file a row of this report reads."""
    claimed = set()
    for runs, actors in ((SPARC_RUNS, SPARC), (STEP_RUNS, STEP)):
        for _l, tpl in runs:
            claimed |= {tpl.format(t=t) for t, _n in actors}
    for b, f in BASE.items():
        claimed |= {f.format(t=t) for t, _n in (SPARC if b == "sparc" else STEP)}
    for b, pair in REFL.items():
        for f in pair:
            claimed |= {f.format(t=t) for t, _n in (SPARC if b == "sparc" else STEP)}
    return claimed


def audit() -> int:
    """List result files on disk that no row claims and no exclusion covers."""
    import re
    claimed = claimed_files()
    keep = re.compile(r"(memp_eval|memorybank_eval|contrastive_eval|"
                      r"reflexion_(root|memory|control)|baseline)_")
    gaps = sorted(
        p.name for p in J.glob("*.json")
        if keep.match(p.name) and "format_failures" not in p.name
        and p.name not in claimed
        # only the `_ex` contrastive configuration is reported
        and not (p.name.startswith("contrastive_eval") and "_ex" not in p.name)
        and not any(x in p.name for x in EXCLUDED_PAT))
    if not gaps:
        print("no unreported result files")
        return 0
    print(f"{len(gaps)} result file(s) on disk that the doc neither reports nor "
          f"excludes:")
    for g in gaps:
        print(f"  {g}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help=f"exit 1 if {DOC.relative_to(ROOT)} differs from this")
    ap.add_argument("--write", action="store_true",
                    help=f"write {DOC.relative_to(ROOT)}")
    ap.add_argument("--audit", action="store_true",
                    help="list result files on disk that no row claims and that "
                         "section 6 does not name as excluded")
    a = ap.parse_args()
    if a.audit:
        return audit()
    text = build()
    if a.check:
        cur = DOC.read_text() if DOC.exists() else ""
        if cur.strip() != text.strip():
            print(f"{DOC.relative_to(ROOT)} is out of date; re-run with --write",
                  file=sys.stderr)
            return 1
        return 0
    if a.write:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(text + "\n")
        print(f"wrote {DOC}", file=sys.stderr)
        return 0
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
