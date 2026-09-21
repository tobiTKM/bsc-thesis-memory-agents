#!/usr/bin/env python
"""Emit the all-runs comparison tables for docs/contrastive.md section 0, as
markdown on stdout.

Same contract as memp_report.py and memorybank_report.py: every table in the doc's
summary section is generated from the result JSONs, so the numbers cannot drift
from the files and a new arm shows up by dropping its result file in results/json/
and adding one line to RUNS.

Seven views per benchmark, mirroring docs/memp.md section 0 so the four methods'
result sections can be read side by side:

  success     solved / accuracy / paired difference against that actor's own baseline,
              bolded at p < 0.05. The contrast that matters for this method -- against
              the `hindsight` control -- is made in docs/contrastive.md section 6.1,
              not here: section 0 is the at-a-glance view
  difficulty  accuracy per SPaRC band 1-5, or per StepGame story length k
  hops        StepGame only: accuracy by TRUE reasoning depth, which k is not
  failure     terminal_reason breakdown -- the mechanism-level test, and the one the
              pre-registered predictions in section 5 are written about
  polarity    share of items whose block held at least one DO -- what the actor SAW,
              not the entry's provenance. A bank's mix is the ACTOR's train accuracy,
              so this bounds every conclusion about the negative half and no effect size
              in the doc is quoted without it (docs/contrastive.md section 2)
  tokens      prompt + completion + retry, per item and per SOLVE
  breakdown   the same tokens split by kind, plus prompt tokens per env step
  steps       SPaRC only: env steps per episode, split by outcome

BENCH=sparc|stepgame selects the campaign; one knob rather than two scripts, because
the two are the same experiment on a different task and their tables must not drift.

Baselines are each model's Reflexion trial 1, read through run_memp's
baseline_outcomes shape rather than re-derived here.

    python scripts/contrastive_report.py
    python scripts/contrastive_report.py --only failure
"""

import argparse
import json
import math
import os
import pathlib
import sys

J = pathlib.Path(__file__).resolve().parents[1] / "results" / "json"

# BENCH picks which campaign to tabulate. One knob rather than two scripts, for the
# same reason run_contrastive_campaign.sh has one: the two campaigns are the same
# experiment on a different task and their tables must not drift apart.
BENCH = os.environ.get("BENCH", "sparc")

if BENCH == "sparc":
    PREFIX, PTAG = "", ""
    # Weakest first, as docs/memp.md section 0 orders them. An actor with no
    # contrastive arm on disk yet is skipped entirely rather than shown as a lone
    # baseline row, so this section grows on its own as jobs land.
    ACTORS = [("qwen06b", "Qwen3-0.6B"), ("qwen32b", "Qwen3-32B"),
              ("gptoss120b", "gpt-oss-120b")]
    BASELINE = "reflexion_root_{tag}_test.json"
    MEMP = "memp_eval_retrieved_test_{tag}_v2.json"
elif BENCH == "stepgame":
    PREFIX, PTAG = "stepgame_", "_upstream-header"
    ACTORS = [("qwen32b", "Qwen3-32B")]
    BASELINE = "reflexion_root_stepgame_{tag}_test" + PTAG + ".json"
    MEMP = "memp_eval_stepgame_retrieved_test_{tag}" + PTAG + ".json"
else:
    raise SystemExit(f"BENCH must be sparc or stepgame (got {BENCH!r})")

# (label, granularity, arm, exemplars, framing). Wave 1 is the 2026-09-14 campaign;
# wave 2 (2026-09-16) repairs the two renderer defects docs/contrastive.md 6.6 traced
# the control's 10-puzzle shortfall to. The filename is DERIVED from the config, so a
# row here cannot name a file the campaign script would not have written.
# Grouped by ARM, not by when it was run: the four `avoid` rows are the method and
# read as one block (main / relevance ablation / polarity ablation / repaired), then
# its control, then the arm that withholds the failed entries entirely.
ARMS = [
    ("avoid, retrieved",                   "avoid",         "retrieved", ""),
    ("avoid, random",                      "avoid",         "random",    ""),
    ("avoid, BALANCED",                    "avoid",         "balanced",  "_v2"),
    ("avoid + exemplar  [@ best]",         "avoid",         "retrieved", "_ex"),
    ("hindsight, retrieved",               "hindsight",     "retrieved", ""),
    ("hindsight + both  [@ best]",         "hindsight",     "retrieved", "_ex_mf"),
    ("positive-only, retrieved",           "positive-only", "retrieved", ""),
]
# The two single-factor wave-2 arms -- `hindsight` with only the exemplar, and with
# only the matched header -- are deliberately NOT here. They exist to decompose the
# apparatus gap, not as campaign results, and they are tabulated where that argument
# is made (docs/contrastive.md section 6.6). Section 0 carries the arms the campaign
# is about, plus the repaired pairing of the method and its control.
DIAGNOSTIC_ARMS = [
    ("hindsight + exemplar only",       "hindsight", "retrieved", 1, 0),
    ("hindsight + matched header only", "hindsight", "retrieved", 0, 1),
]


def fname(gran: str, arm: str, tag: str, sfx: str = "") -> str:
    """`_ex` = worked exemplars on, `_mf` = header matched to the arm, `_v2` = the
    balanced arm re-run after its ordering fix (section 6.3). The suffix is the
    configuration, so a row here cannot name a file the campaign would not write."""
    f = J / f"contrastive_eval_{PREFIX}{gran}_{arm}_test_{tag}{PTAG}{sfx}.json"
    # `balanced` only has a _v2 on benchmarks where it was re-run; fall back so the
    # row still appears rather than silently vanishing.
    if sfx == "_v2" and not f.exists():
        return f"contrastive_eval_{PREFIX}{gran}_{arm}_test_{tag}{PTAG}.json"
    return f.name
RUNS = [(label, tag, fname(g, a, tag, sfx))
        for tag, _name in ACTORS for label, g, a, sfx in ARMS]

# Which granularity each result file came from, for views that apply to only some arms.
GRAN_OF = {fname(g, a, tag, sfx): g
           for tag, _name in ACTORS for _l, g, a, sfx in ARMS}

# One arm of each ported cross-task method per actor, for scale. They are NOT
# contrastive arms -- they render through their own method's block -- so a row-to-row
# gap is a method comparison, not a controlled contrast; the controlled one is `avoid`
# against `hindsight`, made in section 6.1.
#
# Memp's reference is its `script` arm, not its trajectory one: every contrastive arm
# is prose only, so script granularity is the like-for-like comparison, and `v2` is the
# rule-correct bank (docs/memp.md 3.9). StepGame has no script arm to point at -- that
# campaign ran Memp at trajectory granularity only, deliberately
# (run_stepgame_campaign.sh memp-eval says why) -- so the two benchmarks' Memp rows are
# different granularities and are labelled as such rather than silently equated.
if BENCH == "sparc":
    REFERENCE = [(lab, tag, fn.format(tag=tag))
                 for tag, _n in ACTORS
                 for lab, fn in (
                     ("*Memp — script v2, retrieved*",
                      "memp_eval_retrieved_test_{tag}_v2.json"),
                     ("*MemoryBank — case-record, retrieved*",
                      "memorybank_eval_retrieved_test_{tag}_norp.json"))]
else:
    REFERENCE = [("*Memp — trajectory, retrieved*", "qwen32b",
                  f"memp_eval_stepgame_retrieved_test_qwen32b{PTAG}.json"),
                 ("*MemoryBank — case-record, retrieved*", "qwen32b",
                  f"memorybank_eval_stepgame_retrieved_test_qwen32b{PTAG}_rp.json")]

# The arm every other one is really tested against -- "does learning from the failure
# beat learning from the gold path?" is a contrast with `hindsight`, not with the
# memoryless run. Section 0 does not tabulate it (one number per row reads better at
# a glance); the contrast is made, with its p, in section 6.1.
REASONS = ["solved", "wrong-solution", "dead-end", "wrong-answer",
           "format-failure", "error"]
INTERACTIVE = BENCH == "sparc"


def mcnemar_exact(a: int, b: int) -> float:
    """Two-sided exact McNemar from the discordant counts (same as run_memp's)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(name: str) -> dict | None:
    """Per-item records, normalised across the eval and reflexion-root shapes."""
    p = J / name
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "config" in d and d["config"].get("arm") == "root":     # baseline = trial 1
        out = {}
        for pid, e in d["puzzles"].items():
            a = (e.get("attempts") or [{}])[0]
            out[pid] = {"success": e.get("solved", False),
                        "difficulty": e.get("difficulty", -1),
                        "terminal_reason": a.get("terminal_reason"),
                        # A reflexion root stores the step count on the attempt, not
                        # on the puzzle; without this the baseline row of the steps
                        # table reads 0.0 and looks like a broken run.
                        "steps": a.get("steps", 0),
                        "prompt_tokens": a.get("prompt_tokens", 0),
                        "completion_tokens": a.get("completion_tokens", 0),
                        "retry_tokens": a.get("retry_tokens", 0),
                        "retrieved": []}
        return out
    if "episodes" in d:
        return {str(e["puzzle_id"]): e for e in d["episodes"]}
    return d["puzzles"]


def summary_of(name: str) -> dict:
    p = J / name
    return json.loads(p.read_text()).get("summary", {}) if p.exists() else {}


def paired(a: dict, b: dict) -> tuple[int, int, float]:
    """(a alone won, b alone won, p) over the items both arms scored."""
    common = sorted(set(a) & set(b))
    a_only = sum(1 for p in common if a[p]["success"] and not b[p]["success"])
    b_only = sum(1 for p in common if b[p]["success"] and not a[p]["success"])
    return a_only, b_only, mcnemar_exact(a_only, b_only)


def bold(text: str, cond: bool) -> str:
    return f"**{text}**" if cond else text


def has_arms(tag: str) -> bool:
    """Does this actor have any contrastive arm on disk? Actors still queued are
    skipped rather than rendered as a lone baseline row."""
    return any(t == tag and (J / fn).exists() for _l, t, fn in RUNS)


def rows_for(tag: str):
    """(label, records) for one actor: its baseline first, then every arm on disk."""
    base = load(BASELINE.format(tag=tag))
    out = [("baseline (memoryless)", base)] if base else []
    for label, t, fn in list(RUNS) + list(REFERENCE):
        if t != tag:
            continue
        recs = load(fn)
        if recs is None:
            print(f"not run yet: {fn}", file=sys.stderr)
            continue
        out.append((label, recs))
    return base, out


def toks(r: dict) -> int:
    return ((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
            + (r.get("retry_tokens") or 0))


def _cell(a, b, keys):
    """Paired McNemar of `a` against `b` over `keys`, as a markdown cell.

    The delta only, bolded when p < 0.05. Section 0 is the at-a-glance view; the
    exact p of anything that matters is quoted where it is argued, in section 6.
    """
    x = sum(1 for k in keys if a[k]["success"] and not b[k]["success"])
    y = sum(1 for k in keys if b[k]["success"] and not a[k]["success"])
    return f"**{x-y:+d}**" if mcnemar_exact(y, x) < 0.05 else f"{x-y:+d}"


def table_success():
    n = "1000" if BENCH == "stepgame" else "500"
    print("#### Success rate\n")
    print("`vs baseline` is the paired difference against that actor's memoryless run "
          "over the same items. **Bold = *p* < 0.05** on an exact McNemar; the p itself "
          "is quoted in section 6 where it is argued. The two *italic* rows are one arm "
          "each of the ported methods, for scale — they use their own renderer, so a "
          "gap to them is a method comparison, not a controlled contrast."
          + (" Memp is shown at **script** granularity, the like-for-like one since "
             "every arm here is prose only." if BENCH == "sparc" else
             " Memp is shown at **trajectory** granularity: the StepGame campaign ran "
             "no script arm, so unlike section 0.1 its Memp row is not prose-only.")
          + "\n")
    print(f"| Actor | Arm | Solved /{n} | Accuracy | vs baseline |")
    print("|---|---|---|---|---|")
    for tag, name in ACTORS:
        if not has_arms(tag):
            continue
        base, rows = rows_for(tag)
        for i, (label, r) in enumerate(rows):
            keys = sorted(set(r) & set(base))
            s_ = sum(r[k]["success"] for k in keys)
            print(f"| {name if i == 0 else ''} | {label} | {s_} | "
                  f"{100*s_/len(keys):.1f}% | {'—' if i == 0 else _cell(r, base, keys)} |")
    print()


def table_difficulty():
    if BENCH == "sparc":
        print("#### Success by difficulty\n")
        print("Accuracy within each SPaRC difficulty band (n = 86 / 118 / 121 / 86 / 89)."
              "\n")
        bands, hdr = range(1, 6), "| d1 | d2 | d3 | d4 | d5 |"
    else:
        print("#### Success by story length *k*\n")
        print("`k` is the number of sentences, which is what every published StepGame "
              "number reports. It is **not** the reasoning depth — see the hop table "
              "below and `docs/stepgame.md` section 2.2.\n")
        bands = range(1, 11)
        hdr = "| " + " | ".join(f"k{d}" for d in bands) + " |"
    print(f"| Actor | Arm {hdr}")
    print("|---|---|" + "---|" * len(list(bands)))
    for tag, name in ACTORS:
        if not has_arms(tag):
            continue
        base, rows = rows_for(tag)
        diff = {k: v.get("difficulty", -1) for k, v in base.items()}
        for i, (label, r) in enumerate(rows):
            cells = []
            for d in bands:
                ks = [k for k in r if diff.get(k) == d]
                cells.append(f"{100*sum(r[k]['success'] for k in ks)/len(ks):.0f}%"
                             if ks else "—")
            print(f"| {name if i == 0 else ''} | {label} | " + " | ".join(cells) + " |")
    print()


def table_hops():
    """StepGame only: the axis that is actually reasoning depth."""
    if BENCH != "stepgame":
        return
    print("#### Success by true hop count\n")
    print("The reasoning depth, as re-derived by the exact solver. The informative band "
          "is **h≥4**; h9/h10 rest on a handful of items each and should not be read "
          "individually.\n")
    hops = {}
    for tag, _n in ACTORS:
        _b, rows = rows_for(tag)
        for _l, r in rows:
            for k, v in r.items():
                h = (v.get("extra") or {}).get("hops")
                if h is not None:
                    hops[k] = int(h)
    if not hops:
        print("*(no arm on disk records `extra.hops`)*\n")
        return
    print("| Actor | Arm | " + " | ".join(f"h{h}" for h in range(1, 11))
          + " | **h≥4** | all |")
    print("|---|---|" + "---|" * 12)
    for tag, name in ACTORS:
        if not has_arms(tag):
            continue
        _b, rows = rows_for(tag)
        for i, (label, r) in enumerate(rows):
            ks = [k for k in r if k in hops]
            cells = []
            for h in range(1, 11):
                sub = [k for k in ks if hops[k] == h]
                cells.append(f"{100*sum(r[k]['success'] for k in sub)/len(sub):.0f}%"
                             if sub else "—")
            deep = [k for k in ks if hops[k] >= 4]
            d = f"**{100*sum(r[k]['success'] for k in deep)/len(deep):.1f}%**" if deep else "—"
            a = f"{100*sum(r[k]['success'] for k in ks)/len(ks):.1f}%" if ks else "—"
            print(f"| {name if i == 0 else ''} | {label} | " + " | ".join(cells)
                  + f" | {d} | {a} |")
    print()


def table_failure():
    print("#### Failure modes\n")
    if BENCH == "sparc":
        print("Terminal reason for all 500 episodes, and **the table the pre-registered "
              "prediction in section 5 is about**. `wrong-solution` = a path was "
              "submitted and broke a rule; `dead-end` = the path was painted into a "
              "corner; `format-failure` = no parseable move.\n")
        cols = ["solved", "wrong-solution", "dead-end", "format-failure", "error"]
    else:
        print("Terminal reason for all 1000 items. Single-turn, so the only failure "
              "modes are a wrong label and an unparseable reply.\n")
        cols = ["solved", "wrong-answer", "format-failure", "error"]
    print("| Actor | Arm | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for tag, name in ACTORS:
        if not has_arms(tag):
            continue
        _b, rows = rows_for(tag)
        for i, (label, r) in enumerate(rows):
            c = {x: 0 for x in cols}
            for v in r.values():
                c[v.get("terminal_reason")] = c.get(v.get("terminal_reason"), 0) + 1
            print(f"| {name if i == 0 else ''} | {label} | "
                  + " | ".join(str(c[x]) for x in cols) + " |")
    print()


def polarity_of(fn: str):
    """(share of injected entries shown as DO, share of items seeing >=1 DO).

    Recomputed from each eval file's stored `retrieved` log rather than read from its
    summary, because the summaries written before 2026-09-16 logged the entry's
    PROVENANCE instead of how the arm rendered it -- the same thing only for `avoid`.
    Under `hindsight` every entry is shown as a DO whatever its provenance, so the
    stored figure understated it as 28.8% when the truth is 100%. Everything needed to
    redo it correctly is already in the file, so no arm has to be re-run.
    """
    path = J / fn
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    gran = d.get("config", {}).get("granularity", "avoid")
    shown_do = injected = n_items = with_do = 0
    for rec in d["puzzles"].values():
        picks = rec.get("retrieved") or []
        if not picks:
            continue
        n_items += 1
        k = 0
        for it in picks:
            injected += 1
            # `shown` is written by runs from 2026-09-16 on; older files carry only
            # provenance, so re-derive the rendering from the arm's granularity.
            sh = it.get("shown") or _shown(it["provenance"], gran)
            k += sh == "do"
        shown_do += k
        with_do += k > 0
    if not injected:
        return None
    return shown_do / injected, with_do / max(n_items, 1)


def _shown(provenance: str, granularity: str) -> str:
    """memory.contrastive.rendered_as, for a log row that has only the provenance."""
    positive = provenance == "agent"
    if positive and granularity == "avoid-only":
        return "withheld"
    if not positive and granularity == "positive-only":
        return "withheld"
    return "do" if (positive or granularity == "hindsight") else "dont"


def table_polarity():
    print("#### Injected polarity\n")
    print("Share of items whose injected block contained at least one **DO** — what "
          "the actor SAW, not where the text came from. Only the `avoid` arms appear: "
          "they are the ones carrying DON'Ts, and a bank's mix of the two signs is the "
          "**actor's train accuracy**, not a knob. gpt-oss solves 71 of 500 SPaRC train "
          "puzzles, so a plain cosine top-3 leaves most items with no worked example at "
          "all; on StepGame the ratio inverts. An arm low in this column cannot be read "
          "as a verdict on negative advice — that is what `balanced` exists for "
          "(section 2). `hindsight` and `positive-only` sit at 100% and 28.8% "
          "trivially, by construction.\n")
    print("| Actor | Arm | items with ≥1 DO |")
    print("|---|---|---|")
    for tag, name in ACTORS:
        if not has_arms(tag):
            continue
        first = True
        for label, t, fn in RUNS:
            # Only the arms that actually carry DON'Ts. `hindsight` shows a DO on
            # every item by construction and `positive-only` shows nothing else, so
            # neither has a polarity question to answer -- listing them invites the
            # column to be read as a quality score rather than as the interpretive
            # caveat it is.
            if t != tag or GRAN_OF.get(fn) != "avoid":
                continue
            pol = polarity_of(fn)
            if pol is None:
                continue
            print(f"| {name if first else ''} | {label} | {pol[1]:.1%} |")
            first = False
    print()


def table_tokens():
    print("#### Token cost and efficiency\n")
    print("Prompt + completion + retry over every episode. `retry` is included per the "
          "accounting rule in `docs/memp.md` section 2 — an arm that provokes more "
          "format failures must not look cheap. **tok/solve** is the efficiency figure: "
          "an arm that solves more can still cost more per solved item.\n")
    unit = "puzzle" if BENCH == "sparc" else "item"
    print(f"| Actor | Arm | Total | ktok/{unit} | Solved | ktok/solve | vs baseline |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in ACTORS:
        if not has_arms(tag):
            continue
        base, rows = rows_for(tag)
        bp = sum(toks(v) for v in base.values()) / max(
            sum(v["success"] for v in base.values()), 1)
        for i, (label, r) in enumerate(rows):
            tot = sum(toks(v) for v in r.values())
            s_ = sum(v["success"] for v in r.values())
            per = tot / max(s_, 1)
            rel = "—" if i == 0 else f"{100*(per/bp - 1):+.0f}%"
            print(f"| {name if i == 0 else ''} | {label} | {tot/1e6:.1f}M | "
                  f"{tot/len(r)/1e3:.1f} | {s_} | {per/1e3:.1f} | {rel} |")
    print()


def table_breakdown():
    print("#### Token breakdown: prompt vs completion\n")
    extra = ("**Prompt tok/step** is the clearest measure of what a memory block "
             "actually costs, since the block is re-sent on every step of the episode. "
             if INTERACTIVE else "")
    print("The same tokens as the previous table, split by kind. " + extra
          + "`retry` is prompt+completion spent on replies that yielded no parseable "
            "answer.\n")
    tail = " Prompt tok/step |" if INTERACTIVE else ""
    print("| Actor | Arm | Prompt | Completion | Retry | Prompt share |" + tail)
    print("|---|---|---|---|---|" + ("---|---|" if INTERACTIVE else "---|"))
    for tag, name in ACTORS:
        if not has_arms(tag):
            continue
        _b, rows = rows_for(tag)
        for i, (label, r) in enumerate(rows):
            pr = sum(v.get("prompt_tokens") or 0 for v in r.values())
            co = sum(v.get("completion_tokens") or 0 for v in r.values())
            rt = sum(v.get("retry_tokens") or 0 for v in r.values())
            st = sum(v.get("steps") or 0 for v in r.values())
            cell = f" {pr/max(st,1):.0f} |" if INTERACTIVE else ""
            print(f"| {name if i == 0 else ''} | {label} | {pr/1e6:.1f}M | "
                  f"{co/1e6:.1f}M | {rt/1e6:.2f}M | "
                  f"{100*pr/max(pr+co+rt,1):.0f}% |" + cell)
    print()


def table_steps():
    if not INTERACTIVE:
        return            # single-turn: every episode is exactly one step
    print("#### Steps in the environment\n")
    print("Env steps per episode (`max_steps=100`), split by outcome: a solved episode's "
          "step count is essentially the path length, while a failed one shows how long "
          "the agent kept moving before dying.\n")
    print("| Actor | Arm | Mean (all) | Mean (solved) | Mean (failed) |")
    print("|---|---|---|---|---|")
    for tag, name in ACTORS:
        if not has_arms(tag):
            continue
        _b, rows = rows_for(tag)
        for i, (label, r) in enumerate(rows):
            g = lambda f: [v.get("steps") or 0 for v in r.values() if f(v)]
            m = lambda xs: f"{sum(xs)/len(xs):.1f}" if xs else "—"
            print(f"| {name if i == 0 else ''} | {label} | {m(g(lambda v: True))} | "
                  f"{m(g(lambda v: v['success']))} | "
                  f"{m(g(lambda v: not v['success']))} |")
    print()


TABLES = {"success": table_success, "difficulty": table_difficulty,
          "hops": table_hops, "failure": table_failure, "polarity": table_polarity,
          "tokens": table_tokens, "breakdown": table_breakdown, "steps": table_steps}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(TABLES), help="emit one table")
    args = ap.parse_args()
    missing = [fn for _l, _t, fn in RUNS if not (J / fn).exists()]
    if len(missing) == len(RUNS):
        print(f"No {BENCH} contrastive arms on disk yet (looked for "
              f"{missing[0]} and {len(missing) - 1} more).", file=sys.stderr)
        return 1
    if missing:
        print(f"note: {len(missing)} arm(s) not on disk, skipped: "
              f"{', '.join(missing)}", file=sys.stderr)
    for name, fn in TABLES.items():
        if args.only in (None, name):
            fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
