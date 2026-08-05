#!/usr/bin/env python
"""One report comparing the Reflexion memory arm against its control, across models.

`run_reflexion.py --phase compare` answers "did memory help on THIS model". This
answers "did memory help", by putting every model's arm pair in one table -- which is
the form the thesis actually needs, and the only form in which the multiple-comparison
problem is visible at all.

Pairs are discovered automatically: every results/json/reflexion_memory_<tag>_test.json
with a matching reflexion_control_<tag>_test.json.

  python scripts/compare_all_arms.py                    # all runs -> results/txt/
  python scripts/compare_all_arms.py --out -            # stdout only
  python scripts/compare_all_arms.py --include-smoke    # keep the 20-puzzle smoke runs

Stdlib only, and it reuses run_reflexion's own mcnemar() rather than reimplementing
it: two divergent versions of the significance test is exactly the bug you never
notice. That import is why this runs on the login node -- run_reflexion guards its
model-stack imports, so nothing here pulls in numpy or the venv.
"""

import argparse
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS_JSON = REPO / "results" / "json"
RESULTS_TXT = REPO / "results" / "txt"

_spec = importlib.util.spec_from_file_location("rr", REPO / "scripts" / "run_reflexion.py")
_rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rr)
mcnemar = _rr.mcnemar


def discover(include_smoke: bool) -> list:
    """Return [(tag, memory_state, control_state)] for every complete arm pair."""
    pairs = []
    for mem_path in sorted(RESULTS_JSON.glob("reflexion_memory_*.json")):
        tag = mem_path.stem[len("reflexion_memory_"):]
        if "smoke" in tag and not include_smoke:
            continue
        ctl_path = mem_path.with_name(f"reflexion_control_{tag}.json")
        if not ctl_path.exists():
            print(f"skipping {tag}: no control arm at {ctl_path.name}", file=sys.stderr)
            continue
        mem = json.loads(mem_path.read_text())
        ctl = json.loads(ctl_path.read_text())
        pairs.append((tag, mem, ctl))
    return pairs


def label(state: dict) -> str:
    cfg = state["config"]
    return cfg.get("model_id") or cfg.get("model") or "?"


def comparable(mem: dict, ctl: dict) -> list:
    """Reasons this pair is not a like-for-like comparison (empty = it is)."""
    problems = []
    mc, cc = mem["config"], ctl["config"]
    for key in ("split", "limit", "base_seed", "temperature", "top_p", "top_k",
                "max_tokens"):
        if mc.get(key) != cc.get(key):
            problems.append(f"{key} differs ({mc.get(key)} vs {cc.get(key)})")
    if mc.get("root_file") != cc.get("root_file"):
        problems.append("arms started from different trial-1 files")
    if len(mem["trials"]) != len(ctl["trials"]):
        problems.append(f"trial counts differ ({len(mem['trials'])} vs {len(ctl['trials'])})")
    errs = sum(t.get("errors", 0) for t in mem["trials"]) + \
        sum(t.get("errors", 0) for t in ctl["trials"])
    if errs:
        problems.append(f"{errs} crashed episodes")
    return problems


def build(pairs: list) -> str:
    out = []
    w = out.append

    w("=" * 92)
    w("  REFLEXION vs TRIAL-MATCHED CONTROL -- all models")
    w("=" * 92)
    w("")
    w("  Each model contributes one arm pair: identical puzzles, identical trial 1,")
    w("  identical per-trial seeds. The arms differ only in whether that puzzle's")
    w("  reflections are injected. 'delta' is memory minus control at the final trial.")
    w("")

    rows = []
    for tag, mem, ctl in pairs:
        mt, ct = mem["trials"][-1], ctl["trials"][-1]
        n_trials = min(len(mem["trials"]), len(ctl["trials"]))
        stats = mcnemar(mem, ctl)
        rows.append({
            "tag": tag, "model": label(mem), "mem": mem, "ctl": ctl,
            "n": len(mem["puzzles"]), "trials": n_trials,
            "t1": mem["trials"][0]["cumulative_accuracy"],
            "m_final": mt["cumulative_accuracy"], "c_final": ct["cumulative_accuracy"],
            "m_solved": mt["cumulative_solved"], "c_solved": ct["cumulative_solved"],
            "stats": stats, "problems": comparable(mem, ctl),
        })
    rows.sort(key=lambda r: -r["t1"])

    k = len(rows)
    # Three arm pairs tested against the same hypothesis is three chances at p<0.05.
    # Stating the corrected threshold next to the raw p-values is the difference
    # between reporting a result and overclaiming one.
    alpha = 0.05 / k if k else 0.05

    w(f"  {'model':<22}{'n':>5}{'trials':>7}{'trial 1':>9}{'memory':>9}{'control':>9}"
      f"{'delta':>8}{'p':>9}  verdict")
    w("  " + "-" * 88)
    for r in rows:
        p = r["stats"]["p_value"]
        if p < alpha:
            verdict = "significant"
        elif p < 0.05:
            verdict = "sig. uncorrected only"
        else:
            verdict = "not significant"
        w(f"  {r['model']:<22}{r['n']:>5}{r['trials']:>7}{r['t1']:>9.3f}"
          f"{r['m_final']:>9.3f}{r['c_final']:>9.3f}{r['m_final'] - r['c_final']:>+8.3f}"
          f"{p:>9.4f}  {verdict}")
    w("")
    w(f"  Significance: McNemar exact two-sided on paired per-puzzle outcomes at the")
    w(f"  final trial. With {k} models tested, the Bonferroni-corrected threshold is")
    w(f"  p < {alpha:.4f}; the 0.05 column is shown uncorrected.")
    w("")

    w("  paired outcome counts (what the test actually uses)")
    w("")
    w(f"  {'model':<22}{'both':>7}{'memory only':>14}{'control only':>14}"
      f"{'neither':>9}{'discordant':>12}")
    w("  " + "-" * 88)
    for r in rows:
        s = r["stats"]
        w(f"  {r['model']:<22}{s['both']:>7}{s['memory_only']:>14}"
          f"{s['control_only']:>14}{s['neither']:>9}{s['discordant']:>12}")
    w("")
    w("  Only the discordant pairs carry information. Puzzles both arms solved, and")
    w("  the far larger set neither solved, say nothing about which arm is better.")
    w("")

    w("  cumulative accuracy by trial")
    w("")
    for r in rows:
        w(f"  {r['model']}  (n={r['n']})")
        w(f"    {'trial':>6}{'memory':>10}{'control':>10}{'delta':>9}{'new (mem)':>11}")
        for mt, ct in zip(r["mem"]["trials"], r["ctl"]["trials"]):
            w(f"    {mt['trial']:>6}{mt['cumulative_accuracy']:>10.3f}"
              f"{ct['cumulative_accuracy']:>10.3f}"
              f"{mt['cumulative_accuracy'] - ct['cumulative_accuracy']:>+9.3f}"
              f"{mt['solved_this_trial']:>11}")
        w("")
    w("  Both curves rise with the trial index by construction: only unsolved puzzles")
    w("  are re-attempted and solved ones are carried forward, so neither curve can")
    w("  fall. The control arm is what separates memory from plain resampling.")
    w("")

    w("  token cost (millions)")
    w("")
    w(f"  {'model':<22}{'mem actor':>12}{'reflection':>12}{'ctl actor':>12}"
      f"{'reflection %':>14}")
    w("  " + "-" * 88)
    tok = lambda st, key: sum(t.get(key, 0) for t in st["trials"])
    for r in rows:
        ma = tok(r["mem"], "actor_prompt_tokens") + tok(r["mem"], "actor_completion_tokens")
        mr = tok(r["mem"], "reflection_prompt_tokens") + tok(r["mem"], "reflection_completion_tokens")
        ca = tok(r["ctl"], "actor_prompt_tokens") + tok(r["ctl"], "actor_completion_tokens")
        pct = 100 * mr / (ma + mr) if (ma + mr) else 0.0
        w(f"  {r['model']:<22}{ma / 1e6:>12.1f}{mr / 1e6:>12.1f}{ca / 1e6:>12.1f}"
          f"{pct:>13.1f}%")
    w("")

    flagged = [r for r in rows if r["problems"]]
    if flagged:
        w("  WARNINGS -- these pairs are not clean comparisons")
        w("")
        for r in flagged:
            for p in r["problems"]:
                w(f"    {r['model']}: {p}")
        w("")
    else:
        w("  All pairs verified comparable: matched split, seeds, sampling and trial")
        w("  counts, shared trial-1 file, and zero crashed episodes.")
        w("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="output path (default results/txt/reflexion_all_models.txt; "
                         "'-' for stdout only)")
    ap.add_argument("--include-smoke", action="store_true",
                    help="also include the small smoke-test runs")
    args = ap.parse_args()

    pairs = discover(args.include_smoke)
    if not pairs:
        print("no complete memory/control arm pairs found in results/json/",
              file=sys.stderr)
        return 1

    report = build(pairs)
    print(report)
    if args.out == "-":
        return 0
    dest = pathlib.Path(args.out) if args.out else RESULTS_TXT / "reflexion_all_models.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report)
    print(f"[written] {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
