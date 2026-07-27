#!/usr/bin/env python
"""Condense a run's results JSON into a short human-readable text report.

The results files are one summary block plus one record per puzzle, which is the
right shape for later analysis but not for reading. This prints the run's
settings (the knobs we might change between runs), the whole-split numbers, the
same numbers per difficulty, and the hardest puzzles that were solved.

  python scripts/summarize_results.py                       # newest run
  python scripts/summarize_results.py results/*.json        # all + comparison
  python scripts/summarize_results.py run.json --out -      # stdout only
"""

import argparse
import collections
import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS_JSON = REPO / "results" / "json"   # run data written by run_baseline.py
RESULTS_TXT = REPO / "results" / "txt"     # reports written by this script


# ---- formatting helpers --------------------------------------------------
def dist(values) -> str:
    """avg/med/min/max of a list, the shape every per-puzzle stat is reported in."""
    if not values:
        return "n/a"
    return (f"avg={statistics.mean(values):.2f}, med={statistics.median(values):.2f}, "
            f"min={min(values):g}, max={max(values):g}")


def share(count: int, total: int) -> str:
    return f"{100 * count / total:6.2f}% ({count})" if total else "n/a"


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s"


def format_failures(summary: dict, replies: int) -> list:
    """One line on unparseable replies -- expanded only when they cost something.

    The count alone is not the interesting number: the agent retries up to
    max_format_retries times, so a failure that recovered cost tokens and
    nothing else. What contaminates the results is an *abort* (every retry
    failed), because the harness ends that episode as truncated, where it is
    indistinguishable from a real dead end. So report aborts, and stay quiet
    when there are none. A run with a high rate here is not trustworthy until
    it is explained -- see the 4096-token run that lost 66% of its episodes.
    """
    ff = summary.get("format_failures") or {}
    if not ff:
        return []
    n, puzzles = ff.get("n", 0), ff.get("puzzles_affected", 0)
    if not n:
        return []

    # The aggregate does not keep attempt numbers, so read the raw log: a record
    # at the last attempt index means that reply was never parsed = episode aborted.
    aborted = None
    log = pathlib.Path(ff.get("log", ""))
    if not log.exists():
        # Runs made before results/ was split into json/ and txt/ recorded the log
        # at the old flat path; it moved with everything else.
        log = RESULTS_JSON / log.name
    if log.exists():
        attempts = collections.defaultdict(int)
        for line in log.read_text().splitlines():
            if line.strip():
                try:
                    attempts[json.loads(line)["puzzle_id"]] += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        retries = summary.get("max_format_retries", 5)
        aborted = sum(1 for c in attempts.values() if c >= retries)

    rate = f"{100 * n / replies:.2f}%" if replies else "n/a"
    tail = f", {aborted} aborted an episode" if aborted is not None else ""
    out = [f"  format failures:     {rate} of replies ({n} over {puzzles} puzzles{tail})"]
    if aborted:
        out.append("    ^ those episodes are counted as truncated but are NOT dead ends")
    if log.exists():
        rel = log.relative_to(REPO) if log.is_relative_to(REPO) else log
        out.append(f"    log: {rel}")
    return out


# ---- analysis ------------------------------------------------------------
def block(episodes: list, indent: str) -> list:
    """The per-puzzle stat block, used for the whole split and for each difficulty.

    `terminated` and `truncated` are reported over *all* puzzles and sum to 100%:
    in Spatial-Gym `terminated` means the agent reached the exit (wins are a
    subset of it) and `truncated` means it had no legal action left, i.e. walked
    into a dead end. The step cap also sets `truncated`, but episodes run ~15
    steps against a cap of 100, so that never fires. So "failed at exit" below
    is the rule-violation bucket, and the two failure modes the memory methods
    have to attack are: lost the path (truncated) vs lost the rules.
    """
    n = len(episodes)
    wins = sum(1 for e in episodes if e.get("success"))
    term = sum(1 for e in episodes if e.get("terminated"))
    trunc = sum(1 for e in episodes if e.get("truncated"))

    steps = [e.get("steps", 0) for e in episodes]
    completion = [e.get("completion_tokens", 0) for e in episodes]
    prompt = [e.get("prompt_tokens", 0) for e in episodes]
    # tokens per model reply, i.e. how long the model thinks on a single move
    per_reply = [e.get("completion_tokens", 0) / e["steps"]
                 for e in episodes if e.get("steps")]

    out = [
        f"{indent}total puzzles:        {n}",
        f"{indent}wins:                {share(wins, n)}",
        f"{indent}fails:               {share(n - wins, n)}",
        f"{indent}terminated runs:     {share(term, n)}   [reached the exit]",
        f"{indent}truncated runs:      {share(trunc, n)}   [dead end, no legal move]",
        f"{indent}failed at exit:      {share(term - wins, n)}   [exit reached, rule broken]",
        "",
        f"{indent}average_completion_tokens_per_puzzle:   [per model reply]",
        f"{indent}  {dist(per_reply)}",
        f"{indent}steps_per_puzzle:",
        f"{indent}  {dist(steps)}",
        f"{indent}total_completion_tokens_per_puzzle:",
        f"{indent}  {dist(completion)}",
        f"{indent}total_prompt_tokens_per_puzzle:",
        f"{indent}  {dist(prompt)}",
    ]
    return out


def analyse(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text())
    episodes = data.get("episodes", [])
    by_diff = collections.defaultdict(list)
    for ep in episodes:
        by_diff[ep.get("difficulty", -1)].append(ep)
    return {
        "path": path,
        "summary": data.get("summary", {}),
        "episodes": episodes,
        "by_diff": dict(sorted(by_diff.items())),
    }


def render(a: dict, top: int) -> str:
    s, eps = a["summary"], a["episodes"]
    n = len(eps) or s.get("n", 0)
    out = []
    w = out.append

    w("=" * 72)
    w(f"  {a['path'].name}")
    w("=" * 72)

    # Older runs predate some of these fields; say so rather than printing a
    # default that was never actually used.
    def setting(label: str, key: str, fmt=str):
        val = s.get(key)
        w(f"  {label:<20} {fmt(val) if val is not None else '(not recorded)'}")

    w("")
    w("RUN")
    served, exact = s.get("model", "?"), s.get("model_id")
    w(f"  {'model:':<20} {exact or served}" + (f'   (served as "{served}")' if exact else ""))
    setting("split:", "split")
    setting("max_tokens:", "max_tokens")
    setting("temperature:", "temperature")
    setting("top_p:", "top_p")
    setting("top_k:", "top_k")
    setting("seed:", "seed")
    setting("concurrency:", "concurrency")
    if s.get("elapsed_sec"):
        w(f"  {'runtime:':<20} {hms(s['elapsed_sec'])}")
        w(f"  {'per puzzle:':<20} {s['elapsed_sec'] / max(n, 1):.1f} s"
          f"   (wall clock / puzzles, so it is divided by concurrency)")

    w("")
    w("FULL RUN")
    out.extend(block(eps, "  "))
    # replies = one per step, plus every retry that had to be thrown away
    replies = sum(e.get("steps", 0) for e in eps) + (s.get("format_failures") or {}).get("n", 0)
    out.extend(format_failures(s, replies))

    w("")
    w("BY DIFFICULTY")
    for d, group in a["by_diff"].items():
        w(f"  Difficulty {d}")
        out.extend(block(group, "    "))
        w("")

    if top:
        solved = [e for e in eps if e.get("success")]
        hardest = sorted(solved, key=lambda e: (-e.get("difficulty", 0), e.get("steps", 0)))
        if hardest:
            w(f"HARDEST SOLVED (top {min(top, len(hardest))})")
            for e in hardest[:top]:
                w(f"  {e['puzzle_id']}  diff {e.get('difficulty')}  "
                  f"{e.get('steps')} steps  {e.get('completion_tokens', 0):,} completion tokens")
            w("")

    return "\n".join(out)


def render_comparison(runs: list) -> str:
    """One row per run: the numbers you compare methods on."""
    out = ["=" * 72, "  COMPARISON", "=" * 72, "",
           f"  {'run':<40} {'n':>5} {'wins':>8} {'steps':>7} {'ktok/pz':>8}"]
    for a in runs:
        eps = a["episodes"]
        n = max(len(eps), 1)
        wins = sum(1 for e in eps if e.get("success"))
        steps = statistics.mean([e.get("steps", 0) for e in eps]) if eps else 0
        kt = sum(e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
                 for e in eps) / n / 1000
        out.append(f"  {a['path'].stem[:40]:<40} {len(eps):>5} "
                   f"{100 * wins / n:>7.2f}% {steps:>7.1f} {kt:>8.1f}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=pathlib.Path,
                    help="results JSON files (default: newest in results/json/)")
    ap.add_argument("--out", default=None,
                    help="write here instead of results/txt/<stem>_summary.txt; '-' for stdout only")
    ap.add_argument("--top", type=int, default=5,
                    help="list this many hardest solved puzzles (0 to skip)")
    args = ap.parse_args()

    paths = args.paths
    if not paths:
        # *.json only: the format-failure logs sit here too but are .jsonl.
        candidates = sorted(RESULTS_JSON.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print(f"no *.json found in {RESULTS_JSON}", file=sys.stderr)
            return 1
        paths = [candidates[-1]]

    runs = []
    for path in paths:
        if not path.exists():
            print(f"skipping missing {path}", file=sys.stderr)
            continue
        try:
            runs.append(analyse(path))
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
    if not runs:
        return 1

    report = "\n".join(render(a, args.top) for a in runs)
    if len(runs) > 1:
        report += "\n" + render_comparison(runs)

    print(report)
    if args.out == "-":
        return 0
    if args.out:
        dest = pathlib.Path(args.out)
    elif len(runs) == 1:
        dest = RESULTS_TXT / f"{runs[0]['path'].stem}_summary.txt"
    else:
        dest = RESULTS_TXT / "comparison.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report)
    print(f"[written] {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
