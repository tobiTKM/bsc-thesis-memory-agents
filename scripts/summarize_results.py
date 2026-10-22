#!/usr/bin/env python
"""Condense a run's results JSON into a short text report: settings, overall and
per-difficulty stats, and the hardest solved puzzles.

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
RESULTS_JSON = REPO / "results" / "json"
RESULTS_TXT = REPO / "results" / "txt"


# ---- formatting helpers --------------------------------------------------
def dist(values) -> str:
    """avg/med/min/max of a list of numbers."""
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
    """Report lines on unparseable replies, including episodes aborted by them.

    An abort (every retry failed) ends the episode as truncated, which is
    indistinguishable from a real dead end in the stats.
    """
    ff = summary.get("format_failures") or {}
    if not ff:
        return []
    n, puzzles = ff.get("n", 0), ff.get("puzzles_affected", 0)
    if not n:
        return []

    # The aggregate has no attempt numbers, so read the raw log.
    aborted = None
    log = pathlib.Path(ff.get("log", ""))
    if not log.exists():
        # Older runs recorded the log at a flat results/ path.
        log = RESULTS_JSON / log.name
    if log.exists():
        # An abort is a record at the last attempt index; several records for one
        # puzzle can be separate moves that each recovered.
        retries = summary.get("max_format_retries", 5)
        aborted_puzzles = set()
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("attempt", 0) >= retries - 1:
                    aborted_puzzles.add(rec["puzzle_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        aborted = len(aborted_puzzles)

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
# Same terminated/truncated arithmetic on every benchmark; only the meaning differs.
_LABELS = {
    "sparc": ("[reached the exit]",
              "[dead end, no legal move]",
              "[exit reached, rule broken]"),
    "stepgame": ("[gave a parseable answer]",
                 "[never produced a parseable answer]",
                 "[answered, but wrongly]"),
}


def block(episodes: list, indent: str, benchmark: str = "sparc") -> list:
    """The stat block for a group of episodes (whole split or one difficulty).

    `terminated` and `truncated` are over all episodes and sum to 100%; wins are a
    subset of `terminated`. On SPaRC `truncated` means a dead end (the step cap never binds).
    """
    reached, cutoff, missed = _LABELS.get(benchmark, _LABELS["sparc"])
    n = len(episodes)
    wins = sum(1 for e in episodes if e.get("success"))
    term = sum(1 for e in episodes if e.get("terminated"))
    trunc = sum(1 for e in episodes if e.get("truncated"))

    steps = [e.get("steps", 0) for e in episodes]
    completion = [e.get("completion_tokens", 0) for e in episodes]
    prompt = [e.get("prompt_tokens", 0) for e in episodes]
    per_reply = [e.get("completion_tokens", 0) / e["steps"]
                 for e in episodes if e.get("steps")]

    out = [
        f"{indent}total puzzles:        {n}",
        f"{indent}wins:                {share(wins, n)}",
        f"{indent}fails:               {share(n - wins, n)}",
        f"{indent}terminated runs:     {share(term, n)}   {reached}",
        f"{indent}truncated runs:      {share(trunc, n)}   {cutoff}",
        f"{indent}failed at exit:      {share(term - wins, n)}   {missed}",
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


def episodes_from_state(state: dict) -> list:
    """Flatten a multi-trial run to one record per puzzle.

    Outcome fields come from the winning attempt (else the last); tokens are
    summed over all attempts.
    """
    out = []
    for pid, e in sorted(state.get("puzzles", {}).items()):
        attempts = e.get("attempts") or []
        if not attempts:
            continue
        shown = next((a for a in attempts if a.get("success")), attempts[-1])
        out.append({
            "puzzle_id": pid,
            "difficulty": e.get("difficulty", -1),
            "success": e.get("solved", False),
            "terminated": shown.get("terminated"),
            "truncated": shown.get("truncated"),
            "terminal_reason": shown.get("terminal_reason"),
            "steps": shown.get("steps", 0),
            "prompt_tokens": sum(a.get("prompt_tokens", 0) for a in attempts),
            "completion_tokens": sum(a.get("completion_tokens", 0) for a in attempts),
            "retry_tokens": sum(a.get("retry_tokens", 0) for a in attempts),
            "attempts": len(attempts),
            "solved_at_trial": e.get("solved_at_trial"),
        })
    return out


def summary_from_state(state: dict) -> dict:
    """Rebuild the settings block render() prints, from a multi-trial state file."""
    cfg = state.get("config", {})
    trials = state.get("trials", [])
    return {
        "model": cfg.get("model", "?"),
        "model_id": cfg.get("model_id"),
        "split": cfg.get("split"),
        "max_tokens": cfg.get("max_tokens"),
        "temperature": cfg.get("temperature"),
        "top_p": cfg.get("top_p"),
        "top_k": cfg.get("top_k"),
        "seed": cfg.get("base_seed"),
        "concurrency": cfg.get("concurrency"),
        "arm": cfg.get("arm"),
        "memory_cap": cfg.get("memory_cap"),
        "elapsed_sec": sum(t.get("elapsed_sec", 0) for t in trials),
        "errors": sum(t.get("errors", 0) for t in trials),
    }


def trial_curve(state: dict) -> list:
    """The per-trial table: what each trial re-attempted, won and cost."""
    trials = state.get("trials", [])
    if not trials:
        return []
    cfg = state.get("config", {})
    n = len(state.get("puzzles", {}))
    out = [
        f"TRIAL CURVE  (arm: {cfg.get('arm', '?')}, "
        f"memory window: {cfg.get('memory_cap', '?')} reflections/puzzle)",
        f"  {'trial':>5} {'seed':>5} {'attempted':>10} {'new':>5} {'solved':>7} "
        f"{'acc':>8} {'refl':>6} {'minutes':>8}",
    ]
    for t in trials:
        out.append(
            f"  {t.get('trial', 0):>5} {t.get('seed', 0):>5} "
            f"{t.get('attempted', 0):>10} {t.get('solved_this_trial', 0):>5} "
            f"{t.get('cumulative_solved', 0):>7} "
            f"{t.get('cumulative_accuracy', 0):>8.3f} "
            f"{t.get('reflections_written', 0):>6} "
            f"{t.get('elapsed_sec', 0) / 60:>8.0f}")
    out += [
        "",
        f"  cumulative accuracy over {n} puzzles; it can only rise with the trial",
        "  index (solved puzzles are carried forward), so it is NOT evidence on its",
        "  own -- compare it against the control arm at the same trial:",
        "    python scripts/run.py reflexion --phase compare --memory <f> --control <f>",
    ]
    actor = sum(t.get("actor_prompt_tokens", 0) + t.get("actor_completion_tokens", 0)
                for t in trials)
    refl = sum(t.get("reflection_prompt_tokens", 0)
               + t.get("reflection_completion_tokens", 0) for t in trials)
    out += ["",
            f"  tokens: actor {actor:,} + reflection {refl:,} = {actor + refl:,}"]
    return out


# The cross-task methods share one {config, puzzles, summary} state shape and differ
# only in schema prefix; a new method only needs its prefix added here.
BANK_FAMILY = ("memp-bank-v1", "memorybank-bank-v1", "contrastive-bank-v1")
STATE_FAMILY = ("memp", "memorybank", "contrastive")


def method_of(schema: str) -> str:
    """"memorybank-eval-v1" -> "memorybank"."""
    return schema.split("-", 1)[0] if schema else ""


def episodes_from_memp_state(state: dict) -> list:
    """Records of a memp/memorybank/contrastive state, minus the bulky episode_context."""
    return [{k: v for k, v in rec.items() if k != "episode_context"}
            for _pid, rec in sorted(state.get("puzzles", {}).items())]


def summary_from_memp_state(state: dict) -> dict:
    """The run's summary (written once the pass completes) merged with its config."""
    cfg = state.get("config", {})
    s = dict(state.get("summary") or {})
    s.update({
        "model": cfg.get("model", "?"),
        "model_id": cfg.get("model_id"),
        "split": cfg.get("split"),
        "max_tokens": cfg.get("max_tokens"),
        "temperature": cfg.get("temperature"),
        "top_p": cfg.get("top_p"),
        "top_k": cfg.get("top_k"),
        "seed": cfg.get("seed"),
        "concurrency": cfg.get("concurrency"),
        "arm": cfg.get("arm"),
        "method": method_of(cfg.get("schema", "")),
        "phase": cfg.get("phase"),
        "top_k_memories": cfg.get("top_k_memories"),
        "bank": cfg.get("bank"),
        "embed_model": cfg.get("embed_model"),
        "bank_entries": cfg.get("bank_entries"),
    })
    # Fall back to the config so an unfinished run reports its real settings.
    for key in ("granularity", "provenance", "failure_coverage_bank",
                "mean_block_chars"):
        if s.get(key) is None and cfg.get(key) is not None:
            s[key] = cfg[key]
    if "errors" not in s:   # pass still running; the summary is written at the end
        s["errors"] = sum(1 for p in state.get("puzzles", {}).values()
                          if p.get("terminal_reason") == "error")
    return s


def analyse(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text())
    # Shapes: single-pass {summary, episodes}, Reflexion {config, trials, puzzles},
    # cross-task {config: {schema}, puzzles}. A memory bank is rejected.
    schema = data.get("config", {}).get("schema", "")
    if data.get("schema") in BANK_FAMILY or "entries" in data:
        stats = data.get("stats", {})
        raise ValueError(
            f"a {data.get('schema', 'memory')} bank "
            f"({len(data.get('entries', []))} entries, stats={stats}), not a run; "
            f"nothing to summarize")
    if method_of(schema) in STATE_FAMILY:
        episodes = episodes_from_memp_state(data)
        summary = summary_from_memp_state(data)
        state = None
    elif "trials" in data and "puzzles" in data:
        episodes = episodes_from_state(data)
        summary = summary_from_state(data)
        state = data
    else:
        episodes = data.get("episodes", [])
        summary = data.get("summary", {})
        state = None
    by_diff = collections.defaultdict(list)
    for ep in episodes:
        by_diff[ep.get("difficulty", -1)].append(ep)
    # StepGame: `difficulty` is k (story length); `extra.hops` is reasoning depth.
    by_hops = collections.defaultdict(list)
    for ep in episodes:
        hops = (ep.get("extra") or {}).get("hops")
        if hops is not None:
            by_hops[int(hops)].append(ep)
    benchmark = summary.get("benchmark") or (
        "stepgame" if by_hops else "sparc")
    return {
        "path": path,
        "summary": summary,
        "episodes": episodes,
        "state": state,
        "benchmark": benchmark,
        "by_diff": dict(sorted(by_diff.items())),
        "by_hops": dict(sorted(by_hops.items())),
    }


def render(a: dict, top: int) -> str:
    s, eps = a["summary"], a["episodes"]
    n = len(eps) or s.get("n", 0)
    out = []
    w = out.append

    w("=" * 72)
    w(f"  {a['path'].name}")
    w("=" * 72)

    # Older runs lack some fields; print "(not recorded)" rather than a default.
    def setting(label: str, key: str, fmt=str):
        val = s.get(key)
        w(f"  {label:<20} {fmt(val) if val is not None else '(not recorded)'}")

    w("")
    w("RUN")
    # `model` is the served alias; model_id is the real checkpoint.
    served, exact = s.get("model", "?"), s.get("model_id")
    if exact:
        w(f"  {'model:':<20} {exact}")
    else:
        w(f"  {'model:':<20} {served}   (API alias; real checkpoint not recorded)")
    setting("split:", "split")
    setting("max_tokens:", "max_tokens")
    setting("temperature:", "temperature")
    setting("top_p:", "top_p")
    setting("top_k:", "top_k")
    setting("seed:", "seed")
    setting("concurrency:", "concurrency")
    if s.get("arm"):
        note = {"memory": "   [Reflexion: reflections injected]",
                "control": "   [CONTROL: same trials, no memory]",
                "retrieved": "   [cosine top-k memories injected]",
                "random": "   [ABLATION: randomly chosen memories injected]",
                "fixed": "   [ABLATION: the same k memories for every item]",
                "avefact": "   [Memp AveFact: keyword-averaged retrieval]"}
        w(f"  {'arm:':<20} {s['arm']}{note.get(s['arm'], '')}")
    if s.get("method") in STATE_FAMILY:
        if s.get("phase") == "build":
            w(f"  {'phase:':<20} build   [memoryless pass over train; feeds distill]")
        else:
            w(f"  {'top-k memories:':<20} {s.get('top_k_memories', '?')}")
            scale = {
                "memp": "   [paper's Build policy: script | trajectory | "
                        "proceduralization]",
                "memorybank": "   [the paper's two storage layers: "
                              "case-record | case]",
                "contrastive": "   [WHICH HALVES ARE SHOWN -- this is the "
                               "independent variable]",
            }.get(s.get("method"), "")
            # Fallbacks must match each driver's argparse default.
            gran = s.get("granularity") or {
                "memp": "script", "memorybank": "case-record",
                "contrastive": "contrastive"}.get(s.get("method"), "(not recorded)")
            w(f"  {'granularity:':<20} {gran}{scale}")
            # Contrastive only.
            if s.get("failure_coverage_injected") is not None:
                w(f"  {'failure coverage:':<20} "
                  f"{s['failure_coverage_injected']:.1%} of injected entries "
                  f"carried a failed attempt"
                  + (f" (bank: {s['failure_coverage_bank']:.1%})"
                     if s.get("failure_coverage_bank") is not None else ""))
            if s.get("mean_block_chars"):
                w(f"  {'injected block:':<20} "
                  f"~{s['mean_block_chars'] // 4} tokens mean")
            if s.get("provenance") and s["provenance"] != "all":
                w(f"  {'bank provenance:':<20} {s['provenance']}"
                  "   [agent = paper-faithful successes-only build]")
            if s.get("update") and s["update"] != "none":
                w(f"  {'update policy:':<20} {s['update']}"
                  f"   [ONLINE: bank evolved over {s.get('update_groups', '?')} groups]")
            w(f"  {'bank:':<20} {s.get('bank', '?')}"
              + (f"   ({s['bank_entries']} entries)" if s.get("bank_entries") else ""))
            if s.get("bank_model") and s["bank_model"] != s.get("model_id"):
                w(f"  {'bank built by:':<20} {s['bank_model']}"
                  "   [TRANSFER: bank and actor are different models]")
            w(f"  {'embed model:':<20} {s.get('embed_model') or '(not recorded)'}")
            if s.get("mean_top1_similarity") is not None:
                w(f"  {'top-1 similarity:':<20} mean {s['mean_top1_similarity']}")
            if s.get("injected_provenance"):
                w(f"  {'injected entries:':<20} {s['injected_provenance']}"
                  f"   [agent = from own success, gold = from gold path, "
                  f"online = added during the run]")
            if s.get("group_curve"):
                w("")
                w("UPDATE CURVE  (the paper's Figure 3: does the bank improve as it learns?)")
                w(f"  {'group':>5} {'attempted':>10} {'solved':>7} {'rate':>7} {'bank':>7}")
                for g in s["group_curve"]:
                    att = g.get("attempted") or 0
                    won = g.get("solved") or 0
                    w(f"  {g.get('group', 0):>5} {att:>10} {won:>7} "
                      f"{(won / att if att else 0):>7.3f} {g.get('bank_after', 0):>7}")
                w(f"  entries added {s.get('entries_added', 0)}, "
                  f"revised {s.get('entries_revised', 0)}, "
                  f"update cost {s.get('update_tokens', 0):,} tokens")
                w("  NOTE: each group is a disjoint slice of the split, so this rate is")
                w("  directly comparable across groups -- unlike Reflexion's cumulative curve.")
    if s.get("elapsed_sec"):
        w(f"  {'runtime:':<20} {hms(s['elapsed_sec'])}")
        w(f"  {'per puzzle:':<20} {s['elapsed_sec'] / max(n, 1):.1f} s"
          f"   (wall clock / puzzles, so it is divided by concurrency)")

    if a.get("state"):
        w("")
        out.extend(trial_curve(a["state"]))

    w("")
    w("FULL RUN")
    out.extend(block(eps, "  ", a.get("benchmark", "sparc")))
    replies = sum(e.get("steps", 0) for e in eps) + (s.get("format_failures") or {}).get("n", 0)
    out.extend(format_failures(s, replies))
    # Crashed episodes are neither terminated nor truncated, so no bucket above counts them.
    if s.get("errors"):
        w(f"  errored episodes:    {share(s['errors'], n)}   [crashed; run is INCOMPLETE]")
    if s.get("retry_tokens"):
        total = s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)
        w(f"  wasted on retries:   {share(s['retry_tokens'], total)} of all tokens"
          f"   [replies that yielded no move]")

    bm = a.get("benchmark", "sparc")
    axis = "k (story length)" if bm == "stepgame" else "Difficulty"

    w("")
    w("BY DIFFICULTY" if bm != "stepgame" else "BY k  (story length)")
    for d, group in a["by_diff"].items():
        w(f"  {axis} {d}")
        out.extend(block(group, "    ", bm))
        w("")

    if a.get("by_hops"):
        w("BY HOPS  (true reasoning depth)")
        for h, group in a["by_hops"].items():
            w(f"  hops {h}")
            out.extend(block(group, "    ", bm))
            w("")

    if top:
        solved = [e for e in eps if e.get("success")]
        # Rank by hops where present: a high-k StepGame item can be a single hop.
        def hardness(e):
            hops = (e.get("extra") or {}).get("hops")
            return (-(hops if hops is not None else e.get("difficulty", 0)),
                    e.get("steps", 0))
        hardest = sorted(solved, key=hardness)
        if hardest:
            w(f"HARDEST SOLVED (top {min(top, len(hardest))})")
            for e in hardest[:top]:
                hops = (e.get("extra") or {}).get("hops")
                depth = (f"diff {e.get('difficulty')}" if hops is None
                         else f"k {e.get('difficulty')} / {hops} hops")
                w(f"  {e['puzzle_id']}  {depth}  "
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
        # *.json only: the format-failure .jsonl logs live here too.
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
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
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
