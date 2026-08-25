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
        # An abort is a reply logged at the LAST attempt index, not simply a puzzle
        # with many records: an episode whose five separate moves each failed once
        # and recovered also produces five records, and counting those as aborts
        # inflated this number with exactly the cases that were harmless.
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


def episodes_from_state(state: dict) -> list:
    """Flatten a multi-trial run to one record per puzzle: its final outcome.

    A puzzle attempted over several trials still has one outcome for the run --
    solved or not -- so the per-puzzle stats stay comparable with a single-pass
    baseline. The representative attempt is the winning one where there is one and
    the last one otherwise (that is what "how did this puzzle end" means), while
    tokens are summed over *every* attempt, because all of them were paid for.
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
    # Only unsolved puzzles are re-attempted and solved ones are carried forward, so
    # this curve cannot fall. Saying so here stops the shape being read as evidence.
    out += [
        "",
        f"  cumulative accuracy over {n} puzzles; it can only rise with the trial",
        "  index (solved puzzles are carried forward), so it is NOT evidence on its",
        "  own -- compare it against the control arm at the same trial:",
        "    python scripts/run_reflexion.py --phase compare --memory <f> --control <f>",
    ]
    actor = sum(t.get("actor_prompt_tokens", 0) + t.get("actor_completion_tokens", 0)
                for t in trials)
    refl = sum(t.get("reflection_prompt_tokens", 0)
               + t.get("reflection_completion_tokens", 0) for t in trials)
    out += ["",
            f"  tokens: actor {actor:,} + reflection {refl:,} = {actor + refl:,}"]
    return out


def episodes_from_memp_state(state: dict) -> list:
    """Flatten a memp build/eval state to one record per puzzle.

    The records are already in run_baseline's shape; only the build pass's
    episode_context (the compact trajectory kept for the distill phase) is
    dropped, since no stat reads it and it dwarfs everything else.
    """
    return [{k: v for k, v in rec.items() if k != "episode_context"}
            for _pid, rec in sorted(state.get("puzzles", {}).items())]


def summary_from_memp_state(state: dict) -> dict:
    """The settings block for a memp state file: the run's own summary (present
    once the pass completed) plus the frozen config knobs render() prints."""
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
        "method": "memp",
        "phase": cfg.get("phase"),
        "top_k_memories": cfg.get("top_k_memories"),
        "bank": cfg.get("bank"),
        "embed_model": cfg.get("embed_model"),
        "bank_entries": cfg.get("bank_entries"),
    })
    if "errors" not in s:   # pass still running; the summary is written at the end
        s["errors"] = sum(1 for p in state.get("puzzles", {}).values()
                          if p.get("terminal_reason") == "error")
    return s


def analyse(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text())
    # Three on-disk shapes: single-pass runs ({summary, episodes}), multi-trial
    # Reflexion state files ({config, trials, puzzles, memory}) and memp state
    # files ({config: {schema: memp-*}, puzzles}). A memp *bank* is none of these
    # -- it holds procedures, not episodes -- so name it rather than render junk.
    schema = data.get("config", {}).get("schema", "")
    if data.get("schema") == "memp-bank-v1" or "entries" in data:
        stats = data.get("stats", {})
        raise ValueError(
            f"a memp bank ({len(data.get('entries', []))} entries, "
            f"stats={stats}), not a run; nothing to summarize")
    if schema.startswith("memp"):
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
    return {
        "path": path,
        "summary": summary,
        "episodes": episodes,
        "state": state,
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
    # `model` is only vLLM's --served-model-name alias, which used to default to
    # "qwen" for every checkpoint. Printing it next to the real model read as though
    # gpt-oss had been served as a Qwen, so it is no longer shown: the checkpoint is
    # the only thing a reader of this report needs. Runs old enough to predate
    # model_id have nothing but the alias, so those say so instead of passing it off
    # as the model.
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
                "retrieved": "   [Memp: cosine top-k procedures injected]",
                "random": "   [Memp ABLATION: random procedures injected]"}
        w(f"  {'arm:':<20} {s['arm']}{note.get(s['arm'], '')}")
    if s.get("method") == "memp":
        if s.get("phase") == "build":
            w(f"  {'phase:':<20} build   [memoryless pass over train; feeds distill]")
        else:
            w(f"  {'top-k memories:':<20} {s.get('top_k_memories', '?')}")
            w(f"  {'granularity:':<20} {s.get('granularity', 'script')}"
              "   [paper's Build policy: script | trajectory | proceduralization]")
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
    out.extend(block(eps, "  "))
    # replies = one per step, plus every retry that had to be thrown away
    replies = sum(e.get("steps", 0) for e in eps) + (s.get("format_failures") or {}).get("n", 0)
    out.extend(format_failures(s, replies))
    # Crashed episodes are neither terminated nor truncated, so they show up in no
    # bucket above. Say so explicitly: a non-zero count means the run is incomplete,
    # not that the model did badly.
    if s.get("errors"):
        w(f"  errored episodes:    {share(s['errors'], n)}   [crashed; run is INCOMPLETE]")
    if s.get("retry_tokens"):
        total = s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)
        w(f"  wasted on retries:   {share(s['retry_tokens'], total)} of all tokens"
          f"   [replies that yielded no move]")

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
