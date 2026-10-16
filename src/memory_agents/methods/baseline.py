"""Run the memoryless baseline over a benchmark split and save results.

Reads the server endpoint from LLM_BASE_URL / LLM_MODEL (set by
serve_and_run.sbatch). Writes a JSON with a summary (accuracy overall and by
difficulty, token totals) plus one record per puzzle.

  python scripts/run.py baseline --split test --concurrency 32 --limit 50
  python scripts/run.py baseline --benchmark stepgame --split test
"""

import argparse
import collections
import json
import os
import pathlib
import time
from datetime import datetime

from memory_agents.agents.base import MAX_FORMAT_RETRIES
from memory_agents.benchmarks import build_benchmark
from memory_agents.config import resolve_server
from memory_agents.llm.client import LLMClient
from memory_agents.methods.common import add_run_args


def summarize_format_failures(path: pathlib.Path) -> dict:
    """Aggregate BaselineAgent's per-reply failure log ("length" = hit max_tokens)."""
    if not path.exists():
        return {}
    reasons, empties, n = collections.Counter(), 0, 0
    puzzles = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        reasons[rec.get("finish_reason") or "none"] += 1
        empties += rec.get("text_len", 0) == 0
        puzzles.add(rec.get("puzzle_id"))
    return {
        "n": n,
        "puzzles_affected": len(puzzles),
        "empty_replies": empties,
        "by_finish_reason": dict(reasons),
        "log": str(path),
    }


def summarize(results) -> dict:
    """Overall + per-difficulty accuracy and token totals."""
    n = len(results)
    wins = sum(r.success for r in results)

    by_diff: dict[int, list] = {}
    for r in results:
        by_diff.setdefault(r.difficulty, []).append(r.success)

    # StepGame only: true hop count (reasoning depth), distinct from k (story length).
    by_hops: dict[int, list] = {}
    for r in results:
        hops = (r.extra or {}).get("hops")
        if hops is not None:
            by_hops.setdefault(int(hops), []).append(r.success)

    return {
        "n": n,
        "success": wins,
        "accuracy": wins / n if n else 0.0,
        "accuracy_by_difficulty": {
            str(d): sum(s) / len(s) for d, s in sorted(by_diff.items())
        },
        **({"accuracy_by_hops": {str(h): sum(v) / len(v)
                                 for h, v in sorted(by_hops.items())}}
           if by_hops else {}),
        "prompt_tokens": sum(r.prompt_tokens() for r in results),
        "completion_tokens": sum(r.completion_tokens() for r in results),
        # Part of the totals above spent on replies that never yielded a move.
        "retry_tokens": sum(r.retry_tokens() for r in results),
        # Crashed, not failed: non-zero means the run is incomplete.
        "errors": sum(1 for r in results if r.terminal_reason == "error"),
        "total_steps": sum(r.steps() for r in results),
    }


def to_record(r) -> dict:
    return {
        "puzzle_id": str(r.puzzle_id),
        "success": r.success,
        "difficulty": r.difficulty,
        "steps": r.steps(),
        "terminated": r.terminated,
        "truncated": r.truncated,
        "terminal_reason": r.terminal_reason,
        "prompt_tokens": r.prompt_tokens(),
        "completion_tokens": r.completion_tokens(),
        "retry_tokens": r.retry_tokens(),
        "error": r.error or None,
        **({"extra": r.extra} if r.extra else {}),
    }


async def main(argv: list[str], prog: str) -> None:
    ap = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Greedy and short, unlike common.SAMPLING. Reasoning models want
    # --temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens 12288.
    add_run_args(ap, max_tokens=1024, temperature=0.0, top_p=1.0, top_k=None,
                 seed_help="decoding seed. Vary it to resample the same split.")
    ap.add_argument("--failure-log", default=None,
                    help="JSONL of unparseable replies (default: alongside --out)")
    args = ap.parse_args(argv)

    limit = args.limit if args.limit > 0 else None
    benchmark = build_benchmark(args)
    base_url, model = resolve_server()

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    fail_log = pathlib.Path(
        args.failure_log or f"results/json/format_failures_{args.split}_{stamp}.jsonl")
    # BaselineAgent reads this env var lazily, so setting it here is enough.
    os.environ["FORMAT_FAILURE_LOG"] = str(fail_log)
    print(f"server={base_url} model={model} split={args.split} "
          f"concurrency={args.concurrency} limit={limit} "
          f"max_tokens={args.max_tokens} temperature={args.temperature} "
          f"top_p={args.top_p} top_k={args.top_k}", flush=True)

    client = LLMClient(base_url=base_url, model=model, max_tokens=args.max_tokens,
                       temperature=args.temperature, top_p=args.top_p,
                       top_k=args.top_k, seed=args.seed)

    t0 = time.time()
    results = await benchmark.run_baseline_suite(
        client, split=args.split, concurrency=args.concurrency, limit=limit)
    elapsed = time.time() - t0
    await client.aclose()

    summary = summarize(results)
    summary["elapsed_sec"] = round(elapsed, 1)
    summary["model"] = model
    # `model` is the served alias; LLM_MODEL_ID is the real checkpoint.
    summary["model_id"] = os.environ.get("LLM_MODEL_ID") or model
    summary["split"] = args.split
    # Protocols are not comparable: `upstream` is 10-shot, so not memoryless.
    if args.benchmark == "stepgame":
        summary["prompt_protocol"] = args.prompt_protocol
    summary["max_tokens"] = args.max_tokens
    summary["temperature"] = args.temperature
    summary["top_p"] = args.top_p
    summary["top_k"] = args.top_k
    summary["seed"] = client.seed
    summary["concurrency"] = args.concurrency
    # Lets summarize_results tell recovered retries from aborted episodes.
    summary["max_format_retries"] = MAX_FORMAT_RETRIES
    summary["benchmark"] = benchmark.name
    # summarize_results reads truncation as a dead end, valid only while episodes
    # stay well under this cap. Omitted for single-turn benchmarks.
    if benchmark.interactive:
        from memory_agents.envs.sparc import MAX_STEPS
        summary["max_steps"] = MAX_STEPS
    summary["format_failures"] = summarize_format_failures(fail_log)

    tag = "" if benchmark.name == "sparc" else f"{benchmark.name}_"
    out = args.out or (
        f"results/json/baseline_{tag}{model.replace('/', '_')}_"
        f"{args.split}_{stamp}.json")
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"summary": summary, "episodes": [to_record(r) for r in results]}, indent=2))

    print(f"\naccuracy={summary['accuracy']:.3f} ({summary['success']}/{summary['n']})  "
          f"by_difficulty={summary['accuracy_by_difficulty']}", flush=True)
    if "accuracy_by_hops" in summary:
        print(f"by_hops={summary['accuracy_by_hops']}", flush=True)
    print(f"tokens: prompt={summary['prompt_tokens']} completion={summary['completion_tokens']}  "
          f"wasted_on_retries={summary['retry_tokens']}  elapsed={elapsed:.1f}s", flush=True)
    if summary["errors"]:
        print(f"WARNING: {summary['errors']} episodes crashed and are counted as "
              f"failures; this run is incomplete", flush=True)
    ff = summary["format_failures"]
    if ff:
        print(f"format failures: {ff['n']} replies over {ff['puzzles_affected']} puzzles, "
              f"{ff['empty_replies']} empty, by_finish_reason={ff['by_finish_reason']}  "
              f"-> {ff['log']}", flush=True)
    print(f"wrote {out_path}", flush=True)

