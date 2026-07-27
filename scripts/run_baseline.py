#!/usr/bin/env python
"""Run the memoryless baseline over a SPaRC split and save results.

Reads the server endpoint from LLM_BASE_URL / LLM_MODEL (set by run_all.sbatch)
or the discovery file. Writes a JSON with a summary (accuracy overall and by
difficulty, token totals) plus one record per puzzle.

  python scripts/run_baseline.py --split test --concurrency 32 --limit 50
"""

import argparse
import asyncio
import collections
import json
import os
import pathlib
import sys
import time
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents.config import resolve_server
from memory_agents.harness.run import run_sparc_suite_baseline
from memory_agents.llm.client import LLMClient


def summarize_format_failures(path: pathlib.Path) -> dict:
    """Aggregate the per-reply failure log written by BaselineAgent.

    `finish_reason == "length"` is the one that matters: it means the reply hit
    max_tokens, so the fix is a bigger budget, not a looser parser.
    """
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

    return {
        "n": n,
        "success": wins,
        "accuracy": wins / n if n else 0.0,
        "accuracy_by_difficulty": {
            str(d): sum(s) / len(s) for d, s in sorted(by_diff.items())
        },
        "prompt_tokens": sum(r.prompt_tokens() for r in results),
        "completion_tokens": sum(r.completion_tokens() for r in results),
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
        "prompt_tokens": r.prompt_tokens(),
        "completion_tokens": r.completion_tokens(),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 or less = whole split")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (deterministic). Reasoning models: use ~0.6.")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None,
                    help="vLLM extra param; unset = disabled. Qwen3 thinking: 20.")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--failure-log", default=None,
                    help="JSONL of unparseable replies (default: alongside --out)")
    args = ap.parse_args()

    limit = args.limit if args.limit > 0 else None
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
                       temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    t0 = time.time()
    results = await run_sparc_suite_baseline(
        client, split=args.split, concurrency=args.concurrency, limit=limit)
    elapsed = time.time() - t0
    await client.aclose()

    summary = summarize(results)
    summary["elapsed_sec"] = round(elapsed, 1)
    summary["model"] = model
    # `model` is only the --served-model-name alias ("qwen"), which is stable on
    # purpose so result filenames are. The real checkpoint comes from the serve
    # job, so a report can name what was actually run.
    summary["model_id"] = os.environ.get("LLM_MODEL_ID") or model
    summary["split"] = args.split
    summary["max_tokens"] = args.max_tokens
    summary["temperature"] = args.temperature
    summary["top_p"] = args.top_p
    summary["top_k"] = args.top_k
    summary["seed"] = client.seed
    summary["concurrency"] = args.concurrency
    summary["format_failures"] = summarize_format_failures(fail_log)

    # results/json holds the machine-readable run data (this file plus the failure
    # log); results/txt holds the reports scripts/summarize_results.py writes.
    out = args.out or (
        f"results/json/baseline_{model.replace('/', '_')}_{args.split}_{stamp}.json")
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"summary": summary, "episodes": [to_record(r) for r in results]}, indent=2))

    print(f"\naccuracy={summary['accuracy']:.3f} ({summary['success']}/{summary['n']})  "
          f"by_difficulty={summary['accuracy_by_difficulty']}", flush=True)
    print(f"tokens: prompt={summary['prompt_tokens']} completion={summary['completion_tokens']}  "
          f"elapsed={elapsed:.1f}s", flush=True)
    ff = summary["format_failures"]
    if ff:
        print(f"format failures: {ff['n']} replies over {ff['puzzles_affected']} puzzles, "
              f"{ff['empty_replies']} empty, by_finish_reason={ff['by_finish_reason']}  "
              f"-> {ff['log']}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
