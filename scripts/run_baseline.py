#!/usr/bin/env python
"""Run the memoryless baseline over a SPaRC split and save results.

Reads the server endpoint from LLM_BASE_URL / LLM_MODEL (set by run_all.sbatch)
or the discovery file. Writes a JSON with a summary (accuracy overall and by
difficulty, token totals) plus one record per puzzle.

  python scripts/run_baseline.py --split test --concurrency 32 --limit 50
"""

import argparse
import asyncio
import json
import pathlib
import sys
import time
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents.config import resolve_server
from memory_agents.harness.run import run_sparc_suite_baseline
from memory_agents.llm.client import LLMClient


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
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    limit = args.limit if args.limit > 0 else None
    base_url, model = resolve_server()
    print(f"server={base_url} model={model} split={args.split} "
          f"concurrency={args.concurrency} limit={limit}", flush=True)

    client = LLMClient(base_url=base_url, model=model, max_tokens=args.max_tokens)

    t0 = time.time()
    results = await run_sparc_suite_baseline(
        client, split=args.split, concurrency=args.concurrency, limit=limit)
    elapsed = time.time() - t0
    await client.aclose()

    summary = summarize(results)
    summary["elapsed_sec"] = round(elapsed, 1)
    summary["model"] = model
    summary["split"] = args.split

    out = args.out or (
        f"results/baseline_{model.replace('/', '_')}_{args.split}_"
        f"{datetime.now():%Y%m%d_%H%M%S}.json")
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"summary": summary, "episodes": [to_record(r) for r in results]}, indent=2))

    print(f"\naccuracy={summary['accuracy']:.3f} ({summary['success']}/{summary['n']})  "
          f"by_difficulty={summary['accuracy_by_difficulty']}", flush=True)
    print(f"tokens: prompt={summary['prompt_tokens']} completion={summary['completion_tokens']}  "
          f"elapsed={elapsed:.1f}s", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
