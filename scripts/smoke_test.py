#!/usr/bin/env python
"""Smoke test for the hosting + async client.

  # start the server first:  sbatch scripts/serve_and_run.sbatch
  python scripts/smoke_test.py                # fire N concurrent chats
  python scripts/smoke_test.py --puzzles 4    # run 4 real SPaRC test puzzles

Default mode needs only the server. --puzzles also needs Spatial_Gym installed.
"""

import argparse
import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents.config import resolve_server
from memory_agents.llm.client import LLMClient


async def chat_smoke(client, n):
    """Send n identical chats at once -- checks the client, concurrency, tokens."""
    messages = [{"role": "user",
                 "content": "Reply with exactly this line and nothing else:\nFinal: 0"}]
    responses = await asyncio.gather(*[client.chat(messages) for _ in range(n)])
    for k, r in enumerate(responses):
        print(f"[{k}] text={r.text!r} prompt_tok={r.prompt_tokens} "
              f"completion_tok={r.completion_tokens}")


async def puzzle_smoke(client, n, concurrency):
    """Run n real puzzles through the baseline harness."""
    from memory_agents.harness.run import run_sparc_suite_baseline
    results = await run_sparc_suite_baseline(
        client, split="test", concurrency=concurrency, limit=n)
    wins = sum(r.success for r in results)
    steps = sum(r.steps() for r in results)
    ptok = sum(r.prompt_tokens() for r in results)
    ctok = sum(r.completion_tokens() for r in results)
    print(f"puzzles={len(results)} success={wins}/{len(results)} steps={steps} "
          f"prompt_tok={ptok} completion_tok={ctok}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=int, default=0, help="run this many SPaRC puzzles")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n", type=int, default=4, help="concurrent chats in default mode")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="bump for reasoning models so the think block can finish")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    base_url, model = resolve_server()
    print(f"server={base_url} model={model}")
    client = LLMClient(base_url=base_url, model=model, max_tokens=args.max_tokens,
                       temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    t0 = time.time()
    if args.puzzles > 0:
        await puzzle_smoke(client, args.puzzles, args.concurrency)
    else:
        await chat_smoke(client, args.n)
    print(f"elapsed={time.time() - t0:.1f}s")
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
