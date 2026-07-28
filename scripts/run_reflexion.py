#!/usr/bin/env python
"""Run Reflexion over a SPaRC split, as two disk-linked phases.

Reflexion adds memory by *retrying*: attempt each puzzle once with no memory,
turn each failure into a short verbal reflection, then re-attempt the failed
puzzles with their own reflection in context. We split that across two SLURM jobs
so the reflections can be inspected (or hand-edited) between them:

  phase pass1:  trial-1 = the memoryless baseline, then reflect on every failure.
                Writes results/json/reflexion_pass1_<...>.json (per-puzzle record
                + that puzzle's reflection).
  phase pass2:  loads the pass1 file, re-runs only the failed puzzles with their
                reflection injected, carries the trial-1 wins forward, and writes
                the final trial-1 vs trial-2 comparison.
  phase control: the same retry over the same puzzles at the same seed, with NO
                reflection injected. Required to interpret pass2: only failures are
                retried, so lift is >= 0 by construction, and a second sample at
                temperature 0.6 converts some failures on its own. Reflexion's real
                effect is pass2's retried_accuracy minus this one's.

  python scripts/run_reflexion.py --phase pass1   --split test --limit 50
  python scripts/run_reflexion.py --phase pass2   --pass1 results/json/reflexion_pass1_*.json
  python scripts/run_reflexion.py --phase control --pass1 results/json/reflexion_pass1_*.json
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
from memory_agents.harness.run import (
    run_sparc_suite_baseline,
    run_sparc_suite_memory_pass,
)
from memory_agents.llm.client import LLMClient
from memory_agents.memory import Reflector, ReflexionMemory


# --------------------------------------------------------------------------- #
# metrics helpers (shared shape with scripts/run_baseline.py)
# --------------------------------------------------------------------------- #
def accuracy_block(records) -> dict:
    """Overall + per-difficulty accuracy over an iterable of {success, difficulty}."""
    n = len(records)
    wins = sum(r["success"] for r in records)
    by_diff: dict[int, list] = {}
    for r in records:
        by_diff.setdefault(r["difficulty"], []).append(r["success"])
    return {
        "n": n,
        "success": wins,
        "accuracy": wins / n if n else 0.0,
        "accuracy_by_difficulty": {
            str(d): sum(s) / len(s) for d, s in sorted(by_diff.items())
        },
    }


def to_record(r) -> dict:
    """Per-puzzle record, with the reflection fields Reflexion needs downstream."""
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
        "retry_tokens": r.retry_tokens(),   # spent on replies that yielded no move
        "error": r.error or None,           # set only when the episode crashed
        "reflection": None,   # filled in for failures during pass1
    }


def make_client(args, max_tokens, seed=None) -> LLMClient:
    base_url, model = resolve_server()
    return LLMClient(base_url=base_url, model=model, max_tokens=max_tokens,
                     temperature=args.temperature, top_p=args.top_p,
                     top_k=args.top_k,
                     seed=args.seed if seed is None else seed)


# --------------------------------------------------------------------------- #
# phase 1: baseline trial + reflect on every failure
# --------------------------------------------------------------------------- #
async def reflect_on_failures(reflector, results, concurrency):
    """Generate one reflection per failed episode, `concurrency` at a time.

    Returns (memory, reflections, tokens, n_failed): a ReflexionMemory keyed by
    puzzle id, the per-puzzle texts, the token totals spent generating them
    (counted as part of Reflexion's cost), and how many calls blew up.
    """
    memory = ReflexionMemory()
    failures = [r for r in results if not r.success]
    tokens = collections.Counter()
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        async with sem:
            text, resp = await reflector.reflect(r)
        tokens["prompt"] += resp.prompt_tokens
        tokens["completion"] += resp.completion_tokens
        # An empty reflection (e.g. a reasoning model that spent its whole budget
        # in <think>) is useless as a hint; skip it so pass2 just carries the
        # trial-1 failure rather than re-running with no guidance.
        if text:
            memory.write(r.puzzle_id, text)
        return str(r.puzzle_id), text

    # return_exceptions is load-bearing: this runs *after* trial-1 has finished, so
    # letting one bad reflection call propagate out of gather would throw away a
    # completed baseline run that cost hours of GPU time, before anything is written
    # to disk. A reflection we could not generate just means that puzzle carries its
    # trial-1 failure forward.
    outcomes = await asyncio.gather(*(one(r) for r in failures), return_exceptions=True)
    reflections, n_failed = {}, 0
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            n_failed += 1
            print(f"[pass1] reflection failed: {outcome!r}", flush=True)
            continue
        pid, text = outcome
        reflections[pid] = text
    return memory, reflections, tokens, n_failed


async def run_pass1(args):
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    fail_log = pathlib.Path(f"results/json/format_failures_reflexion_{args.split}_{stamp}.jsonl")
    os.environ["FORMAT_FAILURE_LOG"] = str(fail_log)

    actor = make_client(args, args.max_tokens)
    reflector = Reflector(make_client(args, args.reflect_max_tokens))

    print(f"[pass1] trial-1 baseline: split={args.split} limit={args.limit} "
          f"concurrency={args.concurrency}", flush=True)
    t0 = time.time()
    results = await run_sparc_suite_baseline(
        actor, split=args.split, concurrency=args.concurrency, limit=args.limit)

    n_fail = sum(not r.success for r in results)
    print(f"[pass1] trial-1 done: {len(results)-n_fail}/{len(results)} solved; "
          f"reflecting on {n_fail} failures...", flush=True)
    memory, reflections, rtokens, rfailed = await reflect_on_failures(
        reflector, results, args.concurrency)
    elapsed = time.time() - t0

    await actor.aclose()
    await reflector.client.aclose()

    episodes = []
    for r in results:
        rec = to_record(r)
        rec["reflection"] = reflections.get(rec["puzzle_id"])
        episodes.append(rec)

    summary = accuracy_block(episodes)
    summary.update({
        "phase": "pass1",
        "model": actor.model,
        "model_id": os.environ.get("LLM_MODEL_ID") or actor.model,
        "split": args.split,
        "elapsed_sec": round(elapsed, 1),
        "seed": actor.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reflect_max_tokens": args.reflect_max_tokens,
        "reflections_written": len(memory),
        "reflections_failed": rfailed,
        # Episodes that crashed rather than failed; non-zero means trial-1 is
        # incomplete, not that the model did badly.
        "errors": sum(1 for r in results if r.terminal_reason == "error"),
        # actor (trial-1) tokens vs reflection-generation tokens, kept apart so
        # the final report can show what memory actually costs.
        "actor_prompt_tokens": sum(e["prompt_tokens"] for e in episodes),
        "actor_completion_tokens": sum(e["completion_tokens"] for e in episodes),
        "reflection_prompt_tokens": rtokens["prompt"],
        "reflection_completion_tokens": rtokens["completion"],
    })

    out = args.out or f"results/json/reflexion_pass1_{args.split}_{stamp}.json"
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=2))

    print(f"[pass1] trial-1 accuracy={summary['accuracy']:.3f} "
          f"({summary['success']}/{summary['n']}); {len(memory)} reflections written",
          flush=True)
    print(f"[pass1] wrote {out_path}\n[pass1] next: "
          f"python scripts/run_reflexion.py --phase pass2 --pass1 {out_path}", flush=True)


# --------------------------------------------------------------------------- #
# phase pass2 / control: re-attempt the failures, with or without the reflection
# --------------------------------------------------------------------------- #
async def run_retry_pass(args, with_memory: bool):
    """Second attempt at every puzzle trial-1 failed, carrying trial-1 wins forward.

    `with_memory=True` is Reflexion proper: each puzzle gets its own reflection in
    the actor's context. `with_memory=False` is the CONTROL, and it is what makes
    the headline number mean anything.

    Why the control is not optional: only trial-1 *failures* are retried and trial-1
    wins are carried forward untouched, so trial-2 accuracy can never drop below
    trial-1 and `lift` is >= 0 by construction. The actor also samples at
    temperature 0.6, so a second attempt converts some failures on its own. Without
    a no-memory arm the measured lift is (memory effect + resampling effect) with no
    way to separate the two, which is the first thing an examiner will ask about.

    Both arms run at --retry-seed, which MUST differ from the seed trial-1 used:
    given the same prompt and the same seed, vLLM reproduces trial-1's reply, so a
    control at the trial-1 seed would measure a replay rather than a resample and
    flatter Reflexion by default. With that held equal, the two arms differ only in
    whether the reflection block is injected.
    """
    label = "pass2" if with_memory else "control"
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    pass1 = json.loads(pathlib.Path(args.pass1).read_text())
    ep1 = pass1["episodes"]

    # The retried set: failed puzzles that got a (possibly hand-edited) reflection.
    # The control re-attempts exactly this set so the two arms stay comparable.
    memory = {e["puzzle_id"]: [e["reflection"]]
              for e in ep1 if not e["success"] and e["reflection"]}
    how = "with reflections" if with_memory else "WITHOUT memory (control arm)"
    print(f"[{label}] loaded {args.pass1}: trial-1 solved "
          f"{sum(e['success'] for e in ep1)}/{len(ep1)}; "
          f"re-attempting {len(memory)} {how} at seed {args.retry_seed}", flush=True)

    fail_log = pathlib.Path(
        f"results/json/format_failures_reflexion_{label}_{args.split}_{stamp}.jsonl")
    os.environ["FORMAT_FAILURE_LOG"] = str(fail_log)
    actor = make_client(args, args.max_tokens, seed=args.retry_seed)

    # A MemoryAgent with an empty block behaves identically to the memoryless
    # baseline (see MemoryAgent), so the control needs nothing but this.
    block_kwargs = {} if with_memory else {"block_of": lambda _: ""}

    t0 = time.time()
    results2 = await run_sparc_suite_memory_pass(
        actor, memory, split=args.split, concurrency=args.concurrency, **block_kwargs)
    elapsed = time.time() - t0
    await actor.aclose()

    # Build the final per-puzzle outcome: trial-2 result where we re-ran, else the
    # carried-forward trial-1 outcome (solved puzzles + any failure we could not
    # reflect on).
    retried = {str(r.puzzle_id): r for r in results2}
    final = []
    for e in ep1:
        pid = e["puzzle_id"]
        if pid in retried:
            r = retried[pid]
            rec = {"puzzle_id": pid, "success": r.success, "difficulty": r.difficulty,
                   "steps": r.steps(), "terminal_reason": r.terminal_reason,
                   "prompt_tokens": r.prompt_tokens(),
                   "completion_tokens": r.completion_tokens(),
                   "retry_tokens": r.retry_tokens(), "from": "trial2"}
        else:
            rec = {"puzzle_id": pid, "success": e["success"], "difficulty": e["difficulty"],
                   "steps": e["steps"], "terminal_reason": e["terminal_reason"],
                   "prompt_tokens": 0, "completion_tokens": 0,
                   "retry_tokens": 0, "from": "carried"}
        final.append(rec)

    trial1 = accuracy_block(ep1)
    trial2 = accuracy_block(final)
    s1 = pass1["summary"]
    actor2_prompt = sum(r.prompt_tokens() for r in results2)
    actor2_completion = sum(r.completion_tokens() for r in results2)
    # The number to actually compare arms on. Whole-split accuracy is diluted by the
    # carried trial-1 wins, which compresses the effect; on the retried subset
    # trial-1 accuracy is 0 by construction, so this fraction *is* the effect:
    # of N re-attempted failures, how many did this arm solve.
    retried_solved = sum(1 for r in results2 if r.success)

    summary = {
        "phase": label,
        "with_memory": with_memory,
        "model": actor.model,
        "model_id": s1.get("model_id", actor.model),
        "split": args.split,
        "elapsed_sec": round(elapsed, 1),
        "pass1_file": str(args.pass1),
        "retried": len(memory),
        "trial1_seed": s1.get("seed"),
        "retry_seed": args.retry_seed,
        "trial1_accuracy": trial1["accuracy"],
        "trial2_accuracy": trial2["accuracy"],
        "lift": trial2["accuracy"] - trial1["accuracy"],
        # Compare this across the pass2 and control files; the difference is the
        # part attributable to memory rather than to resampling.
        "retried_solved": retried_solved,
        "retried_accuracy": retried_solved / len(results2) if results2 else 0.0,
        "errors": sum(1 for r in results2 if r.terminal_reason == "error"),
        "trial1": trial1,
        "trial2": trial2,
        "newly_solved": trial2["success"] - trial1["success"],
        # total token cost of the whole Reflexion run = trial-1 actor + reflection
        # generation + trial-2 actor.
        "tokens": {
            "trial1_actor_prompt": s1.get("actor_prompt_tokens", 0),
            "trial1_actor_completion": s1.get("actor_completion_tokens", 0),
            "reflection_prompt": s1.get("reflection_prompt_tokens", 0),
            "reflection_completion": s1.get("reflection_completion_tokens", 0),
            "trial2_actor_prompt": actor2_prompt,
            "trial2_actor_completion": actor2_completion,
        },
    }

    stem = "reflexion_final" if with_memory else "reflexion_control"
    out = args.out or f"results/json/{stem}_{args.split}_{stamp}.json"
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"summary": summary, "episodes": final}, indent=2))

    print(f"[{label}] trial-1={trial1['accuracy']:.3f} -> trial-2={trial2['accuracy']:.3f} "
          f"(lift {summary['lift']:+.3f}, {summary['newly_solved']} newly solved)", flush=True)
    print(f"[{label}] on the retried subset: {retried_solved}/{len(results2)} solved "
          f"({summary['retried_accuracy']:.3f})", flush=True)
    print(f"[{label}] by_difficulty trial-2={trial2['accuracy_by_difficulty']}", flush=True)
    if summary["errors"]:
        print(f"[{label}] WARNING: {summary['errors']} episodes crashed; "
              f"this pass is incomplete", flush=True)
    print(f"[{label}] wrote {out_path}", flush=True)
    if with_memory:
        print(f"[{label}] next, for the control arm: python scripts/run_reflexion.py "
              f"--phase control --pass1 {args.pass1}", flush=True)
    else:
        print(f"[{label}] Reflexion's effect = pass2 retried_accuracy - "
              f"{summary['retried_accuracy']:.3f} (this control)", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pass1", "pass2", "control"], required=True)
    ap.add_argument("--pass1", default=None,
                    help="pass1 JSON (required for --phase pass2 and control)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 or less = whole split")
    ap.add_argument("--max-tokens", type=int, default=1024, help="actor budget")
    ap.add_argument("--reflect-max-tokens", type=int, default=1024,
                    help="self-reflection budget (pass1 only)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="decoding seed for trial-1")
    ap.add_argument("--retry-seed", type=int, default=1,
                    help="seed for pass2 AND control. Must differ from --seed, or "
                         "the retry replays trial-1 instead of resampling it.")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()
    args.limit = args.limit if args.limit > 0 else None

    if args.phase == "pass1":
        await run_pass1(args)
    else:
        if not args.pass1:
            ap.error(f"--phase {args.phase} requires --pass1 <reflexion_pass1_*.json>")
        if args.retry_seed == args.seed:
            ap.error("--retry-seed must differ from --seed: with the same prompt and "
                     "seed vLLM reproduces trial-1's reply, so the retry measures a "
                     "replay rather than a second attempt.")
        await run_retry_pass(args, with_memory=args.phase == "pass2")


if __name__ == "__main__":
    asyncio.run(main())
