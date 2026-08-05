#!/usr/bin/env python
"""Run Reflexion over a SPaRC split for up to N trials, with a matched control arm.

This is the paper's loop (Shinn et al., 2023, Algorithm 1): attempt a task, and while
it is still unsolved and trials remain, turn the failed trajectory into a short verbal
reflection, append it to that task's episodic memory (bounded to the last `cap`), and
re-attempt the SAME task with that memory in context. Solved tasks are never
re-attempted, exactly like the reference implementation's `skip` flag.

Three phases, because a 10-trial run over 500 puzzles is ~20 h of GPU per arm and the
SLURM limit is 12 h:

  root   trial 1 only: the memoryless baseline over the split. Written once and
         shared by BOTH arms, so they start from the *identical* trial-1 episodes
         rather than from two independent samples.
  arm    trials 2..N for one arm, resumable. --arm memory is Reflexion proper;
         --arm control is the same loop with no reflection injected and no
         reflection generated -- the paper's own `use_memory=False` baseline.
  compare  no GPU: prints the two arms' trial curves side by side and the per-trial
         difference, which is the headline number.

  # 1. shared trial 1
  python scripts/run_reflexion.py --phase root --split test --limit 0
  # 2. the two arms (separate jobs; run them in either order or in parallel)
  python scripts/run_reflexion.py --phase arm --arm memory  --root results/json/reflexion_root_*.json --trials 10
  python scripts/run_reflexion.py --phase arm --arm control --root results/json/reflexion_root_*.json --trials 10
  # 3. hit the wall clock? continue the same file, no work repeated
  python scripts/run_reflexion.py --phase arm --resume results/json/reflexion_memory_*.json --trials 10
  # 4. the result
  python scripts/run_reflexion.py --phase compare --memory results/json/reflexion_memory_*.json \
                                                  --control results/json/reflexion_control_*.json

Why the control arm mirrors the trial count: only *unsolved* puzzles are re-attempted
and solved ones are carried forward, so cumulative accuracy can only rise with the
trial index -- an upward curve is guaranteed even with no memory whatsoever. At
temperature 0.6 a plain resample converts some failures on its own, and ten resamples
convert more. So "accuracy went up over 10 trials" measures nothing on its own; the
control arm, run for the same number of trials over the same puzzles at the same
per-trial seeds, is what turns it into a claim about memory.
"""

# Annotations must stay lazy: several signatures below name types from the optional
# model stack (LLMClient, ...), and an eagerly-evaluated annotation would raise
# NameError at import time on the login node, where that stack is deliberately absent.
from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import json
import math
import os
import pathlib
import sys
import time
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# Only `root` and `arm` need the model stack; `compare` is pure JSON arithmetic and is
# meant to run on the LOGIN node, which has neither numpy nor a working project venv
# (the venv is built for the compute nodes and dies with an illegal instruction here).
# Importing these eagerly made --phase compare unusable in the one place it was
# designed for, so the failure is deferred to the phases that actually need them.
try:
    from memory_agents.config import resolve_server
    from memory_agents.envs.base import EpisodeResult
    from memory_agents.harness.run import (
        run_sparc_suite_baseline,
        run_sparc_suite_memory_pass,
    )
    from memory_agents.llm.client import LLMClient
    from memory_agents.memory import Reflector, ReflexionMemory
    from memory_agents.prompts import format_reflections
    MODEL_STACK_ERROR = None
except ImportError as exc:      # noqa: BLE001 - re-raised per phase in main()
    MODEL_STACK_ERROR = exc

SCHEMA = "reflexion-multitrial-v1"

# Knobs that define what a run *is*. On --resume these come from the state file and
# the CLI cannot override them: trials 6-10 silently running at a different
# temperature (or against a different split) would corrupt the curve they extend.
FROZEN = ("split", "limit", "temperature", "top_p", "top_k", "max_tokens",
          "base_seed", "memory_cap")


# --------------------------------------------------------------------------- #
# state: one JSON file per arm, checkpointed after every trial
# --------------------------------------------------------------------------- #
def save_state(path, state) -> None:
    """Write the state file atomically.

    A checkpoint is written after each trial, i.e. after ~2 h of GPU. A partial
    write (job killed mid-dump, wall clock hit) would leave unparseable JSON and
    cost the whole run, so write to a sibling temp file and rename -- os.replace is
    atomic within a filesystem.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        blob = json.dumps(state, indent=2)
    except TypeError as exc:
        # A value the env produced is not JSON-encodable (this cost a 2 h trial once,
        # to a numpy ndarray). Losing hours of GPU to an encoder error is far worse
        # than a lossily-encoded field, so degrade instead of raising -- but say so
        # loudly, because it means something upstream needs make_json_safe.
        print(f"[save_state] WARNING: {exc}; falling back to str() for unencodable "
              f"values. The run is preserved but some field is degraded.", flush=True)
        blob = json.dumps(state, indent=2, default=str)
    tmp.write_text(blob)
    os.replace(tmp, path)


def puzzle_entry(difficulty: int) -> dict:
    return {"difficulty": difficulty, "solved": False, "solved_at_trial": None,
            "attempts": [], "last_episode": None}


def record_attempt(state: dict, trial: int, r) -> None:
    """Fold one episode into the per-puzzle history."""
    pid = str(r.puzzle_id)
    entry = state["puzzles"].setdefault(pid, puzzle_entry(r.difficulty))
    # A crashed episode reports difficulty -1; never let it overwrite a real one.
    if r.difficulty is not None and r.difficulty >= 0:
        entry["difficulty"] = r.difficulty
    entry["attempts"].append({
        "trial": trial, "success": r.success, "steps": r.steps(),
        "terminated": r.terminated, "truncated": r.truncated,
        "terminal_reason": r.terminal_reason,
        "prompt_tokens": r.prompt_tokens(),
        "completion_tokens": r.completion_tokens(),
        "retry_tokens": r.retry_tokens(),
        "error": r.error or None,
    })
    if r.success:
        entry["solved"] = True
        entry["solved_at_trial"] = trial
        entry["last_episode"] = None      # never reflected on again; drop the bulk
    else:
        # Overwritten every trial: the next reflection is written from the most
        # recent failure, as in the paper. None for crashes (see reflection_context).
        entry["last_episode"] = r.reflection_context()


def unsolved_ids(state: dict) -> list:
    """The puzzles the next trial attempts, in a stable order."""
    return sorted(pid for pid, e in state["puzzles"].items() if not e["solved"])


def cumulative_block(state: dict) -> dict:
    """Solved-by-now over the whole split -- the paper's per-trial curve.

    Cumulative, not per-trial: a puzzle solved at trial 3 stays solved at trial 4.
    """
    puzzles = state["puzzles"]
    n = len(puzzles)
    wins = sum(1 for e in puzzles.values() if e["solved"])
    by_diff: dict[int, list] = {}
    for e in puzzles.values():
        by_diff.setdefault(e["difficulty"], []).append(e["solved"])
    return {
        "n": n,
        "solved": wins,
        "accuracy": wins / n if n else 0.0,
        "accuracy_by_difficulty": {
            str(d): sum(v) / len(v) for d, v in sorted(by_diff.items())
        },
    }


def trial_seed(base_seed: int, trial: int) -> int:
    """Decoding seed for trial t. Distinct per trial, identical across arms.

    Load-bearing in both directions. Across arms: the two arms must differ only in
    the injected block, so trial t gets the same seed in each. Across trials: in the
    control arm every trial sends a byte-identical prompt, so a fixed seed would make
    vLLM replay trial 1's reply ten times and the control would measure nothing at
    all rather than the resampling effect it exists to measure.
    """
    return base_seed + trial - 1


def make_client(cfg: dict, max_tokens: int, seed: int) -> LLMClient:
    base_url, model = resolve_server()
    return LLMClient(base_url=base_url, model=model, max_tokens=max_tokens,
                     temperature=cfg["temperature"], top_p=cfg["top_p"],
                     top_k=cfg["top_k"], seed=seed)


def set_failure_log(cfg: dict, tag: str, trial: int) -> None:
    """Point the agent's format-failure log at this trial's own file."""
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    os.environ["FORMAT_FAILURE_LOG"] = str(pathlib.Path(
        f"results/json/format_failures_reflexion_{tag}_{cfg['split']}_t{trial}_{stamp}.jsonl"))


def trial_block(trial: int, seed: int, results, state: dict, elapsed: float,
                reflection: dict | None = None) -> dict:
    """One row of the trial curve: what this trial attempted, won and cost."""
    cum = cumulative_block(state)
    attempted = len(results)
    solved_now = sum(1 for r in results if r.success)
    block = {
        "trial": trial,
        "seed": seed,
        "attempted": attempted,
        "solved_this_trial": solved_now,
        # Of the puzzles this trial actually re-attempted, how many it converted.
        # Trial-1 accuracy on that set is 0 by construction from trial 2 on, so this
        # is the per-trial conversion rate the arms are compared on.
        "trial_accuracy": solved_now / attempted if attempted else 0.0,
        "cumulative_solved": cum["solved"],
        "cumulative_accuracy": cum["accuracy"],
        "accuracy_by_difficulty": cum["accuracy_by_difficulty"],
        # Crashed, not failed: a non-zero count means this trial is incomplete.
        "errors": sum(1 for r in results if r.terminal_reason == "error"),
        "elapsed_sec": round(elapsed, 1),
        "actor_prompt_tokens": sum(r.prompt_tokens() for r in results),
        "actor_completion_tokens": sum(r.completion_tokens() for r in results),
        "actor_retry_tokens": sum(r.retry_tokens() for r in results),
        # Zero for the control arm, which never calls the reflection model at all.
        "reflection_prompt_tokens": 0,
        "reflection_completion_tokens": 0,
        "reflections_written": 0,
        "reflections_empty": 0,
        "reflections_failed": 0,
        "reflections_skipped": 0,
    }
    if reflection:
        block.update(reflection)
    return block


# --------------------------------------------------------------------------- #
# phase root: trial 1, shared by both arms
# --------------------------------------------------------------------------- #
async def run_root(args) -> None:
    cfg = {
        "schema": SCHEMA, "arm": "root", "split": args.split,
        "limit": args.limit or 0, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens,
        "base_seed": args.seed, "memory_cap": args.memory_cap,
        "concurrency": args.concurrency,
    }
    seed = trial_seed(cfg["base_seed"], 1)
    set_failure_log(cfg, "root", 1)

    print(f"[root] trial 1 (memoryless baseline): split={cfg['split']} "
          f"limit={args.limit} seed={seed} concurrency={args.concurrency}", flush=True)
    client = make_client(cfg, cfg["max_tokens"], seed)
    t0 = time.time()
    results = await run_sparc_suite_baseline(
        client, split=cfg["split"], concurrency=args.concurrency, limit=args.limit)
    elapsed = time.time() - t0
    await client.aclose()

    cfg["model"] = client.model
    cfg["model_id"] = os.environ.get("LLM_MODEL_ID") or client.model
    state = {"config": cfg, "trials": [], "puzzles": {}, "memory": {}}
    for r in results:
        record_attempt(state, 1, r)
    state["trials"].append(trial_block(1, seed, results, state, elapsed))

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out = args.out or f"results/json/reflexion_root_{cfg['split']}_{stamp}.json"
    save_state(out, state)

    t1 = state["trials"][0]
    print(f"[root] trial 1: {t1['cumulative_solved']}/{len(state['puzzles'])} solved "
          f"({t1['cumulative_accuracy']:.3f})", flush=True)
    if t1["errors"]:
        print(f"[root] WARNING: {t1['errors']} episodes crashed; trial 1 is incomplete",
              flush=True)
    print(f"[root] wrote {out}\n[root] next:\n"
          f"  python scripts/run_reflexion.py --phase arm --arm memory  --root {out} --trials {args.trials}\n"
          f"  python scripts/run_reflexion.py --phase arm --arm control --root {out} --trials {args.trials}",
          flush=True)


# --------------------------------------------------------------------------- #
# phase arm: trials 2..N, resumable
# --------------------------------------------------------------------------- #
async def reflect_pass(reflector, state, memory, todo, concurrency) -> dict:
    """Write one fresh reflection per unsolved puzzle, from its latest failure.

    Runs *before* the trial that consumes it, so a resumed job can always rebuild
    the memory it needs from the state file alone. Reflection is per-puzzle: the
    lesson from puzzle X is only ever shown when re-attempting X.
    """
    sem = asyncio.Semaphore(concurrency)
    # A puzzle whose last attempt crashed has no trajectory to reflect on; it is
    # still re-attempted, just without a new reflection. Reflecting on it would ask
    # the model to diagnose a failure that never happened.
    targets = [pid for pid in todo if state["puzzles"][pid]["last_episode"]]
    skipped = len(todo) - len(targets)

    async def one(pid):
        entry = state["puzzles"][pid]
        episode = EpisodeResult.from_reflection_context(entry["last_episode"], pid)
        async with sem:
            text, resp = await reflector.reflect(episode, previous=memory.read(pid))
        return pid, text, resp

    # return_exceptions: this runs after a completed trial that cost hours. One bad
    # reflection call must not take the run down before the checkpoint is written --
    # that puzzle just carries the memory it already had into the next trial.
    outcomes = await asyncio.gather(*(one(p) for p in targets), return_exceptions=True)

    tokens = collections.Counter()
    written = empty = failed = 0
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            failed += 1
            print(f"[arm] reflection failed: {outcome!r}", flush=True)
            continue
        pid, text, resp = outcome
        tokens["prompt"] += resp.prompt_tokens
        tokens["completion"] += resp.completion_tokens
        if text:
            memory.write(pid, text)   # applies the cap (keeps the last `cap`)
            written += 1
        else:
            # e.g. a reasoning model that spent its whole budget inside <think>.
            empty += 1
    return {
        "reflections_written": written,
        "reflections_empty": empty,
        "reflections_failed": failed,
        "reflections_skipped": skipped,
        "reflection_prompt_tokens": tokens["prompt"],
        "reflection_completion_tokens": tokens["completion"],
    }


def load_arm_state(args) -> tuple[dict, str]:
    """Return (state, output_path) for a fresh arm or a resumed one."""
    if args.resume:
        state = json.loads(pathlib.Path(args.resume).read_text())
        if state["config"]["arm"] == "root":
            raise SystemExit(
                f"--resume expects an arm file; {args.resume} is a root file. "
                f"Start an arm from it with --root {args.resume} --arm <memory|control>.")
        if args.arm and args.arm != state["config"]["arm"]:
            raise SystemExit(
                f"--arm {args.arm} contradicts the resumed file's arm "
                f"({state['config']['arm']}); drop --arm to continue it.")
        return state, args.resume

    if not args.arm:
        raise SystemExit("--phase arm needs --arm memory|control")
    # Deep copy so the shared root file is never mutated by either arm.
    state = copy.deepcopy(json.loads(pathlib.Path(args.root).read_text()))
    if state["config"]["arm"] != "root":
        raise SystemExit(
            f"--root expects a root file; {args.root} is a '{state['config']['arm']}' "
            f"arm file. Continue that one with --resume instead.")
    state["config"]["arm"] = args.arm
    state["config"]["root_file"] = str(args.root)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out = args.out or (
        f"results/json/reflexion_{args.arm}_{state['config']['split']}_{stamp}.json")
    return state, out


async def run_arm(args) -> None:
    state, out = load_arm_state(args)
    cfg = state["config"]
    arm = cfg["arm"]
    with_memory = arm == "memory"

    # Frozen knobs come from the state, never the CLI: a resumed job must extend the
    # curve it is continuing, not start a differently-configured one halfway through.
    overridden = [k for k in FROZEN
                  if getattr(args, k, None) is not None and args.explicit.get(k)
                  and getattr(args, k) != cfg.get(k)]
    if overridden:
        print(f"[{arm}] ignoring CLI {', '.join(overridden)}: the run's stored values "
              f"win (" + ", ".join(f"{k}={cfg.get(k)}" for k in overridden) + ")",
              flush=True)
    cfg["concurrency"] = args.concurrency

    memory = ReflexionMemory.from_dict(state.get("memory", {}), cap=cfg["memory_cap"])
    done = len(state["trials"])
    if done >= args.trials:
        print(f"[{arm}] {out} already has {done} trials (>= --trials {args.trials}); "
              f"nothing to do", flush=True)
        return
    print(f"[{arm}] arm={arm} memory={'on' if with_memory else 'OFF (control)'} "
          f"cap={cfg['memory_cap']} trials {done + 1}..{args.trials} "
          f"temp={cfg['temperature']} base_seed={cfg['base_seed']}", flush=True)

    for trial in range(done + 1, args.trials + 1):
        todo = unsolved_ids(state)
        if not todo:
            print(f"[{arm}] every puzzle solved by trial {trial - 1}; stopping early",
                  flush=True)
            break

        seed = trial_seed(cfg["base_seed"], trial)
        reflection_stats = None
        if with_memory:
            print(f"[{arm}] trial {trial}: reflecting on {len(todo)} unsolved puzzles...",
                  flush=True)
            # Seeded per trial like the actor: a puzzle that repeats a trajectory and
            # whose reflection came back empty would otherwise re-send an identical
            # prompt at an identical seed and get the same empty reply forever.
            reflector = Reflector(make_client(cfg, args.reflect_max_tokens, seed))
            try:
                reflection_stats = await reflect_pass(
                    reflector, state, memory, todo, args.concurrency)
            finally:
                await reflector.client.aclose()
            print(f"[{arm}] trial {trial}: {reflection_stats['reflections_written']} "
                  f"reflections written "
                  f"({reflection_stats['reflections_empty']} empty, "
                  f"{reflection_stats['reflections_failed']} failed, "
                  f"{reflection_stats['reflections_skipped']} skipped)", flush=True)

        # The two arms differ here and nowhere else.
        if with_memory:
            def block_for(pid, _m=memory):
                items = _m.read(pid)
                return format_reflections(items) if items else ""
        else:
            def block_for(pid):
                return ""

        set_failure_log(cfg, arm, trial)
        print(f"[{arm}] trial {trial}: attempting {len(todo)} puzzles at seed {seed}",
              flush=True)
        client = make_client(cfg, cfg["max_tokens"], seed)
        t0 = time.time()
        results = await run_sparc_suite_memory_pass(
            client, todo, block_for, split=cfg["split"], concurrency=args.concurrency)
        elapsed = time.time() - t0
        await client.aclose()

        for r in results:
            record_attempt(state, trial, r)
        block = trial_block(trial, seed, results, state, elapsed, reflection_stats)
        state["trials"].append(block)
        state["memory"] = memory.to_dict()
        save_state(out, state)   # checkpoint: a killed job resumes from here

        print(f"[{arm}] trial {trial}: solved {block['solved_this_trial']}/"
              f"{block['attempted']} re-attempted ({block['trial_accuracy']:.3f}); "
              f"cumulative {block['cumulative_solved']}/{len(state['puzzles'])} "
              f"= {block['cumulative_accuracy']:.3f}  [{elapsed / 60:.0f} min]", flush=True)
        if block["errors"]:
            print(f"[{arm}] WARNING: {block['errors']} episodes crashed this trial",
                  flush=True)

    # Also covers the case where the loop broke immediately (everything already
    # solved): a fresh arm must still leave its file on disk, not just claim to.
    save_state(out, state)
    print_curve(state, arm)
    print(f"[{arm}] wrote {out}", flush=True)
    if len(state["trials"]) < args.trials:
        print(f"[{arm}] {len(state['trials'])}/{args.trials} trials done; continue with:\n"
              f"  python scripts/run_reflexion.py --phase arm --resume {out} "
              f"--trials {args.trials}", flush=True)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_curve(state: dict, label: str) -> None:
    n = len(state["puzzles"])
    print(f"\n[{label}] trial curve over {n} puzzles", flush=True)
    print(f"[{label}]  trial  attempted  new  cumulative   accuracy", flush=True)
    for t in state["trials"]:
        print(f"[{label}]  {t['trial']:>5}  {t['attempted']:>9}  {t['solved_this_trial']:>3}  "
              f"{t['cumulative_solved']:>10}   {t['cumulative_accuracy']:>7.3f}", flush=True)


def mcnemar(memory_state: dict, control_state: dict) -> dict:
    """Paired significance test on the final per-puzzle outcomes.

    McNemar rather than a two-sample proportion test, because the arms are *paired*:
    both attempt the same puzzles from the same trial 1, so the puzzles solved by
    neither (the large majority here) carry no information about which arm is better.
    Only the discordant pairs do -- solved by memory but not control, and vice versa.
    Exact two-sided binomial rather than the chi-square approximation, since the
    discordant count is small enough for the approximation to be unreliable.
    """
    ids = sorted(set(memory_state["puzzles"]) & set(control_state["puzzles"]))
    both = mem_only = ctl_only = neither = 0
    for pid in ids:
        m = memory_state["puzzles"][pid]["solved"]
        c = control_state["puzzles"][pid]["solved"]
        if m and c:
            both += 1
        elif m:
            mem_only += 1
        elif c:
            ctl_only += 1
        else:
            neither += 1

    n = mem_only + ctl_only
    if n:
        tail = sum(math.comb(n, k) for k in range(min(mem_only, ctl_only) + 1))
        p = min(1.0, 2 * tail / (2 ** n))
    else:
        p = 1.0
    return {"n_puzzles": len(ids), "both": both, "memory_only": mem_only,
            "control_only": ctl_only, "neither": neither, "discordant": n, "p_value": p}


def run_compare(args) -> None:
    """The headline table: both arms' curves and the memory-minus-control delta."""
    mem = json.loads(pathlib.Path(args.memory).read_text())
    ctl = json.loads(pathlib.Path(args.control).read_text())
    for name, st, want in (("--memory", mem, "memory"), ("--control", ctl, "control")):
        if st["config"]["arm"] != want:
            raise SystemExit(f"{name} points at a '{st['config']['arm']}' file, not {want}")

    mc, cc = mem["config"], ctl["config"]
    for key in ("split", "base_seed", "temperature", "max_tokens", "limit"):
        if mc.get(key) != cc.get(key):
            print(f"WARNING: arms disagree on {key}: {mc.get(key)} vs {cc.get(key)} "
                  f"-- they are not comparable", flush=True)
    if mc.get("root_file") != cc.get("root_file"):
        print(f"WARNING: arms started from different trial-1 files "
              f"({mc.get('root_file')} vs {cc.get('root_file')})", flush=True)

    n = len(mem["puzzles"])
    rows = []
    lines = [
        "=" * 74,
        f"  REFLEXION vs CONTROL over {n} puzzles ({mc.get('split')}, "
        f"{mc.get('model_id', mc.get('model'))})",
        "=" * 74,
        "",
        "  cumulative accuracy (solved at least once by trial t)",
        "",
        f"  {'trial':>5}  {'seed':>5} | {'memory':>16} | {'control':>16} | {'delta':>8}",
        f"  {'':>5}  {'':>5} | {'solved':>7} {'acc':>8} | {'solved':>7} {'acc':>8} | {'':>8}",
        "  " + "-" * 70,
    ]
    for m, c in zip(mem["trials"], ctl["trials"]):
        delta = m["cumulative_accuracy"] - c["cumulative_accuracy"]
        rows.append((m["trial"], delta))
        lines.append(
            f"  {m['trial']:>5}  {m['seed']:>5} | {m['cumulative_solved']:>7} "
            f"{m['cumulative_accuracy']:>8.3f} | {c['cumulative_solved']:>7} "
            f"{c['cumulative_accuracy']:>8.3f} | {delta:>+8.3f}")

    if len(mem["trials"]) != len(ctl["trials"]):
        lines += ["", f"  NOTE: unequal trial counts (memory {len(mem['trials'])}, "
                      f"control {len(ctl['trials'])}); compared over the common prefix."]

    m_last, c_last = mem["trials"][len(rows) - 1], ctl["trials"][len(rows) - 1]
    lines += [
        "",
        f"  After {rows[-1][0]} trials: memory {m_last['cumulative_accuracy']:.3f} vs "
        f"control {c_last['cumulative_accuracy']:.3f}  "
        f"(delta {rows[-1][1]:+.3f}, "
        f"{m_last['cumulative_solved'] - c_last['cumulative_solved']:+d} puzzles)",
        "",
        "  Reflexion's effect is the delta column: both arms retry the same unsolved",
        "  puzzles at the same per-trial seeds and carry solved ones forward, so each",
        "  arm's own curve rises with the trial index whether or not memory helps.",
        "",
    ]

    mc = mcnemar(mem, ctl)
    verdict = ("SIGNIFICANT at p<0.05" if mc["p_value"] < 0.05
               else "NOT significant at p<0.05")
    lines += [
        "  paired significance (McNemar, exact two-sided)",
        "",
        f"  {'solved by both':<22} {mc['both']:>5}",
        f"  {'memory only':<22} {mc['memory_only']:>5}   <- discordant",
        f"  {'control only':<22} {mc['control_only']:>5}   <- discordant",
        f"  {'solved by neither':<22} {mc['neither']:>5}",
        "",
        f"  discordant pairs: {mc['discordant']}     p = {mc['p_value']:.4f}     {verdict}",
        "",
        "  Only the discordant pairs carry information: the puzzles both arms solved,",
        "  and the far larger set neither solved, say nothing about which arm is better.",
        "",
    ]

    tok = lambda st, k: sum(t.get(k, 0) for t in st["trials"])
    lines += ["  token cost", ""]
    for name, st in (("memory", mem), ("control", ctl)):
        actor = tok(st, "actor_prompt_tokens") + tok(st, "actor_completion_tokens")
        refl = tok(st, "reflection_prompt_tokens") + tok(st, "reflection_completion_tokens")
        lines.append(f"  {name:<8} actor {actor:>12,}   reflection {refl:>12,}   "
                     f"total {actor + refl:>12,}")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out != "-":
        # Name the report after the arm it came from. A fixed default would have each
        # model's comparison silently overwrite the last one's, and the three runs
        # differ only by model -- the fastest way to lose a finished 20 h result.
        default = pathlib.Path("results/txt") / (
            pathlib.Path(args.memory).stem.replace("memory", "vs_control") + ".txt")
        dest = pathlib.Path(args.out) if args.out else default
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(report)
        print(f"[written] {dest}", file=sys.stderr)


# --------------------------------------------------------------------------- #
async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["root", "arm", "compare"], required=True)
    ap.add_argument("--arm", choices=["memory", "control"],
                    help="which arm to run (--phase arm)")
    ap.add_argument("--root", default=None, help="root JSON to start an arm from")
    ap.add_argument("--resume", default=None, help="arm JSON to continue in place")
    ap.add_argument("--trials", type=int, default=10,
                    help="total trials INCLUDING trial 1 (the paper runs ~10)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 or less = whole split")
    ap.add_argument("--max-tokens", type=int, default=12288, help="actor budget")
    ap.add_argument("--reflect-max-tokens", type=int, default=12288,
                    help="self-reflection budget (memory arm only)")
    # Qwen3 in thinking mode loops at temperature 0, and the control arm is only
    # meaningful under sampling, so the thinking-mode defaults are the defaults here.
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, dest="base_seed", default=0,
                    help="seed for trial 1; trial t uses seed + t - 1")
    ap.add_argument("--memory-cap", type=int, default=3,
                    help="reflections kept per puzzle (the paper's Omega, 1-3)")
    ap.add_argument("--memory", default=None, help="memory-arm JSON (--phase compare)")
    ap.add_argument("--control", default=None, help="control-arm JSON (--phase compare)")
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args()

    # Which knobs the user actually typed, so a resume can tell "left at its default"
    # from "explicitly asked for" when it reports what it is ignoring.
    typed = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    args.explicit = {k: k in typed or (k == "base_seed" and "seed" in typed)
                     for k in FROZEN}
    args.seed = args.base_seed
    args.limit = args.limit if args.limit > 0 else None

    if args.phase in ("root", "arm") and MODEL_STACK_ERROR is not None:
        ap.error(f"--phase {args.phase} needs the model stack, which failed to import "
                 f"({MODEL_STACK_ERROR}). Run it from a compute node with the project "
                 f"venv; only --phase compare works on the login node.")

    if args.phase == "root":
        await run_root(args)
    elif args.phase == "arm":
        if not (args.root or args.resume):
            ap.error("--phase arm needs --root <root.json> or --resume <arm.json>")
        if args.root and args.resume:
            ap.error("pass either --root (start) or --resume (continue), not both")
        if args.trials < 2:
            ap.error("--trials must be >= 2 for an arm (trial 1 is the root phase)")
        await run_arm(args)
    else:
        if not (args.memory and args.control):
            ap.error("--phase compare needs --memory <file> --control <file>")
        run_compare(args)


if __name__ == "__main__":
    asyncio.run(main())
