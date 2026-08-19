#!/usr/bin/env python
"""Memp (Fang et al., 2026) over SPaRC: build procedural memory on the train
split, freeze it, retrieve it for the test split.

The paper's loop is Build -> Retrieve -> Update. Here the bank is built offline
from the train split and FROZEN at test time (docs/architecture.md section 5):
each test puzzle is attempted exactly once, so there is no lift>=0 artifact and
no control arm -- the comparison is two-sided against the memoryless baseline at
the same seed, with a random-retrieval arm to separate "retrieval quality" from
"any memory at all".

Five phases, because build and eval are ~8 h of GPU each and the embedding model
is a different server than the chat model (vLLM serves one model per process):

  build    attempt all 500 TRAIN puzzles memoryless at baseline params, keeping
           each episode's compact trajectory. Checkpointed every --chunk puzzles.
  distill  turn the build pass into the bank: a solved puzzle's own trajectory
           -> procedure ("agent" provenance); a failed/crashed puzzle's gold
           solution path -> procedure in hindsight ("gold" provenance).
  embed    (embedding model job) embed puzzle_key_text for ALL train+test
           puzzles -> one .npz of vectors + a .json meta. Depends only on the
           dataset, so it can run in parallel with build/distill.
  eval     (chat model job) cosine top-k (or seeded-random k) entries per test
           puzzle -> injected procedure block -> one pass over the test split.
  compare  no GPU: baseline vs retrieved vs random, with McNemar exact tests.

  sbatch -J memp-build --export=ALL,DRIVER=memp,PHASE=build,SPLIT=train,LIMIT=0 scripts/serve_and_run.sbatch
  sbatch -J memp-dist  --export=ALL,DRIVER=memp,PHASE=distill,BUILD=results/json/memp_build_train_<stamp>.json scripts/serve_and_run.sbatch
  sbatch -J memp-embed -G A100:1 -t 01:00:00 --export=ALL,DRIVER=memp,PHASE=embed,TP=1 scripts/serve_and_run.sbatch
  sbatch -J memp-ret   --export=ALL,DRIVER=memp,PHASE=eval,ARM=retrieved,LIMIT=0,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
  sbatch -J memp-rnd   --export=ALL,DRIVER=memp,PHASE=eval,ARM=random,LIMIT=0,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
  python scripts/run_memp.py --phase compare --baseline <baseline.json> --retrieved <ret.json> --random <rnd.json>
"""

import argparse
import asyncio
import collections
import json
import math
import os
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from memory_agents.config import resolve_server
from memory_agents.envs.base import EpisodeResult, make_json_safe
from memory_agents.envs.sparc import grid_summary, path_to_actions, puzzle_key_text
from memory_agents.harness.run import run_sparc_suite_memory_pass
from memory_agents.llm.client import EmbeddingClient, LLMClient
from memory_agents.memory import Distiller, MempMemory
from memory_agents.prompts import format_procedures

BUILD_SCHEMA = "memp-build-v1"
EVAL_SCHEMA = "memp-eval-v1"
KEY_FN = "puzzle-key-v1"   # must match puzzle_key_text's docstring version

# Knobs that define what a run *is*, per phase. On --resume these come from the
# state file and the CLI cannot override them (same discipline as run_reflexion:
# a resumed 8 h pass silently continuing at a different temperature would corrupt
# the very numbers it exists to produce).
FROZEN = {
    "build": ("split", "limit", "temperature", "top_p", "top_k", "max_tokens",
              "seed"),
    "distill": ("build", "distill_max_tokens", "temperature", "top_p", "top_k",
                "seed"),
    "eval": ("bank", "embeddings", "arm", "top_k_memories", "split", "limit",
             "temperature", "top_p", "top_k", "max_tokens", "seed",
             "granularity", "provenance", "update", "groups"),
}


# --------------------------------------------------------------------------- #
# shared plumbing (the run_reflexion patterns, reused verbatim where possible)
# --------------------------------------------------------------------------- #
def save_state(path, state) -> None:
    """Atomic checkpoint: write a sibling temp file, then os.replace."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def make_client(cfg: dict, max_tokens: int, seed: int) -> LLMClient:
    base_url, model = resolve_server()
    return LLMClient(base_url=base_url, model=model, max_tokens=max_tokens,
                     temperature=cfg["temperature"], top_p=cfg["top_p"],
                     top_k=cfg["top_k"], seed=seed)


def set_failure_log(tag: str) -> None:
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    os.environ["FORMAT_FAILURE_LOG"] = str(pathlib.Path(
        f"results/json/format_failures_memp_{tag}_{stamp}.jsonl"))


def report_frozen(args, cfg: dict, phase: str) -> None:
    """On resume, say which typed CLI flags are being ignored (stored values win)."""
    overridden = [k for k in FROZEN[phase]
                  if args.explicit.get(k) and getattr(args, k, None) is not None
                  and getattr(args, k) != cfg.get(k)]
    if overridden:
        print(f"[{phase}] ignoring CLI {', '.join(overridden)}: the run's stored "
              f"values win (" + ", ".join(f"{k}={cfg.get(k)}" for k in overridden)
              + ")", flush=True)


def split_ids(split: str, limit) -> list[str]:
    """The puzzle ids a pass runs over, in dataset order."""
    df = load_dataset("lkaesberg/SPaRC", "all", split=split).to_pandas()
    ids = [str(pid) for pid in df["id"]]
    return ids[:limit] if limit else ids


def episode_record(r) -> dict:
    """One puzzle's outcome, in run_baseline's record shape plus the compact
    trajectory. Unlike Reflexion -- which drops the trajectory once a puzzle is
    solved -- the build pass keeps it FOR successes: the successful trajectory is
    exactly what the distill phase turns into a procedure."""
    return make_json_safe({
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
        "episode_context": r.reflection_context(),
    })


def accuracy_block(records: list[dict]) -> dict:
    """run_baseline.summarize() over already-serialised records."""
    n = len(records)
    wins = sum(1 for rec in records if rec["success"])
    by_diff: dict[int, list] = {}
    for rec in records:
        by_diff.setdefault(rec["difficulty"], []).append(rec["success"])
    return {
        "n": n,
        "success": wins,
        "accuracy": wins / n if n else 0.0,
        "accuracy_by_difficulty": {
            str(d): sum(s) / len(s) for d, s in sorted(by_diff.items())
        },
        "prompt_tokens": sum(rec["prompt_tokens"] for rec in records),
        "completion_tokens": sum(rec["completion_tokens"] for rec in records),
        "retry_tokens": sum(rec["retry_tokens"] for rec in records),
        "errors": sum(1 for rec in records if rec["terminal_reason"] == "error"),
        "total_steps": sum(rec["steps"] for rec in records),
    }


# --------------------------------------------------------------------------- #
# phase build: one memoryless pass over the train split, chunk-checkpointed
# --------------------------------------------------------------------------- #
async def run_build(args) -> None:
    if args.resume:
        state = json.loads(pathlib.Path(args.resume).read_text())
        if state["config"].get("schema") != BUILD_SCHEMA:
            raise SystemExit(f"--resume expects a {BUILD_SCHEMA} file; "
                             f"{args.resume} is {state['config'].get('schema')!r}")
        cfg, out = state["config"], args.resume
        report_frozen(args, cfg, "build")
    else:
        # Memories must come from puzzles the eval will never see; train and test
        # are disjoint by construction, so anything else is a leakage bug.
        if args.split != "train":
            raise SystemExit(f"the build pass runs on the train split (got "
                             f"--split {args.split}); submit with SPLIT=train")
        cfg = {
            "schema": BUILD_SCHEMA, "phase": "build", "split": args.split,
            "limit": args.limit or 0, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k,
            "max_tokens": args.max_tokens, "seed": args.seed,
        }
        state = {"config": cfg, "puzzles": {}}
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        out = args.out or f"results/json/memp_build_train_{stamp}.json"
    cfg["concurrency"] = args.concurrency

    todo = [pid for pid in split_ids(cfg["split"], cfg["limit"])
            if pid not in state["puzzles"]]
    total = len(state["puzzles"]) + len(todo)
    if not todo:
        print(f"[build] {out} already covers every puzzle; nothing to do", flush=True)
        return
    print(f"[build] attempting {len(todo)} train puzzles memoryless "
          f"(chunk={args.chunk}, seed={cfg['seed']}, temp={cfg['temperature']})",
          flush=True)

    set_failure_log("build")
    client = make_client(cfg, cfg["max_tokens"], cfg["seed"])
    cfg["model"] = client.model
    cfg["model_id"] = os.environ.get("LLM_MODEL_ID") or client.model
    try:
        # Chunked, unlike Reflexion's per-trial checkpoint: this is ONE ~8 h pass,
        # so waiting for the end to write state would make a wall-clock kill cost
        # the whole job. Re-creating the env pool per chunk costs ~2 min in total.
        for lo in range(0, len(todo), args.chunk):
            chunk = todo[lo:lo + args.chunk]
            t0 = time.time()
            results = await run_sparc_suite_memory_pass(
                client, chunk, lambda pid: "",   # empty block == baseline actor
                split=cfg["split"], concurrency=args.concurrency)
            for r in results:
                state["puzzles"][str(r.puzzle_id)] = episode_record(r)
            save_state(out, state)
            solved = sum(1 for p in state["puzzles"].values() if p["success"])
            print(f"[build] {len(state['puzzles'])}/{total} done, "
                  f"{solved} solved so far  [{(time.time() - t0) / 60:.0f} min/chunk]",
                  flush=True)
    finally:
        await client.aclose()

    block = accuracy_block(list(state["puzzles"].values()))
    print(f"[build] train pass: {block['success']}/{block['n']} solved "
          f"({block['accuracy']:.3f}) by_difficulty={block['accuracy_by_difficulty']}",
          flush=True)
    if block["errors"]:
        print(f"[build] WARNING: {block['errors']} episodes crashed; they will be "
              f"distilled from the gold path like other failures", flush=True)
    print(f"[build] wrote {out}\n[build] next:\n"
          f"  sbatch -J memp-dist --export=ALL,DRIVER=memp,PHASE=distill,"
          f"BUILD={out} scripts/serve_and_run.sbatch", flush=True)


# --------------------------------------------------------------------------- #
# phase distill: build pass -> procedure bank
# --------------------------------------------------------------------------- #
def gold_exemplar(row) -> dict:
    """The concrete solved example attached to a gold-provenance entry.

    Assembled in code, never by the LLM: the action sequence comes straight from
    the dataset's gold path, so it cannot be hallucinated."""
    puzzle = yaml.safe_load(row["text_visualization"])["puzzle"]
    actions = path_to_actions(list(row["solutions"][0]["path"]))
    return {
        "actions": actions, "n_moves": len(actions),
        "width": puzzle["dimensions"]["width"],
        "height": puzzle["dimensions"]["height"],
        "start": [puzzle["start"]["x"], puzzle["start"]["y"]],
        "end": [puzzle["end"]["x"], puzzle["end"]["y"]],
        "grid_summary": grid_summary(row),
    }


def agent_exemplar(row, episode_ctx: dict) -> dict:
    """Same shape, but the actions are the agent's own winning moves. Success in
    SPaRC is an exact match to an enumerated solution, so this sequence is
    guaranteed to be a valid solution path too."""
    ex = gold_exemplar(row)
    actions = [int(action) for _step, action, _loc in episode_ctx["moves"]]
    ex["actions"], ex["n_moves"] = actions, len(actions)
    return ex


async def run_distill(args) -> None:
    if args.resume:
        bank_data = json.loads(pathlib.Path(args.resume).read_text())
        memory = MempMemory.from_dict(bank_data)
        cfg, out = bank_data["config"], args.resume
        report_frozen(args, cfg, "distill")
    else:
        if not args.build:
            raise SystemExit("--phase distill needs --build <memp_build_*.json> "
                             "or --resume <memp_bank_*.json>")
        memory = MempMemory()
        cfg = {
            "schema": BUILD_SCHEMA.replace("build", "bank"),  # informational only
            "phase": "distill", "build": str(args.build),
            "distill_max_tokens": args.distill_max_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "seed": args.seed,
        }
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        out = args.out or f"results/json/memp_bank_{stamp}.json"
    cfg["concurrency"] = args.concurrency

    build = json.loads(pathlib.Path(cfg["build"]).read_text())
    if build["config"].get("schema") != BUILD_SCHEMA:
        raise SystemExit(f"--build expects a {BUILD_SCHEMA} file; "
                         f"{cfg['build']} is {build['config'].get('schema')!r}")

    df = load_dataset("lkaesberg/SPaRC", "all", split="train").to_pandas()
    row_of = {str(df["id"][i]): i for i in range(len(df))}

    if args.verify_gold:
        verify_gold_paths(df, n=args.verify_gold)

    todo = sorted(pid for pid in build["puzzles"] if pid not in memory)
    if not todo:
        print(f"[distill] {out} already covers the whole build file", flush=True)
        return
    stats = memory.stats
    for key in ("n_agent", "n_gold", "empty", "failed",
                "distill_prompt_tokens", "distill_completion_tokens"):
        stats.setdefault(key, 0)
    print(f"[distill] distilling {len(todo)} puzzles "
          f"({sum(1 for p in todo if build['puzzles'][p]['success'])} from agent "
          f"trajectories, the rest from gold paths)", flush=True)

    async def distill_one(distiller, sem, pid):
        rec = build["puzzles"][pid]
        row = df.iloc[row_of[pid]]
        key_text = puzzle_key_text(row)
        # Route on provenance. A crashed episode has no episode_context, which is
        # exactly why the gold prompt reads dataset fields only.
        from_agent = bool(rec["success"] and rec.get("episode_context"))
        async with sem:
            if from_agent:
                episode = EpisodeResult.from_reflection_context(
                    rec["episode_context"], pid)
                text, resp = await distiller.distill_from_trajectory(episode, key_text)
            else:
                gold_actions = path_to_actions(list(row["solutions"][0]["path"]))
                text, resp = await distiller.distill_from_gold(
                    key_text, row["text_visualization"], gold_actions)
        return pid, from_agent, text, resp

    async def distill_pass(pids, seed):
        """One pass over `pids`; returns the pids whose reply came back empty."""
        distiller = Distiller(make_client(cfg, cfg["distill_max_tokens"], seed))
        cfg.setdefault("model", distiller.client.model)
        cfg.setdefault("model_id",
                       os.environ.get("LLM_MODEL_ID") or distiller.client.model)
        empties = []
        try:
            sem = asyncio.Semaphore(args.concurrency)
            for lo in range(0, len(pids), args.chunk):
                chunk = pids[lo:lo + args.chunk]
                outcomes = await asyncio.gather(
                    *(distill_one(distiller, sem, p) for p in chunk),
                    return_exceptions=True)
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        stats["failed"] += 1
                        print(f"[distill] call failed: {outcome!r}", flush=True)
                        continue
                    pid, from_agent, text, resp = outcome
                    stats["distill_prompt_tokens"] += resp.prompt_tokens
                    stats["distill_completion_tokens"] += resp.completion_tokens
                    if not text:
                        empties.append(pid)
                        continue
                    rec = build["puzzles"][pid]
                    row = df.iloc[row_of[pid]]
                    memory.write(pid, {
                        "provenance": "agent" if from_agent else "gold",
                        # From the dataset, not the build record: a crashed
                        # episode reports difficulty -1.
                        "difficulty": int(row["difficulty_level"]),
                        "key_text": puzzle_key_text(row),
                        "script": text,
                        "exemplar": (agent_exemplar(row, rec["episode_context"])
                                     if from_agent else gold_exemplar(row)),
                        "distill_prompt_tokens": resp.prompt_tokens,
                        "distill_completion_tokens": resp.completion_tokens,
                    })
                    stats["n_agent" if from_agent else "n_gold"] += 1
                save_state(out, {**memory.to_dict(), "config": cfg})
                print(f"[distill] {len(memory)} entries banked "
                      f"({stats['n_agent']} agent, {stats['n_gold']} gold)", flush=True)
        finally:
            await distiller.client.aclose()
        return empties

    empties = await distill_pass(todo, cfg["seed"])
    if empties:
        # Same prompt at the same seed would replay the same empty reply (usually a
        # thinking budget exhausted inside <think>), so the one retry reseeds.
        print(f"[distill] retrying {len(empties)} empty replies at seed+1", flush=True)
        empties = await distill_pass(empties, cfg["seed"] + 1)
    stats["empty"] = len(empties)
    save_state(out, {**memory.to_dict(), "config": cfg})

    print(f"[distill] bank: {len(memory)} entries "
          f"({stats['n_agent']} agent, {stats['n_gold']} gold, "
          f"{stats['empty']} empty after retry, {stats['failed']} failed); "
          f"distill tokens prompt={stats['distill_prompt_tokens']:,} "
          f"completion={stats['distill_completion_tokens']:,}", flush=True)
    print(f"[distill] wrote {out}", flush=True)


def verify_gold_paths(df, n: int) -> None:
    """Replay the first n gold action sequences through a real env and assert
    success -- proves the (dx,dy)->action table against the live env once, with
    no LLM in the loop. Cheap enough to run on every distill job."""
    from memory_agents.envs.sparc import SparcEnv, is_success

    env = SparcEnv(split="train")
    try:
        for i in range(min(n, len(df))):
            pid = str(df["id"][i])
            _obs, info = env.reset(puzzle_id=df["id"][i])
            actions = path_to_actions(list(df.iloc[i]["solutions"][0]["path"]))
            for action in actions:
                _obs, _reward, terminated, truncated, info = env.step(action)
            if not is_success(info):
                raise SystemExit(f"[verify-gold] replaying the gold path of {pid} "
                                 f"did NOT solve it -- path_to_actions is wrong "
                                 f"(terminated={terminated} truncated={truncated})")
        print(f"[verify-gold] {min(n, len(df))} gold paths replayed successfully",
              flush=True)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# phase upgrade: add round-by-round trajectories to an existing bank (no GPU)
# --------------------------------------------------------------------------- #
def run_upgrade(args) -> None:
    """Attach the paper's `Trajectory` granularity to every entry of a bank.

    Replays each entry's own exemplar action sequence through a real env and
    stores the resulting round-by-round trace. Uniform across provenance: an
    agent entry replays the moves the model actually made, a gold entry replays
    the dataset's solution path. Doing it for the whole bank (rather than only
    the handful of agent successes) keeps granularity from being confounded with
    bank size when the arms are compared.

    Pure env replay -- no model, no GPU. Runs on the login node in a minute.
    """
    from memory_agents.envs.sparc import SparcEnv, replay_path

    src = pathlib.Path(args.bank)
    data = json.loads(src.read_text())
    memory = MempMemory.from_dict(data)
    out = pathlib.Path(args.out or src.with_name(src.stem + "_traj.json"))

    env = SparcEnv(split="train")
    upgraded = failed = solved = 0
    try:
        for e in memory.entries:
            actions = (e.get("exemplar") or {}).get("actions") or []
            if not actions:
                failed += 1
                continue
            try:
                traj = replay_path(env, e["source_pid"], actions)
            except Exception as exc:            # a bad exemplar must not kill the run
                print(f"[upgrade] {e['id']} ({e['source_pid']}): {exc!r}", flush=True)
                failed += 1
                continue
            e["trajectory"] = traj
            upgraded += 1
            solved += traj["terminal_reason"] == "solved"
    finally:
        env.close()

    save_state(out, {**memory.to_dict(), "config": data.get("config", {})})
    print(f"[upgrade] {upgraded}/{len(memory)} entries now carry a trajectory "
          f"({solved} of those replays reached a solved state, {failed} failed)",
          flush=True)
    # A replay that does not end solved means the stored exemplar is not a winning
    # line -- worth knowing, since that entry now shows the agent a losing trace.
    if upgraded and solved < upgraded:
        print(f"[upgrade] NOTE: {upgraded - solved} replays did not end solved; "
              f"those entries' exemplars were not winning paths", flush=True)
    print(f"[upgrade] wrote {out}", flush=True)


# --------------------------------------------------------------------------- #
# phase keywords: AveFact key construction (LLM), then embed-keywords (embeddings)
# --------------------------------------------------------------------------- #
async def run_keywords(args) -> None:
    """Extract per-puzzle keywords for the AveFact retriever (one LLM call each)."""
    cfg = {"temperature": args.temperature, "top_p": args.top_p,
           "top_k": args.top_k}
    distiller = Distiller(make_client(cfg, args.distill_max_tokens, args.seed))
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out = pathlib.Path(args.out or f"results/json/memp_keywords_{stamp}.json")
    result = {"model_id": os.environ.get("LLM_MODEL_ID") or distiller.client.model,
              "key_fn": KEY_FN, "keywords": {}}
    if out.exists():                      # resume: keyword extraction is 1000 calls
        result = json.loads(out.read_text())
    sem = asyncio.Semaphore(args.concurrency)

    async def one(pid, key_text):
        async with sem:
            return pid, *(await distiller.keywords(key_text))

    try:
        for split in ("train", "test"):
            df = load_dataset("lkaesberg/SPaRC", "all", split=split).to_pandas()
            n = min(args.limit, len(df)) if args.limit else len(df)
            todo = [(str(df["id"][i]), puzzle_key_text(df.iloc[i]))
                    for i in range(n) if str(df["id"][i]) not in result["keywords"]]
            for lo in range(0, len(todo), args.chunk):
                outcomes = await asyncio.gather(
                    *(one(p, t) for p, t in todo[lo:lo + args.chunk]),
                    return_exceptions=True)
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        print(f"[keywords] failed: {outcome!r}", flush=True)
                        continue
                    pid, kws, _resp = outcome
                    if kws:
                        result["keywords"][pid] = kws
                save_state(out, result)
                print(f"[keywords] {len(result['keywords'])} puzzles done",
                      flush=True)
    finally:
        await distiller.client.aclose()
    empty = sum(1 for v in result["keywords"].values() if not v)
    print(f"[keywords] wrote {out} ({len(result['keywords'])} puzzles, "
          f"{empty} empty)", flush=True)


async def run_embed_keywords(args) -> None:
    """Embed the extracted keywords -- the vectors AveFact scores against."""
    base_url, model = resolve_server()
    client = EmbeddingClient(base_url=base_url, model=model)
    data = json.loads(pathlib.Path(args.keywords).read_text())
    kws = data["keywords"]
    pids = sorted(kws)
    flat = [(pid, kw) for pid in pids for kw in kws[pid]]
    print(f"[embed-kw] {len(flat)} keywords over {len(pids)} puzzles", flush=True)

    vecs: list[list[float]] = []
    try:
        for lo in range(0, len(flat), args.batch):
            vecs.extend(await client.embed([kw for _p, kw in flat[lo:lo + args.batch]]))
    finally:
        await client.aclose()
    mat = np.asarray(vecs, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)

    out = pathlib.Path(args.out or
                       pathlib.Path(args.keywords).with_suffix(".npz"))
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, kw_pids=np.asarray([p for p, _k in flat]), kw_vecs=mat)
    os.replace(tmp, out)
    print(f"[embed-kw] wrote {out}  ({mat.shape[0]} x {mat.shape[1]})", flush=True)


def load_keywords(path):
    """(keyword vectors per puzzle id) -- used for both queries and bank entries."""
    npz = np.load(str(pathlib.Path(path).with_suffix(".npz")))
    by_pid: dict[str, list] = {}
    for pid, vec in zip(npz["kw_pids"], npz["kw_vecs"]):
        by_pid.setdefault(str(pid), []).append(vec)
    grouped = {pid: np.asarray(v, dtype=np.float32) for pid, v in by_pid.items()}
    return grouped, grouped


# --------------------------------------------------------------------------- #
# phase embed: puzzle_key_text for both splits -> vectors on disk
# --------------------------------------------------------------------------- #
async def run_embed(args) -> None:
    base_url, model = resolve_server()
    client = EmbeddingClient(base_url=base_url, model=model)
    embed_model = os.environ.get("LLM_MODEL_ID") or model

    # The embed phase's own smoke: the sbatch chat smoke cannot run against a
    # pooling server, so prove the endpoint works before spending any real calls.
    probe = await client.embed(["smoke"])
    dim = len(probe[0])
    norm = float(np.linalg.norm(probe[0]))
    if dim <= 0 or norm <= 0:
        raise SystemExit(f"embedding smoke failed: dim={dim} norm={norm}")
    print(f"[embed] {embed_model}: dim={dim}", flush=True)

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out = pathlib.Path(args.out or f"results/json/memp_embeddings_{stamp}.npz")
    arrays: dict[str, np.ndarray] = {}
    try:
        for split in ("train", "test"):
            df = load_dataset("lkaesberg/SPaRC", "all", split=split).to_pandas()
            n = min(args.limit, len(df)) if args.limit else len(df)
            texts = [puzzle_key_text(df.iloc[i]) for i in range(n)]
            vecs: list[list[float]] = []
            for lo in range(0, n, args.batch):
                vecs.extend(await client.embed(texts[lo:lo + args.batch]))
            mat = np.asarray(vecs, dtype=np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True)
            arrays[f"{split}_ids"] = np.asarray(
                [str(pid) for pid in df["id"][:n]])
            arrays[f"{split}_vecs"] = mat
            print(f"[embed] {split}: {mat.shape[0]} x {mat.shape[1]}", flush=True)
    finally:
        await client.aclose()

    # Atomic like save_state: a torn npz would fail eval's np.load with a message
    # that says nothing about why.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    meta = {"embed_model": embed_model, "dim": dim, "normalized": True,
            "key_fn": KEY_FN,
            "n_train": int(arrays["train_ids"].shape[0]),
            "n_test": int(arrays["test_ids"].shape[0])}
    # .meta.json, not .json: results/json/*.json is what summarize_results globs,
    # and this sidecar is a contract for the eval phase, not a run.
    meta_path = out.with_name(out.stem + ".meta.json")
    save_state(meta_path, meta)

    # Sanity: a key must be its own nearest neighbour (cosine 1.0 with itself is
    # trivial; this checks the matrix rows line up with the ids).
    tv = arrays["train_vecs"]
    self_top1 = (tv[:5] @ tv.T).argmax(axis=1)
    if not (self_top1 == np.arange(len(self_top1))).all():
        raise SystemExit("[embed] sanity failed: a train key is not its own "
                         "top-1 neighbour; id/vector rows are misaligned")
    print(f"[embed] wrote {out} + {meta_path}", flush=True)


# --------------------------------------------------------------------------- #
# phase eval: one retrieval-augmented pass over the test split
# --------------------------------------------------------------------------- #
def load_embeddings(path) -> tuple[dict, dict]:
    npz = np.load(path)
    path = pathlib.Path(path)
    meta_path = path.with_name(path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("key_fn") not in (None, KEY_FN):
        raise SystemExit(f"{path} was embedded with key_fn={meta.get('key_fn')!r} "
                         f"but this code renders {KEY_FN}; re-run the embed phase")
    return {k: npz[k] for k in npz.files}, meta


def build_blocks(memory, arrays, cfg, test_ids, keyword_queries=None):
    """Precompute the injected block + retrieval log for a set of test puzzles.

    With a frozen bank, retrieval is a pure function of (bank, embeddings, arm, k,
    seed), so precomputing makes block_for a dict lookup (trivially safe under
    concurrency) and means a resumed job injects byte-identical blocks. The Update
    policies mutate the bank between groups, so they call this again per group --
    hence `test_ids` is a parameter rather than the whole split.
    """
    arm, k, seed = cfg["arm"], cfg["top_k_memories"], cfg["seed"]
    granularity = cfg.get("granularity", "script")
    qrow = {str(pid): i for i, pid in enumerate(arrays["test_ids"])}
    blocks, retrieved_log = {}, {}
    for pid in test_ids:
        if arm == "retrieved":
            if pid not in qrow:
                raise SystemExit(f"test puzzle {pid} has no query embedding; "
                                 f"re-run the embed phase without a limit")
            picks = memory.retrieve(arrays["test_vecs"][qrow[pid]], k)
        elif arm == "avefact":
            kws = (keyword_queries or {}).get(pid)
            # `not kws` would raise: these are numpy arrays, not lists.
            if kws is None or len(kws) == 0:
                raise SystemExit(f"test puzzle {pid} has no keyword vectors; "
                                 f"re-run --phase keywords")
            picks = memory.avefact_retrieve(kws, k)
        elif arm == "fixed":
            picks = memory.fixed_retrieve(k)
        else:
            picks = memory.random_retrieve(k, seed_key=f"{seed}:{pid}")
        blocks[pid] = format_procedures([entry for entry, _sim in picks],
                                        granularity=granularity)
        retrieved_log[pid] = [
            {"id": entry["id"], "source_pid": entry["source_pid"],
             "provenance": entry["provenance"],
             "similarity": None if sim is None else round(float(sim), 4)}
            for entry, sim in picks]
    return blocks, retrieved_log


# --------------------------------------------------------------------------- #
# the Update stage (the paper's section 4.3): the bank evolves across the test
# split, refreshed after every group of tasks.
# --------------------------------------------------------------------------- #
async def update_pass(policy, memory, arrays, group_results, retrieved_log,
                      df_test, row_of, cfg, args, seed) -> dict:
    """Fold one finished group of test episodes back into the bank.

      vanilla     every trajectory in the group becomes a new memory
      validation  only the successful ones do; failures are discarded
      adjustment  each failure revises the memory that was retrieved for it

    Returns counters for the run record. New entries carry the *test* puzzle's own
    embedding (already in the sidecar), so retrieval for later groups sees them.
    """
    stats = collections.Counter()
    qrow = {str(pid): i for i, pid in enumerate(arrays["test_ids"])}
    sem = asyncio.Semaphore(args.concurrency)
    distiller = Distiller(make_client(cfg, cfg["distill_max_tokens"], seed))

    async def one(r):
        pid = str(r.puzzle_id)
        ctx = r.reflection_context()
        if ctx is None:                      # crashed: nothing to learn from
            return ("skipped", pid, None, None)
        episode = EpisodeResult.from_reflection_context(ctx, pid)
        row = df_test.iloc[row_of[pid]]
        key_text = puzzle_key_text(row)
        if policy == "adjustment":
            if r.success:
                return ("skipped", pid, None, None)   # only failures trigger a fix
            picks = retrieved_log.get(pid) or []
            if not picks:
                return ("skipped", pid, None, None)
            async with sem:
                text, resp = await distiller.revise(
                    _script_of(memory, picks[0]["id"]), episode, key_text)
            return ("revise", pid, (picks[0]["id"], text), resp)
        # vanilla / validation both distil the trajectory into a NEW memory
        async with sem:
            text, resp = await distiller.distill_from_trajectory(
                episode, key_text, succeeded=bool(r.success))
        return ("add", pid, (text, row, ctx), resp)

    targets = [r for r in group_results
               if policy != "validation" or r.success]
    stats["considered"] = len(targets)
    outcomes = await asyncio.gather(*(one(r) for r in targets),
                                    return_exceptions=True)
    try:
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                stats["failed"] += 1
                print(f"[update] call failed: {outcome!r}", flush=True)
                continue
            kind, pid, payload, resp = outcome
            if resp is not None:
                stats["prompt_tokens"] += resp.prompt_tokens
                stats["completion_tokens"] += resp.completion_tokens
            if kind == "skipped":
                stats["skipped"] += 1
            elif kind == "revise":
                entry_id, text = payload
                if text:
                    memory.revise(entry_id, text)
                    stats["revised"] += 1
                else:
                    stats["empty"] += 1
            else:
                text, row, ctx = payload
                if not text:
                    stats["empty"] += 1
                    continue
                # The agent's OWN moves, never the dataset's gold path: an online
                # entry is built from a test puzzle, and a gold exemplar would put
                # that puzzle's official solution into the context of every later
                # test puzzle that retrieves it.
                memory.add_with_vector(pid, {
                    "provenance": "online",
                    "difficulty": int(row["difficulty_level"]),
                    "key_text": puzzle_key_text(row),
                    "script": text,
                    "exemplar": agent_exemplar(row, ctx),
                    "origin": f"update:{policy}",
                }, arrays["test_vecs"][qrow[pid]])
                stats["added"] += 1
    finally:
        await distiller.client.aclose()
    return dict(stats)


def _script_of(memory, entry_id: str) -> str:
    for e in memory.entries:
        if e["id"] == entry_id:
            return e["script"]
    raise KeyError(entry_id)


async def run_eval(args) -> None:
    if args.resume:
        state = json.loads(pathlib.Path(args.resume).read_text())
        if state["config"].get("schema") != EVAL_SCHEMA:
            raise SystemExit(f"--resume expects a {EVAL_SCHEMA} file; "
                             f"{args.resume} is {state['config'].get('schema')!r}")
        cfg, out = state["config"], args.resume
        report_frozen(args, cfg, "eval")
    else:
        if not (args.bank and args.embeddings and args.arm):
            raise SystemExit("--phase eval needs --bank, --embeddings and "
                             "--arm retrieved|random (or --resume <eval file>)")
        cfg = {
            "schema": EVAL_SCHEMA, "phase": "eval", "arm": args.arm,
            "bank": str(args.bank), "embeddings": str(args.embeddings),
            "top_k_memories": args.top_k_memories, "split": args.split,
            "limit": args.limit or 0, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k,
            "max_tokens": args.max_tokens, "seed": args.seed,
            "granularity": args.granularity, "provenance": args.provenance,
            "update": args.update, "groups": args.groups,
            "keywords": str(args.keywords) if args.keywords else None,
            "distill_max_tokens": args.distill_max_tokens,
        }
        state = {"config": cfg, "puzzles": {}}
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        out = args.out or (f"results/json/memp_eval_{cfg['arm']}_"
                           f"{cfg['split']}_{stamp}.json")
    cfg["concurrency"] = args.concurrency

    # An Update run rewrites its bank as it goes, so a resumed job must continue
    # from the memory the finished groups produced, not from the on-disk seed bank.
    bank_src = state.get("bank") or json.loads(pathlib.Path(cfg["bank"]).read_text())
    memory = MempMemory.from_dict(bank_src)
    arrays, meta = load_embeddings(cfg["embeddings"])
    memory.embed_model = cfg["embed_model"] = meta.get("embed_model", "")
    if not state.get("bank"):     # filtering already applied before it was saved
        memory = memory.filtered(cfg.get("provenance", "all"))
    if not len(memory):
        raise SystemExit(f"provenance={cfg.get('provenance')} leaves the bank empty")
    # Pool both splits: a resumed Update run holds entries sourced from *test*
    # puzzles (added online), which are absent from the train arrays alone. For a
    # frozen run nothing changes -- no entry has a test source_pid.
    memory.attach_vectors(
        np.concatenate([arrays["train_ids"], arrays["test_ids"]]),
        np.concatenate([arrays["train_vecs"], arrays["test_vecs"]]))
    cfg["bank_entries"] = len(memory)
    cfg["bank_model"] = json.loads(
        pathlib.Path(cfg["bank"]).read_text()).get("config", {}).get("model_id")

    if cfg.get("granularity", "script") != "script":
        missing = sum(1 for e in memory.entries if not e.get("trajectory"))
        if missing:
            raise SystemExit(
                f"granularity={cfg['granularity']} needs stored trajectories but "
                f"{missing}/{len(memory)} entries lack one; run --phase upgrade "
                f"on {cfg['bank']} first")

    keyword_queries = None
    if cfg["arm"] == "avefact":
        keyword_queries, entry_kw = load_keywords(cfg["keywords"])
        memory.attach_keyword_vectors(entry_kw)

    test_ids = split_ids(cfg["split"], cfg["limit"])
    state.setdefault("updates", [])

    def blocks_for(ids):
        return build_blocks(memory, arrays, cfg, ids, keyword_queries)

    # A frozen bank can precompute every block once. The Update policies rewrite
    # the bank between groups, so those recompute per group instead.
    online = cfg.get("update", "none") != "none"
    blocks, retrieved_log = blocks_for(test_ids)

    longest = max(blocks.values(), key=len)
    print(f"[eval] sample injected block (for {test_ids[0]}):"
          f"{blocks[test_ids[0]]}\n", flush=True)
    if len(longest) / 4 > 6000:
        print(f"[eval] WARNING: largest block ~{len(longest) // 4} tokens; "
              f"check it still fits under MAXLEN with --max-tokens "
              f"{cfg['max_tokens']}", flush=True)

    todo = [pid for pid in test_ids if pid not in state["puzzles"]]
    if not todo:
        print(f"[eval] {out} already covers every puzzle; nothing to do", flush=True)
        finalize_eval(state, out)
        return
    print(f"[eval] arm={cfg['arm']} k={cfg['top_k_memories']} "
          f"granularity={cfg.get('granularity')} provenance={cfg.get('provenance')} "
          f"update={cfg.get('update')} bank={len(memory)} entries; attempting "
          f"{len(todo)} {cfg['split']} puzzles at seed {cfg['seed']}", flush=True)

    # The Update stage processes the split in groups (the paper refreshes memory
    # "after every t test-set tasks"); a frozen run just uses the chunk size.
    group_size = max(1, math.ceil(len(test_ids) / cfg["groups"])) if online \
        else args.chunk

    df_test = load_dataset("lkaesberg/SPaRC", "all", split=cfg["split"]).to_pandas()
    row_of = {str(df_test["id"][i]): i for i in range(len(df_test))}

    set_failure_log(f"eval_{cfg['arm']}")
    client = make_client(cfg, cfg["max_tokens"], cfg["seed"])
    cfg["model"] = client.model
    cfg["model_id"] = os.environ.get("LLM_MODEL_ID") or client.model
    try:
        for lo in range(0, len(todo), group_size):
            chunk = todo[lo:lo + group_size]
            t0 = time.time()
            results = await run_sparc_suite_memory_pass(
                client, chunk, lambda pid: blocks[pid],
                split=cfg["split"], concurrency=args.concurrency)
            for r in results:
                rec = episode_record(r)
                rec.pop("episode_context", None)   # eval keeps no trajectories
                rec["retrieved"] = retrieved_log[str(r.puzzle_id)]
                state["puzzles"][str(r.puzzle_id)] = rec
            solved = sum(1 for p in state["puzzles"].values() if p["success"])
            print(f"[eval] {len(state['puzzles'])}/{len(test_ids)} done, "
                  f"{solved} solved  [{(time.time() - t0) / 60:.0f} min/group]",
                  flush=True)

            if online:
                group_solved = sum(1 for r in results if r.success)
                stats = await update_pass(
                    cfg["update"], memory, arrays, results, retrieved_log,
                    df_test, row_of, cfg, args, cfg["seed"])
                stats.update({"group": len(state["updates"]) + 1,
                              "attempted": len(results),
                              "solved_this_group": group_solved,
                              "bank_after": len(memory),
                              "cumulative_solved": solved})
                state["updates"].append(stats)
                # Rewrite the bank alongside the episodes so a resume continues
                # with the memory the finished groups actually produced.
                state["bank"] = memory.to_dict()
                print(f"[update] group {stats['group']}: +{stats.get('added', 0)} "
                      f"added, {stats.get('revised', 0)} revised, "
                      f"{stats.get('empty', 0)} empty -> bank {len(memory)}",
                      flush=True)
                remaining = [p for p in todo[lo + group_size:]]
                if remaining:
                    blocks, retrieved_log = blocks_for(remaining)
                    # keep the finished groups' logs for the record
                    for pid, rec in state["puzzles"].items():
                        retrieved_log.setdefault(pid, rec.get("retrieved", []))
            save_state(out, state)
    finally:
        await client.aclose()

    finalize_eval(state, out)


def finalize_eval(state: dict, out) -> None:
    records = list(state["puzzles"].values())
    summary = accuracy_block(records)
    summary["method"] = "memp"
    summary["arm"] = state["config"]["arm"]
    sims = [r["retrieved"][0]["similarity"] for r in records
            if r.get("retrieved") and r["retrieved"][0]["similarity"] is not None]
    if sims:
        summary["mean_top1_similarity"] = round(sum(sims) / len(sims), 4)
    prov: dict[str, int] = {}
    for rec in records:
        for item in rec.get("retrieved", []):
            prov[item["provenance"]] = prov.get(item["provenance"], 0) + 1
    summary["injected_provenance"] = prov
    cfg = state.get("config", {})
    for key in ("granularity", "provenance", "update", "top_k_memories",
                "bank_entries", "bank_model"):
        if cfg.get(key) is not None:
            summary[key] = cfg[key]
    if state.get("updates"):
        ups = state["updates"]
        summary["update_groups"] = len(ups)
        summary["entries_added"] = sum(u.get("added", 0) for u in ups)
        summary["entries_revised"] = sum(u.get("revised", 0) for u in ups)
        summary["update_tokens"] = sum(
            u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) for u in ups)
        # The paper's Figure 3 reads as a per-group curve, so keep it renderable.
        summary["group_curve"] = [
            {"group": u.get("group"), "solved": u.get("solved_this_group"),
             "attempted": u.get("attempted"), "bank_after": u.get("bank_after")}
            for u in ups]
    state["summary"] = summary
    save_state(out, state)
    print(f"[eval] {summary['success']}/{summary['n']} solved "
          f"({summary['accuracy']:.3f}) "
          f"by_difficulty={summary['accuracy_by_difficulty']}", flush=True)
    if summary["errors"]:
        print(f"[eval] WARNING: {summary['errors']} episodes crashed; the run is "
              f"incomplete", flush=True)
    print(f"[eval] wrote {out}", flush=True)


# --------------------------------------------------------------------------- #
# phase compare: baseline vs retrieved vs random
# --------------------------------------------------------------------------- #
def mcnemar_exact(flips_a: int, flips_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    Same 500 puzzles, one attempt per run => paired binary outcomes, and only the
    discordant pairs are informative. Under H0 each discordant pair is a fair
    coin, so the p-value is an exact binomial tail -- math.comb, no scipy.
    """
    n = flips_a + flips_b
    if n == 0:
        return 1.0
    k = min(flips_a, flips_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def eval_outcomes(path) -> tuple[dict, dict]:
    data = json.loads(pathlib.Path(path).read_text())
    if data.get("config", {}).get("schema") != EVAL_SCHEMA:
        raise SystemExit(f"{path} is not a {EVAL_SCHEMA} file")
    return ({pid: rec["success"] for pid, rec in data["puzzles"].items()}, data)


def baseline_outcomes(path) -> tuple[dict, dict]:
    """Wins + a {summary, episodes}-shaped view of the baseline run.

    Accepts either a run_baseline output or a reflexion ROOT state file: the
    root IS the memoryless baseline over the split (trial 1 at the base seed),
    so the one root job per model serves as both Reflexion's shared trial 1 and
    Memp's baseline -- no separate baseline pass needed.
    """
    data = json.loads(pathlib.Path(path).read_text())
    if "episodes" in data:
        return ({str(e["puzzle_id"]): e["success"] for e in data["episodes"]}, data)
    cfg = data.get("config", {})
    if cfg.get("arm") == "root" and "puzzles" in data:
        episodes = []
        for pid, e in sorted(data["puzzles"].items()):
            a = (e.get("attempts") or [{}])[0]   # a root holds exactly one attempt
            episodes.append({
                "puzzle_id": pid, "success": e.get("solved", False),
                "difficulty": e.get("difficulty", -1),
                "steps": a.get("steps", 0),
                "terminated": a.get("terminated"), "truncated": a.get("truncated"),
                "terminal_reason": a.get("terminal_reason"),
                "prompt_tokens": a.get("prompt_tokens", 0),
                "completion_tokens": a.get("completion_tokens", 0),
                "retry_tokens": a.get("retry_tokens", 0),
                "error": a.get("error")})
        summary = {"split": cfg.get("split"), "seed": cfg.get("base_seed"),
                   "temperature": cfg.get("temperature"),
                   "max_tokens": cfg.get("max_tokens"),
                   "model": cfg.get("model"), "model_id": cfg.get("model_id")}
        return ({e["puzzle_id"]: e["success"] for e in episodes},
                {"summary": summary, "episodes": episodes})
    raise SystemExit(f"{path} is neither a run_baseline output (no 'episodes') "
                     f"nor a reflexion root state file")


def compare_row(label: str, records: list[dict]) -> str:
    block = accuracy_block(records)
    by_diff = " ".join(f"{d}:{a:.2f}"
                       for d, a in block["accuracy_by_difficulty"].items())
    steps = block["total_steps"] / block["n"] if block["n"] else 0.0
    ktok = block["completion_tokens"] / block["n"] / 1000 if block["n"] else 0.0
    return (f"  {label:<16} {block['success']:>4}/{block['n']:<4} "
            f"{block['accuracy']:>7.3f}   {by_diff:<40} {steps:>6.1f} {ktok:>9.1f}")


def run_compare(args) -> None:
    base_wins, base = baseline_outcomes(args.baseline)
    ret_wins, ret = eval_outcomes(args.retrieved)
    arms = [("baseline", base_wins,
             [e for e in base["episodes"]]),
            ("memp-retrieved", ret_wins, list(ret["puzzles"].values()))]
    rnd = None
    if args.random:
        rnd_wins, rnd = eval_outcomes(args.random)
        arms.insert(1, ("memp-random", rnd_wins, list(rnd["puzzles"].values())))

    # The whole comparison rests on config matching; warn loudly when it doesn't.
    bs, rc = base.get("summary", {}), ret["config"]
    for key in ("split", "seed", "temperature", "max_tokens"):
        if bs.get(key) != rc.get(key):
            print(f"WARNING: baseline and retrieved disagree on {key}: "
                  f"{bs.get(key)} vs {rc.get(key)} -- not comparable", flush=True)
    if rnd is not None:
        for key in ("bank", "top_k_memories", "seed", "split"):
            if rc.get(key) != rnd["config"].get(key):
                print(f"WARNING: retrieved and random disagree on {key}: "
                      f"{rc.get(key)} vs {rnd['config'].get(key)}", flush=True)

    common = set(base_wins) & set(ret_wins)
    for _label, wins, _recs in arms:
        common &= set(wins)
    common = sorted(common)

    lines = [
        "=" * 96,
        f"  MEMP vs BASELINE over {len(common)} puzzles "
        f"({rc.get('split')}, {rc.get('model_id', rc.get('model'))}, "
        f"k={rc.get('top_k_memories')}, bank={rc.get('bank')})",
        "=" * 96,
        "",
        f"  {'run':<16} {'solved':>9} {'acc':>7}   {'by difficulty':<40} "
        f"{'steps':>6} {'ktok/puz':>9}",
        "  " + "-" * 92,
    ]
    for label, _wins, recs in arms:
        lines.append(compare_row(label, recs))
    lines.append("")

    def pair(name_a, wins_a, name_b, wins_b):
        a_only = sum(1 for p in common if wins_a[p] and not wins_b[p])
        b_only = sum(1 for p in common if wins_b[p] and not wins_a[p])
        p = mcnemar_exact(a_only, b_only)
        return (f"  {name_a} vs {name_b}: {name_a} alone won {a_only}, "
                f"{name_b} alone won {b_only}  (McNemar exact p={p:.4f})")

    lines += ["  paired tests over the common puzzles (discordant pairs only)", ""]
    lines.append(pair("retrieved", ret_wins, "baseline", base_wins))
    if rnd is not None:
        lines.append(pair("retrieved", ret_wins, "random", rnd_wins))
    lines.append("")

    # Memp's price of admission: the bank was not free. Reported separately from
    # eval cost because it amortises over every future eval against this bank.
    bank_path = pathlib.Path(rc.get("bank", ""))
    if bank_path.exists():
        stats = json.loads(bank_path.read_text()).get("stats", {})
        lines += [
            "  bank construction cost (one-off, amortised over evals)",
            "",
            f"  distill tokens: prompt {stats.get('distill_prompt_tokens', 0):,}  "
            f"completion {stats.get('distill_completion_tokens', 0):,}  "
            f"({stats.get('n_agent', 0)} agent + {stats.get('n_gold', 0)} gold "
            f"entries)",
            "  (plus the build pass: one memoryless run over the train split, and "
            "the embed job)",
            "",
        ]

    report = "\n".join(lines)
    print(report)
    if args.out != "-":
        dest = pathlib.Path(args.out or "results/txt/memp_vs_baseline.txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(report)
        print(f"[written] {dest}", file=sys.stderr)


# --------------------------------------------------------------------------- #
async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True,
                    choices=["build", "distill", "embed", "eval", "compare",
                             "upgrade", "keywords", "embed-keywords"])
    ap.add_argument("--split", default="test",
                    help="build refuses anything but train; eval defaults to test")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 or less = whole split")
    ap.add_argument("--chunk", type=int, default=100,
                    help="puzzles per checkpoint in build/eval, calls per "
                         "checkpoint in distill")
    ap.add_argument("--max-tokens", type=int, default=12288, help="actor budget")
    ap.add_argument("--distill-max-tokens", type=int, default=12288,
                    help="distillation budget (distill phase)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20,
                    help="sampling top-k (NOT the number of memories)")
    ap.add_argument("--seed", type=int, default=0,
                    help="decoding seed; keep equal to the baseline's for eval")
    ap.add_argument("--top-k-memories", type=int, default=3,
                    help="memories injected per test puzzle (eval)")
    ap.add_argument("--arm", choices=["retrieved", "random", "avefact", "fixed"],
                    help="eval retrieval policy: cosine top-k (the paper's "
                         "Key=Query), the random-sample ablation, AveFact "
                         "keyword matching, or a fixed block for every puzzle")
    ap.add_argument("--granularity", default="script",
                    choices=["script", "trajectory", "proceduralization"],
                    help="eval: the paper's Build-policy conditions (Table 1). "
                         "Anything but 'script' needs a bank run through --phase upgrade")
    ap.add_argument("--provenance", default="all", choices=["all", "agent", "gold"],
                    help="eval: restrict the bank. 'agent' is the paper-faithful "
                         "successes-only build source")
    ap.add_argument("--update", default="none",
                    choices=["none", "vanilla", "validation", "adjustment"],
                    help="eval: the paper's online Update policies (section 4.3). "
                         "'none' keeps the bank frozen")
    ap.add_argument("--groups", type=int, default=5,
                    help="eval: test-set groups between memory refreshes (update arms)")
    ap.add_argument("--keywords", default=None,
                    help="memp_keywords_*.json for --arm avefact / --phase embed-keywords")
    ap.add_argument("--build", default=None, help="build JSON (distill phase)")
    ap.add_argument("--bank", default=None, help="bank JSON (eval phase)")
    ap.add_argument("--embeddings", default=None, help="embeddings .npz (eval)")
    ap.add_argument("--resume", default=None, help="state file to continue in place")
    ap.add_argument("--batch", type=int, default=64, help="embed batch size")
    ap.add_argument("--verify-gold", type=int, default=10, metavar="N",
                    help="distill: replay N gold paths through the env first "
                         "(0 disables)")
    ap.add_argument("--baseline", default=None, help="baseline JSON (compare)")
    ap.add_argument("--retrieved", default=None, help="retrieved-arm JSON (compare)")
    ap.add_argument("--random", default=None, help="random-arm JSON (compare)")
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args()

    # Which frozen knobs the user actually typed, so a resume can tell "left at
    # its default" from "explicitly asked for" (run_reflexion's trick).
    typed = {a.lstrip("-").replace("-", "_").split("=")[0]
             for a in sys.argv[1:] if a.startswith("--")}
    args.explicit = {k: k in typed for keys in FROZEN.values() for k in keys}
    args.limit = args.limit if args.limit > 0 else None

    if args.phase == "build":
        await run_build(args)
    elif args.phase == "distill":
        await run_distill(args)
    elif args.phase == "embed":
        await run_embed(args)
    elif args.phase == "upgrade":
        if not args.bank:
            ap.error("--phase upgrade needs --bank <memp_bank_*.json>")
        run_upgrade(args)
    elif args.phase == "keywords":
        await run_keywords(args)
    elif args.phase == "embed-keywords":
        if not args.keywords:
            ap.error("--phase embed-keywords needs --keywords <memp_keywords_*.json>")
        await run_embed_keywords(args)
    elif args.phase == "eval":
        if args.arm == "avefact" and not (args.keywords or args.resume):
            ap.error("--arm avefact needs --keywords <memp_keywords_*.json>")
        await run_eval(args)
    else:
        if not (args.baseline and args.retrieved):
            ap.error("--phase compare needs --baseline and --retrieved "
                     "(--random is optional but recommended)")
        run_compare(args)


if __name__ == "__main__":
    asyncio.run(main())
