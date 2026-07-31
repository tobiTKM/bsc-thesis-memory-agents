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
             "temperature", "top_p", "top_k", "max_tokens", "seed"),
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


def build_blocks(memory, arrays, arm, k, seed, test_ids):
    """Precompute the injected block + retrieval log for every test puzzle.

    Done up front, before any episode runs: the bank is frozen, so retrieval is
    a pure function of (bank, embeddings, arm, k, seed) -- precomputing makes
    block_for a dict lookup (trivially safe under concurrency) and means a
    resumed job injects byte-identical blocks."""
    qrow = {str(pid): i for i, pid in enumerate(arrays["test_ids"])}
    blocks, retrieved_log = {}, {}
    for pid in test_ids:
        if arm == "retrieved":
            if pid not in qrow:
                raise SystemExit(f"test puzzle {pid} has no query embedding; "
                                 f"re-run the embed phase without a limit")
            picks = memory.retrieve(arrays["test_vecs"][qrow[pid]], k)
        else:
            picks = memory.random_retrieve(k, seed_key=f"{seed}:{pid}")
        blocks[pid] = format_procedures([entry for entry, _sim in picks])
        retrieved_log[pid] = [
            {"id": entry["id"], "source_pid": entry["source_pid"],
             "provenance": entry["provenance"],
             "similarity": None if sim is None else round(float(sim), 4)}
            for entry, sim in picks]
    return blocks, retrieved_log


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
        }
        state = {"config": cfg, "puzzles": {}}
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        out = args.out or (f"results/json/memp_eval_{cfg['arm']}_"
                           f"{cfg['split']}_{stamp}.json")
    cfg["concurrency"] = args.concurrency

    memory = MempMemory.from_dict(json.loads(pathlib.Path(cfg["bank"]).read_text()))
    arrays, meta = load_embeddings(cfg["embeddings"])
    memory.embed_model = cfg["embed_model"] = meta.get("embed_model", "")
    cfg["bank_entries"] = len(memory)
    memory.attach_vectors(arrays["train_ids"], arrays["train_vecs"])

    test_ids = split_ids(cfg["split"], cfg["limit"])
    blocks, retrieved_log = build_blocks(
        memory, arrays, cfg["arm"], cfg["top_k_memories"], cfg["seed"], test_ids)

    # One fully rendered block in the log for eyeballing, plus a budget guard:
    # entries are word-capped at distill time, but a prompt that quietly grew past
    # ~2k tokens (chars/4) would eat into the actor's thinking budget.
    longest = max(blocks.values(), key=len)
    print(f"[eval] sample injected block (for {test_ids[0]}):"
          f"{blocks[test_ids[0]]}\n", flush=True)
    if len(longest) / 4 > 2000:
        print(f"[eval] WARNING: largest block ~{len(longest) // 4} tokens; "
              f"consider a smaller --top-k-memories", flush=True)

    todo = [pid for pid in test_ids if pid not in state["puzzles"]]
    if not todo:
        print(f"[eval] {out} already covers every puzzle; nothing to do", flush=True)
        finalize_eval(state, out)
        return
    print(f"[eval] arm={cfg['arm']} k={cfg['top_k_memories']} bank={len(memory)} "
          f"entries; attempting {len(todo)} {cfg['split']} puzzles at "
          f"seed {cfg['seed']}", flush=True)

    set_failure_log(f"eval_{cfg['arm']}")
    client = make_client(cfg, cfg["max_tokens"], cfg["seed"])
    cfg["model"] = client.model
    cfg["model_id"] = os.environ.get("LLM_MODEL_ID") or client.model
    try:
        for lo in range(0, len(todo), args.chunk):
            chunk = todo[lo:lo + args.chunk]
            t0 = time.time()
            results = await run_sparc_suite_memory_pass(
                client, chunk, lambda pid: blocks[pid],
                split=cfg["split"], concurrency=args.concurrency)
            for r in results:
                rec = episode_record(r)
                rec.pop("episode_context", None)   # eval keeps no trajectories
                rec["retrieved"] = retrieved_log[str(r.puzzle_id)]
                state["puzzles"][str(r.puzzle_id)] = rec
            save_state(out, state)
            solved = sum(1 for p in state["puzzles"].values() if p["success"])
            print(f"[eval] {len(state['puzzles'])}/{len(test_ids)} done, "
                  f"{solved} solved  [{(time.time() - t0) / 60:.0f} min/chunk]",
                  flush=True)
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
    data = json.loads(pathlib.Path(path).read_text())
    if "episodes" not in data:
        raise SystemExit(f"{path} does not look like a run_baseline output "
                         f"(no 'episodes')")
    return ({str(e["puzzle_id"]): e["success"] for e in data["episodes"]}, data)


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
                    choices=["build", "distill", "embed", "eval", "compare"])
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
    ap.add_argument("--arm", choices=["retrieved", "random"],
                    help="eval: cosine top-k, or the random-retrieval ablation")
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
    elif args.phase == "eval":
        await run_eval(args)
    else:
        if not (args.baseline and args.retrieved):
            ap.error("--phase compare needs --baseline and --retrieved "
                     "(--random is optional but recommended)")
        run_compare(args)


if __name__ == "__main__":
    asyncio.run(main())
