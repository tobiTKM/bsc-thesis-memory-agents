"""Memp (Fang et al., 2026): build procedural memory on the train split, freeze
it, and retrieve from it on the test split. Compared against the memoryless
baseline at the same seed and a random-retrieval arm.
memorybank.py and contrastive.py reuse this module's plumbing.

Phases:
  build           memoryless pass over the train split, keeping each trajectory
  distill         build pass -> bank: own trajectory if solved, else the gold path
  embed           embed every train+test retrieval key (embedding-model server)
  eval            one test pass with injected entries; --arm retrieved|random|avefact|fixed
  compare         baseline vs retrieved vs random, McNemar exact tests (no GPU)
  upgrade         attach replayed trajectories to a bank, for --granularity (no GPU)
  keywords        extract per-item keywords for --arm avefact
  embed-keywords  embed those keywords (embedding-model server)

  sbatch -J memp-build --export=ALL,DRIVER=memp,PHASE=build,SPLIT=train,LIMIT=0 scripts/serve_and_run.sbatch
  sbatch -J memp-dist  --export=ALL,DRIVER=memp,PHASE=distill,BUILD=results/json/memp_build_train_<stamp>.json scripts/serve_and_run.sbatch
  sbatch -J memp-embed -G A100:1 -t 01:00:00 --export=ALL,DRIVER=memp,PHASE=embed,TP=1 scripts/serve_and_run.sbatch
  sbatch -J memp-ret   --export=ALL,DRIVER=memp,PHASE=eval,ARM=retrieved,LIMIT=0,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
  python scripts/run.py memp --phase compare --baseline <baseline.json> --retrieved <ret.json> --random <rnd.json>
  python scripts/run.py memp --phase upgrade --bank <bank.json>
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

from memory_agents.benchmarks import build_benchmark
from memory_agents.config import resolve_server
from memory_agents.envs.base import EpisodeResult, make_json_safe
from memory_agents.llm.client import EmbeddingClient, LLMClient
from memory_agents.memory import Distiller, MempMemory
from memory_agents.methods.common import SAMPLING, add_run_args, typed_flags

BUILD_SCHEMA = "memp-build-v1"
EVAL_SCHEMA = "memp-eval-v1"
# SPaRC's key-fn name; runs use bench().key_fn.
KEY_FN = "puzzle-key-v1"


# The process's one Benchmark, set in main(); memorybank and contrastive set it too.
BENCH = None


def bench():
    """The active Benchmark, defaulting to SPaRC for callers that never set one."""
    global BENCH
    if BENCH is None:
        from memory_agents.benchmarks import SparcBenchmark
        BENCH = SparcBenchmark()
    return BENCH

# Knobs that define a run, per phase; on --resume the stored values win over the CLI.
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
# shared plumbing
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
    """The item ids a pass runs over, in split order."""
    return bench().ids(split, limit or None)


def episode_record(r) -> dict:
    """One puzzle's outcome in baseline's record shape, plus the compact
    trajectory (kept for successes too: distill turns it into a procedure)."""
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
        # StepGame only; not in reflection_context(), so no prompt sees it.
        **({"extra": r.extra} if getattr(r, "extra", None) else {}),
    })


def accuracy_block(records: list[dict]) -> dict:
    """baseline.summarize() over already-serialised records."""
    n = len(records)
    wins = sum(1 for rec in records if rec["success"])
    by_diff: dict[int, list] = {}
    for rec in records:
        by_diff.setdefault(rec["difficulty"], []).append(rec["success"])
    # StepGame's second difficulty axis (the true hop count, not k).
    by_hops: dict[int, list] = {}
    for rec in records:
        hops = (rec.get("extra") or {}).get("hops")
        if hops is not None:
            by_hops.setdefault(int(hops), []).append(rec["success"])
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
        # Memories must come from puzzles the eval never sees.
        if args.split != "train":
            raise SystemExit(f"the build pass runs on the train split (got "
                             f"--split {args.split}); submit with SPLIT=train")
        cfg = {
            "schema": BUILD_SCHEMA, "phase": "build", "benchmark": bench().name,
            "stepgame_train": args.stepgame_train,
            "stepgame_test": args.stepgame_test,
            "key_variant": args.key_variant, "split": args.split,
            "limit": args.limit or 0, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k,
            "max_tokens": args.max_tokens, "seed": args.seed,
        }
        state = {"config": cfg, "puzzles": {}}
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        tag = "" if bench().name == "sparc" else f"{bench().name}_"
        out = args.out or f"results/json/memp_build_{tag}train_{stamp}.json"
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
        # Checkpoint per chunk so a wall-clock kill loses at most one chunk.
        for lo in range(0, len(todo), args.chunk):
            chunk = todo[lo:lo + args.chunk]
            t0 = time.time()
            results = await bench().run_memory_pass(
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
def polyshapes_of(row):
    """SPaRC's shape table for a row (used by the Update policies)."""
    return bench().polyshapes(row)


def gold_exemplar(record) -> dict:
    """The solved example of a gold entry, built in code (gold path or solver),
    never by the LLM."""
    return bench().gold_exemplar(record)


def agent_exemplar(record, episode_ctx: dict) -> dict:
    """Same shape, from the agent's own successful attempt."""
    return bench().agent_exemplar(record, episode_ctx)


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
            "benchmark": bench().name, "key_fn": bench().key_fn,
            "stepgame_train": args.stepgame_train,
            "stepgame_test": args.stepgame_test,
            "key_variant": args.key_variant,
            "distill_max_tokens": args.distill_max_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "seed": args.seed,
        }
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        tag = "" if bench().name == "sparc" else f"{bench().name}_"
        out = args.out or f"results/json/memp_bank_{tag}{stamp}.json"
    cfg["concurrency"] = args.concurrency

    build = json.loads(pathlib.Path(cfg["build"]).read_text())
    if build["config"].get("schema") != BUILD_SCHEMA:
        raise SystemExit(f"--build expects a {BUILD_SCHEMA} file; "
                         f"{cfg['build']} is {build['config'].get('schema')!r}")

    if args.verify_gold:
        bench().verify_gold("train", args.verify_gold)

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
        row = bench().record("train", pid)
        key_text = bench().key_text(row)
        # A crashed episode has no episode_context, so it takes the gold route.
        from_agent = bool(rec["success"] and rec.get("episode_context"))
        async with sem:
            if from_agent:
                episode = EpisodeResult.from_reflection_context(
                    rec["episode_context"], pid)
                text, resp = await distiller.distill_from_trajectory(
                    episode, key_text, row)
            else:
                text, resp = await distiller.distill_from_gold(key_text, row)
        return pid, from_agent, text, resp

    async def distill_pass(pids, seed):
        """One pass over `pids`; returns the pids whose reply came back empty."""
        distiller = Distiller(make_client(cfg, cfg["distill_max_tokens"], seed),
                              benchmark=bench())
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
                    row = bench().record("train", pid)
                    memory.write(pid, {
                        "provenance": "agent" if from_agent else "gold",
                        # From the dataset: a crashed episode records difficulty -1.
                        "difficulty": bench().difficulty(row),
                        "key_text": bench().key_text(row),
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
        # The same seed would replay the same empty reply, so the retry reseeds.
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


def verify_gold_paths(df, n: int) -> None:   # SPaRC only
    """Replay the first n gold paths through the env and assert each one solves
    (checks path_to_actions, no LLM)."""
    from memory_agents.envs.sparc import path_to_actions
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
        if env is not None:
            env.close()


# --------------------------------------------------------------------------- #
# phase upgrade: add round-by-round trajectories to an existing bank (no GPU)
# --------------------------------------------------------------------------- #
def run_upgrade(args) -> None:
    """Attach a round-by-round `trajectory` to every bank entry, without a model.

    Interactive benchmarks replay the entry's exemplar actions through the env;
    single-turn ones store the item's question plus its worked derivation. Every
    entry is upgraded so granularity is not confounded with bank size.
    """
    src = pathlib.Path(args.bank)
    data = json.loads(src.read_text())
    memory = MempMemory.from_dict(data)
    out = pathlib.Path(args.out or src.with_name(src.stem + "_traj.json"))

    interactive = bench().interactive
    env = None
    if interactive:
        from memory_agents.envs.sparc import SparcEnv, replay_path
        env = SparcEnv(split="train")
    upgraded = failed = solved = 0
    try:
        for e in memory.entries:
            try:
                if interactive:
                    actions = (e.get("exemplar") or {}).get("actions") or []
                    if not actions:
                        failed += 1
                        continue
                    traj = replay_path(env, e["source_pid"], actions)
                else:
                    traj = bench().gold_trajectory(
                        bench().record("train", e["source_pid"]))
            except Exception as exc:            # a bad exemplar must not kill the run
                print(f"[upgrade] {e['id']} ({e['source_pid']}): {exc!r}", flush=True)
                failed += 1
                continue
            e["trajectory"] = traj
            upgraded += 1
            solved += traj["terminal_reason"] == "solved"
    finally:
        if env is not None:
            env.close()

    save_state(out, {**memory.to_dict(), "config": data.get("config", {})})
    print(f"[upgrade] {upgraded}/{len(memory)} entries now carry a trajectory "
          f"({solved} of those replays reached a solved state, {failed} failed)",
          flush=True)
    if upgraded and solved < upgraded:
        print(f"[upgrade] NOTE: {upgraded - solved} replays did not end solved; "
              f"those entries' exemplars were not winning paths", flush=True)
    print(f"[upgrade] wrote {out}", flush=True)


# --------------------------------------------------------------------------- #
# phase keywords: AveFact key construction (LLM), then embed-keywords (embeddings)
# --------------------------------------------------------------------------- #
async def run_keywords(args) -> None:
    """Extract per-puzzle keywords for the AveFact retriever (one LLM call each).

    The extraction prompt is SPaRC-shaped; other benchmarks run but prompt poorly.
    """
    cfg = {"temperature": args.temperature, "top_p": args.top_p,
           "top_k": args.top_k}
    distiller = Distiller(make_client(cfg, args.distill_max_tokens, args.seed))
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out = pathlib.Path(args.out or f"results/json/memp_keywords_{stamp}.json")
    result = {"model_id": os.environ.get("LLM_MODEL_ID") or distiller.client.model,
              "key_fn": bench().key_fn, "benchmark": bench().name, "keywords": {}}
    if out.exists():                      # resume a partial file
        result = json.loads(out.read_text())
    sem = asyncio.Semaphore(args.concurrency)

    async def one(pid, key_text):
        async with sem:
            return pid, *(await distiller.keywords(key_text))

    try:
        for split in ("train", "test"):
            n = None if not args.limit else args.limit
            todo = [(pid, bench().key_text(bench().record(split, pid)))
                    for pid in bench().ids(split, n)
                    if pid not in result["keywords"]]
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
    """Keyword vectors per puzzle id, returned twice: as queries and as entry keys."""
    npz = np.load(str(pathlib.Path(path).with_suffix(".npz")))
    by_pid: dict[str, list] = {}
    for pid, vec in zip(npz["kw_pids"], npz["kw_vecs"]):
        by_pid.setdefault(str(pid), []).append(vec)
    grouped = {pid: np.asarray(v, dtype=np.float32) for pid, v in by_pid.items()}
    return grouped, grouped


# --------------------------------------------------------------------------- #
# phase embed: key text for both splits -> vectors on disk
# --------------------------------------------------------------------------- #
async def run_embed(args) -> None:
    base_url, model = resolve_server()
    client = EmbeddingClient(base_url=base_url, model=model)
    embed_model = os.environ.get("LLM_MODEL_ID") or model

    # Smoke-test the pooling endpoint here; the sbatch chat smoke cannot.
    probe = await client.embed(["smoke"])
    dim = len(probe[0])
    norm = float(np.linalg.norm(probe[0]))
    if dim <= 0 or norm <= 0:
        raise SystemExit(f"embedding smoke failed: dim={dim} norm={norm}")
    print(f"[embed] {embed_model}: dim={dim}", flush=True)

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    tag = "" if bench().name == "sparc" else f"{bench().name}_"
    if bench().name != "sparc" and getattr(args, "key_variant", "") == "lexical":
        tag += "lex_"
    out = pathlib.Path(args.out or f"results/json/memp_embeddings_{tag}{stamp}.npz")
    arrays: dict[str, np.ndarray] = {}
    try:
        for split in ("train", "test"):
            ids = bench().ids(split, args.limit or None)
            n = len(ids)
            texts = [bench().key_text(bench().record(split, pid)) for pid in ids]
            vecs: list[list[float]] = []
            for lo in range(0, n, args.batch):
                vecs.extend(await client.embed(texts[lo:lo + args.batch]))
            mat = np.asarray(vecs, dtype=np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True)
            arrays[f"{split}_ids"] = np.asarray([str(pid) for pid in ids])
            arrays[f"{split}_vecs"] = mat
            print(f"[embed] {split}: {mat.shape[0]} x {mat.shape[1]}", flush=True)
    finally:
        await client.aclose()

    # Atomic write, as in save_state.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    meta = {"embed_model": embed_model, "dim": dim, "normalized": True,
            "key_fn": bench().key_fn, "benchmark": bench().name,
            "n_train": int(arrays["train_ids"].shape[0]),
            "n_test": int(arrays["test_ids"].shape[0])}
    # .meta.json, not .json: summarize_results globs results/json/*.json.
    meta_path = out.with_name(out.stem + ".meta.json")
    save_state(meta_path, meta)

    # Catch misaligned id/vector rows. Not `argmax == own row`: items with an
    # identical key text have identical vectors and tie.
    tv = arrays["train_vecs"]
    probe = min(64, tv.shape[0])
    sims = tv[:probe] @ tv.T
    rows = np.arange(probe)
    if not np.all(sims[rows, sims.argmax(axis=1)] <= sims[rows, rows] + 1e-5):
        raise SystemExit("[embed] sanity failed: some train key is strictly closer "
                         "to another key than to itself; id/vector rows are "
                         "misaligned")
    for split in ("train", "test"):
        ids_ = arrays[f"{split}_ids"]
        distinct = len({tuple(np.round(v, 6)) for v in arrays[f"{split}_vecs"]})
        if distinct < len(ids_):
            # Not an error (ties break by id), but it limits retrieval.
            print(f"[embed] note: {split} has {distinct} distinct vectors for "
                  f"{len(ids_)} items ({len(ids_) - distinct} share a key text "
                  f"with another item)", flush=True)
    print(f"[embed] wrote {out} + {meta_path}", flush=True)


# --------------------------------------------------------------------------- #
# phase eval: one retrieval-augmented pass over the test split
# --------------------------------------------------------------------------- #
def load_embeddings(path) -> tuple[dict, dict]:
    npz = np.load(path)
    path = pathlib.Path(path)
    meta_path = path.with_name(path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    expected = bench().key_fn
    if meta.get("key_fn") not in (None, expected):
        raise SystemExit(f"{path} was embedded with key_fn={meta.get('key_fn')!r} "
                         f"but this run renders {expected}; re-run the embed phase "
                         f"(or pass the matching --benchmark/--key-variant)")
    return {k: npz[k] for k in npz.files}, meta


def build_blocks(memory, arrays, cfg, test_ids, keyword_queries=None):
    """Injected block and retrieval log per test puzzle.

    Precomputed so a resumed job injects identical blocks; the Update policies
    call it again per group.
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
        blocks[pid] = bench().format_procedures(
            [entry for entry, _sim in picks], granularity=granularity)
        retrieved_log[pid] = [
            {"id": entry["id"], "source_pid": entry["source_pid"],
             "provenance": entry["provenance"],
             "similarity": None if sim is None else round(float(sim), 4)}
            for entry, sim in picks]
    return blocks, retrieved_log


# --------------------------------------------------------------------------- #
# the Update stage: the bank evolves across the test split, group by group
# --------------------------------------------------------------------------- #
async def update_pass(policy, memory, arrays, group_results, retrieved_log,
                      df_test, row_of, cfg, args, seed) -> dict:
    """Fold one finished group of test episodes back into the bank.

      vanilla     every trajectory in the group becomes a new memory
      validation  only the successful ones do; failures are discarded
      adjustment  each failure revises the memory that was retrieved for it

    Returns counters. New entries carry the test puzzle's own embedding, so later
    groups can retrieve them.
    """
    from memory_agents.envs.sparc import puzzle_key_text   # SPaRC-only ablation
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
                    _script_of(memory, picks[0]["id"]), episode, key_text,
                    polyshapes_of(row))
            return ("revise", pid, (picks[0]["id"], text), resp)
        # vanilla / validation: distil into a new memory
        async with sem:
            text, resp = await distiller.distill_from_trajectory(
                episode, key_text, polyshapes_of(row), succeeded=bool(r.success))
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
                # The agent's own moves, never the gold path: that would leak a
                # test puzzle's solution into later test puzzles.
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
            "benchmark": bench().name, "key_variant": args.key_variant,
            "stepgame_train": args.stepgame_train,
            "stepgame_test": args.stepgame_test,
            "granularity": args.granularity, "provenance": args.provenance,
            "update": args.update, "groups": args.groups,
            "keywords": str(args.keywords) if args.keywords else None,
            "distill_max_tokens": args.distill_max_tokens,
        }
        state = {"config": cfg, "puzzles": {}}
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        tag = "" if bench().name == "sparc" else f"{bench().name}_"
        out = args.out or (f"results/json/memp_eval_{tag}{cfg['arm']}_"
                           f"{cfg['split']}_{stamp}.json")
    cfg["concurrency"] = args.concurrency

    # An Update run saves its evolving bank; a resume continues from that one.
    bank_src = state.get("bank") or json.loads(pathlib.Path(cfg["bank"]).read_text())
    memory = MempMemory.from_dict(bank_src)
    arrays, meta = load_embeddings(cfg["embeddings"])
    memory.embed_model = cfg["embed_model"] = meta.get("embed_model", "")
    if not state.get("bank"):     # a saved bank is already filtered
        memory = memory.filtered(cfg.get("provenance", "all"))
    if not len(memory):
        raise SystemExit(f"provenance={cfg.get('provenance')} leaves the bank empty")
    # Both splits: a resumed Update bank holds entries sourced from test puzzles.
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

    # Update policies rewrite the bank, so they recompute blocks per group.
    online = cfg.get("update", "none") != "none"
    if online and bench().name != "sparc":
        raise SystemExit(
            f"--update {cfg['update']} is implemented for SPaRC only (it revises "
            f"procedures against replayed env trajectories); use --update none "
            f"on {bench().name}")
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

    # Update runs split the test set into --groups; a frozen run uses --chunk.
    group_size = max(1, math.ceil(len(test_ids) / cfg["groups"])) if online \
        else args.chunk

    # Only the (SPaRC-only) Update policies need dataset rows here.
    df_test = row_of = None
    if online:
        from datasets import load_dataset
        df_test = load_dataset("lkaesberg/SPaRC", "all",
                               split=cfg["split"]).to_pandas()
        row_of = {str(df_test["id"][i]): i for i in range(len(df_test))}

    set_failure_log(f"eval_{cfg['arm']}")
    client = make_client(cfg, cfg["max_tokens"], cfg["seed"])
    cfg["model"] = client.model
    cfg["model_id"] = os.environ.get("LLM_MODEL_ID") or client.model
    try:
        for lo in range(0, len(todo), group_size):
            chunk = todo[lo:lo + group_size]
            t0 = time.time()
            results = await bench().run_memory_pass(
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
                # Saved with the episodes so a resume continues from this bank.
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
    """Two-sided exact McNemar p-value (binomial tail) from the discordant counts."""
    n = flips_a + flips_b
    if n == 0:
        return 1.0
    k = min(flips_a, flips_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def eval_outcomes(path, schema: str = EVAL_SCHEMA) -> tuple[dict, dict]:
    """Per-puzzle wins and raw data of one eval file (`schema` for other methods)."""
    data = json.loads(pathlib.Path(path).read_text())
    if data.get("config", {}).get("schema") != schema:
        raise SystemExit(f"{path} is not a {schema} file")
    return ({pid: rec["success"] for pid, rec in data["puzzles"].items()}, data)


def baseline_outcomes(path) -> tuple[dict, dict]:
    """Wins plus a {summary, episodes} view of a baseline output or of a reflexion
    root state file (whose single attempt is the memoryless baseline)."""
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
    raise SystemExit(f"{path} is neither a baseline output (no 'episodes') "
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

    # The comparison is only valid if the configs match.
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
def _resumed_config(args) -> dict | None:
    """The stored config of the file this invocation continues, or None.

    --resume covers build/distill/eval; --bank covers upgrade, which must re-read
    its bank under the benchmark it was written with.
    """
    for attr in ("resume", "bank"):
        path = getattr(args, attr, None)
        if not path or not pathlib.Path(path).exists():
            continue
        try:
            data = json.loads(pathlib.Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return None
        cfg = data.get("config")
        if isinstance(cfg, dict) and cfg.get("benchmark"):
            return cfg
    return None


async def main(argv: list[str], prog: str) -> None:
    ap = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True,
                    choices=["build", "distill", "embed", "eval", "compare",
                             "upgrade", "keywords", "embed-keywords"])
    add_run_args(ap, **SAMPLING,
                 split_help="build refuses anything but train; eval defaults to test",
                 seed_help="decoding seed; keep equal to the baseline's for eval")
    ap.add_argument("--chunk", type=int, default=100,
                    help="puzzles per checkpoint in build/eval, calls per "
                         "checkpoint in distill")
    ap.add_argument("--distill-max-tokens", type=int, default=12288,
                    help="distillation budget (distill phase)")
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
    args = ap.parse_args(argv)

    # Typed flags, so a resume can report which ones it ignores.
    typed = typed_flags(argv)
    args.explicit = {k: k in typed for keys in FROZEN.values() for k in keys}
    args.limit = args.limit if args.limit > 0 else None

    # On resume the benchmark comes from the state file, so prompts and key-fn
    # match the partial output.
    global BENCH
    stored = _resumed_config(args)
    BENCH = build_benchmark(stored if stored else args)
    if stored and stored.get("benchmark", "sparc") != getattr(args, "benchmark",
                                                             "sparc"):
        print(f"[memp] resuming a {stored.get('benchmark', 'sparc')} run; "
              f"ignoring --benchmark {args.benchmark}", flush=True)

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

