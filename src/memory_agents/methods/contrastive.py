"""Contrastive memory over SPaRC / StepGame: one bank holding DOs distilled from the
actor's successes and DON'Ts distilled from its failures.

The entry's side follows the train outcome: a solved puzzle yields a `strategy`
(Memp's distill prompt); a failed one yields an `avoid` text distilled from the
failure plus a `hindsight` text (Memp's gold prompt). --granularity picks what a
failed entry contributes: avoid (the method), hindsight (the control),
positive-only (nothing), avoid-only (DON'Ts, DOs withheld). Bank, retriever and k
are shared, and Memp's build file and embeddings are reused as-is.

  distill    memp_build_train_*.json -> contrastive_bank_*.json, all texts in one pass
  exemplars  attach winning/losing lines to a bank built without them (CPU, no model)
  eval       top-k entries per test item -> injected block -> one pass over the split
  compare    baseline vs the arms (exact McNemar) plus each arm's injected polarity

  sbatch -J co-dist --export=ALL,DRIVER=contrastive,PHASE=distill,BUILD=<memp_build_train_*.json> scripts/serve_and_run.sbatch
  python scripts/run.py contrastive --phase exemplars --bank <bank.json>
  sbatch -J co-avo  --export=ALL,DRIVER=contrastive,PHASE=eval,ARM=retrieved,CO_GRANULARITY=avoid,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
  python scripts/run.py contrastive --phase compare --baseline <root.json> --retrieved <avoid.json> --hindsight <hindsight.json> --positive <pos.json> --random <rnd.json>
"""

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from datetime import datetime

# Plumbing is imported from memp, not copied, so all cross-task methods share it.
from memory_agents.methods import memp
from memory_agents.methods.memp import (accuracy_block, agent_exemplar, baseline_outcomes, bench,
                      compare_row, episode_record, eval_outcomes, gold_exemplar,
                      load_embeddings, make_client, mcnemar_exact, save_state,
                      split_ids)

from memory_agents.benchmarks import build_benchmark
from memory_agents.envs.base import EpisodeResult
from memory_agents.memory import AvoidDistiller, ContrastiveMemory, Distiller
from memory_agents.memory.contrastive import (AGENT, FAILED, GRANULARITIES,
                                              broken_rules, rendered_as)
from memory_agents.methods.common import SAMPLING, add_run_args, typed_flags

BANK_SCHEMA = "contrastive-bank-v1"     # == ContrastiveMemory.SCHEMA
EVAL_SCHEMA = "contrastive-eval-v1"
BUILD_SCHEMA = memp.BUILD_SCHEMA        # the build file is Memp's, reused as-is

FROZEN = {
    "distill": ("build", "limit", "distill_max_tokens",
                "temperature", "top_p", "top_k", "seed"),
    "eval": ("bank", "embeddings", "arm", "top_k_memories", "split", "limit",
             "temperature", "top_p", "top_k", "max_tokens", "seed",
             "granularity", "provenance", "min_positive", "exemplars", "framing"),
}


def set_failure_log(tag: str) -> None:
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    os.environ["FORMAT_FAILURE_LOG"] = str(pathlib.Path(
        f"results/json/format_failures_contrastive_{tag}_{stamp}.jsonl"))


def report_frozen(args, cfg: dict, phase: str) -> None:
    """On resume, say which typed CLI flags are being ignored (stored values win)."""
    overridden = [k for k in FROZEN[phase]
                  if args.explicit.get(k) and getattr(args, k, None) is not None
                  and getattr(args, k) != cfg.get(k)]
    if overridden:
        print(f"[{phase}] ignoring CLI {', '.join(overridden)}: the run's stored "
              f"values win (" + ", ".join(f"{k}={cfg.get(k)}" for k in overridden)
              + ")", flush=True)


# --------------------------------------------------------------------------- #
# phase distill: Memp's build pass -> a bank of signed advice
# --------------------------------------------------------------------------- #
async def run_distill(args) -> None:
    if args.resume:
        bank_data = json.loads(pathlib.Path(args.resume).read_text())
        memory = ContrastiveMemory.from_dict(bank_data)
        cfg, out = bank_data["config"], args.resume
        report_frozen(args, cfg, "distill")
    else:
        if not args.build:
            raise SystemExit("--phase distill needs --build <memp_build_*.json> "
                             "or --resume <contrastive_bank_*.json>")
        memory = ContrastiveMemory()
        cfg = {
            "schema": BANK_SCHEMA, "phase": "distill", "build": str(args.build),
            "benchmark": bench().name, "key_fn": bench().key_fn,
            "stepgame_train": args.stepgame_train,
            "stepgame_test": args.stepgame_test,
            "key_variant": args.key_variant, "limit": args.limit or 0,
            "distill_max_tokens": args.distill_max_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "seed": args.seed,
        }
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        bm = "" if bench().name == "sparc" else f"{bench().name}_"
        out = args.out or f"results/json/contrastive_bank_{bm}{stamp}.json"
    cfg["concurrency"] = args.concurrency

    build = json.loads(pathlib.Path(cfg["build"]).read_text())
    if build["config"].get("schema") != BUILD_SCHEMA:
        raise SystemExit(f"--build expects a {BUILD_SCHEMA} file; "
                         f"{cfg['build']} is {build['config'].get('schema')!r}")
    # Bank polarity is the build actor's train accuracy; eval warns on a mismatch.
    cfg.setdefault("build_model", build["config"].get("model_id"))

    if args.verify_gold:
        bench().verify_gold("train", args.verify_gold)

    stats = memory.stats
    for key in ("n_agent", "n_failed", "n_hindsight", "empty", "failed",
                "distill_prompt_tokens", "distill_completion_tokens"):
        stats.setdefault(key, 0)

    todo = sorted(pid for pid in build["puzzles"] if pid not in memory)
    if cfg["limit"]:
        todo = todo[:max(0, cfg["limit"] - len(memory))]
    if not todo:
        print(f"[distill] {out} already covers the whole build file", flush=True)
    else:
        n_solved = sum(1 for p in todo if build["puzzles"][p]["success"])
        print(f"[distill] {len(todo)} entries to write: {n_solved} DOs from solved "
              f"puzzles, {len(todo) - n_solved} DON'Ts from failures "
              f"(+ a hindsight control text for each of those)", flush=True)

    async def distill_one(pos, neg, sem, pid):
        """All texts for one entry; the sign follows the episode outcome, and
        `hindsight` is written in the same pass as `avoid` so both arms share a bank."""
        rec = build["puzzles"][pid]
        row = bench().record("train", pid)
        key_text = bench().key_text(row)
        ctx = rec.get("episode_context") or {}
        # Mirrors Memp's routing: a crashed episode has no context to distil from.
        solved = bool(rec["success"] and ctx)
        async with sem:
            if solved:
                episode = EpisodeResult.from_reflection_context(ctx, pid)
                text, resp = await pos.distill_from_trajectory(
                    episode, key_text, row)
                return pid, AGENT, {"strategy": text}, [resp]
            rules = []
            avoid_text, avoid_resp = "", None
            if ctx.get("moves"):
                failed = bench().failed_trajectory(row, ctx)
                rules = broken_rules(failed)
                episode = EpisodeResult.from_reflection_context(failed, pid)
                avoid_text, avoid_resp = await neg.distill_from_failure(
                    episode, key_text, row, rules=rules)
            # Written even without a trajectory, so `hindsight` covers the whole bank.
            hind_text, hind_resp = await pos.distill_from_gold(key_text, row)
            return pid, FAILED, {
                "avoid": avoid_text, "hindsight": hind_text,
                "broken_rules": rules,
                "failure_reason": rec.get("terminal_reason"),
            }, [r for r in (avoid_resp, hind_resp) if r is not None]

    async def distill_pass(pids, seed):
        """One pass over `pids`; returns the pids whose reply came back empty."""
        client_pos = make_client(cfg, cfg["distill_max_tokens"], seed)
        client_neg = make_client(cfg, cfg["distill_max_tokens"], seed)
        pos = Distiller(client_pos, benchmark=bench())
        neg = AvoidDistiller(client_neg, benchmark=bench())
        cfg.setdefault("model", client_pos.model)
        cfg.setdefault("model_id",
                       os.environ.get("LLM_MODEL_ID") or client_pos.model)
        empties = []
        try:
            sem = asyncio.Semaphore(args.concurrency)
            for lo in range(0, len(pids), args.chunk):
                chunk = pids[lo:lo + args.chunk]
                outcomes = await asyncio.gather(
                    *(distill_one(pos, neg, sem, p) for p in chunk),
                    return_exceptions=True)
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        stats["failed"] += 1
                        print(f"[distill] call failed: {outcome!r}", flush=True)
                        continue
                    pid, provenance, fields, resps = outcome
                    for r in resps:
                        stats["distill_prompt_tokens"] += r.prompt_tokens
                        stats["distill_completion_tokens"] += r.completion_tokens
                    # No default-arm text would render a blank Item: retry, don't bank.
                    primary = fields.get("strategy") or fields.get("avoid")
                    if not primary:
                        empties.append(pid)
                        continue
                    row = bench().record("train", pid)
                    memory.write(pid, {
                        "provenance": provenance,
                        "difficulty": bench().difficulty(row),
                        "key_text": bench().key_text(row),
                        **fields,
                    })
                    stats["n_agent" if provenance == AGENT else "n_failed"] += 1
                    if fields.get("hindsight"):
                        stats["n_hindsight"] += 1
                save_state(out, {**memory.to_dict(), "config": cfg})
                pol = memory.polarity()
                print(f"[distill] {len(memory)} entries banked "
                      f"({pol[AGENT]} DO, {pol[FAILED]} DON'T)", flush=True)
        finally:
            await client_pos.aclose()
            await client_neg.aclose()
        return empties

    if todo:
        empties = await distill_pass(todo, cfg["seed"])
        if empties:
            # The same seed would replay the same empty reply, so the retry reseeds.
            print(f"[distill] retrying {len(empties)} empty replies at seed+1",
                  flush=True)
            empties = await distill_pass(empties, cfg["seed"] + 1)
        stats["empty"] = len(empties)
        save_state(out, {**memory.to_dict(), "config": cfg})

    # Attached here so `--exemplars on` needs no separate `--phase exemplars` pass.
    n_win, n_fail, _m = attach_exemplars(memory, build)
    cfg["exemplars_attached"] = True
    print(f"[distill] attached {n_win} winning lines and {n_fail} losing lines "
          f"(no model; --exemplars on can read this bank directly)", flush=True)
    pol = memory.polarity()
    cfg["positive_share"] = round(memory.positive_share(), 4)
    save_state(out, {**memory.to_dict(), "config": cfg})
    print(f"[distill] bank: {len(memory)} entries -- {pol[AGENT]} DO, "
          f"{pol[FAILED]} DON'T ({memory.positive_share():.1%} positive); "
          f"{stats['n_hindsight']} control texts; {stats['empty']} empty after "
          f"retry, {stats['failed']} failed. Tokens prompt="
          f"{stats['distill_prompt_tokens']:,} "
          f"completion={stats['distill_completion_tokens']:,}", flush=True)
    # A lopsided bank lets plain top-k inject almost no DOs; ARM=balanced guards this.
    if memory.positive_share() < 0.25 and len(memory):
        exp = memory.positive_share() * args.top_k_memories
        print(f"[distill] NOTE: only {memory.positive_share():.1%} of this bank is "
              f"positive, so a plain top-{args.top_k_memories} injects ~{exp:.1f} DO "
              f"entries per item. Run ARM=balanced alongside ARM=retrieved, and "
              f"quote the injected polarity with every effect size.", flush=True)
    for prov in (AGENT, FAILED):
        e = next((x for x in memory.entries if x["provenance"] == prov), None)
        if e:
            print(f"[distill] sample {prov} entry:\n"
                  f"{bench().format_contrastive([e], 'avoid')}", flush=True)
    print(f"[distill] wrote {out}", flush=True)


# --------------------------------------------------------------------------- #
# phase exemplars: attach the concrete artifacts to an existing bank (CPU, no model)
# --------------------------------------------------------------------------- #
def attach_exemplars(memory, build) -> tuple[int, int, int]:
    """Write `exemplar` and `failed_exemplar` onto every entry (no model).

    Returns (winning lines, losing lines, entries without a build record).
    """
    n_win = n_fail = missing = 0
    for e in memory.entries:
        rec = build["puzzles"].get(e["source_pid"])
        if not rec:
            missing += 1
            continue
        row = bench().record("train", e["source_pid"])
        ctx = rec.get("episode_context") or {}
        solved = bool(rec["success"] and ctx)
        # Routed exactly as distill routes provenance.
        e["exemplar"] = agent_exemplar(row, ctx) if solved else gold_exemplar(row)
        n_win += 1
        # The losing line, not the gold path: a warning carrying the solution
        # would turn `avoid` into `hindsight`.
        if not solved and ctx.get("moves"):
            e["failed_exemplar"] = agent_exemplar(row, ctx)
            n_fail += 1
    return n_win, n_fail, missing


def run_exemplars(args) -> None:
    """Add `exemplar` and `failed_exemplar` to a bank built without them.

    Leaves every text byte-identical; re-running distill would resample the prose.
    exemplar = the winning line (actor's moves if solved, else the gold path, as in
    Memp's `script`); failed_exemplar = the losing line the DON'T came from.
    """
    src = pathlib.Path(args.bank)
    data = json.loads(src.read_text())
    memory = ContrastiveMemory.from_dict(data)
    cfg = data.get("config", {})
    build_path = args.build or cfg.get("build")
    if not build_path or not pathlib.Path(build_path).exists():
        raise SystemExit(f"--phase exemplars needs the build file the bank came from "
                         f"(--build); {build_path!r} is not on disk")
    build = json.loads(pathlib.Path(build_path).read_text())
    out = pathlib.Path(args.out or src.with_name(src.stem + "_ex.json"))

    n_win, n_fail, missing = attach_exemplars(memory, build)
    save_state(out, {**memory.to_dict(), "config": {**cfg, "exemplars_attached": True}})
    print(f"[exemplars] {n_win} winning lines, {n_fail} failed lines attached"
          + (f", {missing} entries had no build record" if missing else ""), flush=True)
    gap = sum(1 for e in memory.entries
              if e["provenance"] == FAILED and not e.get("failed_exemplar"))
    if gap:
        print(f"[exemplars] NOTE: {gap} failed entries have no losing line to show "
              f"(crashed episodes); they render their warning without one", flush=True)
    if memory.entries:
        e = next((x for x in memory.entries if x.get("failed_exemplar")), memory.entries[0])
        print(f"[exemplars] sample avoid item:\n"
              f"{bench().format_contrastive([e], 'avoid', exemplars=True)}", flush=True)
    print(f"[exemplars] wrote {out}", flush=True)


# --------------------------------------------------------------------------- #
# phase eval: one retrieval-augmented pass over the test split
# --------------------------------------------------------------------------- #
def build_blocks(memory, arrays, cfg, test_ids):
    """Precompute the injected block + retrieval log for the test split.

    Retrieval is deterministic over a frozen bank, so a resumed job injects
    byte-identical blocks.
    """
    arm, k, seed = cfg["arm"], cfg["top_k_memories"], cfg["seed"]
    granularity, min_pos = cfg["granularity"], cfg.get("min_positive", 1)
    exemplars, framing = cfg.get("exemplars", False), cfg.get("framing", "mixed")
    qrow = {str(pid): i for i, pid in enumerate(arrays["test_ids"])}
    blocks, retrieved_log = {}, {}
    for pid in test_ids:
        if arm in ("retrieved", "balanced"):
            if pid not in qrow:
                raise SystemExit(f"test item {pid} has no query embedding; "
                                 f"re-run Memp's embed phase without a limit")
            vec = arrays["test_vecs"][qrow[pid]]
            picks = (memory.balanced_retrieve(vec, k, min_pos)
                     if arm == "balanced" else memory.retrieve(vec, k))
        elif arm == "random-balanced":
            # Matched to `balanced`, not `retrieved`, so only relevance changes.
            picks = memory.stratified_random_retrieve(
                k, seed_key=f"{seed}:{pid}", min_positive=min_pos)
        else:
            picks = memory.random_retrieve(k, seed_key=f"{seed}:{pid}")
        blocks[pid] = bench().format_contrastive(
            [entry for entry, _sim in picks], granularity,
            exemplars=exemplars, framing=framing)
        retrieved_log[pid] = [
            {"id": entry["id"], "source_pid": entry["source_pid"],
             "provenance": entry["provenance"],
             # How it was rendered (do/dont/withheld), not its provenance: under
             # `hindsight` a failed entry is shown as a DO.
             "shown": rendered_as(entry, granularity),
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
            "granularity": args.granularity, "provenance": args.provenance,
            "min_positive": args.min_positive,
            "exemplars": args.exemplars, "framing": args.framing,
            "benchmark": bench().name, "key_variant": args.key_variant,
            "stepgame_train": args.stepgame_train,
            "stepgame_test": args.stepgame_test,
        }
        state = {"config": cfg, "puzzles": {}}
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        bm = "" if bench().name == "sparc" else f"{bench().name}_"
        out = args.out or (f"results/json/contrastive_eval_{bm}"
                           f"{cfg['granularity']}_{cfg['arm']}_{cfg['split']}_"
                           f"{stamp}.json")
    cfg["concurrency"] = args.concurrency

    bank_data = json.loads(pathlib.Path(cfg["bank"]).read_text())
    try:
        memory = ContrastiveMemory.from_dict(bank_data)
    except ValueError as exc:
        # Usually a Memp or MemoryBank bank passed by mistake.
        raise SystemExit(f"{cfg['bank']}: {exc}. --phase eval needs a bank built "
                         f"by `run.py contrastive --phase distill`.") from None
    arrays, meta = load_embeddings(cfg["embeddings"])
    memory.embed_model = cfg["embed_model"] = meta.get("embed_model", "")
    memory = memory.filtered(cfg.get("provenance", "all"))
    if not len(memory):
        raise SystemExit(f"provenance={cfg.get('provenance')} leaves the bank empty")
    memory.attach_vectors(arrays["train_ids"], arrays["train_vecs"])
    cfg["bank_entries"] = len(memory)
    cfg["bank_model"] = bank_data.get("config", {}).get("build_model")
    cfg["positive_share_bank"] = round(memory.positive_share(), 4)
    cfg["bank_polarity"] = memory.polarity()
    # A bank built by another model is a transfer arm; its polarity describes that model.
    if cfg["bank_model"] and cfg["bank_model"] != (
            os.environ.get("LLM_MODEL_ID") or ""):
        print(f"[eval] NOTE: bank was built by {cfg['bank_model']}; its "
              f"{cfg['positive_share_bank']:.1%} positive share is that model's "
              f"train accuracy, not this actor's", flush=True)

    # A granularity whose text the bank lacks would render blank Items and silently
    # become another arm, so refuse.
    gran = cfg["granularity"]
    need = {"avoid": ("avoid", FAILED), "avoid-only": ("avoid", FAILED),
            "hindsight": ("hindsight", FAILED),
            "positive-only": ("strategy", AGENT)}[gran]
    field, side = need
    pool = [e for e in memory.entries if e["provenance"] == side]
    blank = sum(1 for e in pool if not (e.get(field) or "").strip())
    if pool and blank == len(pool):
        raise SystemExit(
            f"granularity={gran} renders the {field!r} text of {side} entries and "
            f"none of the {len(pool)} in {cfg['bank']} has one; wrong bank?")
    if blank:
        print(f"[eval] WARNING: {blank}/{len(pool)} {side} entries have no "
              f"{field!r} text and will render as a placeholder", flush=True)
    if not pool:
        print(f"[eval] WARNING: this bank has no {side} entries at all, so "
              f"granularity={gran} injects nothing but placeholders", flush=True)
    # Without exemplars in the bank this would silently duplicate the non-exemplar arm.
    if cfg.get("exemplars"):
        need = "exemplar" if gran != "avoid-only" else "failed_exemplar"
        if not any(e.get("exemplar") or e.get("failed_exemplar")
                   for e in memory.entries):
            raise SystemExit(
                f"--exemplars on, but no entry in {cfg['bank']} carries one; run "
                f"`--phase exemplars --bank {cfg['bank']}` first (CPU, no model)")
        blank = sum(1 for e in memory.entries if not e.get(need))
        if blank:
            print(f"[eval] NOTE: {blank}/{len(memory)} entries lack {need!r} and "
                  f"render their text without a concrete artifact", flush=True)

    test_ids = split_ids(cfg["split"], cfg["limit"])
    blocks, retrieved_log = build_blocks(memory, arrays, cfg, test_ids)

    longest = max(blocks.values(), key=len)
    print(f"[eval] sample injected block (for {test_ids[0]}):"
          f"{blocks[test_ids[0]]}\n", flush=True)
    # Block size is a token-budget confound against the other methods, so record it.
    cfg["mean_block_chars"] = round(sum(len(b) for b in blocks.values())
                                    / max(len(blocks), 1))
    print(f"[eval] mean block ~{cfg['mean_block_chars'] // 4} tokens, "
          f"largest ~{len(longest) // 4}", flush=True)
    if len(longest) / 4 > 6000:
        print(f"[eval] WARNING: largest block ~{len(longest) // 4} tokens; check it "
              f"still fits under MAXLEN with --max-tokens {cfg['max_tokens']}",
              flush=True)

    todo = [pid for pid in test_ids if pid not in state["puzzles"]]
    if not todo:
        print(f"[eval] {out} already covers every item; nothing to do", flush=True)
        finalize_eval(state, out)
        return
    print(f"[eval] arm={cfg['arm']} k={cfg['top_k_memories']} "
          f"granularity={gran} provenance={cfg.get('provenance')} "
          f"bank={len(memory)} entries ({cfg['positive_share_bank']:.1%} positive); "
          f"attempting {len(todo)} {cfg['split']} items at seed {cfg['seed']}",
          flush=True)

    set_failure_log(f"eval_{gran}_{cfg['arm']}")
    client = make_client(cfg, cfg["max_tokens"], cfg["seed"])
    cfg["model"] = client.model
    cfg["model_id"] = os.environ.get("LLM_MODEL_ID") or client.model
    try:
        for lo in range(0, len(todo), args.chunk):
            chunk = todo[lo:lo + args.chunk]
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
                  f"{solved} solved  [{(time.time() - t0) / 60:.0f} min/chunk]",
                  flush=True)
            save_state(out, state)
    finally:
        await client.aclose()

    finalize_eval(state, out)


def finalize_eval(state: dict, out) -> None:
    records = list(state["puzzles"].values())
    summary = accuracy_block(records)
    summary["method"] = "contrastive"
    summary["arm"] = state["config"]["arm"]
    sims = [r["retrieved"][0]["similarity"] for r in records
            if r.get("retrieved") and r["retrieved"][0]["similarity"] is not None]
    if sims:
        summary["mean_top1_similarity"] = round(sum(sims) / len(sims), 4)
    prov: dict[str, int] = {}
    shown: dict[str, int] = {}
    n_items = with_positive = injected = 0
    for rec in records:
        picks = rec.get("retrieved", [])
        if not picks:
            continue
        n_items += 1
        pos = 0
        for item in picks:
            prov[item["provenance"]] = prov.get(item["provenance"], 0) + 1
            sh = item.get("shown")
            if sh:
                shown[sh] = shown.get(sh, 0) + 1
            injected += 1
            pos += sh == "do" if sh else item["provenance"] == "agent"
        with_positive += pos > 0
    summary["injected_provenance"] = prov
    summary["injected_shown"] = shown
    # Polarity actually shown: with no DO in context, a null on `avoid` says
    # nothing about the DON'Ts.
    summary["injected_positive_share"] = (
        round(shown.get("do", prov.get("agent", 0)) / injected, 4)
        if injected else 0.0)
    summary["items_with_a_positive"] = (round(with_positive / n_items, 4)
                                        if n_items else 0.0)
    summary["polarity_basis"] = "shown" if shown else "provenance"
    cfg = state.get("config", {})
    for key in ("granularity", "provenance", "top_k_memories", "bank_entries",
                "bank_model", "positive_share_bank", "min_positive",
                "exemplars", "framing", "mean_block_chars"):
        if cfg.get(key) is not None:
            summary[key] = cfg[key]
    state["summary"] = summary
    save_state(out, state)
    print(f"[eval] {summary['success']}/{summary['n']} solved "
          f"({summary['accuracy']:.3f}) "
          f"by_difficulty={summary['accuracy_by_difficulty']}", flush=True)
    print(f"[eval] injected {summary['injected_positive_share']:.1%} DO entries; "
          f"{summary['items_with_a_positive']:.1%} of items saw at least one",
          flush=True)
    if summary["errors"]:
        print(f"[eval] WARNING: {summary['errors']} episodes crashed; the run is "
              f"incomplete", flush=True)
    print(f"[eval] wrote {out}", flush=True)


# --------------------------------------------------------------------------- #
# phase compare: baseline vs the half-arms vs the contrastive arm
# --------------------------------------------------------------------------- #
def run_compare(args) -> None:
    base_wins, base = baseline_outcomes(args.baseline)
    ret_wins, ret = eval_outcomes(args.retrieved, schema=EVAL_SCHEMA)
    rc = ret["config"]
    arms = [("baseline", base_wins, list(base["episodes"]))]
    extra = {}
    for label, path in (("hindsight", args.hindsight),
                        ("positive-only", args.positive),
                        ("avoid-random", args.random)):
        if path:
            wins, data = eval_outcomes(path, schema=EVAL_SCHEMA)
            extra[label] = wins
            arms.append((label, wins, list(data["puzzles"].values())))
            for key in ("bank", "top_k_memories", "seed", "split"):
                if rc.get(key) != data["config"].get(key):
                    print(f"WARNING: {label} and the contrastive arm disagree on "
                          f"{key}: {data['config'].get(key)} vs {rc.get(key)}",
                          flush=True)
    arms.append(("avoid", ret_wins, list(ret["puzzles"].values())))
    mp_wins = None
    if args.memp:
        # Reference only: Memp's renderer differs, so not a controlled contrast.
        mp_wins, mp = eval_outcomes(args.memp, schema=memp.EVAL_SCHEMA)
        arms.append(("memp-script", mp_wins, list(mp["puzzles"].values())))

    bs = base.get("summary", {})
    for key in ("split", "seed", "temperature", "max_tokens"):
        if bs.get(key) != rc.get(key):
            print(f"WARNING: baseline and contrastive disagree on {key}: "
                  f"{bs.get(key)} vs {rc.get(key)} -- not comparable", flush=True)

    common = set(base_wins) & set(ret_wins)
    for _label, wins, _recs in arms:
        common &= set(wins)
    common = sorted(common)

    lines = [
        "=" * 96,
        f"  CONTRASTIVE vs BASELINE over {len(common)} items "
        f"({rc.get('split')}, {rc.get('model_id', rc.get('model'))}, "
        f"k={rc.get('top_k_memories')}, granularity={rc.get('granularity')}, "
        f"arm={rc.get('arm')})",
        f"  bank: {rc.get('bank')}  "
        f"({rc.get('bank_polarity')}, "
        f"{rc.get('positive_share_bank')} positive)",
        "=" * 96, "",
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

    lines += ["  paired tests over the common items (discordant pairs only)", ""]
    lines.append(pair("avoid", ret_wins, "baseline", base_wins))
    for label in ("hindsight", "positive-only", "avoid-random"):
        if label in extra:
            lines.append(pair(label, extra[label], "baseline", base_wins))
    # avoid vs hindsight differ only in what the failed entries contribute.
    if "hindsight" in extra:
        lines += ["", "  THE HEADLINE TEST: learning from the failure vs from the "
                  "gold path", ""]
        lines.append(pair("avoid", ret_wins, "hindsight", extra["hindsight"]))
    if "positive-only" in extra:
        lines += ["", "  what the non-solved entries contribute at all", ""]
        lines.append(pair("avoid", ret_wins, "positive-only",
                          extra["positive-only"]))
        if "hindsight" in extra:
            lines.append(pair("hindsight", extra["hindsight"], "positive-only",
                              extra["positive-only"]))
    if "avoid-random" in extra:
        lines += ["", "  relevance", ""]
        lines.append(pair("avoid", ret_wins, "avoid-random", extra["avoid-random"]))
    if mp_wins is not None:
        lines += ["", "  external reference (Memp's own renderer, not a controlled "
                  "contrast -- use `hindsight` for that)", ""]
        lines.append(pair("avoid", ret_wins, "memp-script", mp_wins))
    lines.append("")

    lines += ["  injected polarity (read every result above against this)", "",
              f"  {'run':<16} {'DO share':>10} {'items with >=1 DO':>20}",
              "  " + "-" * 50]
    for label, path in (("avoid", args.retrieved), ("hindsight", args.hindsight),
                        ("positive-only", args.positive),
                        ("avoid-random", args.random)):
        if not path:
            continue
        sm = json.loads(pathlib.Path(path).read_text()).get("summary", {})
        lines.append(f"  {label:<16} "
                     f"{sm.get('injected_positive_share', 0):>10.1%} "
                     f"{sm.get('items_with_a_positive', 0):>20.1%}")
    lines.append("")

    reasons = ["solved", "wrong-solution", "dead-end", "format-failure", "error"]
    if any(r["terminal_reason"] in ("wrong-solution", "dead-end")
           for _l, _w, recs in arms for r in recs):
        lines += ["  failure modes (the mechanism-level test)", "",
                  "  " + f"{'run':<16}" + "".join(f"{r:>16}" for r in reasons),
                  "  " + "-" * 92]
        for label, _wins, recs in arms:
            counts = {r: 0 for r in reasons}
            for rec in recs:
                counts[rec["terminal_reason"]] = counts.get(
                    rec["terminal_reason"], 0) + 1
            lines.append("  " + f"{label:<16}"
                         + "".join(f"{counts.get(r, 0):>16}" for r in reasons))
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out != "-":
        dest = pathlib.Path(args.out or "results/txt/contrastive_vs_baseline.txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(report)
        print(f"[written] {dest}", file=sys.stderr)


# --------------------------------------------------------------------------- #
async def main(argv: list[str], prog: str) -> None:
    ap = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True,
                    choices=["distill", "exemplars", "eval", "compare"])
    add_run_args(ap, **SAMPLING,
                 split_help="eval split (the bank always comes from train)",
                 seed_help="decoding seed; keep equal to the baseline's for eval")
    ap.add_argument("--chunk", type=int, default=100,
                    help="items per checkpoint in eval, entries per checkpoint "
                         "in distill")
    ap.add_argument("--distill-max-tokens", type=int, default=12288,
                    help="budget for the strategy / avoid / hindsight texts")
    ap.add_argument("--top-k-memories", type=int, default=3,
                    help="entries injected per test item; 3 matches Memp and "
                         "MemoryBank. Use 2 for the token-matched control")
    ap.add_argument("--arm",
                    choices=["retrieved", "balanced", "random", "random-balanced"],
                    help="how entries are chosen. retrieved = cosine top-k; "
                         "balanced = top-k but with at least --min-positive DO "
                         "entries, which on a weak actor is the difference between "
                         "testing negative advice and testing its absence; random "
                         "and random-balanced are the matched relevance ablations")
    ap.add_argument("--min-positive", type=int, default=1,
                    help="balanced arms: DO entries guaranteed in each block")
    ap.add_argument("--exemplars", default="off", choices=["on", "off"],
                    help="eval: append the concrete artifact to each item -- a winning "
                         "line to a DO, the losing line to a DON'T. Needs a bank run "
                         "through --phase exemplars. Memp's `script` granularity does "
                         "the same")
    ap.add_argument("--framing", default="mixed", choices=["mixed", "matched"],
                    help="eval: `mixed` gives every arm the two-sign header (a mismatch "
                         "for `hindsight`, whose items are all DO); `matched` gives "
                         "each arm a header "
                         "describing its own content")
    ap.add_argument("--granularity", default="avoid", choices=list(GRANULARITIES),
                    help="what a NON-SOLVED train puzzle contributes -- this IS the "
                         "experiment. avoid = a DON'T distilled from its failure "
                         "(the method); hindsight = a procedure distilled from its "
                         "gold path (Memp's policy, the control); positive-only = "
                         "nothing; avoid-only = the DON'T, with the DOs withheld")
    ap.add_argument("--provenance", default="all", choices=["all", "agent", "failed"],
                    help="eval: restrict the bank to one sign. Changes bank SIZE as "
                         "well as content, so it is not a clean single-factor arm -- "
                         "prefer --granularity positive-only / avoid-only, which "
                         "hold the bank at 500")
    ap.add_argument("--build", default=None,
                    help="memp_build_train_*.json (distill phase)")
    ap.add_argument("--bank", default=None, help="bank JSON (eval phase)")
    ap.add_argument("--embeddings", default=None,
                    help="memp_embeddings_*.npz, reused verbatim (eval)")
    ap.add_argument("--resume", default=None, help="state file to continue in place")
    ap.add_argument("--verify-gold", type=int, default=10, metavar="N",
                    help="distill: replay N gold paths through the env first "
                         "(0 disables)")
    ap.add_argument("--baseline", default=None, help="baseline JSON (compare)")
    ap.add_argument("--retrieved", default=None,
                    help="the `avoid` arm's eval JSON (compare)")
    ap.add_argument("--hindsight", default=None,
                    help="the `hindsight` arm (compare); THE CONTROL")
    ap.add_argument("--positive", default=None,
                    help="the `positive-only` arm (compare)")
    ap.add_argument("--random", default=None,
                    help="the random-retrieval ablation (compare)")
    ap.add_argument("--memp", default=None,
                    help="a memp_eval_*.json to reference against (compare)")
    args = ap.parse_args(argv)

    typed = typed_flags(argv)
    args.explicit = {k: k in typed for keys in FROZEN.values() for k in keys}
    args.limit = args.limit if args.limit > 0 else None
    args.exemplars = args.exemplars == "on"

    stored = memp._resumed_config(args)
    memp.BENCH = build_benchmark(stored if stored else args)
    if stored and stored.get("benchmark", "sparc") != getattr(args, "benchmark",
                                                             "sparc"):
        print(f"[contrastive] resuming a {stored.get('benchmark', 'sparc')} run; "
              f"ignoring --benchmark {args.benchmark}", flush=True)

    if args.phase == "distill":
        await run_distill(args)
    elif args.phase == "exemplars":
        if not args.bank:
            raise SystemExit("--phase exemplars needs --bank <contrastive_bank_*.json>")
        run_exemplars(args)
    elif args.phase == "eval":
        await run_eval(args)
    else:
        if not (args.baseline and args.retrieved):
            ap.error("--phase compare needs --baseline and --retrieved "
                     "(--hindsight is the control and is strongly recommended)")
        run_compare(args)

