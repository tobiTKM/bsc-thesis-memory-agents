"""MemoryBank (Zhong et al., 2024): a frozen declarative bank of past cases built on the
train split and retrieved for the test split. Each entry holds an LLM-written case note,
the verbatim record (its solution replayed through the env) and the compact exemplar.
The train build pass and the embeddings are Memp's and are reused, not re-run; both
methods key on the same puzzle_key_text, so retrieval is identical between them.

  distill  memp_build_train_*.json -> case notes + verbatim records -> memorybank_bank_*.json
           (--rules-preamble on|off picks the prompt variant).
  eval     cosine top-k (or seeded-random k) cases per test puzzle, one pass over the split.
  compare  no GPU: baseline vs retrieved vs random (McNemar exact), optionally vs a Memp eval.

  sbatch -J mb-dist --export=ALL,DRIVER=memorybank,PHASE=distill,RULES_PREAMBLE=on,BUILD=results/json/memp_build_train_<tag>.json scripts/serve_and_run.sbatch
  sbatch -J mb-ret  --export=ALL,DRIVER=memorybank,PHASE=eval,ARM=retrieved,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
  sbatch -J mb-rnd  --export=ALL,DRIVER=memorybank,PHASE=eval,ARM=random,BANK=<bank.json>,EMBEDDINGS=<emb.npz> scripts/serve_and_run.sbatch
  python scripts/run.py memorybank --phase compare --baseline <root.json> --retrieved <ret.json> --random <rnd.json> [--memp <memp_eval_retrieved_*.json>]
"""

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from datetime import datetime


# Run plumbing is shared with Memp verbatim, so it is imported rather than mirrored.
from memory_agents.methods import memp
from memory_agents.methods.memp import (accuracy_block, agent_exemplar, baseline_outcomes, bench,
                      compare_row, episode_record, eval_outcomes, gold_exemplar,
                      load_embeddings, make_client, mcnemar_exact,
                      save_state, split_ids)

from memory_agents.benchmarks import build_benchmark
from memory_agents.envs.base import EpisodeResult
from memory_agents.memory import CaseDistiller, MemoryBankMemory
from memory_agents.methods.common import SAMPLING, add_run_args, typed_flags

BANK_SCHEMA = "memorybank-bank-v1"      # == MemoryBankMemory.SCHEMA
EVAL_SCHEMA = "memorybank-eval-v1"
BUILD_SCHEMA = memp.BUILD_SCHEMA        # the build file is Memp's, reused as-is

# Fixed per run: on --resume these come from the state file, never the CLI.
FROZEN = {
    "distill": ("build", "rules_preamble", "limit", "distill_max_tokens",
                "temperature", "top_p", "top_k", "seed"),
    "eval": ("bank", "embeddings", "arm", "top_k_memories", "split", "limit",
             "temperature", "top_p", "top_k", "max_tokens", "seed",
             "granularity", "provenance"),
}


def set_failure_log(tag: str) -> None:
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    os.environ["FORMAT_FAILURE_LOG"] = str(pathlib.Path(
        f"results/json/format_failures_memorybank_{tag}_{stamp}.jsonl"))


def report_frozen(args, cfg: dict, phase: str) -> None:
    """On resume, report which typed CLI flags are ignored (stored values win).

    A local copy of Memp's, because it must read this module's FROZEN table.
    """
    overridden = [k for k in FROZEN[phase]
                  if args.explicit.get(k) and getattr(args, k, None) is not None
                  and getattr(args, k) != cfg.get(k)]
    if overridden:
        print(f"[{phase}] ignoring CLI {', '.join(overridden)}: the run's stored "
              f"values win (" + ", ".join(f"{k}={cfg.get(k)}" for k in overridden)
              + ")", flush=True)


# --------------------------------------------------------------------------- #
# phase distill: Memp's build pass -> case bank (notes + verbatim records)
# --------------------------------------------------------------------------- #
async def run_distill(args) -> None:
    if args.resume:
        bank_data = json.loads(pathlib.Path(args.resume).read_text())
        memory = MemoryBankMemory.from_dict(bank_data)
        cfg, out = bank_data["config"], args.resume
        report_frozen(args, cfg, "distill")
    else:
        if not args.build:
            raise SystemExit("--phase distill needs --build <memp_build_*.json> "
                             "or --resume <memorybank_bank_*.json>")
        memory = MemoryBankMemory()
        cfg = {
            "schema": BANK_SCHEMA, "phase": "distill", "build": str(args.build),
            "benchmark": bench().name, "key_fn": bench().key_fn,
            "stepgame_train": args.stepgame_train,
            "stepgame_test": args.stepgame_test,
            "key_variant": args.key_variant,
            "rules_preamble": args.rules_preamble,
            "limit": args.limit or 0,
            "distill_max_tokens": args.distill_max_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "seed": args.seed,
        }
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        # Variant in the filename: both banks coexist and eval jobs pick one by path.
        tag = "rp" if args.rules_preamble else "norp"
        bm = "" if bench().name == "sparc" else f"{bench().name}_"
        out = args.out or f"results/json/memorybank_bank_{bm}{tag}_{stamp}.json"
    cfg["concurrency"] = args.concurrency

    build = json.loads(pathlib.Path(cfg["build"]).read_text())
    if build["config"].get("schema") != BUILD_SCHEMA:
        raise SystemExit(f"--build expects a {BUILD_SCHEMA} file; "
                         f"{cfg['build']} is {build['config'].get('schema')!r}")

    if args.verify_gold:
        bench().verify_gold("train", args.verify_gold)

    todo = sorted(pid for pid in build["puzzles"] if pid not in memory)
    if cfg["limit"]:
        todo = todo[:max(0, cfg["limit"] - len(memory))]
    stats = memory.stats
    for key in ("n_agent", "n_gold", "empty", "failed",
                "distill_prompt_tokens", "distill_completion_tokens"):
        stats.setdefault(key, 0)

    if todo:
        print(f"[distill] {len(todo)} puzzles to write "
              f"({sum(1 for p in todo if build['puzzles'][p]['success'])} from agent "
              f"trajectories, the rest from gold paths); "
              f"rules_preamble={cfg['rules_preamble']}", flush=True)

    async def distill_one(distiller, sem, pid):
        rec = build["puzzles"][pid]
        row = bench().record("train", pid)
        key_text = bench().key_text(row)
        # Route on provenance as Memp does; a crashed episode has no episode_context.
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
        distiller = CaseDistiller(
            make_client(cfg, cfg["distill_max_tokens"], seed),
            rules_preamble=cfg["rules_preamble"], benchmark=bench())
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
                        # From the dataset: a crashed episode reports -1.
                        "difficulty": bench().difficulty(row),
                        "key_text": bench().key_text(row),
                        "case_note": text,
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

    if todo:
        empties = await distill_pass(todo, cfg["seed"])
        if empties:
            # The same prompt at the same seed replays the same empty reply.
            print(f"[distill] retrying {len(empties)} empty replies at seed+1",
                  flush=True)
            empties = await distill_pass(empties, cfg["seed"] + 1)
        stats["empty"] = len(empties)
        save_state(out, {**memory.to_dict(), "config": cfg})

    # The verbatim record is half of every entry, so distill always attaches it.
    attach_records(memory, out, cfg)

    print(f"[distill] bank: {len(memory)} entries "
          f"({stats['n_agent']} agent, {stats['n_gold']} gold, "
          f"{stats['empty']} empty after retry, {stats['failed']} failed); "
          f"distill tokens prompt={stats['distill_prompt_tokens']:,} "
          f"completion={stats['distill_completion_tokens']:,}", flush=True)
    if memory.entries:
        e = memory.entries[0]
        print(f"[distill] sample case note ({e['id']}, {e['provenance']}):\n"
              f"  {e['case_note']}", flush=True)
    print(f"[distill] wrote {out}", flush=True)


def attach_records(memory, out, cfg) -> None:
    """Attach each entry's verbatim record; idempotent, so a resume adds only missing ones.

    Interactive benchmarks replay the entry's exemplar (agent or gold moves) through
    the env; single-turn ones use the item's question plus its worked derivation.
    """
    todo = [e for e in memory.entries if not e.get("trajectory")]
    if not todo:
        return
    print(f"[records] replaying {len(todo)} solutions for the verbatim records",
          flush=True)
    interactive = bench().interactive
    env = None
    if interactive:
        from memory_agents.envs.sparc import SparcEnv, replay_path
        env = SparcEnv(split="train")
    attached = failed = solved = 0
    try:
        for e in todo:
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
            except Exception as exc:      # a bad exemplar must not kill the run
                print(f"[records] {e['id']} ({e['source_pid']}): {exc!r}", flush=True)
                failed += 1
                continue
            e["trajectory"] = traj
            attached += 1
            solved += traj["terminal_reason"] == "solved"
    finally:
        if env is not None:
            env.close()
    memory.stats["records_attached"] = memory.stats.get("records_attached", 0) + attached
    memory.stats["records_failed"] = memory.stats.get("records_failed", 0) + failed
    save_state(out, {**memory.to_dict(), "config": cfg})
    print(f"[records] {attached} attached ({solved} replays reached a solved "
          f"state), {failed} failed", flush=True)
    if attached and solved < attached:
        print(f"[records] NOTE: {attached - solved} replays did not end solved; "
              f"those entries' exemplars were not winning paths", flush=True)


# --------------------------------------------------------------------------- #
# phase eval: one retrieval-augmented pass over the test split
# --------------------------------------------------------------------------- #
def build_blocks(memory, arrays, cfg, test_ids):
    """Precompute each test puzzle's injected block and retrieval log.

    Retrieval over a frozen bank is deterministic, so a resumed job injects identical blocks.
    """
    arm, k, seed = cfg["arm"], cfg["top_k_memories"], cfg["seed"]
    granularity = cfg.get("granularity", "case-record")
    qrow = {str(pid): i for i, pid in enumerate(arrays["test_ids"])}
    blocks, retrieved_log = {}, {}
    for pid in test_ids:
        if arm == "retrieved":
            if pid not in qrow:
                raise SystemExit(f"test puzzle {pid} has no query embedding; "
                                 f"re-run Memp's embed phase without a limit")
            picks = memory.retrieve(arrays["test_vecs"][qrow[pid]], k)
        else:
            picks = memory.random_retrieve(k, seed_key=f"{seed}:{pid}")
        blocks[pid] = bench().format_cases([entry for entry, _sim in picks],
                                   granularity=granularity)
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
            "granularity": args.granularity, "provenance": args.provenance,
            "benchmark": bench().name, "key_variant": args.key_variant,
            "stepgame_train": args.stepgame_train,
            "stepgame_test": args.stepgame_test,
        }
        state = {"config": cfg, "puzzles": {}}
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        bm = "" if bench().name == "sparc" else f"{bench().name}_"
        out = args.out or (f"results/json/memorybank_eval_{bm}{cfg['arm']}_"
                           f"{cfg['split']}_{stamp}.json")
    cfg["concurrency"] = args.concurrency

    bank_data = json.loads(pathlib.Path(cfg["bank"]).read_text())
    try:
        memory = MemoryBankMemory.from_dict(bank_data)
    except ValueError as exc:
        # Usually a Memp bank passed by mistake.
        raise SystemExit(f"{cfg['bank']}: {exc}. --phase eval needs a bank built "
                         f"by `run.py memorybank --phase distill`.") from None
    arrays, meta = load_embeddings(cfg["embeddings"])
    memory.embed_model = cfg["embed_model"] = meta.get("embed_model", "")
    memory = memory.filtered(cfg.get("provenance", "all"))
    if not len(memory):
        raise SystemExit(f"provenance={cfg.get('provenance')} leaves the bank empty")
    memory.attach_vectors(arrays["train_ids"], arrays["train_vecs"])
    cfg["bank_entries"] = len(memory)
    bank_cfg = bank_data.get("config", {})
    cfg["bank_model"] = bank_cfg.get("model_id")
    # Records which ablation bank the eval used.
    cfg["rules_preamble"] = bank_cfg.get("rules_preamble")

    if cfg.get("granularity", "case-record") == "case-record":
        missing = sum(1 for e in memory.entries if not e.get("trajectory"))
        if missing:
            raise SystemExit(
                f"granularity=case-record needs the verbatim records but "
                f"{missing}/{len(memory)} entries lack one; re-run --phase distill "
                f"--resume {cfg['bank']} to attach them")

    test_ids = split_ids(cfg["split"], cfg["limit"])
    blocks, retrieved_log = build_blocks(memory, arrays, cfg, test_ids)

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
          f"rules_preamble={cfg.get('rules_preamble')} bank={len(memory)} entries; "
          f"attempting {len(todo)} {cfg['split']} puzzles at seed {cfg['seed']}",
          flush=True)

    set_failure_log(f"eval_{cfg['arm']}")
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
    summary["method"] = "memorybank"
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
    for key in ("granularity", "provenance", "rules_preamble", "top_k_memories",
                "bank_entries", "bank_model"):
        if cfg.get(key) is not None:
            summary[key] = cfg[key]
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
# phase compare: baseline vs retrieved vs random (+ optional vs Memp)
# --------------------------------------------------------------------------- #
def run_compare(args) -> None:
    base_wins, base = baseline_outcomes(args.baseline)
    ret_wins, ret = eval_outcomes(args.retrieved, schema=EVAL_SCHEMA)
    arms = [("baseline", base_wins, list(base["episodes"])),
            ("memorybank-retr", ret_wins, list(ret["puzzles"].values()))]
    rnd = None
    if args.random:
        rnd_wins, rnd = eval_outcomes(args.random, schema=EVAL_SCHEMA)
        arms.insert(1, ("memorybank-rand", rnd_wins, list(rnd["puzzles"].values())))
    mp = None
    if args.memp:
        # Same retrieval and k; only case vs procedure memory differs.
        mp_wins, mp = eval_outcomes(args.memp, schema=memp.EVAL_SCHEMA)
        arms.append(("memp-retrieved", mp_wins, list(mp["puzzles"].values())))

    bs, rc = base.get("summary", {}), ret["config"]
    for key in ("split", "seed", "temperature", "max_tokens"):
        if bs.get(key) != rc.get(key):
            print(f"WARNING: baseline and retrieved disagree on {key}: "
                  f"{bs.get(key)} vs {rc.get(key)} -- not comparable", flush=True)
    if rnd is not None:
        for key in ("bank", "top_k_memories", "seed", "split", "granularity"):
            if rc.get(key) != rnd["config"].get(key):
                print(f"WARNING: retrieved and random disagree on {key}: "
                      f"{rc.get(key)} vs {rnd['config'].get(key)}", flush=True)
    if mp is not None:
        for key in ("seed", "split", "top_k_memories", "max_tokens"):
            if rc.get(key) != mp["config"].get(key):
                print(f"WARNING: memorybank and memp disagree on {key}: "
                      f"{rc.get(key)} vs {mp['config'].get(key)}", flush=True)

    common = set(base_wins) & set(ret_wins)
    for _label, wins, _recs in arms:
        common &= set(wins)
    common = sorted(common)

    lines = [
        "=" * 96,
        f"  MEMORYBANK vs BASELINE over {len(common)} puzzles "
        f"({rc.get('split')}, {rc.get('model_id', rc.get('model'))}, "
        f"k={rc.get('top_k_memories')}, granularity={rc.get('granularity')}, "
        f"rules_preamble={rc.get('rules_preamble')})",
        f"  bank: {rc.get('bank')}",
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
        lines.append(pair("random", rnd_wins, "baseline", base_wins))
    if mp is not None:
        lines.append(pair("memorybank", ret_wins, "memp", mp_wins))
    lines.append("")

    # Reported apart from eval cost: it amortises over every eval against this bank.
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
            "  (the train build pass and the embed job are shared with Memp and "
            "were not re-run)",
            "",
        ]

    report = "\n".join(lines)
    print(report)
    if args.out != "-":
        dest = pathlib.Path(args.out or "results/txt/memorybank_vs_baseline.txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(report)
        print(f"[written] {dest}", file=sys.stderr)


# --------------------------------------------------------------------------- #
async def main(argv: list[str], prog: str) -> None:
    ap = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["distill", "eval", "compare"])
    add_run_args(ap, **SAMPLING,
                 split_help="eval split (the bank always comes from train)",
                 seed_help="decoding seed; keep equal to the baseline's for eval")
    ap.add_argument("--chunk", type=int, default=100,
                    help="puzzles per checkpoint in eval, calls per checkpoint "
                         "in distill")
    ap.add_argument("--distill-max-tokens", type=int, default=12288,
                    help="case-note budget (distill phase)")
    ap.add_argument("--top-k-memories", type=int, default=3,
                    help="cases injected per test puzzle (eval); 3 matches Memp")
    ap.add_argument("--arm", choices=["retrieved", "random"],
                    help="eval retrieval policy: cosine top-k, or the "
                         "seeded-random ablation that isolates relevance")
    ap.add_argument("--granularity", default="case-record",
                    choices=["case-record", "case"],
                    help="what rides along with the case note: the verbatim "
                         "record (the paper's in-depth storage, default) or the "
                         "compact exemplar")
    ap.add_argument("--provenance", default="all", choices=["all", "agent", "gold"],
                    help="eval: restrict the bank by where its cases came from")
    ap.add_argument("--rules-preamble", default="on", choices=["on", "off"],
                    help="distill: prepend the verbatim rulebook to the case-note "
                         "prompts (the ablation; frozen into the bank)")
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
    ap.add_argument("--retrieved", default=None, help="retrieved-arm JSON (compare)")
    ap.add_argument("--random", default=None, help="random-arm JSON (compare)")
    ap.add_argument("--memp", default=None,
                    help="a memp_eval_*.json to contrast against (compare)")
    args = ap.parse_args(argv)

    # Typed flags, so a resume can tell a default from an explicit value.
    typed = typed_flags(argv)
    args.explicit = {k: k in typed for keys in FROZEN.values() for k in keys}
    args.limit = args.limit if args.limit > 0 else None
    args.rules_preamble = args.rules_preamble == "on"

    # Set on the memp module, whose imported plumbing reads memp.BENCH; on --resume
    # it comes from the state file so prompts match the partial bank.
    stored = memp._resumed_config(args)
    memp.BENCH = build_benchmark(stored if stored else args)
    if stored and stored.get("benchmark", "sparc") != getattr(args, "benchmark",
                                                             "sparc"):
        print(f"[memorybank] resuming a {stored.get('benchmark', 'sparc')} run; "
              f"ignoring --benchmark {args.benchmark}", flush=True)

    if args.phase == "distill":
        await run_distill(args)
    elif args.phase == "eval":
        await run_eval(args)
    else:
        if not (args.baseline and args.retrieved):
            ap.error("--phase compare needs --baseline and --retrieved "
                     "(--random is optional but recommended)")
        run_compare(args)

