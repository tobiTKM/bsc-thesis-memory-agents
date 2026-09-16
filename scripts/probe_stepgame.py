#!/usr/bin/env python
"""Adversarial probe: is the StepGame baseline accuracy real, or an artefact?

gpt-oss-120b scored 994/1000 on the clean test split -- higher than any *native*
LLM reasoner in Li et al. (AAAI-24), whose GPT-4 base row collapses 100 -> 25 from
k=1 to k=10, and level with their neuro-symbolic Map+ASP / Davinci+ASP rows. A
static audit of the pipeline (prompt contents, parser, grader, gold, retry policy)
found nothing wrong, so this settles it empirically instead.

Four arms over the SAME sampled items, one call each, replies saved:

  full       the production prompt. Replicates the baseline on this subsample and
             anchors the other three. Expect ~the baseline number.
  bare       a minimal system prompt: the task, the nine labels, the output
             contract, and nothing else. The production prompt also states the
             direction convention ("A relative to B") and what `overlap` means,
             which is the one place our setup is more generous than a plain
             input-output prompt. This measures what that generosity is worth.
  renamed    entity letters permuted by a fixed random bijection, so every story
             is a string that appears in no corpus, while the reasoning task and
             the gold label are identical. Contamination test: memorised answers
             break, real reasoning does not.
  scrambled  the story is replaced by a DIFFERENT item's story of the same k; the
             question, and the gold, are kept. The queried entities are then
             almost never present, so the item is unanswerable. THE LEAK
             DETECTOR: anything above the ~11% floor here means the answer is
             reaching the model by a path other than the story.

Run it the way any driver runs, against a served model:

  sbatch -J sg-probe -G A100:4 -t 02:00:00 \
    --export=ALL,BENCHMARK=stepgame,DRIVER=probe,MODEL=openai/gpt-oss-120b,TP=4,\
REASONING_PARSER=openai_gptoss scripts/serve_and_run.sbatch

or directly, with LLM_BASE_URL / LLM_MODEL already pointing at one.
"""

import argparse
import asyncio
import json
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents.benchmarks import add_benchmark_args, build_benchmark
from memory_agents.config import resolve_server
from memory_agents.envs import stepgame as SG
from memory_agents.llm.client import LLMClient

LABELS = "above, below, left, right, upper-left, upper-right, lower-left, lower-right, overlap"

# Deliberately minimal: the task, the label set, the output contract. No direction
# convention, no note on `overlap`, no "only direction matters". Everything the
# production prompt says beyond this is what the `bare` arm removes.
BARE_SYSTEM = f"""You are solving a spatial reasoning problem.

You are given a story describing where objects sit relative to one another, and a
question asking for the spatial relation between two of them.

The answer is exactly one of nine relations:
{LABELS}

You MAY think step-by-step, but you MUST end your reply with:
Final: <label>
where <label> is exactly one of: {LABELS}
"""

_ENT = re.compile(r"\b([A-Z])\b")


def rename(text: str, mapping: dict) -> str:
    return _ENT.sub(lambda m: mapping.get(m.group(1), m.group(1)), text)


def build_arms(items, n_per_k, seed):
    """Sample n_per_k items per file-k and build the four variants of each."""
    rng = random.Random(seed)
    by_k = defaultdict(list)
    for it in items.values():
        by_k[it.k].append(it)
    sample = []
    for k in sorted(by_k):
        pool = sorted(by_k[k], key=lambda x: x.id)
        sample.extend(rng.sample(pool, min(n_per_k, len(pool))))

    # A donor story of the same k, from a DIFFERENT item, for the scrambled arm.
    donors = {}
    for it in sample:
        pool = [o for o in by_k[it.k] if o.id != it.id]
        donors[it.id] = rng.choice(pool)

    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    cases = []
    for it in sample:
        shuffled = letters[:]
        rng.shuffle(shuffled)
        mapping = dict(zip(letters, shuffled))
        donor = donors[it.id]
        # Entities are sampled from 26 letters, so a donor story sometimes happens
        # to contain both queried entities. Those items are answerable -- just
        # against a different story, so the original gold is right only by chance.
        # Flagged rather than resampled, so the leak test can be read both ways.
        d_ents = set(_ENT.findall(" ".join(donor.story)))
        q_ents = set(re.findall(r"\bagent ([A-Z])\b", it.question))
        cases.append({
            "id": it.id, "k": it.k, "hops": it.hops, "gold": it.gold,
            "scrambled_entities_present": bool(q_ents) and q_ents <= d_ents,
            "arms": {
                "full":      (it.story, it.question, "prod"),
                "bare":      (it.story, it.question, "bare"),
                "renamed":   ([rename(s, mapping) for s in it.story],
                              rename(it.question, mapping), "prod"),
                "scrambled": (donor.story, it.question, "prod"),
            },
        })
    return cases


def render(story, question) -> str:
    lines = "\n".join(f"{i}. {s}" for i, s in enumerate(story, 1))
    return f"Story:\n{lines}\n\n{question}"


async def main():
    ap = argparse.ArgumentParser()
    add_benchmark_args(ap)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-k", type=int, default=20, help="items sampled per file-k")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="full,bare,renamed,scrambled")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench = build_benchmark(args)
    if bench.name != "stepgame":
        raise SystemExit("this probe is StepGame-specific (--benchmark stepgame)")
    prod_system = bench.system_prompt(None)
    systems = {"prod": prod_system, "bare": BARE_SYSTEM}

    items = bench.items(args.split)
    cases = build_arms(items, args.per_k, args.seed)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    base_url, model = resolve_server()
    client = LLMClient(base_url=base_url, model=model, max_tokens=args.max_tokens,
                       temperature=args.temperature, top_p=args.top_p,
                       top_k=args.top_k, seed=args.seed)
    print(f"server={base_url} model={model} arms={arms} "
          f"items={len(cases)} calls={len(cases) * len(arms)}", flush=True)

    sem = asyncio.Semaphore(max(args.concurrency, 1))
    results = []

    async def one(case, arm):
        story, question, sys_key = case["arms"][arm]
        async with sem:
            try:
                resp = await client.chat([
                    {"role": "system", "content": systems[sys_key]},
                    {"role": "user", "content": render(story, question)},
                ])
            except Exception as exc:                       # noqa: BLE001
                return {"id": case["id"], "arm": arm, "error": repr(exc),
                        "predicted": None, "success": False}
        predicted = SG.parse_answer(resp.text)
        # The reasoning trace is where a leak would be visible, so keep a tail of
        # it -- but only for the arm that is supposed to be unanswerable, since it
        # is thousands of tokens and we do not need it for the others.
        trace = (resp.reasoning_text or "")[-1200:] if arm == "scrambled" else ""
        return {
            "id": case["id"], "arm": arm, "k": case["k"], "hops": case["hops"],
            "gold": case["gold"], "predicted": predicted,
            "success": predicted is not None and predicted == case["gold"],
            "format_failure": predicted is None,
            "scrambled_entities_present": case["scrambled_entities_present"],
            "finish_reason": resp.finish_reason,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "reply": (resp.text or "")[-600:],
            "reasoning_tail": trace,
        }

    t0 = time.time()
    tasks = [asyncio.create_task(one(c, a)) for c in cases for a in arms]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        results.append(await fut)
        if i % 100 == 0:
            print(f"  {i}/{len(tasks)} calls done", flush=True)
    await client.aclose()
    elapsed = time.time() - t0

    # ---- report -----------------------------------------------------------
    print(f"\n{'arm':<12} {'n':>5} {'solved':>7} {'acc':>7} {'fmt_fail':>9}   interpretation")
    floor = 1 / 9
    summary = {}
    for arm in arms:
        rows = [r for r in results if r["arm"] == arm]
        n = len(rows)
        ok = sum(1 for r in rows if r["success"])
        ff = sum(1 for r in rows if r.get("format_failure"))
        acc = ok / n if n else float("nan")
        note = {
            "full": "anchor: should match the 1000-item baseline",
            "bare": "drop vs full = what the prompt's extra guidance buys",
            "renamed": "should equal full; a drop means memorisation",
            "scrambled": ("AT FLOOR -> no leak, pipeline sound"
                          if acc < 0.2 else "ABOVE FLOOR -> LEAK, investigate"),
        }.get(arm, "")
        summary[arm] = {"n": n, "solved": ok, "accuracy": acc, "format_failures": ff}
        print(f"{arm:<12} {n:>5} {ok:>7} {acc:>7.3f} {ff:>9}   {note}")
    print(f"\nchance floor = {floor:.3f} (1/9)   elapsed = {elapsed:.0f}s")

    # The clean read of the leak test: items whose donor story does not even
    # mention the queried entities, so no honest answer exists.
    unans = [r for r in results
             if r["arm"] == "scrambled" and not r.get("scrambled_entities_present")]
    if unans:
        ok = sum(1 for r in unans if r["success"])
        print(f"scrambled, queried entities ABSENT from the story: "
              f"{ok}/{len(unans)} = {ok / len(unans):.3f}  "
              f"(no honest answer exists; anything well above {floor:.3f} is a leak)")

    print("\nper-k accuracy:")
    print(f"{'arm':<12} " + " ".join(f"k={k:<3}" for k in range(1, 11)))
    for arm in arms:
        rows = [r for r in results if r["arm"] == arm]
        n, ok = Counter(), Counter()
        for r in rows:
            n[r.get("k")] += 1
            ok[r.get("k")] += bool(r["success"])
        print(f"{arm:<12} " + " ".join(
            f"{(100 * ok[k] / n[k]) if n[k] else float('nan'):<5.0f}" for k in range(1, 11)))

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out = pathlib.Path(args.out or f"results/json/probe_stepgame_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": {"arms": summary, "per_k_items": args.per_k, "seed": args.seed,
                    "model": model, "split": args.split, "elapsed_sec": round(elapsed, 1),
                    "temperature": args.temperature, "max_tokens": args.max_tokens},
        "results": results}, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
