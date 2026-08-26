#!/usr/bin/env python3
"""Freeze the StepGame train and test splits every arm runs against.

The corpus is far too large to evaluate on -- 113k clean test items alone -- so
the experiment runs on a stratified sample, and that sample has to be *frozen*:
the baseline, both Reflexion arms, and every MemP / MemoryBank eval must score the
identical items or the paired tests (exact McNemar over the same items) are not
paired at all. So the sample is drawn once, with a recorded seed, written to JSONL
and committed. Nothing downstream re-samples.

Defaults, and why:

* test = 100 items per k, k=1..10 (1000 items). Li et al. showed 30- and
  100-item evaluations of the *whole* benchmark fluctuate enough to invert the
  ordering between hop levels, while 1000 gives the clean monotone decay; 100 per
  k is that budget spread so every hop level is separately reportable. It is also
  the same order as the 500-puzzle SPaRC split, so per-item costs stay comparable.
* train = 100 items per k, k=1..5 (500 items), which is exactly SPaRC's bank size.

The train/test asymmetry (k<=5 built, k<=10 evaluated) is deliberate and is the
original StepGame generalization protocol: it asks whether memory distilled from
short chains transfers to chains longer than any it was built on. Half the test
set is out of the bank's range by construction.

**k is not the reasoning depth.** StepGame builds a chain of k relations and then
picks the two queried entities *at random from that chain*, so the number of hops
actually separating them is <= k (the original paper says so in passing; nothing
downstream of it seems to). Measured over the rectified clean test corpus, the
k=10 file averages 3.7 true hops and 23% of its items are solvable in one. The
whole published literature reports accuracy per k, which is therefore accuracy
over a mixture of depths, not a depth.

The solver gives us the true hop count for free, so both axes are available:

* `--stratify k` (default) samples per file, which is what every published
  StepGame number is stratified by, and is what our results must be comparable to.
* `--stratify hops` samples per *true* hop count, which is the honest difficulty
  axis for "does memory help more as reasoning gets deeper?".

Every item carries both `k` and `hops` either way, so a single run can be broken
down along both; the two splits differ only in which axis is balanced.

Every sampled item is solved before it is written, and its gold chain stored, so
a bank never has to re-derive one and a mis-parsed item can never reach an arm.

    python scripts/build_stepgame_splits.py                      # both defaults
    python scripts/build_stepgame_splits.py --variant noise --split test
"""

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents.envs import stepgame          # noqa: E402
from memory_agents.envs.stepgame_solver import (  # noqa: E402
    SolverError, edges_of, gold_chain, hop_count, solve,
)

# Sampling is per (variant, split, k) and seeded with the base seed mixed with k,
# so adding a hop level to a split does not reshuffle the levels already drawn.
DEFAULT_SEED = 0


def _pool(variant: str, split: str, ks) -> list[stepgame.StepGameItem]:
    """Every corpus item across the given files, with hops and chain resolved."""
    items = []
    for k in ks:
        path = stepgame.corpus_path(variant, split, k)
        if not path.exists():
            raise SystemExit(
                f"{path} not found. Run `python scripts/fetch_stepgame.py "
                f"--variants {variant} --splits {split}` first.")
        items.extend(stepgame.read_babi(path, variant, split, k))
    return items


def _verify_and_annotate(item, dropped: list) -> bool:
    """Solve the item, store its chain, or record why it cannot be trusted."""
    try:
        predicted = solve(item.story, item.question)
        if predicted != item.gold:
            # Cannot happen while verify_stepgame_solver.py passes; if it ever
            # does, the item is dropped rather than trusted, because we no longer
            # know which of the two labels is right.
            dropped.append((item.id, f"solver={predicted} gold={item.gold}"))
            return False
        item.hops = hop_count(item.story, item.question)
        item.gold_chain = gold_chain(item.story, item.question)
        item.extra = {
            "relations": [label for _a, label, _b in edges_of(item.story)],
            "key_fn": stepgame.KEY_FN,
        }
        return True
    except SolverError as exc:
        dropped.append((item.id, repr(exc)))
        return False


def build(variant: str, split: str, ks, per_bucket: int, seed: int,
          out_dir: pathlib.Path, stratify: str = "k",
          hops: list[int] | None = None) -> pathlib.Path:
    dropped: list[tuple[str, str]] = []
    kept: list[stepgame.StepGameItem] = []

    if stratify == "k":
        # Sample per file. Cheap: only the sampled items need solving.
        for k in ks:
            pool = _pool(variant, split, [k])
            rng = random.Random(f"{seed}:{variant}:{split}:k{k}")
            # Sample without replacement from the *whole* level rather than
            # taking a prefix: the corpus is generated in order and a prefix
            # would correlate with whatever the generator's sampler did first.
            for item in sorted(rng.sample(pool, min(per_bucket, len(pool))),
                               key=lambda it: it.id):
                if _verify_and_annotate(item, dropped):
                    kept.append(item)
    else:
        # Sample per true hop count. Every candidate must be solved to know
        # which bucket it belongs in, so this reads the whole pool first.
        pool = _pool(variant, split, ks)
        for item in pool:
            item.hops = hop_count(item.story, item.question)
        buckets: dict[int, list] = {}
        for item in pool:
            buckets.setdefault(item.hops, []).append(item)
        wanted = hops or sorted(buckets)
        for h in wanted:
            candidates = buckets.get(h, [])
            if not candidates:
                print(f"[split]   note: no items with hops={h}")
                continue
            if len(candidates) < per_bucket:
                print(f"[split]   note: hops={h} has only {len(candidates)} "
                      f"items (wanted {per_bucket})")
            rng = random.Random(f"{seed}:{variant}:{split}:h{h}")
            for item in sorted(rng.sample(candidates,
                                          min(per_bucket, len(candidates))),
                               key=lambda it: it.id):
                if _verify_and_annotate(item, dropped):
                    kept.append(item)

    kept.sort(key=lambda it: it.id)
    by_k = {k: sum(1 for i in kept if i.k == k) for k in sorted({i.k for i in kept})}
    by_hops = {h: sum(1 for i in kept if i.hops == h)
               for h in sorted({i.hops for i in kept})}

    out_dir.mkdir(parents=True, exist_ok=True)
    axis = "" if stratify == "k" else "_byhops"
    stem = (f"stepgame_{variant}_{split}{axis}"
            f"_k{min(ks)}-{max(ks)}_n{len(kept)}")
    out = out_dir / f"{stem}.jsonl"
    out.write_text("".join(json.dumps(it.to_dict()) + "\n" for it in kept))

    meta = {
        "variant": variant, "split": split, "ks": list(ks),
        "stratify": stratify, "per_bucket": per_bucket, "seed": seed,
        "n": len(kept), "by_k": by_k, "by_hops": by_hops,
        "dropped": len(dropped), "dropped_detail": dropped[:20],
        "key_fn": stepgame.KEY_FN,
        "solver_verified": True,
        "source_commit_manifest": str(stepgame.data_dir() / "raw" / "MANIFEST.json"),
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[split] {out}")
    print(f"[split]   {len(kept)} items  by k: {by_k}")
    print(f"[split]   {len(kept)} items  by hops: {by_hops}")
    if dropped:
        print(f"[split]   WARNING: dropped {len(dropped)} unverifiable items")
        for item_id, why in dropped[:5]:
            print(f"[split]     {item_id}: {why}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=None, choices=list(stepgame.VARIANTS),
                    help="default: build the standard train+test pair")
    ap.add_argument("--split", default=None, choices=list(stepgame.SPLITS))
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--per-k", dest="per_bucket", type=int, default=100,
                    help="items per bucket (per k, or per hop count)")
    ap.add_argument("--stratify", default="k", choices=["k", "hops"],
                    help="which difficulty axis to balance (default: k, the "
                         "axis the published StepGame numbers use)")
    ap.add_argument("--hops", nargs="+", type=int, default=None,
                    help="hop buckets for --stratify hops (default: all present)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=None,
                    help="default: <repo>/data/stepgame/splits")
    args = ap.parse_args()

    out_dir = (pathlib.Path(args.out) if args.out
               else stepgame.data_dir() / "splits")

    if args.variant or args.split:
        variant = args.variant or "clean"
        split = args.split or "test"
        ks = args.ks or (list(range(1, 6)) if split == "train" else list(stepgame.KS))
        build(variant, split, ks, args.per_bucket, args.seed, out_dir,
              stratify=args.stratify, hops=args.hops)
        return 0

    # The standard pair: the bank is built on short chains, the evaluation spans
    # all of them.
    build("clean", "train", list(range(1, 6)), args.per_bucket, args.seed, out_dir)
    # A hop-balanced bank source. Stratifying the train split by k leaves it 58%
    # one-hop items, because most queried pairs sit close together on the chain --
    # a bank distilled from that is mostly worked examples of a task with no
    # chaining in it at all. This one holds 100 per true hop depth instead.
    build("clean", "train", list(range(1, 6)), args.per_bucket, args.seed, out_dir,
          stratify="hops", hops=list(range(1, 6)))
    build("clean", "test", list(stepgame.KS), args.per_bucket, args.seed, out_dir)
    # The difficulty-analysis split. h=10 is capped at what the corpus holds (78
    # items over the whole clean test set), so it is excluded rather than
    # reported as an underpowered bucket.
    build("clean", "test", list(stepgame.KS), args.per_bucket, args.seed, out_dir,
          stratify="hops", hops=list(range(1, 10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
