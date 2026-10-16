#!/usr/bin/env python3
"""Freeze the StepGame train and test splits every arm runs against.

The splits are drawn once with a recorded seed and committed, so every arm scores the
identical items for the paired tests; nothing downstream re-samples. By default the
bank source is train k=1..5 and the eval set is test k=1..10, 100 items per k.
k is not the reasoning depth (the queried pair is drawn from the k-relation chain, so
true hops <= k); `--stratify hops` balances by true hop count instead. Every item is
solved before it is written and carries both `k` and `hops` plus its gold chain.

    python scripts/build_stepgame_splits.py                      # the standard splits
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

# Each (variant, split, bucket) has its own RNG, so adding a bucket keeps the others.
DEFAULT_SEED = 0


def _pool(variant: str, split: str, ks) -> list[stepgame.StepGameItem]:
    """Every corpus item across the given k files."""
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
            # Cannot happen while verify_stepgame_solver.py passes; dropped, not trusted.
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
        for k in ks:
            pool = _pool(variant, split, [k])
            rng = random.Random(f"{seed}:{variant}:{split}:k{k}")
            # Sample the whole level, not a prefix: the corpus is generated in order.
            for item in sorted(rng.sample(pool, min(per_bucket, len(pool))),
                               key=lambda it: it.id):
                if _verify_and_annotate(item, dropped):
                    kept.append(item)
    else:
        # The hop bucket is only known after solving, so the whole pool is solved first.
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

    # Standard pair: bank built on k<=5, evaluation on k<=10.
    build("clean", "train", list(range(1, 6)), args.per_bucket, args.seed, out_dir)
    # Hop-balanced bank source: stratifying train by k leaves it mostly one-hop items.
    build("clean", "train", list(range(1, 6)), args.per_bucket, args.seed, out_dir,
          stratify="hops", hops=list(range(1, 6)))
    build("clean", "test", list(stepgame.KS), args.per_bucket, args.seed, out_dir)
    # Hop-balanced test split; hops=10 is excluded, the corpus holds too few such items.
    build("clean", "test", list(stepgame.KS), args.per_bucket, args.seed, out_dir,
          stratify="hops", hops=list(range(1, 10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
