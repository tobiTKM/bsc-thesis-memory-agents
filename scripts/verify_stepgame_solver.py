#!/usr/bin/env python3
"""Gate: the symbolic solver must answer every StepGame item correctly, else exit 1.

The solver is the oracle and the source of gold chains for bank entries, so it must be
exact. Test-file item counts are also checked against Li, Hogg & Cohn's Table 2 removal
percentages, the only cheap way to tell the rectified corpus from the original.

    python scripts/verify_stepgame_solver.py                 # everything on disk
    python scripts/verify_stepgame_solver.py --splits test --sample 500
"""

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents.envs import stepgame          # noqa: E402
from memory_agents.envs import stepgame_solver   # noqa: E402

# Li et al. (AAAI-24) Table 2: % of test instances removed by rectification.
TABLE_2_REMOVED = {
    "clean": {1: 7.64, 2: 15.03, 3: 20.87, 4: 26.39, 5: 32.54,
              6: 37.66, 7: 41.71, 8: 47.20, 9: 51.50, 10: 54.29},
    "noise": {1: 20.43, 2: 30.19, 3: 34.59, 4: 48.18, 5: 57.13,
              6: 61.14, 7: 63.60, 8: 69.45, 9: 72.84, 10: 74.21},
}
ORIGINAL_TEST_SIZE = 10_000
TOLERANCE_PP = 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(stepgame.VARIANTS))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--ks", nargs="+", type=int, default=list(stepgame.KS))
    ap.add_argument("--sample", type=int, default=0,
                    help="items per file (0 = all; the gate wants all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"templates: {stepgame_solver.load_templates.__module__} "
          f"<- {stepgame.data_dir() / 'raw' / 'sentence_to_relation.py'}")
    print(f"{len(stepgame_solver.load_templates())} templates loaded\n")

    header = (f"{'file':22s} {'n':>7s} {'correct':>8s} {'wrong':>6s} "
              f"{'unparsed':>9s} {'removed%':>9s} {'expect%':>8s}")
    print(header)
    print("-" * len(header))

    failed = False
    table_2_ok = True
    total = total_correct = 0

    for variant in args.variants:
        for split in args.splits:
            for k in args.ks:
                path = stepgame.corpus_path(variant, split, k)
                if not path.exists():
                    continue
                items = stepgame.read_babi(path, variant, split, k)
                full_n = len(items)
                if args.sample and args.sample < full_n:
                    items = random.Random(args.seed).sample(items, args.sample)

                report = stepgame_solver.verify(items)
                total += report["n"]
                total_correct += report["correct"]

                removed = expected = float("nan")
                flag = ""
                if split == "test":
                    removed = 100 * (1 - full_n / ORIGINAL_TEST_SIZE)
                    expected = TABLE_2_REMOVED[variant][k]
                    if abs(removed - expected) > TOLERANCE_PP:
                        flag, table_2_ok = "  <-- NOT the rectified corpus?", False

                if report["wrong"] or report["unparsed"]:
                    failed = True

                print(f"{variant}/{split}/qa{k:<12d} {report['n']:7d} "
                      f"{report['correct']:8d} {report['wrong']:6d} "
                      f"{report['unparsed']:9d} "
                      f"{removed:9.2f} {expected:8.2f}{flag}")
                for item_id, why in report["failures"]:
                    print(f"    FAIL {item_id}: {why}")

    print("-" * len(header))
    rate = 100 * total_correct / total if total else 0.0
    print(f"{'TOTAL':22s} {total:7d} {total_correct:8d} "
          f"({rate:.4f}% correct)\n")

    # Every solver label must round-trip through the actor's output format.
    for label in stepgame_solver.LABELS:
        parsed = stepgame.parse_answer(f"Some reasoning.\nFinal: {label}")
        if parsed != label:
            print(f"[gate] parse_answer round-trip failed: {label} -> {parsed}")
            failed = True

    if failed:
        print("[gate] FAILED: the solver does not decode the corpus exactly.")
        return 1
    if not table_2_ok:
        print("[gate] FAILED: item counts do not match the rectified corpus.")
        return 1
    print("[gate] PASS: solver is exact and the corpus is the rectified one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
