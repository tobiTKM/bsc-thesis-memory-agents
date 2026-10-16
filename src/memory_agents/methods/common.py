"""CLI plumbing shared by every method driver. Stdlib only, so parsers build
without the model stack installed."""

from memory_agents.benchmarks import add_benchmark_args

# Decoding settings of every memory method (Qwen3 thinking mode loops at
# temperature 0). The baseline keeps its own greedy defaults.
SAMPLING = dict(max_tokens=12288, temperature=0.6, top_p=0.95, top_k=20)


def add_run_args(ap, *, max_tokens: int, temperature: float, top_p: float,
                 top_k: int | None, split_help: str | None = None,
                 seed_help: str | None = None, seed_dest: str = "seed") -> None:
    """Split, sampling, seed, output and benchmark flags; defaults are passed
    explicitly so each driver keeps its own."""
    ap.add_argument("--split", default="test", help=split_help)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 or less = whole split")
    ap.add_argument("--max-tokens", type=int, default=max_tokens, help="actor budget")
    ap.add_argument("--temperature", type=float, default=temperature)
    ap.add_argument("--top-p", type=float, default=top_p)
    ap.add_argument("--top-k", type=int, default=top_k,
                    help="sampling top-k (NOT the number of memories); "
                         "unset = disabled")
    ap.add_argument("--seed", type=int, dest=seed_dest, default=0, help=seed_help)
    ap.add_argument("--out", default=None, help="output path")
    add_benchmark_args(ap)


def typed_flags(argv: list[str]) -> set[str]:
    """The typed flags as argparse dests, so a resume can report which ones it ignores."""
    return {a.lstrip("-").replace("-", "_").split("=")[0]
            for a in argv if a.startswith("--")}
