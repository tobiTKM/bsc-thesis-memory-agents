#!/usr/bin/env python
"""Run one experiment: a memory method (or the memoryless baseline) on a benchmark.

  python scripts/run.py <method> [--phase <phase>] [flags]
  python scripts/run.py <method> --help          # that method's phases and flags

Reads the server endpoint from LLM_BASE_URL / LLM_MODEL, which
scripts/serve_and_run.sbatch sets after starting vLLM. Phases that need no model
(every `compare`, contrastive `exemplars`, memp `upgrade`) run anywhere.
"""

import asyncio
import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memory_agents.methods import METHODS  # noqa: E402


def usage() -> str:
    width = max(map(len, METHODS))
    rows = "\n".join(f"  {name:<{width}}  {summary}" for name, summary in METHODS.items())
    return f"{__doc__}\nmethods:\n{rows}\n"


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(usage())
        return 0 if len(sys.argv) >= 2 else 2
    method, argv = sys.argv[1], sys.argv[2:]
    if method not in METHODS:
        print(f"run.py: unknown method {method!r} (choose from: "
              f"{', '.join(METHODS)})", file=sys.stderr)
        return 2
    # Imported lazily: some drivers pull heavy dependencies at import time.
    module = importlib.import_module(f"memory_agents.methods.{method}")
    asyncio.run(module.main(argv, prog=f"run.py {method}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
