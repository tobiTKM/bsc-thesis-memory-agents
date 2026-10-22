"""The experiment drivers, one module per method, dispatched by scripts/run.py:

  python scripts/run.py <method> [--phase <phase>] [flags]

Each module exposes `async def main(argv, prog)`. This package imports none of
them, so model-free phases run where numpy and openai are not installed.
"""

# name -> one-line summary, shown by `scripts/run.py --help`.
METHODS = {
    "baseline":    "memoryless baseline over one split (no phases)",
    "reflexion":   "episodic, per task (Shinn et al. 2023): root | arm | compare",
    "memp":        "procedural, cross-task (Fang et al. 2026): build | distill | embed "
                   "| eval | compare | upgrade | keywords | embed-keywords",
    "memorybank":  "declarative, cross-task (Zhong et al. 2024): distill | eval | compare",
    "contrastive": "DOs from successes + DON'Ts from failures (ours): distill "
                   "| exemplars | eval | compare",
}
