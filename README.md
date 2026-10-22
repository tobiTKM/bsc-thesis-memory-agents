# Does Memory Help? Episodic, Declarative and Procedural Memory for LLM Agents on Spatial and Rule-Based Reasoning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code for the bachelor thesis of the same title (Georg-August-Universität Göttingen, 2026).

## Overview

An agent built on a large language model (LLM) can feed experience from earlier episodes
back into its prompt, as episodic, declarative or procedural memory. These memory methods
were developed on tasks that tolerate many routes to the goal. It is unknown whether they
help where every step must satisfy explicit rules and one broken rule invalidates an
otherwise complete solution.

This repository ports three memory methods, **Reflexion** (episodic), **Memp** (procedural)
and **MemoryBank** (declarative), onto one harness, so that two experiments differ only in
what their memory says. It adds a fourth, **SAW** (Strategies And Warnings), which
keeps a strategy from every solved training task and a warning from every failed one in the
same bank. Memp, MemoryBank and SAW are cross-task methods. They distill the actor's
memoryless training episodes into a bank, freeze it, and place the three entries most
similar to each test task in its prompt. Reflexion instead writes a reflection after each
failed attempt and retries the same task with it.

All four methods run on two benchmarks that differ in horizon.

- **[Spatial-Gym](https://github.com/lkaesberg/Spatial-Gym)** poses rule-constrained grid
  puzzles, solved one move per call.
- **StepGame** asks questions about chains of spatial relations, answered in one call per
  item.

Two open-weight actors, gpt-oss-120b and Qwen3-32B, are each compared item by item with
themselves without memory. Two controls separate the memory from its side effects. One shows
three entries drawn at random from the same bank instead of the most similar ones. The
other, for Reflexion, is a trial-matched control that retries the same tasks without
reflections.

## Key findings

- **Whether memory helps depends on the actor and the benchmark, and the kind of memory
  makes no measurable difference.** Every cross-task method lifts gpt-oss-120b on
  Spatial-Gym by 3.8 to 7.0 points and Qwen3-32B on StepGame by 15 to 20 points, yet leaves
  Qwen3-32B below its accuracy without memory on Spatial-Gym. No kind of memory separates
  from another beyond the run-to-run floor (about 3 points on Spatial-Gym, 3.9 on StepGame).
  Reflexion leads its trial-matched control only for gpt-oss-120b, and most of its rising
  curve comes from retrying rather than reflecting.
- **Which entry the actor sees hardly matters.** Entries drawn at random from the same bank
  stay within the run-to-run floor of the most similar ones in all 21 comparisons.
- **Memory is expensive where the actor reads it at every move.** On Spatial-Gym,
  cross-task memory makes each solved puzzle 1.2 to 2.0 times more expensive for
  gpt-oss-120b and 1.6 to 3.1 times for Qwen3-32B. On StepGame, Qwen3-32B pays only 1.2 to
  1.3 times as much per solved item. For gpt-oss-120b, SAW comes within noise of the most
  accurate configuration, Memp with both forms together, for about 30% fewer tokens (0.49
  against 0.69 million tokens per solved puzzle).
- **Memory helps where a capable actor already partly succeeds.** gpt-oss-120b's gains on
  Spatial-Gym come from the two easiest difficulty levels, and the hardest puzzles stay
  unsolved, whereas Qwen3-32B loses most on the easiest puzzles. A Spatial-Gym episode fails
  either as a wrong solution, which reaches the end but breaks a rule, or as a dead end,
  where no legal move is left. Solutions replayed move by move leave more wrong solutions
  and fewer dead ends, whereas written guidance cuts wrong solutions.

Accuracy (%) with retrieved entries. Cross-task rows give the range over each method's
configurations. Reflexion rows give in parentheses its trial-matched control, which retries
without memory.

| | Spatial-Gym, gpt-oss-120b | Spatial-Gym, Qwen3-32B | StepGame, Qwen3-32B |
|---|---:|---:|---:|
| Memoryless baseline | 14.0 | 11.0 | 73.3 |
| Memp (4 configurations) | 18.0 to 21.0 | 5.8 to 8.6 | 88.8 to 93.3 |
| MemoryBank (2 banks) | 17.8 to 19.6 | 9.6 | 88.6 to 90.0 |
| SAW | 19.0 | 8.2 | 88.8 |
| Reflexion, one reflection (control) | 24.0 (19.6) | 15.8 (15.4) | 87.4 (87.6) |
| Reflexion, ten trials (control) | 35.8 (31.8) | 25.6 (26.6) | 94.0 (96.3) |

The test sets hold 500 Spatial-Gym puzzles and 1000 StepGame items. gpt-oss-120b is not run on
StepGame because it answers 99.4% of the items without memory. The thesis reports all
configurations, random-entry controls, costs, failure modes and a third actor (Qwen3-0.6B).

## Methods

| Method | Memory type | What one entry shows the actor | Learns from |
|---|---|---|---|
| Reflexion | episodic | up to three reflections on earlier failed attempts at the same task | failures, same task only |
| Memp | procedural | a written procedure, the solution replayed step by step, or both | training tasks |
| MemoryBank | declarative | the actor's recollection of one past task, with its solution replayed move by move | training tasks |
| SAW | new in this thesis | a strategy and its moves if the actor solved the training task, a warning and the failed attempt's moves if it failed. On Spatial-Gym, the warning is grounded in the environment's verdict | successes and failures, across tasks |

The cross-task banks hold one entry per training task. An entry comes from the actor's own
run when it solved the task. Otherwise Memp and MemoryBank write it from the dataset's
solution, and SAW writes a warning from the failed attempt. Retrieval embeds every task
with Qwen3-Embedding-0.6B and selects the three entries with the highest cosine similarity.
Every cross-task configuration also runs with three random entries from the same bank.
Because Reflexion retries only unsolved tasks, its accuracy rises with the number of trials
even without memory, so every Reflexion run is paired with a trial-matched control.

## Benchmarks

**Spatial-Gym** builds a path between two nodes of a grid, one move per call, with
backtracking disabled. Seven rule types constrain the regions the path creates, and the
environment checks them against the finished path. An episode ends solved, as a wrong
solution (end reached, rule broken) or as a dead end (no legal move left). The banks are
built from the 500 training puzzles and every experiment runs on the same 500 test puzzles.
The puzzles load from the SPaRC dataset on Hugging Face.

**StepGame** tells a story of *k* sentences that chain spatial relations between entities,
then asks for the relation between two of them, one of nine labels. This repository uses
the clean variant of the rectified corpus by Li, Hogg and Cohn (AAAI-24). Because the queried
pair is drawn at random from the chain, *k* is not the reasoning depth, so difficulty is
reported by hop count. The frozen splits in `data/stepgame/splits/` hold 500 training stories
(*k* = 1 to 5) and 1000 test items (100 per *k* from 1 to 10).

## Installation

```bash
git clone https://github.com/tobiTKM/bsc-thesis-memory-agents.git
cd bsc-thesis-memory-agents
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/fetch_stepgame.py      # StepGame corpus into data/stepgame/raw/
```

`pip install -e .` also installs the Spatial-Gym environment from GitHub, pinned to the
commit the thesis used. vLLM is not a dependency, because the experiments need
only an OpenAI-compatible endpoint (see below). The StepGame corpus has no upstream license, so
`fetch_stepgame.py` downloads it from a pinned commit instead of the repository shipping it.

## Quick start

Every driver talks to an OpenAI-compatible chat endpoint named by two environment variables.

```bash
vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3 --max-model-len 32768   # in another shell
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=Qwen/Qwen3-0.6B

python scripts/run.py baseline --limit 10                       # 10 Spatial-Gym test puzzles
python scripts/run.py baseline --benchmark stepgame --limit 10  # 10 StepGame test items
```

Results go to `results/json/`. `python scripts/summarize_results.py` prints a readable
summary of the newest run.

## Running experiments

`scripts/run.py` is the single entry point, called as `python scripts/run.py <method>
--phase <phase> [flags]`. `python scripts/run.py --help` lists the methods, and `run.py <method> --help`
lists a method's phases and flags. Phases that call no model (`compare`, contrastive
`exemplars`, memp `upgrade`) run anywhere.

| Method | Phases, in order |
|---|---|
| `baseline` | none |
| `reflexion` | `root` (trial 1) → `arm --arm memory` and `arm --arm control` → `compare` |
| `memp` | `build` (train pass) → `distill` → `embed` → [`upgrade`] → `eval` → `compare` |
| `memorybank` | `distill` → `eval` → `compare`, reusing Memp's build file and embeddings |
| `contrastive` (SAW) | `distill` → `eval` → `compare`, reusing Memp's build file and embeddings |

The commands below run Reflexion and Memp on Spatial-Gym from start to finish.

```bash
# Reflexion. Its trial 1 is also the memoryless baseline every method is compared with.
python scripts/run.py reflexion --phase root --out results/json/root.json
python scripts/run.py reflexion --phase arm --arm memory  --root results/json/root.json --trials 10 --out results/json/refl_memory.json
python scripts/run.py reflexion --phase arm --arm control --root results/json/root.json --trials 10 --out results/json/refl_control.json
python scripts/run.py reflexion --phase compare --memory results/json/refl_memory.json --control results/json/refl_control.json

# One memoryless pass over the training split, shared by all cross-task methods.
python scripts/run.py memp --phase build --split train --out results/json/build.json
python scripts/run.py memp --phase distill --build results/json/build.json --out results/json/memp_bank.json

# Embeddings come from a second server running the embedding model, started with
#   vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --port 8001
LLM_BASE_URL=http://localhost:8001/v1 LLM_MODEL=Qwen/Qwen3-Embedding-0.6B \
  python scripts/run.py memp --phase embed --out results/json/emb.npz

python scripts/run.py memp --phase eval --arm retrieved --bank results/json/memp_bank.json --embeddings results/json/emb.npz --out results/json/memp_ret.json
python scripts/run.py memp --phase eval --arm random    --bank results/json/memp_bank.json --embeddings results/json/emb.npz --out results/json/memp_rnd.json
python scripts/run.py memp --phase compare --baseline results/json/root.json --retrieved results/json/memp_ret.json --random results/json/memp_rnd.json
```

MemoryBank and SAW (driver `contrastive`) follow the same pattern. They run `distill
--build results/json/build.json`, then `eval` with their own bank and the same embeddings.
Every long phase checkpoints its output file, and `--resume <file>` continues it. A resumed
run keeps the settings stored in its file and ignores conflicting flags.

### On a SLURM cluster

`scripts/serve_and_run.sbatch` starts vLLM on the allocated GPU node, waits until it is
healthy, sets `LLM_BASE_URL` and `LLM_MODEL`, and runs one phase, chosen by environment
variables.

```bash
sbatch --export=ALL,DRIVER=memp,PHASE=eval,ARM=retrieved,MODEL=openai/gpt-oss-120b,TP=4,LIMIT=0,BANK=<bank.json>,EMBEDDINGS=<emb.npz> -G A100:4 scripts/serve_and_run.sbatch
```

The script is written for the GWDG SCC cluster (account, partition and Apptainer image), so
adapt its header before using it elsewhere. Its header lists an example for every phase. The
`run_*_campaign.sh` scripts submit the jobs of each campaign in the thesis.

## Reproducing the thesis

Every experiment in the thesis uses the default settings of the memory methods. These are
temperature 0.6, top-p 0.95, top-k 20, at most 12 288 generated tokens per call, seed 0,
thinking mode on, and up to five resamples of an unparseable reply. The `baseline` driver
defaults to greedy decoding with 1024 tokens instead. The thesis compares every method with
Reflexion's trial 1, which uses the settings above.

| Thesis configuration | Flags |
|---|---|
| Memp, written procedure | `memp --phase eval --granularity script` (default) |
| Memp, step by step | `--granularity trajectory`, on a bank passed through `memp --phase upgrade --bank <bank>` |
| Memp, both together | `--granularity proceduralization`, on the upgraded bank |
| Memp, own successes | `--granularity script --provenance agent` (on StepGame `--granularity trajectory --provenance agent`) |
| MemoryBank, rules given / withheld | `memorybank --phase distill --rules-preamble on` / `off` |
| SAW | `contrastive --phase eval --granularity avoid --exemplars on` |
| Random-entry control | `--arm random` instead of `--arm retrieved` |
| Reflexion | `reflexion --phase arm --arm memory` / `--arm control`, `--trials 10` |

On StepGame, every phase takes three more flags, `--benchmark stepgame`, `--prompt-protocol
upstream-header` (Li et al.'s instruction paragraph without their worked examples) and
`--stepgame-train data/stepgame/splits/stepgame_clean_train_byhops_k1-5_n500.jsonl` (the
hop-stratified training split the banks were built from). The test split defaults to the
frozen 1000-item file.

`scripts/thesis_report.py` turns the result files in `results/json/` into the tables of the
thesis. It expects the file names that the campaign scripts write.

```bash
python scripts/thesis_report.py --audit   # list result files that no table claims
python scripts/thesis_report.py --write   # -> results/thesis/thesis_reported_runs.md
python scripts/thesis_tables.py all       # the LaTeX result tables
python scripts/thesis_bootstrap.py        # paired bootstrap 95% intervals
```

## Repository structure

```
scripts/
  run.py                    entry point for every experiment
  serve_and_run.sbatch      SLURM job that serves a model with vLLM and runs one phase
  run_*_campaign.sh         job submissions for each thesis campaign
  fetch_stepgame.py         download the StepGame corpus
  build_stepgame_splits.py  build the frozen StepGame splits (already committed)
  thesis_*.py               thesis tables, bootstrap intervals and example blocks
  summarize_results.py      readable summary of a result file
src/memory_agents/
  methods/                  one driver per method (baseline, reflexion, memp, memorybank, contrastive)
  memory/                   the memory stores and the model calls that write them
  envs/                     Spatial-Gym wrapper, StepGame loader and exact solver
  benchmarks.py             what differs per benchmark, behind one interface
  prompts.py                Spatial-Gym prompts and memory-block renderers
  prompts_stepgame.py       StepGame prompts and memory-block renderers
  agents/, harness/, llm/   the actor, the episode loops, the async vLLM client
tests/                      invariant checks and an end-to-end dry run with a mock model
data/stepgame/splits/       frozen StepGame train and test splits
```

## Tests

No GPU or model is needed for these checks.

```bash
python tests/smoke_stepgame.py           # StepGame data layer and prompt invariants (seconds)
python tests/smoke_contrastive.py        # what each SAW arm renders (seconds)
python tests/regress_sparc.py --n 25     # Spatial-Gym prompts byte-identical to the reported runs
python tests/dryrun_stepgame.py          # every phase of every method against a mock model
python scripts/verify_stepgame_solver.py # the exact solver against the whole corpus (minutes)
```

## References

These papers describe the ported methods and the benchmarks.

- Shinn et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS.
- Fang et al. (2026). Memp: Exploring Agent Procedural Memory. Findings of ACL.
- Zhong et al. (2024). MemoryBank: Enhancing Large Language Models with Long-Term Memory. AAAI.
- Kaesberg et al. (2026). Mind the Gap Between Spatial Reasoning and Acting! Step-by-Step
  Evaluation of Agents With Spatial-Gym. arXiv:2604.09338.
  [Code](https://github.com/lkaesberg/Spatial-Gym)
- Kaesberg et al. (2025). SPaRC: A Spatial Pathfinding Reasoning Challenge. EMNLP.
  [Code](https://github.com/lkaesberg/SPaRC)
- Shi, Zhang and Lipani (2022). StepGame: A New Benchmark for Robust Multi-Hop Spatial
  Reasoning in Texts. AAAI.
- Li, Hogg and Cohn (2024). Advancing Spatial Reasoning in Large Language Models: An
  In-Depth Evaluation and Enhancement Using the StepGame Benchmark. AAAI.
  [Code](https://github.com/Fangjun-Li/SpatialLM-StepGame)

## Acknowledgements

Supervised by Lars Benedikt Kaesberg and examined by PD Dr. Terry Lima Ruas and Dr. Jan
Philip Wahle at the Institute of Computer Science, Georg-August-Universität Göttingen.
Experiments ran on the GWDG Scientific Compute Cluster.

## License

MIT, see [LICENSE](LICENSE).
