"""The Benchmark interface: what a memory method needs that differs per task.

SPaRC is an interactive grid MDP and StepGame single-turn text QA; each driver in
memory_agents/methods/ (baseline, reflexion, memp, memorybank, contrastive) is
written once against this interface.
"""

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod


class Benchmark(ABC):
    """What a memory method needs to know about a task."""

    name: str = ""
    key_fn: str = ""
    interactive: bool = True        # multi-step env vs. one reply per item
    # Granularities the injected block supports; the first is the default.
    granularities: tuple = ("script", "trajectory", "proceduralization")
    case_granularities: tuple = ("case-record", "case")
    # Must match memory/contrastive.GRANULARITIES.
    contrastive_granularities: tuple = (
        "avoid", "hindsight", "positive-only", "avoid-only")

    # ---- dataset ---------------------------------------------------------
    @abstractmethod
    def ids(self, split: str, limit=None) -> list[str]:
        """Item ids for a split, in a stable order."""

    @abstractmethod
    def record(self, split: str, item_id: str):
        """The opaque per-item record the other methods consume."""

    @abstractmethod
    def difficulty(self, record) -> int:
        """The primary difficulty label (SPaRC: level; StepGame: k)."""

    def secondary_difficulty(self, record):
        """An optional second difficulty axis, or None (StepGame: hop count, not k)."""
        return None

    @abstractmethod
    def key_text(self, record) -> str:
        """The canonical text an item is embedded from, for retrieval."""

    @abstractmethod
    def parse_reply(self, text):
        """Reply -> action/answer, or None for a format failure."""

    # ---- running ---------------------------------------------------------
    @abstractmethod
    async def run_baseline_suite(self, client, split: str, concurrency: int,
                                 limit=None) -> list:
        """A memoryless pass over the whole split."""

    @abstractmethod
    async def run_memory_pass(self, client, item_ids, block_for, split: str,
                              concurrency: int) -> list:
        """A pass over an explicit item set with a per-item block injected."""

    # ---- bank construction ----------------------------------------------
    @abstractmethod
    def gold_exemplar(self, record) -> dict:
        """The compact solved example for a hindsight (gold-provenance) entry."""

    @abstractmethod
    def agent_exemplar(self, record, episode_ctx: dict) -> dict:
        """Same shape, from the agent's own successful attempt."""

    @abstractmethod
    def gold_trajectory(self, record) -> dict:
        """The verbatim record for a gold entry, in `reflection_context` shape."""

    def failed_trajectory(self, record, episode_ctx: dict) -> dict:
        """The verbatim record of a failed attempt, in `reflection_context` shape.

        Contrastive DON'Ts are distilled from it. Returns {} for a crashed episode.
        """
        return dict(episode_ctx) if episode_ctx else {}

    def verify_gold(self, split: str, n: int) -> None:
        """Optional self-check that the gold source is sound. Runs per distill job."""

    # ---- prompts ---------------------------------------------------------
    @abstractmethod
    def system_prompt(self, record) -> str: ...
    @abstractmethod
    def reflection_prompt(self, episode, previous=()) -> str: ...
    @abstractmethod
    def format_reflections(self, items) -> str: ...
    @abstractmethod
    def distill_trajectory_prompt(self, episode, key_text, record,
                                  succeeded: bool = True) -> str: ...
    @abstractmethod
    def distill_gold_prompt(self, key_text, record) -> str: ...
    @abstractmethod
    def case_trajectory_prompt(self, episode, key_text, record,
                               rules_preamble: bool = True) -> str: ...
    @abstractmethod
    def case_gold_prompt(self, key_text, record,
                         rules_preamble: bool = True) -> str: ...
    @abstractmethod
    def format_procedures(self, entries, granularity: str) -> str: ...
    @abstractmethod
    def format_cases(self, entries, granularity: str) -> str: ...
    @abstractmethod
    def format_contrastive(self, entries, granularity: str, exemplars: bool = False,
                           framing: str = "mixed") -> str: ...
    @abstractmethod
    def avoid_prompt(self, episode, key_text, record, rules=()) -> str: ...


# =========================================================================== #
# SPaRC -- forwards to harness.run and prompts.py; this path must stay
# byte-identical (tests/regress_sparc.py).
# =========================================================================== #
class SparcBenchmark(Benchmark):
    name = "sparc"
    key_fn = "puzzle-key-v1"
    interactive = True

    def __init__(self, config: str = "all", df_name: str = "lkaesberg/SPaRC"):
        self.config, self.df_name = config, df_name
        self._frames: dict[str, object] = {}
        self._rows: dict[str, dict] = {}

    # ---- dataset ---------------------------------------------------------
    def frame(self, split: str):
        """The split's dataframe, loaded once per process."""
        if split not in self._frames:
            from datasets import load_dataset
            df = load_dataset(self.df_name, self.config, split=split).to_pandas()
            self._frames[split] = df
            self._rows[split] = {str(df["id"][i]): i for i in range(len(df))}
        return self._frames[split]

    def ids(self, split, limit=None):
        df = self.frame(split)
        ids = [str(pid) for pid in df["id"]]
        return ids[:limit] if limit else ids

    def record(self, split, item_id):
        self.frame(split)
        return self.frame(split).iloc[self._rows[split][str(item_id)]]

    def difficulty(self, record) -> int:
        return int(record["difficulty_level"])

    def key_text(self, record) -> str:
        from .envs.sparc import puzzle_key_text
        return puzzle_key_text(record)

    def parse_reply(self, text):
        from .envs.sparc import parse_action
        return parse_action(text)

    def polyshapes(self, record):
        import yaml
        return yaml.safe_load(record["polyshapes"])

    # ---- running ---------------------------------------------------------
    async def run_baseline_suite(self, client, split, concurrency, limit=None):
        from .harness.run import run_sparc_suite_baseline
        return await run_sparc_suite_baseline(
            client, split=split, concurrency=concurrency, limit=limit)

    async def run_memory_pass(self, client, item_ids, block_for, split, concurrency):
        from .harness.run import run_sparc_suite_memory_pass
        return await run_sparc_suite_memory_pass(
            client, item_ids, block_for, split=split, concurrency=concurrency)

    # ---- bank construction ----------------------------------------------
    def gold_actions(self, record) -> list:
        from .envs.sparc import path_to_actions
        return path_to_actions(list(record["solutions"][0]["path"]))

    def gold_exemplar(self, record) -> dict:
        import yaml
        from .envs.sparc import grid_summary
        puzzle = yaml.safe_load(record["text_visualization"])["puzzle"]
        actions = self.gold_actions(record)
        return {
            "actions": actions, "n_moves": len(actions),
            "width": puzzle["dimensions"]["width"],
            "height": puzzle["dimensions"]["height"],
            "start": [puzzle["start"]["x"], puzzle["start"]["y"]],
            "end": [puzzle["end"]["x"], puzzle["end"]["y"]],
            "grid_summary": grid_summary(record),
        }

    def agent_exemplar(self, record, episode_ctx: dict) -> dict:
        ex = self.gold_exemplar(record)
        actions = [int(a) for _step, a, _loc in episode_ctx["moves"]]
        ex["actions"], ex["n_moves"] = actions, len(actions)
        return ex

    def gold_trajectory(self, record) -> dict:
        """Replay the gold path through a real env to capture its trace."""
        from .envs.sparc import SparcEnv, replay_path
        env = SparcEnv(split="train")
        try:
            return replay_path(env, record["id"], self.gold_actions(record))
        finally:
            env.close()

    # ---- prompts ---------------------------------------------------------
    def system_prompt(self, record) -> str:
        from .prompts import sparc_system_prompt
        return sparc_system_prompt(self.polyshapes(record))

    def reflection_prompt(self, episode, previous=()):
        from .prompts import reflexion_reflection_prompt
        return reflexion_reflection_prompt(episode, previous=previous)

    def format_reflections(self, items):
        from .prompts import format_reflections
        return format_reflections(items)

    def distill_trajectory_prompt(self, episode, key_text, record, succeeded=True):
        from .prompts import memp_distill_trajectory_prompt
        return memp_distill_trajectory_prompt(
            episode, key_text, self.polyshapes(record), succeeded=succeeded)

    def distill_gold_prompt(self, key_text, record):
        from .prompts import memp_distill_gold_prompt
        return memp_distill_gold_prompt(
            key_text, record["text_visualization"], self.gold_actions(record),
            self.polyshapes(record))

    def case_trajectory_prompt(self, episode, key_text, record, rules_preamble=True):
        from .prompts import memorybank_case_trajectory_prompt
        return memorybank_case_trajectory_prompt(
            episode, key_text, self.polyshapes(record),
            rules_preamble=rules_preamble)

    def case_gold_prompt(self, key_text, record, rules_preamble=True):
        from .prompts import memorybank_case_gold_prompt
        return memorybank_case_gold_prompt(
            key_text, record["text_visualization"], self.gold_actions(record),
            self.polyshapes(record), rules_preamble=rules_preamble)

    def format_procedures(self, entries, granularity):
        from .prompts import format_procedures
        return format_procedures(entries, granularity=granularity)

    def format_cases(self, entries, granularity):
        from .prompts import format_cases
        return format_cases(entries, granularity=granularity)

    def format_contrastive(self, entries, granularity, exemplars=False,
                           framing="mixed"):
        from .prompts import format_contrastive
        return format_contrastive(entries, granularity=granularity,
                                  exemplars=exemplars, framing=framing)

    def avoid_prompt(self, episode, key_text, record, rules=()):
        from .prompts import contrastive_avoid_prompt
        return contrastive_avoid_prompt(
            episode, key_text, self.polyshapes(record), rules=rules)


# =========================================================================== #
# StepGame -- single-turn
# =========================================================================== #
class StepGameBenchmark(Benchmark):
    """StepGame over the frozen splits built by scripts/build_stepgame_splits.py.

    `split` is a logical name ("train" / "test") resolved to a split file, so
    `--stepgame-test` can point at the hop-stratified split instead.
    """

    name = "stepgame"
    interactive = False

    def __init__(self, train_split=None, test_split=None, variant: str = "clean",
                 key_variant: str = "structural", prompt_protocol: str = "ours"):
        from .envs import stepgame
        from .prompts_stepgame_upstream import PROTOCOLS
        self._sg = stepgame
        self.variant = variant
        # "upstream*" protocols reproduce Li et al.'s prompt; see prompts_stepgame_upstream.
        if prompt_protocol not in PROTOCOLS:
            raise SystemExit(
                f"unknown --prompt-protocol {prompt_protocol!r} "
                f"(expected one of {', '.join(PROTOCOLS)})")
        self.prompt_protocol = prompt_protocol
        # Retrieval key: "structural" = reasoning shape, "lexical" = phrasing.
        if key_variant not in ("structural", "lexical"):
            raise ValueError(f"unknown key_variant {key_variant!r}")
        self.key_variant = key_variant
        self.key_fn = (stepgame.KEY_FN if key_variant == "structural"
                       else stepgame.KEY_FN_LEXICAL)
        self._paths = {"train": train_split or self._default("train"),
                       "test": test_split or self._default("test")}
        self._items: dict[str, dict] = {}

    def _default(self, split: str) -> pathlib.Path:
        """The standard k-stratified split file for this variant.

        Several candidates is an error, so arms compared against each other cannot
        silently score different item sets.
        """
        from .envs import stepgame
        d = stepgame.data_dir() / "splits"
        pattern = f"stepgame_{self.variant}_{split}_k*.jsonl"
        matches = sorted(p for p in d.glob(pattern) if "byhops" not in p.name)
        if not matches:
            raise FileNotFoundError(
                f"no {pattern} in {d}. Run "
                f"`python scripts/build_stepgame_splits.py` first.")
        if len(matches) > 1:
            names = "\n  ".join(p.name for p in matches)
            raise SystemExit(
                f"{len(matches)} candidate {split} splits in {d}:\n  {names}\n"
                f"Name one explicitly with --stepgame-{split} so every arm scores "
                f"the same items.")
        return matches[0]

    # ---- dataset ---------------------------------------------------------
    def items(self, split: str) -> dict:
        if split not in self._items:
            path = self._paths[split]
            self._items[split] = {i.id: i for i in self._sg.read_split(path)}
        return self._items[split]

    def split_path(self, split: str):
        return self._paths[split]

    def ids(self, split, limit=None):
        ids = list(self.items(split))
        return ids[:limit] if limit else ids

    def record(self, split, item_id):
        return self.items(split)[str(item_id)]

    def difficulty(self, record) -> int:
        return int(record.k)

    def secondary_difficulty(self, record) -> int:
        return int(record.hops)

    def key_text(self, record) -> str:
        if self.key_variant == "lexical":
            return self._sg.stepgame_key_text_lexical(record)
        return self._sg.stepgame_key_text(record)

    def parse_reply(self, text):
        return self._sg.parse_answer(text)

    def grade(self, record, predicted) -> bool:
        return self._sg.is_success(predicted, record.gold)

    def format_item(self, record) -> str:
        if self.prompt_protocol != "ours":
            from .prompts_stepgame_upstream import format_item_upstream
            return format_item_upstream(record)
        return self._sg.format_item(record)

    # ---- running ---------------------------------------------------------
    async def run_baseline_suite(self, client, split, concurrency, limit=None):
        from .harness.run import run_single_turn_suite
        return await run_single_turn_suite(
            self, client, self.ids(split, limit), lambda _id: "",
            split=split, concurrency=concurrency)

    async def run_memory_pass(self, client, item_ids, block_for, split, concurrency):
        from .harness.run import run_single_turn_suite
        return await run_single_turn_suite(
            self, client, item_ids, block_for, split=split, concurrency=concurrency)

    # ---- bank construction ----------------------------------------------
    def gold_exemplar(self, record) -> dict:
        """Assembled in code from the solver's own output, never by the LLM."""
        return {
            "hops": record.hops,
            "sentences": len(record.story),
            "unused": max(len(record.story) - record.hops, 0),
            "relations": list(record.extra.get("relations", [])),
            "answer": record.gold,
            "k": record.k,
        }

    def agent_exemplar(self, record, episode_ctx: dict) -> dict:
        """Identical to the gold one: StepGame has a single correct answer and no path."""
        return self.gold_exemplar(record)

    def gold_trajectory(self, record) -> dict:
        """The verbatim record: the question and the exact solver's worked derivation."""
        return {
            "first_obs": self._sg.format_item(record),
            "last_obs": record.gold_chain,
            "terminal_reason": "solved",
            "moves": [[0, record.gold, None]],
            "rule_status": {},
        }

    def verify_gold(self, split: str, n: int) -> None:
        """Re-derive n stored gold answers and exit if the split and solver disagree."""
        from .envs.stepgame_solver import solve
        items = list(self.items(split).values())[:n]
        for item in items:
            predicted = solve(item.story, item.question)
            if predicted != item.gold:
                raise SystemExit(
                    f"[verify-gold] {item.id}: solver says {predicted}, split file "
                    f"says {item.gold}. The split and the solver disagree; rebuild "
                    f"the splits or re-run scripts/verify_stepgame_solver.py.")
        print(f"[verify-gold] {len(items)} stored chains re-derived exactly",
              flush=True)

    # ---- prompts ---------------------------------------------------------
    def system_prompt(self, record=None) -> str:
        from .prompts_stepgame_upstream import system_prompt_for
        return system_prompt_for(self.prompt_protocol)

    def reflection_prompt(self, episode, previous=()):
        from .prompts_stepgame import stepgame_reflection_prompt
        return stepgame_reflection_prompt(episode, previous=previous)

    def format_reflections(self, items):
        from .prompts_stepgame import format_reflections
        return format_reflections(items)

    def distill_trajectory_prompt(self, episode, key_text, record, succeeded=True):
        from .prompts_stepgame import stepgame_distill_trajectory_prompt
        return stepgame_distill_trajectory_prompt(
            episode, key_text, record, succeeded=succeeded)

    def distill_gold_prompt(self, key_text, record):
        from .prompts_stepgame import stepgame_distill_gold_prompt
        return stepgame_distill_gold_prompt(key_text, record, record.gold_chain)

    def case_trajectory_prompt(self, episode, key_text, record, rules_preamble=True):
        from .prompts_stepgame import stepgame_case_trajectory_prompt
        return stepgame_case_trajectory_prompt(
            episode, key_text, record, rules_preamble=rules_preamble)

    def case_gold_prompt(self, key_text, record, rules_preamble=True):
        from .prompts_stepgame import stepgame_case_gold_prompt
        return stepgame_case_gold_prompt(
            key_text, record, record.gold_chain, rules_preamble=rules_preamble)

    def format_procedures(self, entries, granularity):
        from .prompts_stepgame import format_procedures
        return format_procedures(entries, granularity=granularity)

    def format_cases(self, entries, granularity):
        from .prompts_stepgame import format_cases
        return format_cases(entries, granularity=granularity)

    def format_contrastive(self, entries, granularity, exemplars=False,
                           framing="mixed"):
        from .prompts_stepgame import format_contrastive
        return format_contrastive(entries, granularity=granularity,
                                  exemplars=exemplars, framing=framing)

    def avoid_prompt(self, episode, key_text, record, rules=()):
        from .prompts_stepgame import stepgame_avoid_prompt
        return stepgame_avoid_prompt(episode, key_text, record, rules=rules)


# =========================================================================== #
BENCHMARKS = {"sparc": SparcBenchmark, "stepgame": StepGameBenchmark}


def add_benchmark_args(ap) -> None:
    """The benchmark-selection flags shared by all drivers.

    Keep this module free of heavy top-level imports: drivers import it on the login node.
    """
    ap.add_argument("--benchmark", default="sparc", choices=sorted(BENCHMARKS))
    ap.add_argument("--stepgame-train", default=None,
                    help="train split JSONL (default: the standard k-stratified one)")
    ap.add_argument("--stepgame-test", default=None,
                    help="test split JSONL; point at the *_byhops_* file for the "
                         "hop-stratified difficulty analysis")
    ap.add_argument("--key-variant", default="structural",
                    choices=["structural", "lexical"],
                    help="StepGame retrieval key: reasoning shape, or phrasing")
    ap.add_argument("--prompt-protocol", default="ours",
                    choices=["ours", "upstream-header", "upstream"],
                    help="StepGame actor prompt. `ours` is zero-shot and the "
                         "thesis protocol. `upstream-header` adds Li et al.'s "
                         "clock/cardinal decoders, still zero-shot. `upstream` "
                         "is their full 10-shot prompt -- a calibration point "
                         "only: its exemplars are injected memory, so it is NOT "
                         "a memoryless baseline.")


def build_benchmark(args) -> Benchmark:
    """Resolve the flags above into a Benchmark.

    `args` may be a Namespace or a run's stored config dict, so a resumed job
    rebuilds the benchmark it started with.
    """
    get = (args.get if isinstance(args, dict)
           else lambda k, d=None: getattr(args, k, d))
    if get("benchmark", "sparc") == "stepgame":
        return get_benchmark(
            "stepgame",
            train_split=get("stepgame_train"),
            test_split=get("stepgame_test"),
            key_variant=get("key_variant", "structural") or "structural",
            prompt_protocol=get("prompt_protocol", "ours") or "ours")
    return get_benchmark("sparc")


def get_benchmark(name: str, **kwargs) -> Benchmark:
    """Resolve a --benchmark flag. Unknown names fail loudly, not by defaulting."""
    try:
        cls = BENCHMARKS[name]
    except KeyError:
        raise SystemExit(
            f"unknown --benchmark {name!r} (expected one of "
            f"{', '.join(sorted(BENCHMARKS))})") from None
    return cls(**kwargs)
