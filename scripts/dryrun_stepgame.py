#!/usr/bin/env python3
"""Run the whole StepGame pipeline end to end against a mock model. No GPU.

Every phase of every arm -- baseline, Reflexion root/arm/compare, MemP
build/distill/embed/upgrade/eval, MemoryBank distill/eval -- exercised on a small
slice with a fake LLM and fake embeddings, so a plumbing error surfaces here in
seconds instead of an hour into a SLURM job. It checks that the pipeline RUNS and
that its files chain together; it says nothing about whether memory helps.

The mock answers a configurable fraction of items correctly (by reading the gold
label, which is exactly what makes it a mock and not an agent), so both the
solved and the failed paths through distillation get taken.

    python scripts/dryrun_stepgame.py
    python scripts/dryrun_stepgame.py --keep      # leave the artefacts for inspection
"""

import argparse
import asyncio
import hashlib
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from memory_agents.envs import stepgame          # noqa: E402
from memory_agents.envs.stepgame_solver import LABELS  # noqa: E402
from memory_agents.llm.client import LLMResponse  # noqa: E402


class MockClient:
    """Stands in for LLMClient. Deterministic given (prompt, seed).

    Answers correctly for `accuracy` of items and returns a plausible wrong label
    otherwise; for a distill/reflect prompt (no story to answer) it returns a
    short piece of prose. Token counts are made up but consistent, so the
    accounting code is exercised rather than bypassed.
    """

    def __init__(self, gold_by_story: dict, accuracy: float = 0.5,
                 model: str = "mock", seed: int = 0):
        self.gold_by_story = gold_by_story
        self.accuracy = accuracy
        self.model = model
        self.seed = seed
        self.calls = 0

    async def chat(self, messages):
        self.calls += 1
        text = "\n".join(m["content"] for m in messages)
        rng = random.Random(hashlib.sha256(
            f"{self.seed}:{text}".encode()).hexdigest())

        gold = next((g for story, g in self.gold_by_story.items()
                     if story in text), None)
        if gold is None:                       # a distill / reflection prompt
            body = ("Applies to: short chains of cardinal relations\n"
                    "1. Fix the reference object and place it at the origin.\n"
                    "2. Convert each sentence to a label before chaining.\n"
                    "3. Sum the offsets and read off the sign.")
        elif rng.random() < self.accuracy:
            body = f"Chaining the relations.\nFinal: {gold}"
        else:
            wrong = [l for l in LABELS if l != gold]
            body = f"Chaining the relations.\nFinal: {rng.choice(wrong)}"
        return LLMResponse(text=body, prompt_tokens=len(text) // 4,
                           completion_tokens=len(body) // 4,
                           finish_reason="stop", reasoning_tokens=0,
                           reasoning_text="some deliberation")

    async def aclose(self):
        pass


class MockEmbeddingClient:
    """Deterministic pseudo-embeddings: a hash of the key text, unit-normalised.

    Meaningless as semantics, but they exercise the real retrieval code -- shape
    checks, the id/row alignment sanity check, cosine top-k, the tie-break -- which
    is what the dry run is for.
    """

    DIM = 32

    def __init__(self, base_url=None, model="mock-embed", **_kw):
        self.base_url = base_url
        self.model = model

    async def embed(self, texts):
        import numpy as np
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer((h * (self.DIM // 8 + 1))[:self.DIM * 4],
                              dtype=np.uint32).astype(np.float32)
            v = v / (np.linalg.norm(v) or 1.0)
            out.append(v.tolist())
        return out

    async def aclose(self):
        pass


def build_splits(tmp: pathlib.Path, n_train: int, n_test: int):
    """Small frozen splits carved out of the real ones, so ids and gold chains
    are real rather than synthesised."""
    src = stepgame.data_dir() / "splits"
    out = tmp / "splits"
    out.mkdir(parents=True)
    pairs = []
    for name, n in (("train", n_train), ("test", n_test)):
        matches = sorted(p for p in src.glob(f"stepgame_clean_{name}_k*.jsonl")
                         if "byhops" not in p.name)
        if not matches:
            raise SystemExit("build the real splits first: "
                             "python scripts/build_stepgame_splits.py")
        items = stepgame.read_split(matches[0])[:n]
        dst = out / matches[0].name
        dst.write_text("".join(__import__("json").dumps(i.to_dict()) + "\n"
                               for i in items))
        pairs.append((dst, items))
    return pairs


def run(cmd: list[str], env: dict, cwd: pathlib.Path) -> None:
    """Run a driver with `cwd` as its working directory.

    Not the repo: every driver writes to a relative `results/json/...`, so running
    from the scratch directory keeps the mock's output out of the real results
    tree instead of leaving files that look like runs but came from a fake model.
    """
    shown = " ".join(str(c) for c in cmd)
    print(f"\n$ {shown}", flush=True)
    proc = subprocess.run([sys.executable, *[str(REPO / cmd[0])],
                           *[str(c) for c in cmd[1:]]],
                          cwd=cwd, env=env, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()[-6:]
    for line in tail:
        print(f"  | {line}")
    if proc.returncode != 0:
        print((proc.stderr or "").strip()[-3000:])
        raise SystemExit(f"FAILED: {' '.join(str(c) for c in cmd)}")


def newest(pattern: str, root: pathlib.Path) -> pathlib.Path:
    matches = sorted((root / "results/json").glob(pattern),
                     key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(f"expected a file matching {pattern}")
    return matches[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=12)
    ap.add_argument("--test", type=int, default=12)
    ap.add_argument("--accuracy", type=float, default=0.5)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stepgame_dryrun_"))
    print(f"scratch: {tmp}")
    (train_path, train_items), (test_path, test_items) = build_splits(
        tmp, args.train, args.test)

    # The mock is installed by an env var the drivers read through a sitecustomize
    # shim, so the drivers themselves stay untouched by test-only code.
    shim = tmp / "sitecustomize.py"
    shim.write_text(f'''
import sys, pathlib, json
sys.path.insert(0, {str(REPO / "src")!r})
sys.path.insert(0, {str(REPO / "scripts")!r})
import dryrun_stepgame as D
from memory_agents.envs import stepgame as _sg
_gold = {{}}
for _p in ({str(train_path)!r}, {str(test_path)!r}):
    for _i in _sg.read_split(_p):
        _gold[_i.question.strip()] = _i.gold
import memory_agents.llm.client as _c
_acc = {args.accuracy!r}
class _MC(D.MockClient):
    def __init__(self, *a, **k):
        super().__init__(_gold, accuracy=_acc, model="mock",
                         seed=k.get("seed", 0) or 0)
_c.LLMClient = _MC
_c.EmbeddingClient = D.MockEmbeddingClient
for _m in ("run_baseline", "run_reflexion", "run_memp", "run_memorybank"):
    try:
        _mod = __import__(_m)
        if hasattr(_mod, "LLMClient"):
            _mod.LLMClient = _MC
        if hasattr(_mod, "EmbeddingClient"):
            _mod.EmbeddingClient = D.MockEmbeddingClient
    except Exception:
        pass
''')

    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tmp}:{REPO / 'src'}:{REPO / 'scripts'}"
    env["LLM_BASE_URL"] = "http://mock"
    env["LLM_MODEL"] = "mock"
    env["LLM_MODEL_ID"] = "mock/dryrun"
    env["STEPGAME_DATA"] = str(stepgame.data_dir())

    common = ["--benchmark", "stepgame",
              "--stepgame-train", train_path, "--stepgame-test", test_path,
              "--concurrency", "4"]

    print("\n" + "=" * 70 + "\n  1. BASELINE\n" + "=" * 70)
    run(["scripts/run_baseline.py", *common, "--split", "test"], env, tmp)
    baseline = newest("baseline_stepgame_*.json", tmp)

    print("\n" + "=" * 70 + "\n  2. REFLEXION\n" + "=" * 70)
    run(["scripts/run_reflexion.py", "--phase", "root", *common,
         "--split", "test"], env, tmp)
    root = newest("reflexion_root_stepgame_*.json", tmp)
    for arm in ("memory", "control"):
        run(["scripts/run_reflexion.py", "--phase", "arm", "--arm", arm,
             "--root", root, "--trials", "3", "--concurrency", "4"], env, tmp)
    mem = newest("reflexion_memory_stepgame_*.json", tmp)
    ctl = newest("reflexion_control_stepgame_*.json", tmp)
    run(["scripts/run_reflexion.py", "--phase", "compare",
         "--memory", mem, "--control", ctl], env, tmp)

    print("\n" + "=" * 70 + "\n  3. MEMP\n" + "=" * 70)
    run(["scripts/run_memp.py", "--phase", "build", *common,
         "--split", "train", "--chunk", "8"], env, tmp)
    build = newest("memp_build_stepgame_train_*.json", tmp)
    run(["scripts/run_memp.py", "--phase", "distill", *common,
         "--build", build, "--chunk", "8", "--verify-gold", "5"], env, tmp)
    bank = newest("memp_bank_stepgame_*.json", tmp)
    run(["scripts/run_memp.py", "--phase", "embed", *common], env, tmp)
    emb = newest("memp_embeddings_stepgame_*.npz", tmp)
    run(["scripts/run_memp.py", "--phase", "upgrade", *common, "--bank", bank], env, tmp)
    bank_traj = pathlib.Path(str(bank).replace(".json", "_traj.json"))
    for arm in ("retrieved", "random", "fixed"):
        run(["scripts/run_memp.py", "--phase", "eval", *common, "--arm", arm,
             "--split", "test", "--bank", bank_traj, "--embeddings", emb,
             "--chunk", "8"], env, tmp)
    retrieved = newest("memp_eval_stepgame_retrieved_*.json", tmp)
    rnd = newest("memp_eval_stepgame_random_*.json", tmp)
    run(["scripts/run_memp.py", "--phase", "compare", "--baseline", baseline,
         "--retrieved", retrieved, "--random", rnd], env, tmp)

    print("\n" + "=" * 70 + "\n  4. MEMORYBANK\n" + "=" * 70)
    run(["scripts/run_memorybank.py", "--phase", "distill", *common,
         "--build", build, "--chunk", "8", "--rules-preamble", "on",
         "--verify-gold", "5"], env, tmp)
    mb_bank = newest("memorybank_bank_stepgame_*.json", tmp)
    for arm in ("retrieved", "random"):
        run(["scripts/run_memorybank.py", "--phase", "eval", *common, "--arm", arm,
             "--split", "test", "--bank", mb_bank, "--embeddings", emb,
             "--chunk", "8"], env, tmp)
    mb_ret = newest("memorybank_eval_stepgame_retrieved_*.json", tmp)
    mb_rnd = newest("memorybank_eval_stepgame_random_*.json", tmp)
    run(["scripts/run_memorybank.py", "--phase", "compare", "--baseline", baseline,
         "--retrieved", mb_ret, "--random", mb_rnd], env, tmp)

    print("\n" + "=" * 70)
    print("  ALL PHASES OF ALL THREE MEMORY ARMS RAN END TO END")
    print("=" * 70)
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"artefacts kept: {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
