"""Episode runners: SPaRC suites over a pool of envs, and single-turn suites
under a semaphore, both run concurrently against one LLM server.
"""

from ..agents.base import BaselineAgent
from ..agents.memory_agent import MemoryAgent
from ..envs.base import EpisodeResult, Transition, make_json_safe

import asyncio

# SPaRC deps are imported lazily so single-turn runs and --phase compare work
# without them installed.


def _sparc_deps():
    """The SPaRC-only imports, resolved on first use."""
    import yaml
    from datasets import load_dataset
    from ..envs.sparc import SparcEnv, format_obs, is_success
    from ..prompts import sparc_system_prompt
    return SparcEnv, format_obs, is_success, sparc_system_prompt, load_dataset, yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bar is optional
    def tqdm(x, **_):
        return x


def _terminal_reason(success, terminated, format_failed) -> str:
    """Why the episode ended; on SPaRC a truncation is a dead end, not a step limit."""
    if success:
        return "solved"
    if format_failed:
        return "format-failure"      # never produced a parseable move
    if terminated:
        return "wrong-solution"      # submitted a path that broke a rule
    return "dead-end"                 # no legal move left


def _drain_and_close(env_pool: asyncio.Queue) -> None:
    """Close every env in the pool (each episode returns its env in a finally)."""
    while not env_pool.empty():
        env_pool.get_nowait().close()


def errored_episode(puzzle_id, exc) -> EpisodeResult:
    """Record a crashed episode as a failure instead of letting it kill the suite.

    Neither terminated nor truncated, so it is not counted as a dead end.
    """
    return EpisodeResult(
        puzzle_id=str(puzzle_id), success=False, terminated=False, truncated=False,
        difficulty=-1, terminal_reason="error", error=repr(exc))


async def run_sparc_episode_baseline(env, agent, obs, info, puzzle_id):
    """Run one SPaRC puzzle to termination with any agent; return its EpisodeResult."""
    _E, format_obs, is_success, _sp, _ld, _y = _sparc_deps()
    reward = 0.0
    transitions = []
    terminated = truncated = format_failed = False
    aborted_prompt = aborted_completion = 0
    first_obs = format_obs(obs)

    while True:
        user_msg = env.user_message(obs, info, reward)
        action, response = await agent.act(user_msg)

        if action is None:
            # Repeated malformed output: abort as a truncation. No Transition is
            # appended, so retry_* (incl. response's own counts) is charged here.
            truncated = format_failed = True
            if response is not None:
                aborted_prompt = response.retry_prompt_tokens
                aborted_completion = response.retry_completion_tokens
            break

        obs, reward, terminated, truncated, info = env.step(action)
        transitions.append(Transition(
            step=len(transitions),
            action=action,
            reward=reward,
            # numpy -> plain Python here, so trajectories serialise and render the
            # same in-process and after a resume.
            legal_actions=make_json_safe(list(info.get("legal_actions", []))),
            rule_status=make_json_safe(info.get("rule_status", {})),
            agent_location=make_json_safe(info.get("agent_location")),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            retry_prompt_tokens=response.retry_prompt_tokens,
            retry_completion_tokens=response.retry_completion_tokens,
        ))

        if terminated or truncated:
            break

    success = is_success(info)
    return EpisodeResult(
        puzzle_id=puzzle_id,
        success=success,
        terminated=terminated,
        truncated=truncated,
        difficulty=int(info.get("difficulty", -1)),
        transitions=transitions,
        first_obs=first_obs,
        last_obs=format_obs(obs),     # final grid, showing the full path drawn
        terminal_reason=_terminal_reason(success, terminated, format_failed),
        aborted_prompt_tokens=aborted_prompt,
        aborted_completion_tokens=aborted_completion,
    )


async def run_sparc_suite_baseline(client, split="test", concurrency=32, limit=None):
    """Run the memoryless baseline over the SPaRC split, `concurrency` at a time.

    Each in-flight episode needs its own env (a shared one would be overwritten
    while another awaits its LLM call) and its own agent; the env queue is also
    the concurrency limiter.
    """
    SparcEnv, _fo, _is, sparc_system_prompt, load_dataset, yaml = _sparc_deps()
    ds = load_dataset("lkaesberg/SPaRC", "all", split=split)
    df = ds.to_pandas()
    n = len(df) if limit is None else min(limit, len(df))

    # Each SparcEnv loads the dataset once, so the pool is created up front.
    pool_size = min(concurrency, n)
    env_pool: asyncio.Queue = asyncio.Queue()
    for _ in range(pool_size):
        await env_pool.put(SparcEnv(split=split))

    async def run_one(i):
        env = await env_pool.get()
        puzzle_id = None
        try:
            puzzle_id = df["id"][i]
            obs, info = env.reset(puzzle_id=puzzle_id)

            polyshapes = yaml.safe_load(df["polyshapes"][i])
            system_prompt = sparc_system_prompt(polyshapes)

            agent = BaselineAgent(client)
            agent.reset(system_prompt, label=puzzle_id)

            return await run_sparc_episode_baseline(env, agent, obs, info, puzzle_id)
        except Exception as exc:
            print(f"[error] puzzle {puzzle_id}: {exc!r}", flush=True)
            return errored_episode(puzzle_id, exc)
        finally:
            await env_pool.put(env)

    tasks = [asyncio.create_task(run_one(i)) for i in range(n)]
    results = []
    try:
        for finished in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.append(await finished)
    finally:
        _drain_and_close(env_pool)
    return results


async def run_sparc_suite_memory_pass(client, puzzle_ids, block_for, split="test",
                                      concurrency=32):
    """Re-attempt `puzzle_ids` with `block_for(pid)` injected, for any memory method.

    The set is passed in, not derived from the store, so a no-memory control arm
    and puzzles with an empty entry still run the same set.
    """
    SparcEnv, _fo, _is, sparc_system_prompt, load_dataset, yaml = _sparc_deps()
    ds = load_dataset("lkaesberg/SPaRC", "all", split=split)
    df = ds.to_pandas()
    row_of = {str(df["id"][i]): i for i in range(len(df))}
    todo = [str(pid) for pid in puzzle_ids if str(pid) in row_of]

    pool_size = min(concurrency, len(todo)) or 1
    env_pool: asyncio.Queue = asyncio.Queue()
    for _ in range(pool_size):
        await env_pool.put(SparcEnv(split=split))

    async def run_one(puzzle_id):
        env = await env_pool.get()
        try:
            i = row_of[puzzle_id]
            obs, info = env.reset(puzzle_id=df["id"][i])

            polyshapes = yaml.safe_load(df["polyshapes"][i])
            system_prompt = sparc_system_prompt(polyshapes)

            agent = MemoryAgent(client)
            agent.reset(system_prompt, memory_block=block_for(puzzle_id),
                        label=df["id"][i])

            return await run_sparc_episode_baseline(env, agent, obs, info, df["id"][i])
        except Exception as exc:
            print(f"[error] puzzle {puzzle_id}: {exc!r}", flush=True)
            return errored_episode(puzzle_id, exc)
        finally:
            await env_pool.put(env)

    tasks = [asyncio.create_task(run_one(pid)) for pid in todo]
    results = []
    try:
        for finished in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.append(await finished)
    finally:
        _drain_and_close(env_pool)
    return results


# --------------------------------------------------------------------------- #
# single-turn benchmarks (no env; a semaphore caps in-flight calls)
# --------------------------------------------------------------------------- #

# Cap on reply text kept for later reflection prompts.
_REPLY_KEEP = 4000


def _reply_text(response) -> str:
    """Reply text for reflection; adds the reasoning tail if the reply is short."""
    if response is None:
        return ""
    text = (response.text or "").strip()
    if len(text) < 200 and (response.reasoning_text or "").strip():
        tail = response.reasoning_text.strip()[-_REPLY_KEEP:]
        return (f"{text}\n\n(reasoning leading to it)\n{tail}" if text else tail)
    return text[:_REPLY_KEEP]


def _single_turn_reason(success, format_failed) -> str:
    if success:
        return "solved"
    if format_failed:
        return "format-failure"
    return "wrong-answer"


async def run_single_turn_episode(benchmark, agent, record):
    """One item, one reply, one EpisodeResult.

    As on SPaRC, terminated means an answer was submitted and truncated means a
    format failure; summarize_results relies on this.
    """
    first_obs = benchmark.format_item(record)
    extra = {"hops": benchmark.secondary_difficulty(record), "gold": record.gold}
    action, response = await agent.act(first_obs)

    if action is None:
        # No Transition is appended, so retry_* (incl. this response) is charged here.
        return EpisodeResult(
            puzzle_id=str(record.id), success=False, terminated=False,
            truncated=True, difficulty=benchmark.difficulty(record),
            transitions=[], first_obs=first_obs, last_obs=_reply_text(response),
            terminal_reason="format-failure", extra={**extra, "predicted": None},
            aborted_prompt_tokens=response.retry_prompt_tokens if response else 0,
            aborted_completion_tokens=(response.retry_completion_tokens
                                       if response else 0))

    success = benchmark.grade(record, action)
    transition = Transition(
        step=0, action=action, reward=1.0 if success else 0.0,
        legal_actions=[], rule_status={}, agent_location=None,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        retry_prompt_tokens=response.retry_prompt_tokens,
        retry_completion_tokens=response.retry_completion_tokens)
    return EpisodeResult(
        puzzle_id=str(record.id), success=success, terminated=True, truncated=False,
        difficulty=benchmark.difficulty(record), transitions=[transition],
        first_obs=first_obs, last_obs=_reply_text(response),
        terminal_reason=_single_turn_reason(success, False),
        extra={**extra, "predicted": action})


async def run_single_turn_suite(benchmark, client, item_ids, block_for,
                                split="test", concurrency=32):
    """Run a single-turn benchmark over `item_ids`, `concurrency` at a time.

    For the baseline pass `block_for = lambda _id: ""`.
    """
    todo = [str(i) for i in item_ids]
    sem = asyncio.Semaphore(max(concurrency, 1))
    system_prompt = benchmark.system_prompt(None)

    async def run_one(item_id):
        async with sem:
            try:
                record = benchmark.record(split, item_id)
                agent = MemoryAgent(client, parse_reply=benchmark.parse_reply)
                agent.reset(system_prompt, memory_block=block_for(item_id),
                            label=item_id)
                return await run_single_turn_episode(benchmark, agent, record)
            except Exception as exc:
                print(f"[error] item {item_id}: {exc!r}", flush=True)
                return errored_episode(item_id, exc)

    tasks = [asyncio.create_task(run_one(i)) for i in todo]
    results = []
    for finished in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await finished)
    return results
