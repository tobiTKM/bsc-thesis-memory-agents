from ..agents.base import BaselineAgent
from ..agents.memory_agent import MemoryAgent
from ..envs.base import EpisodeResult, Transition
from ..envs.sparc import SparcEnv, format_obs, is_success
from ..prompts import format_reflections, sparc_system_prompt

import asyncio
import yaml
from datasets import load_dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bar is optional
    def tqdm(x, **_):
        return x


def _terminal_reason(success, terminated, format_failed) -> str:
    """Why the episode ended -- the single most useful line for a Reflector.

    Kept coarse on purpose: the exact env signals are `success` (outcome_reward),
    `terminated` (a solution was submitted) and truncation. Per this project's
    established finding, a truncation here is a dead end, not a step-limit cutoff.
    """
    if success:
        return "solved"
    if format_failed:
        return "format-failure"      # the model never produced a parseable move
    if terminated:
        return "wrong-solution"      # submitted a path, but it broke a rule
    return "dead-end"                 # painted into a corner: no legal way forward


def _drain_and_close(env_pool: asyncio.Queue) -> None:
    """Close every env in the pool. Each episode returns its env in a finally, so
    by the time the gather loop is done the pool holds all of them again."""
    while not env_pool.empty():
        env_pool.get_nowait().close()


def errored_episode(puzzle_id, exc) -> EpisodeResult:
    """Record a crashed episode as a failure instead of letting it kill the suite.

    There is no try/except between client.chat() and asyncio.as_completed below, so
    a single transient error (a connection reset, a 5xx from vLLM) used to propagate
    out of the gather loop and take the whole run down -- losing every result from a
    multi-hour job because one call blipped at hour eleven. Recording it keeps the
    run going and makes the error count visible in the summary.

    Deliberately neither terminated nor truncated: an error is not a dead end and
    must not be counted as one. `terminal_reason` is what tells them apart.
    """
    return EpisodeResult(
        puzzle_id=str(puzzle_id), success=False, terminated=False, truncated=False,
        difficulty=-1, terminal_reason="error", error=repr(exc))


async def run_sparc_episode_baseline(env, agent, obs, info, puzzle_id):
    """Run one puzzle to termination and return its EpisodeResult.

    The loop is sequential by nature: each move needs the grid state the env
    returned for the previous move. Concurrency happens *across* episodes (see
    run_sparc_suite_baseline), not inside one. The same loop drives both the
    memoryless baseline and a Reflexion trial -- only the `agent` differs.
    """
    reward = 0.0
    transitions = []
    terminated = truncated = format_failed = False
    aborted_prompt = aborted_completion = 0
    first_obs = format_obs(obs)      # the starting grid, kept for reflection

    while True:
        user_msg = env.user_message(obs, info, reward)
        action, response = await agent.act(user_msg)

        if action is None:
            # Repeated malformed output: abort as a (failed) truncation, matching
            # the established host's behaviour. No Transition is appended for this
            # move, so the replies that burned the budget are charged to the episode
            # here instead -- retry_* already covers all of them, response's own
            # counts included (see BaselineAgent.act).
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
            legal_actions=list(info.get("legal_actions", [])),
            rule_status=info.get("rule_status", {}),
            agent_location=info.get("agent_location"),
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
        last_obs=format_obs(obs),     # the final grid = the full path the agent drew
        terminal_reason=_terminal_reason(success, terminated, format_failed),
        aborted_prompt_tokens=aborted_prompt,
        aborted_completion_tokens=aborted_completion,
    )


async def run_sparc_suite_baseline(client, split="test", concurrency=32, limit=None):
    """Run the memoryless baseline over the SPaRC split, `concurrency` at a time.

    Why a pool of envs instead of one shared env: while episode A is awaiting its
    LLM call, episode B runs env.step on its own env. A single shared env would
    let B overwrite A's puzzle state. So we keep exactly `concurrency` envs -- one
    per in-flight episode -- and hand them out through a queue. The queue also
    *is* the concurrency limiter: only `concurrency` episodes can hold an env at
    once, which is exactly how many LLM calls are in flight against the server.
    Each episode also gets its own agent, since an agent owns one message list.
    """
    ds = load_dataset("lkaesberg/SPaRC", "all", split=split)
    df = ds.to_pandas()
    n = len(df) if limit is None else min(limit, len(df))

    # Pre-create the env pool. Each SparcEnv loads the dataset once, so this
    # costs `pool_size` loads up front (not one per puzzle). No point making
    # more envs than puzzles.
    pool_size = min(concurrency, n)
    env_pool: asyncio.Queue = asyncio.Queue()
    for _ in range(pool_size):
        await env_pool.put(SparcEnv(split=split))

    async def run_one(i):
        env = await env_pool.get()          # blocks until an env is free
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
            await env_pool.put(env)         # return the env even if the episode raised

    # Launch every puzzle as a task; the queue caps how many actually run at once.
    tasks = [asyncio.create_task(run_one(i)) for i in range(n)]
    results = []
    try:
        for finished in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.append(await finished)
    finally:
        _drain_and_close(env_pool)
    return results


async def run_sparc_suite_memory_pass(client, memory, split="test", concurrency=32,
                                      block_of=format_reflections):
    """Memory-augmented pass: re-attempt only the puzzles that have stored memory,
    each with its items injected into the actor's context.

    This is method-agnostic on purpose (see the Memory ABC). Reflexion is the
    first caller -- `memory` maps puzzle_id -> its reflections, and `block_of`
    renders those into a system-prompt block -- but MemoryBank / MemP reuse this
    same pass by producing `memory` differently and, if needed, passing their own
    `block_of`. Puzzles solved in the memoryless pass are absent from `memory`:
    the caller carries their win forward rather than re-running a deterministic
    success. The env-pool / queue concurrency model is identical to the baseline
    suite; only which puzzles run and the agent's injected block differ.
    """
    ds = load_dataset("lkaesberg/SPaRC", "all", split=split)
    df = ds.to_pandas()
    # id -> row index, so we can fetch the polyshapes for exactly the stored set.
    row_of = {str(df["id"][i]): i for i in range(len(df))}
    todo = [pid for pid in memory if pid in row_of]

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
            agent.reset(system_prompt, memory_block=block_of(memory[puzzle_id]),
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
