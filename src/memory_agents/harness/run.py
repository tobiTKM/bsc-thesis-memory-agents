from ..agents.base import BaselineAgent
from ..envs.base import EpisodeResult, Transition
from ..envs.sparc import SparcEnv, is_success
from ..prompts import sparc_system_prompt

import asyncio
import yaml
from datasets import load_dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bar is optional
    def tqdm(x, **_):
        return x


async def run_sparc_episode_baseline(env, agent, obs, info, puzzle_id):
    """Run one puzzle to termination and return its EpisodeResult.

    The loop is sequential by nature: each move needs the grid state the env
    returned for the previous move. Concurrency happens *across* episodes (see
    run_sparc_suite_baseline), not inside one.
    """
    reward = 0.0
    transitions = []
    terminated = truncated = False

    while True:
        user_msg = env.user_message(obs, info, reward)
        action, response = await agent.act(user_msg)

        if action is None:
            # Repeated malformed output: abort as a (failed) truncation, matching
            # the established host's behaviour.
            truncated = True
            break

        obs, reward, terminated, truncated, info = env.step(action)
        transitions.append(Transition(
            step=len(transitions),
            action=action,
            reward=reward,
            legal_actions=list(info.get("legal_actions", [])),
            rule_status=info.get("rule_status", {}),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        ))

        if terminated or truncated:
            break

    return EpisodeResult(
        puzzle_id=puzzle_id,
        success=is_success(info),
        terminated=terminated,
        truncated=truncated,
        difficulty=int(info.get("difficulty", -1)),
        transitions=transitions,
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
        try:
            puzzle_id = df["id"][i]
            obs, info = env.reset(puzzle_id=puzzle_id)

            polyshapes = yaml.safe_load(df["polyshapes"][i])
            system_prompt = sparc_system_prompt(polyshapes)

            agent = BaselineAgent(client)
            agent.reset(system_prompt, label=puzzle_id)

            return await run_sparc_episode_baseline(env, agent, obs, info, puzzle_id)
        finally:
            await env_pool.put(env)         # return the env even if the episode raised

    # Launch every puzzle as a task; the queue caps how many actually run at once.
    tasks = [asyncio.create_task(run_one(i)) for i in range(n)]
    results = []
    for finished in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await finished)
    return results
