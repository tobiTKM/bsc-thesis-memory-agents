from ..agents.base import BaselineAgent
from ..agents.memory_agent import MemoryAgent
from ..envs.base import EpisodeResult, Transition, make_json_safe

import asyncio

# The SPaRC env stack (gymnasium + Spatial_Gym) and `datasets` are imported inside
# the SPaRC functions, not here. A single-turn benchmark needs none of them, and
# neither does the login node where --phase compare is meant to run; importing
# them at module scope made every driver that touches the harness depend on the
# whole SPaRC stack being installed and importable.


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
    _E, format_obs, is_success, _sp, _ld, _y = _sparc_deps()
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
            # Convert numpy -> plain python at the boundary where env data enters.
            # The env hands back ndarrays and numpy bools; leaving them raw made a
            # trajectory unserialisable (it killed a 2 h trial in json.dumps) and made
            # the reflection prompt render `[0 2]` in-process but `[0, 2]` after a
            # resume -- the same reflection formatted two ways depending on whether
            # the job happened to restart.
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
    SparcEnv, _fo, _is, sparc_system_prompt, load_dataset, yaml = _sparc_deps()
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


async def run_sparc_suite_memory_pass(client, puzzle_ids, block_for, split="test",
                                      concurrency=32):
    """Memory-augmented pass: re-attempt an explicit set of puzzles, each with its
    own memory block injected into the actor's context.

    This is method-agnostic on purpose (see the Memory ABC). Reflexion is the first
    caller -- `puzzle_ids` is the still-unsolved set and `block_for(pid)` renders
    that puzzle's reflections -- but MemoryBank / MemP reuse this same pass with a
    `block_for` that retrieves from a bank instead. The env-pool / queue concurrency
    model is identical to the baseline suite; only which puzzles run and the agent's
    injected block differ.

    The puzzle set is passed in rather than derived from the memory store because
    the two must be able to disagree. Both Reflexion arms have to re-attempt the
    *same* set for the comparison to hold, yet the control arm has no memory at all
    and even the memory arm can end up with an empty entry (a reflection call that
    came back empty). Deriving the set from the store would silently shrink the
    control arm to nothing and drop exactly the puzzles whose reflection failed.
    """
    SparcEnv, _fo, _is, sparc_system_prompt, load_dataset, yaml = _sparc_deps()
    ds = load_dataset("lkaesberg/SPaRC", "all", split=split)
    df = ds.to_pandas()
    # id -> row index, so we can fetch the polyshapes for exactly the requested set.
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
# single-turn benchmarks (StepGame now, LogicGame next)
# --------------------------------------------------------------------------- #
# The two suites above drive an interactive env: each move needs the state the
# env returned for the previous one, and concurrency comes from a pool of envs.
# A single-turn benchmark has no env and no state to protect, so the pool is gone
# and a plain semaphore caps in-flight calls. Everything downstream -- the
# EpisodeResult shape, the token accounting, the format-failure retry loop, the
# crash handling -- is shared, which is what keeps the two benchmarks comparable.

# The reply text kept on the episode for a later reflection to read. Capped
# because a thinking model's visible reply can run to thousands of tokens, and
# every one of them would be re-sent inside the reflection prompt of every
# subsequent trial.
_REPLY_KEEP = 4000


def _reply_text(response) -> str:
    """What the model produced, for the reflection prompt.

    Prefers the visible reply. Falls back to the tail of the stripped <think>
    block when the visible part is a bare "Final: <label>" -- which is the normal
    case for a reasoning model, and reflecting on "you answered left" with no
    trace of *why* would ask the reflector to diagnose a decision it cannot see.
    """
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
        return "format-failure"    # never produced a parseable answer
    return "wrong-answer"          # answered, but wrongly


async def run_single_turn_episode(benchmark, agent, record):
    """One item, one reply, one EpisodeResult.

    The `terminated` / `truncated` pair keeps the same meaning it has on SPaRC:
    terminated means the agent submitted something, truncated means the episode
    was cut short without one. So a graded answer is terminated, and only a format
    failure is truncated -- which is what lets summarize_results read both
    benchmarks with the same code.
    """
    first_obs = benchmark.format_item(record)
    extra = {"hops": benchmark.secondary_difficulty(record), "gold": record.gold}
    action, response = await agent.act(first_obs)

    if action is None:
        # Every reply was unparseable. No Transition is appended, so the calls
        # that burned the budget are charged to the episode here (retry_* already
        # covers all of them, this response's own counts included).
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
    """Run a single-turn benchmark over an explicit item set, `concurrency` at a time.

    One function for both the memoryless pass and the memory-augmented one: the
    baseline is `block_for = lambda _id: ""`, which makes MemoryAgent identical to
    BaselineAgent (its own docstring says so). Unlike the SPaRC pair, there is no
    reason to keep two versions -- there is no env pool whose lifetime differs.
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
