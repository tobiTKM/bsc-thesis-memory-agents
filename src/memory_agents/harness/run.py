"""Episode driver for SPaRC. One LLM call per move; the env enforces the step
budget and truncation. This is the loop every memory method plugs into.
"""
from __future__ import annotations

from ..agents.base import BaselineAgent
from ..envs.base import EpisodeResult, Transition
from ..envs.sparc import SparcEnv, is_success


def run_sparc_episode(env: SparcEnv, agent: BaselineAgent, puzzle_id,
                      system_prompt: str) -> EpisodeResult:
    obs, info = env.reset(puzzle_id)
    agent.reset(system_prompt)

    reward = 0.0
    transitions: list[Transition] = []
    terminated = truncated = False

    while True:
        user_msg = env.user_message(obs, info, reward)
        action, response = agent.act(user_msg)

        if action is None:
            # Repeated malformed output: abort as a (failed) truncation, matching
            # the established host's behaviour. Not counted as a task retry.
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
