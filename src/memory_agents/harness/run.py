from ..agents.base import BaselineAgent
from ..envs.base import EpisodeResult, Transition
from ..envs.sparc import SparcEnv, is_success
from ..prompts import sparc_system_prompt

from datasets import load_dataset


def run_sparc_episode_baseline(env, agent, system_prompt, obs, info, i):

    reward = 0.0
    transitions = []
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
        puzzle_id=i,
        success=is_success(info),
        terminated=terminated,
        truncated=truncated,
        difficulty=int(info.get("difficulty", -1)),
        transitions=transitions,
    )


def run_sparc_suite_baseline():
    ds = load_dataset("lkaesberg/SPaRC", 'all', split="test")
    df = ds.to_pandas()
    
    env = SparcEnv(split="test")
    
    agent = BaselineAgent()
    
    results = []
    for i in range(len(df)):
        obs, info = env.reset()
        
        polyshapes = df['polyshapes'][i]
        polyshapes = yaml.safe_load(polyshapes)
        
        system_prompt = sparc_system_prompt(polyshapes)
        agent.reset(system_prompt)
        
        result = run_sparc_episode_baseline(env, agent, system_prompt, obs, info, i)
        results.append(result)
    return results
