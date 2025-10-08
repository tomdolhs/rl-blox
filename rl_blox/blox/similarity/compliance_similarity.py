import numpy as np
import gymnasium as gym

from rl_blox.blox.similarity.bisimulation_similarity import ModelWrapper


def collect_experiences(env, n_samples=1000, policy=None):
    """
    Collect experience tuples (s,a,s',r) from env.

    Params
    ------
    env: gym.Env
        A Gym environment with discrete state and action spaces.
    n_samples: int
        Number of experience samples to collect.
    policy: function or None
        A function mapping states to actions. If None, random actions are taken.
    """
    D = []
    obs, _ = env.reset()
    for _ in range(n_samples):
        if policy is None:
            a = env.action_space.sample()
        else:
            a = policy(obs)
        next_obs, r, terminated, truncated, _ = env.step(a)
        D.append((obs, a, next_obs, r))
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return D


def compliance_similarity(P_source, R_source, D_target):
    """
    Compute compliance similarity between source MDP and
    target experience dataset D_target.
    """
    probs = []
    for (s, a, s_next, r) in D_target:
        trans_prob = P_source[s, a, s_next]
        reward_prob = 1.0 if np.isclose(R_source[s, a, s_next], r) else 0.0
        probs.append(trans_prob * reward_prob)
    compliance = np.mean(probs)
    distance = 1 - compliance
    return compliance, distance


if __name__ == "__main__":
    env1 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
    env2 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
    model1 = ModelWrapper(env1.unwrapped)
    model2 = ModelWrapper(env2.unwrapped)
    P1, R1 = model1.get_model()
    P2, R2 = model2.get_model()

    D_target = collect_experiences(env2, n_samples=500)
    compliance, distance = compliance_similarity(P1, R1, D_target)

    print("Compliance similarity:", compliance)
    print("Compliance distance:", distance)
