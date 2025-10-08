import numpy as np
import gymnasium as gym

from rl_blox.blox.similarity.bisimulation_similarity import ModelWrapper





def reward_distance(P1, R1, P2, R2):
    R1_exp = expected_rewards(P1, R1)
    R2_exp = expected_rewards(P2, R2)

    assert R1_exp.shape == R2_exp.shape, "Reward matrices must match in shape"

    diff = R1_exp - R2_exp
    mse = np.mean(diff ** 2)
    return mse


if __name__ == "__main__":
    env1 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
    env2 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")

    model1 = ModelWrapper(env1.unwrapped)
    model2 = ModelWrapper(env2.unwrapped)

    P1, R1 = model1.get_model()
    P2, R2 = model2.get_model()

    d = reward_distance(P1, R1, P2, R2)
    print("Reward function distance:", d)
