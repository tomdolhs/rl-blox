import numpy as np
import gymnasium as gym

from bisimulation_similarity import kantorovich_distance, expected_rewards, ModelWrapper


def homomorphism_distance(R_i, R_j, P_i, P_j, c=0.5, tol=1e-6, max_iter=1000):
    """
    Calculates the homomorphism (lax bisimulation) distance between two MDPs.

    Params
    ------
    R_i: np.ndarray
        Reward tensor of MDP i with shape (n_states_i, n_actions_i, n_states_i).
    R_j: np.ndarray
        Reward tensor of MDP j with shape (n_states_j, n_actions_j, n_states_j).
    P_i: np.ndarray
        Transition probability tensor of MDP i with shape (n_states_i, n_actions_i, n_states_i).
    P_j: np.ndarray
        Transition probability tensor of MDP j with shape (n_states_j, n_actions_j, n_states_j).
    c: float
        Weighting factor between reward and transition differences.
    tol: float
        Tolerance for convergence.
    max_iter: int
        Maximum number of iterations.
    """
    n_i_states, n_i_actions, _ = R_i.shape
    n_j_states, n_j_actions, _ = R_j.shape
    expected_R_i, expected_R_j = expected_rewards(P_i, R_i), expected_rewards(P_j, R_j)

    d = np.zeros((n_i_states, n_j_states))

    for it in range(max_iter):
        print("Iteration:", it)
        d_new = np.zeros_like(d)
        for i in range(n_i_states):
            for j in range(n_j_states):
                max_i = 0.0
                for a_i in range(n_i_actions):
                    vals = []
                    for a_j in range(n_j_actions):
                        reward_diff = abs(expected_R_i[i, a_i] - expected_R_j[j, a_j])
                        trans_diff = kantorovich_distance(d, P_i[i, a_i], P_j[j, a_j])
                        vals.append((1 - c) * reward_diff + c * trans_diff)
                    max_i = max(max_i, min(vals))

                max_j = 0.0
                for a_j in range(n_j_actions):
                    vals = []
                    for a_i in range(n_i_actions):
                        reward_diff = abs(expected_R_i[i, a_i] - expected_R_j[j, a_j])
                        trans_diff = kantorovich_distance(d, P_i[i, a_i], P_j[j, a_j])
                        vals.append((1 - c) * reward_diff + c * trans_diff)
                    max_j = max(max_j, min(vals))  # min over a_i, then max over a_j

                d_new[i, j] = max(max_i, max_j)

        if np.max(np.abs(d_new - d)) < tol:
            return d_new
        d = d_new

    return d_new


def compute_task_similarity(env1, env2, c=0.5):
    model1 = ModelWrapper(env1.unwrapped)
    model2 = ModelWrapper(env2.unwrapped)

    P1, R1 = model1.get_model()
    P2, R2 = model2.get_model()

    d_matrix = homomorphism_distance(R1, R2, P1, P2, c=c)
    return d_matrix


if __name__ == "__main__":
    env1 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
    env2 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
    distance_matrix = compute_task_similarity(env1, env2, c=0.5)
    print(kantorovich_distance(distance_matrix))
