import numpy as np
import gymnasium as gym
from scipy.optimize import linprog


def kantorovich_distance(d: np.ndarray, p: np.ndarray = None, q: np.ndarray = None):
    """
    Calculates the Kantorovich distance between two probability distributions.
    The Kantorovich distance is the solution to the linear programming problem:
    min_T sum_{i,j} d_{i,j} * T_{i,j}
    s.t. sum_j T_{i,j} = p_i for all i
            sum_i T_{i,j} = q_j for all j
            d_{i,j} >= 0 for all i,j

    Params
    ------
    d_matrix: np.ndarray
        Cost matrix with shape (n, m).
    p: np.ndarray | None
        Probability distribution over the rows with shape (n,). If None, uniform distribution is used.
    q: np.ndarray | None
        Probability distribution over the columns with shape (m,). If None, uniform distribution is used.
    """
    n, m = d.shape
    if p is None:
        p = np.ones(n) / n
    if q is None:
        q = np.ones(m) / m
    c = d.flatten()
    A_eq, b_eq = [], []
    for i in range(n):  # Row sums
        row = np.zeros((n, m))
        row[i, :] = 1
        A_eq.append(row.flatten())
        b_eq.append(p[i])
    for j in range(m):  # Column sums
        col = np.zeros((n, m))
        col[:, j] = 1
        A_eq.append(col.flatten())
        b_eq.append(q[j])
    A_eq, b_eq = np.array(A_eq), np.array(b_eq)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.fun


def bisimulation_distance(R_i, R_j, P_i, P_j, c=0.5, tol=1e-6, max_iter=1000):
    """
    Calculates the bisimulation distance between two MDPs using value iteration.

    Params
    ------
    R_i: np.ndarray
        Reward matrix of MDP i with shape (n_states, n_actions).
    R_j: np.ndarray
        Reward matrix of MDP j with shape (n_states, n_actions).
    P_i: np.ndarray
        Transition probability tensor of MDP i with shape (n_states, n_actions, n_states
    P_j: np.ndarray
        Transition probability tensor of MDP j with shape (n_states, n_actions, n_states
    c: float
        Weighting factor between reward and transition differences.
    tol: float
        Tolerance for convergence.
    max_iter: int
        Maximum number of iterations.
    """
    n_i_states, n_actions, _ = R_i.shape
    n_j_states = R_j.shape[0]
    expected_R_i, expected_R_j = expected_rewards(P_i, R_i), expected_rewards(P_j, R_j)
    d = np.zeros((n_i_states, n_j_states))
    for it in range(max_iter):
        print("Iteration:", it)
        d_new = np.zeros_like(d)
        for i in range(n_i_states):
            for j in range(n_j_states):
                vals = []
                for a in range(n_actions):
                    reward_diff = abs(expected_R_i[i, a] - expected_R_j[j, a])
                    trans_diff = kantorovich_distance(d, P_i[i, a], P_j[j, a])
                    vals.append((1 - c) * reward_diff + c * trans_diff)
                d_new[i, j] = max(vals)
        if np.max(np.abs(d_new - d)) < tol:
            return d_new
        d = d_new
    return d


def expected_rewards(P, R):
    return np.einsum("san,san->sa", P, R)


class ModelWrapper:
    """
    Wrapper to extract transition and reward matrices from a Gym environment.

    Params
    ------
    env: gym.Env
        A Gym environment with discrete state and action spaces.
    Methods
    -------
    get_model()
        Returns the transition probability matrix P and reward matrix R.
    """

    def __init__(self, env):
        self.env = env
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        self.P = np.zeros((n_states, n_actions, n_states))
        self.R = np.zeros((n_states, n_actions, n_states))

        for s in range(n_states):
            for a in range(n_actions):
                for prob, s_next, reward, done in env.P[s][a]:
                    self.P[s, a, s_next] += prob
                    self.R[s, a, s_next] += reward

    def get_model(self):
        return self.P, self.R


def compute_task_similarity(env1, env2, c=0.5):
    model1 = ModelWrapper(env1.unwrapped)
    model2 = ModelWrapper(env2.unwrapped)

    P1, R1 = model1.get_model()
    P2, R2 = model2.get_model()

    d_matrix = bisimulation_distance(R1, R2, P1, P2, c=c)
    return d_matrix


if __name__ == "__main__":
    env1 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
    env2 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
    distance_matrix = compute_task_similarity(env1, env2, c=0.5)
    print(kantorovich_distance(distance_matrix))
