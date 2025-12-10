import numpy as np
import gymnasium as gym
from scipy.optimize import linprog
from collections import namedtuple
from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path


class Similarity(ABC):
    """Base class for similarity metrics."""

    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, env1, env2, **kwargs) -> str:
        """
        Generates a unique cache key for the similarity computation.

        Params
        ----------
        env1: gym.Env
            The first environment.
        env2: gym.Env
            The second environment.
        kwargs
            Additional arguments for the similarity metric.

        Returns
        -------
        str
            A unique cache key.
        """
        desc1 = "".join(["".join(row) for row in env1.unwrapped.desc.astype(str)])
        desc2 = "".join(["".join(row) for row in env2.unwrapped.desc.astype(str)])
        params = self.__dict__.copy()
        params.pop('cache_dir', None)  # Don't include cache_dir in the key
        params.update(kwargs)
        file_mod_time = Path(__file__).stat().st_mtime
        key_string = (
            f"{self.__class__.__name__}-{desc1}-{desc2}-"
            f"{file_mod_time}-"
            f"{json.dumps(params, sort_keys=True)}"
        )
        hash = hashlib.sha256(key_string.encode()).hexdigest() + ".npy"
        return hash

    def compute(self, env1, env2, **kwargs):
        """
        Computes the similarity/distance between two environments, using a cache.

        Params
        ------
        env1: gym.Env
            The first environment.
        env2: gym.Env
            The second environment.
        kwargs: dict
            Additional arguments for the similarity metric.

        Returns
        -------
        float
            A measure of similarity or distance.
        """
        cache_key = self._generate_cache_key(env1, env2, **kwargs)
        cache_file = self.cache_dir / cache_key

        if cache_file.exists():
            return np.load(cache_file).item()
        else:
            result = self._compute(env1, env2, **kwargs)
            np.save(cache_file, np.array(result))
            return result

    @abstractmethod
    def _compute(self, env1, env2, **kwargs):
        """
        The actual computation logic for the similarity/distance.
        This method must be implemented by subclasses.
        """
        pass

    def compute_matrix(self, envs, **kwargs):
        """
        Computes the similarity matrix between a list of environments.

        Params
        ------
        envs: list[gym.Env]
            A list of Gym environments.
        kwargs: dict
            Additional arguments for the similarity metric.

        Returns
        -------
        np.ndarray
            A similarity matrix.
        """
        n = len(envs)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                matrix[i, j] = self.compute(envs[i], envs[j], **kwargs)
                matrix[j, i] = matrix[i, j]  # Symmetric
        return matrix


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
        """Returns the transition probability matrix P and reward matrix R."""
        return self.P, self.R


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
    return res.fun if res.fun is not None else 0


def expected_rewards(P, R):
    """
    Calculates the expected rewards for each state-action pair.

    Params
    ------
    P: np.ndarray
        Transition probability matrix with shape (n_states, n_actions, n_states).
    R: np.ndarray
        Reward matrix with shape (n_states, n_actions, n_states).

    Returns
    -------
    np.ndarray
        Expected rewards for each state-action pair with shape (n_states, n_actions).
    """
    return np.einsum("san,san->sa", P, R)


class BisimulationSimilarity(Similarity):
    """
    Computes the bisimulation similarity between two environments.
    """

    def __init__(self, c=0.99, tol=1e-3, max_iter=25, **kwargs):
        """
        Params
        ------
        c: float
            Discount factor.
        tol: float
            Tolerance for convergence.
        max_iter: int
            Maximum number of iterations.
        """
        super().__init__(**kwargs)
        self.c = c
        self.tol = tol
        self.max_iter = max_iter

    def _compute(self, env1, env2, **kwargs):
        """
        Computes the bisimulation similarity between two environments.

        Params
        ------
        env1: gym.Env
            The first environment.
        env2: gym.Env
            The second environment.

        Returns
        -------
        float
            The bisimulation similarity between the two environments.
        """
        model1 = ModelWrapper(env1.unwrapped)
        model2 = ModelWrapper(env2.unwrapped)
        P1, R1 = model1.get_model()
        P2, R2 = model2.get_model()
        d_matrix = self._bisimulation_distance(R1, R2, P1, P2)
        return 1 - kantorovich_distance(d_matrix)

    def _bisimulation_distance(self, R_i, R_j, P_i, P_j):
        n_i_states, n_actions, _ = R_i.shape
        n_j_states = R_j.shape[0]
        expected_R_i, expected_R_j = expected_rewards(P_i, R_i), expected_rewards(P_j, R_j)
        d = np.zeros((n_i_states, n_j_states))
        for it in range(self.max_iter):
            d_new = np.zeros_like(d)
            for i in range(n_i_states):
                for j in range(n_j_states):
                    vals = []
                    for a in range(n_actions):
                        reward_diff = abs(expected_R_i[i, a] - expected_R_j[j, a])
                        trans_diff = kantorovich_distance(d, P_i[i, a], P_j[j, a])
                        if trans_diff is None:
                            continue
                        vals.append(reward_diff + self.c * trans_diff)
                    d_new[i, j] = max(vals) if vals else 0
            if np.max(np.abs(d_new - d)) < self.tol:
                return d_new
            d = d_new
        return d


def collect_experiences(env, n_samples=1000, policy=None):
    """
    Collects experiences from an environment using a given policy.

    Params
    ------
    env: gym.Env
        The environment to collect experiences from.
    n_samples: int
        The number of experiences to collect.
    policy: function
        A function that takes an observation as input and returns an action.
        If None, a random policy is used.

    Returns
    -------
    list
        A list of experiences, where each experience is a tuple of (obs, a, next_obs, r).
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


class ComplianceSimilarity(Similarity):
    """
    Computes the compliance similarity between two environments.
    """

    def __init__(self, n_samples=1000, policy=None, **kwargs):
        """
        Params
        ------
        n_samples: int
            The number of samples to collect from the target environment.
        policy: function
            A function that takes an observation as input and returns an action.
            If None, a random policy is used.
        """
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.policy = policy

    def _compute(self, env1, env2, **kwargs):
        """
        Computes the compliance similarity between two environments.

        Params
        ------
        env1: gym.Env
            The first environment.
        env2: gym.Env
            The second environment.

        Returns
        -------
        float
            The compliance similarity between the two environments.
        """
        model1 = ModelWrapper(env1.unwrapped)
        P1, R1 = model1.get_model()
        D_target = collect_experiences(env2, n_samples=self.n_samples, policy=self.policy)
        distance = self._compliance_similarity(P1, R1, D_target)
        return 1 - distance

    def _compliance_similarity(self, P_source, R_source, D_target):
        probs = []
        for (s, a, s_next, r) in D_target:
            trans_prob = P_source[s, a, s_next]
            reward_prob = 1.0 if np.isclose(R_source[s, a, s_next], r) else 0.0
            probs.append(trans_prob * reward_prob)
        compliance = np.mean(probs) if probs else 0.0
        return compliance


def construct_graph(P: np.ndarray, R: np.ndarray):
    """
    Constructs a graph representation of an MDP.

    Params
    ------
    P: np.ndarray
        Transition probability matrix with shape (n_states, n_actions, n_states).
    R: np.ndarray
        Reward matrix with shape (n_states, n_actions, n_states).

    Returns
    -------
    namedtuple
        A namedtuple containing the graph representation of the MDP.
    """
    num_state_nodes = P.shape[0]
    num_action_nodes = P.shape[1] * P.shape[0]
    decision_edges = np.zeros((num_state_nodes, num_action_nodes), dtype=int)
    transition_edges = np.zeros((num_action_nodes, num_state_nodes), dtype=int)
    transition_edge_probs = np.zeros((num_action_nodes, num_state_nodes), dtype=float)
    transition_edge_rewards = np.zeros((num_action_nodes, num_state_nodes), dtype=float)

    for s in range(P.shape[0]):
        for a in range(P.shape[1]):
            action_node = s * (num_action_nodes // num_state_nodes) + a
            decision_edges[s, action_node] = 1
            for s_next in range(P.shape[2]):
                if P[s, a, s_next] > 0:
                    transition_edges[action_node, s_next] = 1
                    transition_edge_probs[action_node, s_next] = P[s, a, s_next]
                    transition_edge_rewards[action_node, s_next] = R[s, a, s_next]

    return namedtuple('MDPGraph', ['num_state_nodes', 'num_action_nodes', 'decision_edges', 'transition_edges', 'transition_edge_probs', 'transition_edge_rewards'])(
        num_state_nodes, num_action_nodes, decision_edges, transition_edges, transition_edge_probs, transition_edge_rewards
    )


def hausdorff_distance(dA: np.ndarray, Nu: list[int], Nv: list[int]) -> float:
    """
    Calculates the Hausdorff distance between two sets of nodes.

    Params
    ------
    dA: np.ndarray
        Distance matrix between the nodes.
    Nu: list[int]
        List of nodes in the first set.
    Nv: list[int]
        List of nodes in the second set.

    Returns
    -------
    float
        The Hausdorff distance between the two sets of nodes.
    """
    if len(Nu) == 0 or len(Nv) == 0:
        return 1.0
    dA_a_Nv = [np.min(dA[a, Nv]) for a in Nu]
    dA_b_Nu = [np.min(dA[Nu, b]) for b in Nv]
    return max(max(dA_a_Nv), max(dA_b_Nu))


class GraphSimilarity(Similarity):
    """
    Computes the graph similarity between two environments.
    """

    def __init__(self, CS=0.9, CA=0.9, max_iter=50, tol=1e-4, **kwargs):
        """
        Params
        ------
        CS: float
            Weight for state similarity.
        CA: float
            Weight for action similarity.
        max_iter: int
            Maximum number of iterations.
        tol: float
            Tolerance for convergence.
        """
        super().__init__(**kwargs)
        self.CS = CS
        self.CA = CA
        self.max_iter = max_iter
        self.tol = tol

    def _compute(self, env1, env2, **kwargs):
        """
        Computes the graph similarity between two environments.

        Params
        ------
        env1: gym.Env
            The first environment.
        env2: gym.Env
            The second environment.

        Returns
        -------
        float
            The graph similarity between the two environments.
        """
        model1 = ModelWrapper(env1.unwrapped)
        model2 = ModelWrapper(env2.unwrapped)
        P1, R1 = model1.get_model()
        P2, R2 = model2.get_model()
        mdpg1 = construct_graph(P1, R1)
        mdpg2 = construct_graph(P2, R2)
        S, _ = self._graph_distance(mdpg1, mdpg2)
        return kantorovich_distance(S)

    def _graph_distance(self, mdpg1, mdpg2):
        S = np.ones((mdpg1.num_state_nodes, mdpg2.num_state_nodes))
        A = np.ones((mdpg1.num_action_nodes, mdpg2.num_action_nodes))

        for it in range(self.max_iter):
            S_old, A_old = S.copy(), A.copy()

            for a in range(mdpg1.num_action_nodes):
                pa = mdpg1.transition_edge_probs[a]
                ra = mdpg1.transition_edge_rewards[a]
                for b in range(mdpg2.num_action_nodes):
                    pb = mdpg2.transition_edge_probs[b]
                    rb = mdpg2.transition_edge_rewards[b]
                    drwd = abs(np.sum(pa * ra) - np.sum(pb * rb))
                    demd = kantorovich_distance(1 - S, pa, pb)
                    A[a, b] = 1 - (1 - self.CA) * drwd - self.CA * demd

            for u in range(mdpg1.num_state_nodes):
                Nu = np.where(mdpg1.decision_edges[u] == 1)[0]
                if len(Nu) == 0: continue
                for v in range(mdpg2.num_state_nodes):
                    Nv = np.where(mdpg2.decision_edges[v] == 1)[0]
                    if len(Nv) == 0: continue
                    dhaus = hausdorff_distance(1 - A, Nu, Nv)
                    S[u, v] = self.CS * (1 - dhaus)

            if np.max(np.abs(S - S_old)) < self.tol and np.max(np.abs(A - A_old)) < self.tol:
                break
        return S, A


class HomomorphismSimilarity(Similarity):
    """
    Computes the homomorphism similarity between two environments.
    """

    def __init__(self, c=0.5, tol=1e-6, max_iter=1000, **kwargs):
        """
        Params
        ------
        c: float
            Discount factor.
        tol: float
            Tolerance for convergence.
        max_iter: int
            Maximum number of iterations.
        """
        super().__init__(**kwargs)
        self.c = c
        self.tol = tol
        self.max_iter = max_iter

    def _compute(self, env1, env2, **kwargs):
        """
        Computes the homomorphism similarity between two environments.

        Params
        ------
        env1: gym.Env
            The first environment.
        env2: gym.Env
            The second environment.

        Returns
        -------
        float
            The homomorphism similarity between the two environments.
        """
        model1 = ModelWrapper(env1.unwrapped)
        model2 = ModelWrapper(env2.unwrapped)
        P1, R1 = model1.get_model()
        P2, R2 = model2.get_model()
        d_matrix = self._homomorphism_distance(R1, R2, P1, P2)
        return kantorovich_distance(d_matrix)

    def _homomorphism_distance(self, R_i, R_j, P_i, P_j):
        n_i_states, n_i_actions, _ = R_i.shape
        n_j_states, n_j_actions, _ = R_j.shape
        expected_R_i, expected_R_j = expected_rewards(P_i, R_i), expected_rewards(P_j, R_j)
        d = np.zeros((n_i_states, n_j_states))

        for it in range(self.max_iter):
            d_new = np.zeros_like(d)
            for i in range(n_i_states):
                for j in range(n_j_states):
                    max_i = 0.0
                    for a_i in range(n_i_actions):
                        vals = []
                        for a_j in range(n_j_actions):
                            reward_diff = abs(expected_R_i[i, a_i] - expected_R_j[j, a_j])
                            trans_diff = kantorovich_distance(d, P_i[i, a_i], P_j[j, a_j])
                            vals.append((1 - self.c) * reward_diff + self.c * trans_diff)
                        max_i = max(max_i, min(vals) if vals else 0)

                    max_j = 0.0
                    for a_j in range(n_j_actions):
                        vals = []
                        for a_i in range(n_i_actions):
                            reward_diff = abs(expected_R_i[i, a_i] - expected_R_j[j, a_j])
                            trans_diff = kantorovich_distance(d, P_i[i, a_i], P_j[j, a_j])
                            vals.append((1 - self.c) * reward_diff + self.c * trans_diff)
                        max_j = max(max_j, min(vals) if vals else 0)

                    d_new[i, j] = max(max_i, max_j)

            if np.max(np.abs(d_new - d)) < self.tol:
                return d_new
            d = d_new
        return d_new


class RewardSimilarity(Similarity):
    """
    Computes the reward similarity between two environments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute(self, env1, env2, **kwargs):
        """
        Computes the reward similarity between two environments.

        Params
        ------
        env1: gym.Env
            The first environment.
        env2: gym.Env
            The second environment.

        Returns
        -------
        float
            The reward similarity between the two environments.
        """
        model1 = ModelWrapper(env1.unwrapped)
        model2 = ModelWrapper(env2.unwrapped)
        P1, R1 = model1.get_model()
        P2, R2 = model2.get_model()
        return self._reward_distance(P1, R1, P2, R2)

    def _reward_distance(self, P1, R1, P2, R2):
        R1_exp = expected_rewards(P1, R1)
        R2_exp = expected_rewards(P2, R2)
        assert R1_exp.shape == R2_exp.shape, "Reward matrices must match in shape"
        diff = R1_exp - R2_exp
        mse = np.mean(diff ** 2)
        return mse


if __name__ == "__main__":
    # I have corrected the `maps` list structure for it to be valid Python.
    maps = [
        ['SFFF',
         'FFFF',
         'FFFF',
         'FFFG'],
        ['SFFF',
         'HFFH',
         'HHFF',
         'FHFG'],
        ['SFFH',
         'HHFF',
         'HFHF',
         'HHHG'],
        ['SHHH',
         'HHHH',
         'HHHH',
         'HHHG']
    ]
    env1 = gym.make("FrozenLake-v1", is_slippery=False, desc=maps[0])
    env2 = gym.make("FrozenLake-v1", is_slippery=False, desc=maps[3])

    print("--- Bisimulation Similarity ---")
    bisim_sim = BisimulationSimilarity(c=0.99, max_iter=25)
    dist = bisim_sim.compute(env1, env2)
    print("Bisimulation distance:", dist)


    print("\n--- Compliance Similarity ---")
    comp_sim = ComplianceSimilarity(n_samples=500)
    dist = comp_sim.compute(env1, env2)
    print("Compliance distance:", dist)

    print("\n--- Graph Similarity ---")
    graph_sim = GraphSimilarity(max_iter=10)
    dist = graph_sim.compute(env1, env2)
    print("Graph Distance:", dist)

    print("\n--- Homomorphism Similarity ---")
    homo_sim = HomomorphismSimilarity(c=0.5, max_iter=10)
    dist = homo_sim.compute(env1, env2)
    print("Homomorphism distance:", dist)

    print("\n--- Reward Similarity ---")
    reward_sim = RewardSimilarity()
    dist = reward_sim.compute(env1, env2)
    print("Reward function distance:", dist)
