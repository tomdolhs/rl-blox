import time
import numpy as np
import gymnasium as gym
import ot
from scipy.optimize import linprog
from collections import namedtuple
from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path


class Similarity(ABC):
    """Abstract base class for environment similarity measures.

    Subclasses should implement the `_compute` method which returns a
    numeric score or distance between two discrete Gym environments.
    This base class provides caching of computed results.
    """

    def __init__(self, cache_dir: str = ".cache"):
        """Create a similarity measure with an optional cache directory.

        Args:
            cache_dir: directory where computed results will be stored.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, env1: gym.Env, env2: gym.Env, **kwargs) -> str:
        """Generate a deterministic cache key for a pair of environments.

        The key is derived from environment grid descriptions, file
        modification time and the instance parameters (except cache_dir).

        Todo: Make cache keys robust: hash model matrices (P/R) rather than relying on env.desc; handle envs without .desc.

        Returns:
            A filename-safe hash string ending with '.npy'.
        """
        desc1 = "".join(["".join(row)
                        for row in env1.unwrapped.desc.astype(str)])
        desc2 = "".join(["".join(row)
                        for row in env2.unwrapped.desc.astype(str)])
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

    def compute(self, env1: gym.Env, env2: gym.Env, **kwargs):
        """Compute the similarity/distance between env1 and env2.

        Results will be loaded from cache if available; otherwise `_compute`
        will be called and the result stored.

        Returns:
            Numeric similarity/distance result as defined by the subclass.
        """
        # cache_key = self._generate_cache_key(env1, env2, **kwargs)
        # cache_file = self.cache_dir / cache_key

        # if cache_file.exists():
        #     return np.load(cache_file).item()
        # else:
        result = self._compute(env1, env2, **kwargs)
        # np.save(cache_file, np.array(result))
        return result

    @abstractmethod
    def _compute(self, env1: gym.Env, env2: gym.Env, **kwargs):
        """Subclass-implemented computation for similarity/distance.

        Must return a numeric value representing similarity/distance.
        """
        pass

    def compute_matrix(self, envs: list[gym.Env], **kwargs):
        """Compute symmetric matrix of pairwise similarities.

        Args:
            envs: list of Gymnasium environments.

        Returns:
            A numpy array of shape (n, n) with symmetric similarity values.
        """
        n = len(envs)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                matrix[i, j] = self.compute(envs[i], envs[j], **kwargs)
                matrix[j, i] = matrix[i, j]  # Symmetric
        return matrix


def get_model(env: gym.Env):
    """Extract model matrices P and R from a discrete Gym env.

    Args:
        env: an environment using the 'P' attribute structure.

    Returns:
        Tuple (P, R):
            - P: array of shape (n_states, n_actions, n_states) of transition probabilities.
            - R: array of same shape of expected rewards for each transition.
    """


def expected_rewards(P: np.ndarray, R: np.ndarray):
    """Compute expected reward for each state-action pair.

    Args:
        P: transition probability tensor (n_states, n_actions, n_states)
        R: reward tensor (n_states, n_actions, n_states)

    Returns:
        expected rewards array with shape (n_states, n_actions).
    """
    return np.einsum("san,san->sa", P, R)


def hausdorff_distance(dA: np.ndarray, Nu: list[int], Nv: list[int]) -> float:
    """Compute Hausdorff distance between subsets of nodes given a distance matrix.

    Args:
        dA: matrix of distances between action nodes (shape: n_a x m_a)
        Nu: indices of action nodes for subset from graph 1
        Nv: indices of action nodes for subset from graph 2

    Returns:
        The Hausdorff distance as a float in [0, 1].
    """
    if len(Nu) == 0 or len(Nv) == 0:
        return 1.0
    dA_a_Nv = [np.min(dA[a, Nv]) for a in Nu]
    dA_b_Nu = [np.min(dA[Nu, b]) for b in Nv]
    return max(max(dA_a_Nv), max(dA_b_Nu))


def collect_experiences(env: gym.Env, n_samples: int = 1000, policy: callable = None):
    """Collect transitions by rolling out an environment.

    Args:
        env: Gym environment to sample from.
        n_samples: number of transitions to collect.
        policy: optional callable taking observations and returning actions.

    Returns:
        A list of tuples (obs, action, next_obs, reward).
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


class BisimulationSimilarity(Similarity):
    """Bisimulation-inspired distance between two MDPs.

    The algorithm computes a fixed point of state-to-state distances
    based on reward differences and the Kantorovich distance over transitions.
    """

    def __init__(self, c: float = 0.95, tol: float = 1e-4, max_iter: int = 75, **kwargs):
        """Initialize the bisimulation similarity.

        Args:
            c: discount-like constant scaling transition differences.
            tol: convergence tolerance on the distance matrix.
            max_iter: maximum number of iterations for fixed point computation.
        """
        super().__init__(**kwargs)
        self.c = c
        self.tol = tol
        self.max_iter = max_iter

    def _compute(self, env1: gym.Env, env2: gym.Env, **kwargs):
        """Compute the bisimulation distance (returns transformed value).

        Returns:
            1 - Kantorovich distance of the bisimulation matrix to produce a similarity value.
        """
        P1, R1 = env1.env.build_model()
        P2, R2 = env2.env.build_model()
        d_matrix = self._bisimulation_distance(R1, R2, P1, P2)
        return 1 - ot.emd2(np.ones(d_matrix.shape[0]) / d_matrix.shape[0],
                           np.ones(d_matrix.shape[1]) / d_matrix.shape[1], d_matrix)

    def _bisimulation_distance(self, R_i: np.ndarray, R_j: np.ndarray, P_i: np.ndarray, P_j: np.ndarray):
        """Compute the bisimulation distance matrix between MDPs.

        Iteratively refines a matrix of state-to-state distances until convergence.

        Args:
            R_i, R_j: reward tensors for the two MDPs.
            P_i, P_j: transition tensor for the two MDPs.

        Returns:
            A distance matrix (n_i x n_j).
        """
        n_i, n_a, _ = R_i.shape
        n_j = R_j.shape[0]
        expected_R_i = np.sum(R_i * P_i, axis=2)
        expected_R_j = np.sum(R_j * P_j, axis=2)
        reward_diff = np.abs(
            expected_R_i[:, None, :] - expected_R_j[None, :, :])
        d = np.zeros((n_i, n_j))
        for it in range(self.max_iter):
            d_new = np.zeros_like(d)
            for i in range(n_i):
                for j in range(n_j):
                    max_d = 0.0
                    for a in range(n_a):
                        trans_diff = ot.emd2(P_i[i, a], P_j[j, a], d)
                        max_d = max(
                            max_d, reward_diff[i, j, a] + self.c * trans_diff)
                    d_new[i, j] = max_d
            if np.max(np.abs(d_new - d)) < self.tol:
                return d_new
            d = d_new
        return d


class ComplianceSimilarity(Similarity):
    """Similarity based on how well a source MDP explains target sampled transitions.

    Samples transitions from `env2` and measures average probability that the
    source MDP would have generated those transitions with the same rewards.
    """

    def __init__(self,
                 n_samples: int = 1000,
                 policy: callable = None,
                 **kwargs):
        """Initialize the compliance similarity.

        Args:
            n_samples: number of transitions to sample from the target env.
            policy: optional policy to use when sampling.
        """
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.policy = policy

    def _compute(self, env1: gym.Env, env2: gym.Env, **kwargs):
        """Compute compliance similarity as 1 - (avg. probability of sampled transitions)."""
        P1, R1 = env1.env.build_model()
        D_target = collect_experiences(
            env2, n_samples=self.n_samples, policy=self.policy)
        similarity = self._compliance_similarity(P1, R1, D_target)
        return similarity

    def _compliance_similarity(self,
                               P_source: np.ndarray,
                               R_source: np.ndarray,
                               D_target: list[tuple[int, int, int, float]]):
        """Compute compliance metric: average probability that source MDP generated target transitions.

        Args:
            P_source: transition probability tensor of the source MDP.
            R_source: reward tensor of the source MDP.
            D_target: list of sampled transitions from the target environment.

        Returns:
            Scalar compliance value in [0, 1].
        """
        probs = []
        for (s, a, s_next, r) in D_target:
            trans_prob = P_source[s, a, s_next]
            try:
                reward_prob = float(
                    np.sum(P_source[s, a, :] * np.isclose(R_source[s, a, :], r)))
            except IndexError:
                reward_prob = 0.0
            probs.append(trans_prob * reward_prob)
        compliance = np.mean(probs)
        return compliance


def construct_graph(P: np.ndarray,
                    R: np.ndarray):
    """Construct a graph representation of an MDP.

    The graph is bipartite: state nodes and action nodes. Connections:
    - decision_edges: which state connects to which action node
    - transition_edges: edges from action nodes to next state nodes with probabilities and rewards.

    # Todo: Small dataclass instead of namedtuple for clarity.

    Returns:
        A namedtuple with graph information including arrays:
        (num_state_nodes, num_action_nodes, decision_edges, transition_edges, transition_edge_probs, transition_edge_rewards).
    """
    num_state_nodes = P.shape[0]
    num_action_nodes = P.shape[1] * P.shape[0]
    decision_edges = np.zeros((num_state_nodes, num_action_nodes), dtype=int)
    transition_edges = np.zeros((num_action_nodes, num_state_nodes), dtype=int)
    transition_edge_probs = np.zeros(
        (num_action_nodes, num_state_nodes), dtype=float)
    transition_edge_rewards = np.zeros(
        (num_action_nodes, num_state_nodes), dtype=float)

    for s in range(P.shape[0]):
        for a in range(P.shape[1]):
            action_node = s * (num_action_nodes // num_state_nodes) + a
            decision_edges[s, action_node] = 1
            for s_next in range(P.shape[2]):
                if P[s, a, s_next] > 0:
                    transition_edges[action_node, s_next] = 1
                    transition_edge_probs[action_node,
                                          s_next] = P[s, a, s_next]
                    transition_edge_rewards[action_node,
                                            s_next] = R[s, a, s_next]

    return namedtuple('MDPGraph',
                      ['num_state_nodes', 'num_action_nodes', 'decision_edges', 'transition_edges', 'transition_edge_probs', 'transition_edge_rewards'])(
                          num_state_nodes, num_action_nodes, decision_edges, transition_edges, transition_edge_probs, transition_edge_rewards
    )


class GraphSimilarity(Similarity):
    """Graph-structure based similarity between MDPs.

    Builds action/state graphs and iteratively computes state and action similarities
    by combining reward differences and the Kantorovich distance on transitions.
    """

    def __init__(self,
                 CS: float = 0.9,
                 CA: float = 0.9,
                 max_iter: int = 50,
                 tol: float = 1e-4,
                 **kwargs):
        """Initialize graph similarity parameters.

        Args:
            CS: weight applied to state similarity update.
            CA: weight applied to action similarity update.
            max_iter: maximum iterations for convergence.
            tol: convergence threshold.
        """
        super().__init__(**kwargs)
        self.CS = CS
        self.CA = CA
        self.max_iter = max_iter
        self.tol = tol

    def _compute(self,
                 env1: gym.Env,
                 env2: gym.Env,
                 **kwargs):
        """Compute final graph-based similarity between envs.

        Returns:
            1 - Kantorovich distance of the final state similarity matrix.
        """
        P1, R1 = env1.env.build_model()
        P2, R2 = env2.env.build_model()
        mdpg1 = construct_graph(P1, R1)
        mdpg2 = construct_graph(P2, R2)
        S, _ = self._graph_distance(mdpg1, mdpg2)
        return 1 - ot.emd2(np.ones(S.shape[0]) / S.shape[0],
                           np.ones(S.shape[1]) / S.shape[1], S)

    def _graph_distance(self,
                        mdpg1: namedtuple,
                        mdpg2: namedtuple):
        """Iteratively compute state (S) and action (A) similarity matrices.

        Returns:
            Tuple (S, A) where:
            - S: state-to-state similarity matrix
            - A: action-to-action similarity matrix
        """
        S = np.ones((mdpg1.num_state_nodes, mdpg2.num_state_nodes))
        A = np.ones((mdpg1.num_action_nodes, mdpg2.num_action_nodes))

        for _ in range(self.max_iter):
            S_old, A_old = S.copy(), A.copy()

            for a in range(mdpg1.num_action_nodes):
                pa = mdpg1.transition_edge_probs[a]
                ra = mdpg1.transition_edge_rewards[a]
                for b in range(mdpg2.num_action_nodes):
                    pb = mdpg2.transition_edge_probs[b]
                    rb = mdpg2.transition_edge_rewards[b]
                    drwd = abs(np.sum(pa * ra) - np.sum(pb * rb))
                    demd = ot.emd2(pa, pb, 1 - S)
                    A[a, b] = 1 - (1 - self.CA) * drwd - self.CA * demd

            for u in range(mdpg1.num_state_nodes):
                Nu = np.where(mdpg1.decision_edges[u] == 1)[0]
                if len(Nu) == 0:
                    continue
                for v in range(mdpg2.num_state_nodes):
                    Nv = np.where(mdpg2.decision_edges[v] == 1)[0]
                    if len(Nv) == 0:
                        continue
                    dhaus = hausdorff_distance(1 - A, Nu, Nv)
                    S[u, v] = self.CS * (1 - dhaus)

            if np.max(np.abs(S - S_old)) < self.tol and np.max(np.abs(A - A_old)) < self.tol:
                break
        return S, A


class HomomorphismSimilarity(Similarity):
    """Similarity based on homomorphism-style matching between MDPs.

    Quantifies distances by matching actions and states in a way similar to
    bisimulation but allowing actions to be matched (minimax fashion).
    """

    def __init__(self,
                 c: float = 0.95,
                 tol: float = 1e-4,
                 max_iter: int = 75,
                 **kwargs):
        """Initialize the homomorphism similarity measure."""
        super().__init__(**kwargs)
        self.c = c
        self.tol = tol
        self.max_iter = max_iter

    def _compute(self, env1: gym.Env, env2: gym.Env, **kwargs):
        """Compute homomorphism-based similarity between `env1` and `env2`.

        Returns:
            1 - Kantorovich distance of the homomorphism distance matrix.
        """
        P1, R1 = env1.env.build_model()
        P2, R2 = env2.env.build_model()
        d_matrix = self._homomorphism_distance(R1, R2, P1, P2)
        return 1 - ot.emd2(np.ones(d_matrix.shape[0]) / d_matrix.shape[0],
                           np.ones(d_matrix.shape[1]) / d_matrix.shape[1], d_matrix)

    def _homomorphism_distance(self,
                               R_i: np.ndarray,
                               R_j: np.ndarray,
                               P_i: np.ndarray,
                               P_j: np.ndarray):
        """Compute pairwise state distance using a homomorphism criterion.

        Returns:
            Matrix of state-to-state distances.
        """
        n_i_states, n_i_actions, _ = R_i.shape
        n_j_states, n_j_actions, _ = R_j.shape
        expected_R_i, expected_R_j = expected_rewards(
            P_i, R_i), expected_rewards(P_j, R_j)
        d = np.zeros((n_i_states, n_j_states))

        for _ in range(self.max_iter):
            d_new = np.zeros_like(d)
            for i in range(n_i_states):
                for j in range(n_j_states):
                    max_i = 0.0
                    for a_i in range(n_i_actions):
                        min_d = float('inf')
                        for a_j in range(n_j_actions):
                            reward_diff = abs(
                                expected_R_i[i, a_i] - expected_R_j[j, a_j])
                            trans_diff = ot.emd2(P_i[i, a_i], P_j[j, a_j], d)
                            min_d = min(min_d, reward_diff +
                                        self.c * trans_diff)
                        max_i = max(max_i, min_d if min_d !=
                                    float('inf') else 0)

                    max_j = 0.0
                    for a_j in range(n_j_actions):
                        min_d = float('inf')
                        for a_i in range(n_i_actions):
                            reward_diff = abs(
                                expected_R_i[i, a_i] - expected_R_j[j, a_j])
                            trans_diff = ot.emd2(P_i[i, a_i], P_j[j, a_j], d)
                            min_d = min(min_d, reward_diff +
                                        self.c * trans_diff)
                        max_j = max(max_j, min_d if min_d !=
                                    float('inf') else 0)

                    d_new[i, j] = max(max_i, max_j)

            if np.max(np.abs(d_new - d)) < self.tol:
                return d_new
            d = d_new
        return d_new


class RewardSimilarity(Similarity):
    """Simple reward-focused similarity, based on mean squared error of expected rewards."""

    def __init__(self, **kwargs):
        """Initialize reward similarity measure."""
        super().__init__(**kwargs)

    def _compute(self, env1: gym.Env, env2: gym.Env, **kwargs):
        """Compute the reward-based similarity between two environments."""
        P1, R1 = env1.env.build_model()
        P2, R2 = env2.env.build_model()
        return 1 - self._reward_distance(P1, R1, P2, R2)

    def _reward_distance(self, P1: np.ndarray, R1: np.ndarray, P2: np.ndarray, R2: np.ndarray):
        """Compute MSE between expected reward matrices of two MDPs."""
        R1_exp = expected_rewards(P1, R1)
        R2_exp = expected_rewards(P2, R2)
        if R1_exp.shape != R2_exp.shape:
            raise ValueError(
                "Environments must have the same state and action spaces for RewardSimilarity.")
        diff = R1_exp - R2_exp
        mse = np.mean(diff ** 2)
        return mse


class DiscreteMountainCarWrapper(gym.ObservationWrapper):
    """Discretizes the continuous observation space of the MountainCar environment into bins."""

    def __init__(self, env: gym.Env, pos_bins: int = 10, vel_bins: int = 10):
        """Initializes the DiscreteMountainCarWrapper.

        Args:
            env (gym.Env): The Gym environment to wrap.
            pos_bins (int, optional): Number of bins for position. Defaults to 10.
            vel_bins (int, optional): Number of bins for velocity. Defaults to 10.
        """
        super().__init__(env)
        self.pos_bins = pos_bins
        self.vel_bins = vel_bins
        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high
        self.pos_step = (self.high[0] - self.low[0]) / pos_bins
        self.vel_step = (self.high[1] - self.low[1]) / vel_bins
        self.n_states = pos_bins * vel_bins
        self.observation_space = gym.spaces.Discrete(self.n_states)

    def observation(self, state):
        state = np.clip(state, self.low, self.high)
        pos_idx = int((state[0] - self.low[0]) / self.pos_step)
        vel_idx = int((state[1] - self.low[1]) / self.vel_step)
        pos_idx = min(pos_idx, self.pos_bins - 1)
        vel_idx = min(vel_idx, self.vel_bins - 1)
        return pos_idx * self.vel_bins + vel_idx

    def get_continuous_state(self, discrete_state):
        pos_idx = discrete_state // self.vel_bins
        vel_idx = discrete_state % self.vel_bins

        pos = self.low[0] + (pos_idx + 0.5) * self.pos_step
        vel = self.low[1] + (vel_idx + 0.5) * self.vel_step
        return np.array([pos, vel], dtype=np.float32)

    def build_model(self, n_samples=1):
        n_states = self.observation_space.n
        n_actions = self.action_space.n
        P = np.zeros((n_states, n_actions, n_states))
        R = np.zeros((n_states, n_actions, n_states))
        self.reset()
        for s in range(n_states):
            cont_state = self.get_continuous_state(s)
            self.state = cont_state
            for a in range(n_actions):
                transitions = {}
                total_reward = 0
                for _ in range(n_samples):
                    self.unwrapped.state = cont_state
                    next_obs, reward, terminated, truncated, _ = self.step(a)
                    reward = reward if not (terminated or truncated) else 0.0
                    next_s = next_obs
                    transitions[next_s] = transitions.get(next_s, 0) + 1
                    total_reward += reward
                for next_s, count in transitions.items():
                    P[s, a, next_s] = count / n_samples
                    R[s, a, next_s] = total_reward / n_samples
        return P, R


# if __name__ == "__main__":
#     env_1 = DiscreteMountainCarWrapper(gym.make("MountainCar-v0"), pos_bins=10, vel_bins=10)
#     env_2 = DiscreteMountainCarWrapper(gym.make("MountainCar-v0"), pos_bins=10, vel_bins=10)
#     env_2.env.unwrapped.goal_position = 0.0
#     env_2.env.unwrapped.force = 0.2

#     metrics = [
#         ("Bisimulation", BisimulationSimilarity(c=0.95, tol=1e-4, max_iter=75)),
#         ("Compliance", ComplianceSimilarity(n_samples=10000)),
#         # ("Graph", GraphSimilarity(max_iter=10)),
#         ("Homomorphism", HomomorphismSimilarity(c=0.95, tol=1e-4, max_iter=75)),
#         ("Reward", RewardSimilarity()),
#     ]

#     empty_cache = False
#     if empty_cache:
#         for file in Path(".cache").glob("*.npy"):
#             file.unlink()

#     for name, measure in metrics:
#         start = time.perf_counter()
#         dist = measure.compute(env_1, env_2)
#         end = time.perf_counter()
#         print(f"{name} similarity: {dist} (computed in {end - start:.4f} seconds)")


if __name__ == "__main__":
    # Simple demo of similarity measures on FrozenLake environments
    maps4x4 = [[
        "SFFF",
        "HFFF",
        "HFFF",
        "HFFG"
    ], [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ], [
        "SHHH",
        "FHHH",
        "FFFH",
        "HFFG"
    ], [
        "SFFH",
        "HFFH",
        "HFFH",
        "HHFG"
    ], [
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG"
    ]]
    maps6x6 = [
        [  # 0: Open / mostly free with a few holes (baseline)
            "SFFFFF",
            "FFFFFF",
            "FFFHFF",
            "FFFFFF",
            "FFFFHF",
            "FFFFFG",
        ],
        [  # 1: Checkerboard-heavy holes (high fragmentation)
            "SHFHFH",
            "HFHFHF",
            "FHFHFH",
            "HFHFHF",
            "FHFHFH",
            "HFHFHG",
        ],
        [  # 2: Snake / narrow-corridor style (constrains paths)
            "SFFFFH",
            "HHHFFH",
            "HFFHFF",
            "HFFHHF",
            "HFFFHF",
            "HFFFFG",
        ],
        [  # 3: Clustered holes creating large blocked region with one detour
            "SFFFFF",
            "FFFFHF",
            "FFHHHF",
            "FFFHHF",
            "FFHHFF",
            "FFFFFG",
        ],
        [  # 4: Mirror of map 2 (tests symmetry-aware similarity)
            row[::-1] for row in [
                "SFFFFH",
                "HHHFFH",
                "HFFHFF",
                "HFFHHF",
                "HFFFHF",
                "HFFFFG",
            ]
        ],
        [  # 5: Goal-guarded trap — holes around G except a single approach
            "SFFFFF",
            "FFFFFF",
            "FFFFFF",
            "FFFFFF",
            "FFFFHH",
            "FFFFHG",
        ],
        [  # 6: Diagonal-offset holes (symmetric pattern along one axis)
            "SFFFFF",
            "HFFFFF",
            "FHFFFF",
            "FFHFFF",
            "FFFHFF",
            "FFFFFG",
        ],
        [  # 7: Two large safe regions connected by a single narrow bridge
            "SFFFFF",
            "HHHHHF",
            "FFFHFF",
            "FFFHFF",
            "FFFHFF",
            "FFFFFG",
        ],
    ]
    env_idx = 2, 4

    start = time.perf_counter()
    matrix = BisimulationSimilarity().compute_matrix([gym.make("FrozenLake-v1", desc=m, is_slippery=False) for m in maps4x4])
    end = time.perf_counter()
    print(f"Bisimulation similarity matrix (4x4 maps) computed in {end - start:.4f} seconds:")
    print(matrix)

    env_1 = gym.make(
        "FrozenLake-v1", desc=maps6x6[env_idx[0]], is_slippery=False)
    env_2 = gym.make(
        "FrozenLake-v1", desc=maps6x6[env_idx[1]], is_slippery=False)
    metrics = [
        ("Bisimulation", BisimulationSimilarity(c=0.95, tol=1e-4, max_iter=75)),
        ("Compliance", ComplianceSimilarity(n_samples=10000)),
        # ("Graph", GraphSimilarity(max_iter=10)),
        ("Homomorphism", HomomorphismSimilarity(c=0.95, tol=1e-4, max_iter=75)),
        ("Reward", RewardSimilarity()),
    ]

    empty_cache = False
    if empty_cache:
        for file in Path(".cache").glob("*.npy"):
            file.unlink()

    for name, measure in metrics:
        start = time.perf_counter()
        dist = measure.compute(env_1, env_2)
        end = time.perf_counter()
        print(f"{name} similarity: {dist} (computed in {end - start:.4f} seconds)")
