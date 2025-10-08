import numpy as np
import gymnasium as gym
from collections import namedtuple

from rl_blox.blox.similarity.bisimulation_similarity import kantorovich_distance, ModelWrapper


def construct_graph(P: np.ndarray, R: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a bipartite graph representation of an MDP.
    The MDP graph for an MDP M = (S, A, P, R) is defined as G_M = (V, Lambda, E, Psi, p, r)
    has two types of nodes S (state nodes) and Lambda (action nodes). E is the set of
    decision edges (from states to actions) and Psi is the set of transition edges
    (from actions to states). Each transition edge (a, v) has a probability p(a, v)
    and a reward r(a, v).

    Construction:
    1. For each state s in S, create a state node v_s into V.
    2. For each state s in S and each a in A, create an action node alpha_a into Lambda
       and a new edge (v_s, alpha_a) into E.
    3. For each (s, a, s') with P(s, a, s') > 0, create a new edge (alpha_a, v_s') into Psi,
       with probability p(alpha_a, v_s') = P(s, a, s') and reward r(alpha_a, v_s') = R(s, a, s').

    Params
    ------
    P: np.ndarray
        Transition probability tensor of shape (n_states, n_actions, n_states).
    R: np.ndarray
        Reward tensor of shape (n_states, n_actions, n_states).
    Returns
    -------
    mdp_graph: namedtuple
        A namedtuple containing:
        - num_state_nodes: int, number of state nodes
        - num_action_nodes: int, number of action nodes
        - decision_edges: np.ndarray, binary matrix of shape (num_state_nodes, num_action_nodes) indicating decision edges
        - transition_edges: np.ndarray, binary matrix of shape (num_action_nodes, num_state_nodes) indicating transition edges
        - transition_edge_probs: np.ndarray, matrix of shape (num_action_nodes, num_state_nodes) with transition probabilities
        - transition_edge_rewards: np.ndarray, matrix of shape (num_action_nodes, num_state_nodes) with transition rewards

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
        num_state_nodes,
        num_action_nodes,
        decision_edges,
        transition_edges,
        transition_edge_probs,
        transition_edge_rewards
    )


def hausdorff_distance(dA: np.ndarray, Nu: list[int], Nv: list[int]) -> float:
    """
    Compute the Hausdorff distance between two sets of nodes Nu and Nv,
    given a distance matrix dA between action nodes.

    Params
    ------
    dA: np.ndarray
        Distance matrix between action nodes.
    Nu: list[int]
        List of action node indices for the first set.
    Nv: list[int]
        List of action node indices for the second set.
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


def graph_distance(mdpg1, mdpg2, CS=0.9, CA=0.9, max_iter=50, tol=1e-4):
    """
    Compute a simple graph distance between two MDP graphs, based on state and action node similarities.
    """
    S = np.ones((mdpg1.num_state_nodes, mdpg2.num_state_nodes))
    A = np.ones((mdpg1.num_action_nodes, mdpg2.num_action_nodes))

    for it in range(max_iter):
        print("Iteration:", it)
        S_old, A_old = S.copy(), A.copy()

        for a in range(mdpg1.num_action_nodes):
            pa = mdpg1.transition_edge_probs[a]
            ra = mdpg1.transition_edge_rewards[a]
            for b in range(mdpg2.num_action_nodes):
                pb = mdpg2.transition_edge_probs[b]
                rb = mdpg2.transition_edge_rewards[b]
                drwd = abs(np.sum(pa * ra) - np.sum(pb * rb))
                demd = kantorovich_distance(1 - S, pa, pb)
                A[a, b] = 1 - (1 - CA) * drwd - CA * demd

        for u in range(mdpg1.num_state_nodes):
            Nu = np.where(mdpg1.decision_edges[u] == 1)[0]
            if len(Nu) == 0: continue
            for v in range(mdpg2.num_state_nodes):
                Nv = np.where(mdpg2.decision_edges[v] == 1)[0]
                if len(Nv) == 0: continue
                dhaus = hausdorff_distance(1 - A, Nu, Nv)
                S[u, v] = CS * (1 - dhaus)

        if np.max(np.abs(S - S_old)) < tol and np.max(np.abs(A - A_old)) < tol:
            break

    return S, A


if __name__ == "__main__":
    env1 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
    env2 = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")

    model1 = ModelWrapper(env1.unwrapped)
    model2 = ModelWrapper(env2.unwrapped)

    P1, R1 = model1.get_model()
    P2, R2 = model2.get_model()

    mdpg1 = construct_graph(P1, R1)
    mdpg2 = construct_graph(P2, R2)

    S, A = graph_distance(mdpg1, mdpg2, max_iter=100)
    print("State node similarities:\n", S)
    print("Action node similarities:\n", A)
    print("Graph Distance:", kantorovich_distance(S))
