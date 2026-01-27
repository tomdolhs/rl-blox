import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Discrete
from functools import partial

from rl_blox.algorithm.active_mt import train_active_mt
from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.nature_dqn import train_nature_dqn
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.multitask import DiscreteTaskSet
from rl_blox.blox.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer
from rl_blox.logging.logger import AIMLogger


class MountainCarWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, pos_bins: int = 10, vel_bins: int = 10):
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
        R = np.zeros((n_states, n_actions))
        for s in range(n_states):
            cont_state = self.get_continuous_state(s)
            for a in range(n_actions):
                transitions = {}
                total_reward = 0
                for _ in range(n_samples):
                    self.unwrapped.state = cont_state
                    next_obs, reward, _, _, _ = self.step(a)
                    next_s = next_obs
                    transitions[next_s] = transitions.get(next_s, 0) + 1
                    total_reward += reward
                for next_s, count in transitions.items():
                    P[s, a, next_s] = count / n_samples
                R[s, a] = total_reward / n_samples
        return P, R


class FrozenLakeWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n = env.observation_space.n
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def observation(self, obs):
        idx = int(np.asarray(obs))
        onehot = np.zeros(self.n, dtype=np.float32)
        onehot[idx] = 1.0
        return onehot

    def build_model(self, n_samples=1):
        n_states = self.n
        n_actions = self.action_space.n
        P = np.zeros((n_states, n_actions, n_states))
        R = np.zeros((n_states, n_actions, n_states))
        for s in range(n_states):
            for a in range(n_actions):
                for prob, s_next, reward, done in self.unwrapped.P[s][a]:
                    P[s, a, s_next] += prob
                    R[s, a, s_next] += reward
        return P, R


def greedy_policy(q_net, obs):
    q_vals = q_net(obs[None, ...])
    return int(np.argmax(q_vals))


def evaluate_policy(env, q_net, episodes=50):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = greedy_policy(q_net, obs)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_ret += r
        returns.append(ep_ret)
    return np.array(returns)


def asymptotic_stats(returns, window_size=100):
    n = len(returns)
    if n == 0:
        return {}
    w = min(window_size, n)
    last = np.array(returns[-w:])
    mean = float(np.mean(last))
    std = float(np.std(last, ddof=1))
    sem = std / np.sqrt(len(last))
    slope = float(np.polyfit(np.arange(len(last)), last, 1)[0])
    return {"mean": mean, "std": std, "sem": sem, "slope": slope, "window_size": w}


def auc_returns(returns, x=None):
    if x is None:
        x = np.arange(len(returns))
    return float(np.trapezoid(returns, x))


def time_to_threshold(returns, threshold, window=10, min_stability=1):
    # returns: episodic mean returns (or moving average with per-timestep values)
    # find earliest index where moving average >= threshold for min_stability consecutive windows
    r = np.array(returns)
    if len(r) < window:
        return None
    mov = np.convolve(r, np.ones(window)/window, mode='valid')
    for i in range(len(mov) - min_stability + 1):
        if np.all(mov[i:i+min_stability] >= threshold):
            return int(i + window - 1)   # index in original returns
    return None


def create_frozen_lake_set():
    maps = [[
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
            "HHFG"
            ],  [
            "SFFH",
            "HFFH",
            "HFFH",
            "HHFG"
            ], [
            "SFFF",
            "FFFF",
            "FFFF",
            "FFFG"
            ]
            ]

    def set_context_fn(env: gym.Env, context: np.ndarray) -> None:
        map_idx = int(context.argmax())
        new_env = gym.make("FrozenLake-v1", desc=maps[map_idx], is_slippery=False)
        env.unwrapped.__dict__.update(new_env.unwrapped.__dict__)
        new_env.close()
    base_env = FrozenLakeWrapper(gym.make("FrozenLake-v1", is_slippery=False))
    contexts = np.eye(len(maps), dtype=np.float32)
    train_set = DiscreteTaskSet(base_env=base_env, set_context=set_context_fn, contexts=contexts, context_aware=True)
    return train_set


def run_experiment(config: dict, logger: AIMLogger = AIMLogger()) -> dict:
    print(f"Start experiment: {config['name']}")

    # Experiment parameters
    seed = config.get("seed", 42)
    env_name = config.get("env_name", "FrozenLake-v1")
    backbone = config.get("backbone", "DDQN")
    task_selector = config.get("task_selector", "Dissimilarity")
    total_timesteps = config.get("total_timesteps", 100_000)
    train_set = create_frozen_lake_set()
    logger.define_experiment(env_name=env_name, algorithm_name=f"AMT-{backbone}-{task_selector}", hparams=config)

    # Train
    env = train_set.get_task(0)
    q_net = MLP(env.observation_space.shape[0], int(env.action_space.n), activation="relu", hidden_nodes=[64, 64], rngs=nnx.Rngs(seed))
    q_target_net = nnx.clone(q_net)
    replay_buffer = MultiTaskReplayBuffer(ReplayBuffer(buffer_size=100_000, discrete_actions=True), len(train_set))
    optimizer = nnx.Optimizer(q_net, optax.adam(0.005), wrt=nnx.Param)
    train_st = partial(train_ddqn, q_net=q_net, optimizer=optimizer, q_target_net=q_target_net)
    result = train_active_mt(
        train_set,
        train_st,
        replay_buffer,
        task_selector=task_selector,
        r_max=config.get("r_max", 1.0),
        ducb_gamma=config.get("ducb_gamma", 0.95),
        temperature=config.get("temperature", 0.99),
        xi=config.get("xi", 1e-5),
        learning_starts=config.get("learning_starts", 0),
        scheduling_interval=config.get("scheduling_interval", 5),
        total_timesteps=total_timesteps,
        logger=logger,
        seed=seed,
        progress_bar=True,
    )

    # Evaluate
    episodes_per_task = 50
    q_net = result[0][0]
    returns = []
    for task_id in range(len(train_set)):
        env = train_set.get_task(task_id)
        returns.extend(evaluate_policy(env, q_net, episodes=episodes_per_task))
        stats = asymptotic_stats(returns[-episodes_per_task:])
        auc = auc_returns(returns[-episodes_per_task:])
        # ttt = time_to_threshold(returns, threshold=0.8, window=10, min_stability=3)
        logger.record_stat(f"eval/task_{task_id}/mean_return", stats["mean"])
        logger.record_stat(f"eval/task_{task_id}/std_return", stats["std"])
        logger.record_stat(f"eval/task_{task_id}/auc_return", auc)
        # logger.record_stat(f"eval/task_{task_id}/time_to_threshold_0.8", ttt if ttt is not None else -1)
        print(f"Task {task_id}: mean return={stats['mean']:.3f}, std={stats['std']:.3f}, AUC={auc:.3f}")
    logger.record_stat("eval/overall/mean_return", np.mean(returns))
    logger.record_stat("eval/overall/std_return", np.std(returns, ddof=1))
    print(f"Experiment {config['name']} completed")
    env.close()
    return result


if __name__ == "__main__":

    experiment_configs = [
        {
            "name": "Dissimilarity",
            "seed": 1,
            "backbone": "DDQN",
            "task_selector": "Dissimilarity",
            "temperature": 0.25,
            "total_timesteps": 25_000
        },
        {
            "name": "Round_Robin",
            "seed": 1,
            "backbone": "DDQN",
            "task_selector": "Round Robin",
            "total_timesteps": 25_000
        },
        {
            "name": "Similarity",
            "seed": 1,
            "backbone": "DDQN",
            "task_selector": "Similarity",
            "temperature": 0.25,
            "total_timesteps": 25_000
        },
        {
            "name": "SimilarityUCB",
            "seed": 1,
            "backbone": "DDQN",
            "task_selector": "Similarity UCB",
            "total_timesteps": 25_000
        },
        {
            "name": "Diversity",
            "seed": 1,
            "backbone": "DDQN",
            "task_selector": "Diversity",
            "total_timesteps": 25_000
        },
    ]

    print(f"Start experiments: {', '.join([cfg['name'] for cfg in experiment_configs])}")

    results = []
    for config in experiment_configs:
        try:
            res = run_experiment(config)
            results.append((config['name'], res))
        except Exception as e:
            print(f"Experiment {config['name']} failed with error: {e}")
            continue

    print(f"Completed experiments: {', '.join([name for name, _ in results])}")
