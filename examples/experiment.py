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


class OneHotObsWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, Discrete), "env.observation_space must be Discrete"
        self.n = env.observation_space.n
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def observation(self, obs):
        idx = int(np.asarray(obs))
        onehot = np.zeros(self.n, dtype=np.float32)
        onehot[idx] = 1.0
        return onehot


def run_experiment(config: dict):
    print(f"Start experiment: {config['name']}")

    seed = config.get("seed", 42)
    env_name = config.get("env_name", "FrozenLake-v1")
    map_probs = config.get("map_probs", [0.9, 0.7, 0.5, 0.3])
    backbone = config.get("backbone", "DDQN")
    task_selector = config.get("task_selector", "Dissimilarity")
    total_timesteps = config.get("total_timesteps", 100_000)

    logger = AIMLogger()
    logger.define_experiment(env_name=env_name, algorithm_name=f"AMT-{backbone}-{task_selector}", hparams=config)

    maps = [generate_random_map(size=4, p=prob, seed=seed) for prob in map_probs]

    base_env = gym.make(env_name, is_slippery=False)
    base_env = OneHotObsWrapper(base_env)

    def set_context_fn(env: gym.Env, context: np.ndarray) -> None:
        map_idx = int(context.argmax())
        new_env = gym.make("FrozenLake-v1", desc=maps[map_idx])
        env.unwrapped.__dict__.update(new_env.unwrapped.__dict__)
        new_env.close()

    contexts = np.eye(len(maps), dtype=np.float32)
    train_set = DiscreteTaskSet(base_env=base_env, set_context=set_context_fn, contexts=contexts, context_aware=True)

    env_dummy = train_set.get_task(0)
    q_net = MLP(env_dummy.observation_space.shape[0], int(env_dummy.action_space.n), activation="relu", hidden_nodes=[128, 128, 128], rngs=nnx.Rngs(seed))
    q_target_net = nnx.clone(q_net)
    replay_buffer = MultiTaskReplayBuffer(ReplayBuffer(buffer_size=100_000, discrete_actions=True), len(train_set))
    optimizer = nnx.Optimizer(q_net, optax.adam(0.003), wrt=nnx.Param)
    train_st = partial(train_ddqn, q_net=q_net, optimizer=optimizer, q_target_net=q_target_net)

    result = train_active_mt(
        train_set,
        train_st,
        replay_buffer,
        task_selector=task_selector,
        r_max=config.get("r_max", 1.0),
        ducb_gamma=config.get("ducb_gamma", 0.95),
        similarity_gamma=config.get("similarity_gamma", 0.99),
        xi=config.get("xi", 1e-5),
        learning_starts=config.get("learning_starts", 0),
        scheduling_interval=config.get("scheduling_interval", 5),
        total_timesteps=total_timesteps,
        logger=logger,
        seed=seed,
        progress_bar=False,
    )

    print(f"Experiment {config['name']} completed")
    return result


if __name__ == "__main__":

    experiment_configs = [
        {
            "name": "Dissimilarity",
            "seed": 1,
            "backbone": "DDQN",
            "task_selector": "Dissimilarity",
            "similarity_gamma": 10.0,
            "map_probs": [0.9, 0.7, 0.5, 0.3],
            "total_timesteps": 10_000
        },
        {
            "name": "Dissimilarity_Harder",
            "seed": 3,
            "backbone": "DDQN",
            "task_selector": "Dissimilarity",
            "similarity_gamma": 10.0,
            "map_probs": [0.4, 0.3, 0.2, 0.1],
            "total_timesteps": 10_000
        },
        {
            "name": "Round_Robin",
            "seed": 1,
            "backbone": "DDQN",
            "task_selector": "Round Robin",
            "map_probs": [0.9, 0.7, 0.5, 0.3],
            "total_timesteps": 10_000
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
