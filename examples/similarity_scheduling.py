from functools import partial

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from rl_blox.algorithm.active_mt import train_active_mt
from rl_blox.algorithm.ddqn import train_ddqn
from rl_blox.algorithm.nature_dqn import train_nature_dqn
from rl_blox.algorithm.smt import ContextualMultiTaskDefinition
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.replay_buffer import (
    MultiTaskReplayBuffer,
    ReplayBuffer,
)
from rl_blox.logging.logger import AIMLogger
import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


class MultiClassFrozenLake(ContextualMultiTaskDefinition):
    def __init__(self, seed: int = 0, render_mode=None):
        super().__init__(
            contexts=[jnp.array([x]) for x in jnp.linspace(0.1, 1.0, num=10)],
            context_in_observation=True,
        )
        self.seed = seed
        self.render_mode = render_mode
        self.env = None

    def _get_env(self, context):
        prob = float(context[0])
        random_map = generate_random_map(size=4, p=prob)
        if self.env is not None:
            self.env.close()
        np.random.seed(self.seed)
        self.env = gym.make(
            "FrozenLake-v1",
            desc=random_map,
            is_slippery=True,
            render_mode=self.render_mode,
        )
        return self.env

    def get_solved_threshold(self, task_id: int) -> float:
        return 0.78

    def get_unsolvable_threshold(self, task_id: int) -> float:
        return 0.2

    def close(self):
        self.env.close()


seed = 42
verbose = 2
backbone = "DDQN"

if verbose:
    print(
        "This example uses the AIM logger. You will not see any output on "
        "stdout. Run 'aim up' to analyze the progress."
    )
logger = AIMLogger()
logger.define_experiment(
    env_name="MountainCar-v0",
    algorithm_name=f"SMT-{backbone}",
    hparams={},
)

mt_def = MultiClassFrozenLake()

env = mt_def.get_task(0)
q_net = MLP(
    env.observation_space.shape[0],
    int(env.action_space.n),
    activation="relu",
    hidden_nodes=[128, 128],
    rngs=nnx.Rngs(seed),
)
q_target_net = nnx.clone(q_net)
replay_buffer = MultiTaskReplayBuffer(
    ReplayBuffer(buffer_size=100_000, discrete_actions=True),
    len(mt_def),
)
optimizer = nnx.Optimizer(q_net, optax.adam(0.003), wrt=nnx.Param)
if backbone == "DDQN":
    train_st = partial(
        train_ddqn,
        q_net=q_net,
        optimizer=optimizer,
        q_target_net=q_target_net,
    )
elif backbone == "NDQN":
    train_st = partial(
        train_nature_dqn,
        q_net=q_net,
        optimizer=optimizer,
        q_target_net=q_target_net,
    )
else:
    raise NotImplementedError(f"Unknown backbone '{backbone}'")

result = train_active_mt(
    mt_def,
    train_st,
    replay_buffer,
    task_selector="Similarity",
    r_max=200,
    ducb_gamma=0.95,
    xi=1e-5,
    learning_starts=0,
    scheduling_interval=5,
    total_timesteps=200_000,
    logger=logger,
    seed=seed,
)
mt_def.close()

# Evaluation
result_st = result[0]
q_net = result_st[0]
mt_env = MultiClassFrozenLake(render_mode="human")
for task_id in range(len(mt_env)):
    print(f"Evaluating task {task_id}")
    env = mt_env.get_task(task_id)
    done = False
    infos = {}
    obs, _ = env.reset()
    while not done:
        action = int(jnp.argmax(q_net([obs])))
        next_obs, reward, termination, truncation, infos = env.step(action)
        done = termination or truncation
        obs = np.asarray(next_obs)
