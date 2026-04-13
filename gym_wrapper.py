import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

class EnergySimGymWrapper(VecEnv):
    """
    A hardware-bridging wrapper that perfectly mimics Stable-Baselines3's native 
    vectorized environment structure, bypassing Gymnasium's VectorEnv conflicts.
    """
    def __init__(self, jax_env, num_envs, obs_dim, action_dim, extract_obs_fn, map_actions_fn):
        self.jax_env = jax_env
        self.extract_obs_fn = extract_obs_fn
        self.map_actions_fn = map_actions_fn
        
        # SB3's VecEnv expects the spaces of a SINGLE environment. 
        # It natively infers the batch dimension from 'num_envs'
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        super().__init__(num_envs, observation_space, action_space)
        
        # Internal state tracking
        self._env_state = None
        self._key = jax.random.PRNGKey(42)
        self._actions = None

    def reset(self):
        """SB3 VecEnv reset returns ONLY the observation array."""
        self._key, reset_key = jax.random.split(self._key)
        self._env_state = self.jax_env.reset(reset_key)
        
        t = self._env_state.time_idx[0]
        exo_batch = jax.tree.map(lambda x: x[t], self.jax_env.shared_exo_data)
        jax_obs = jax.vmap(self.extract_obs_fn, in_axes=(0, None))(
            self._env_state.sim.state, exo_batch
        )
        
        return np.array(jax_obs)

    def step_async(self, actions):
        """SB3 splits step into async and wait for multiprocessing. We just cache the actions."""
        self._actions = actions

    def step_wait(self):
        """Executes the JAX physics and returns obs, reward, done, info."""
        jax_actions = jnp.array(self._actions)
        phys_actions = self.map_actions_fn(jax_actions, self.num_envs)
        
        self._env_state, jax_reward, jax_done, _ = self.jax_env.step(self._env_state, phys_actions)
        
        t = self._env_state.time_idx[0]
        exo_batch = jax.tree.map(lambda x: x[t], self.jax_env.shared_exo_data)
        jax_next_obs = jax.vmap(self.extract_obs_fn, in_axes=(0, None))(
            self._env_state.sim.state, exo_batch
        )
        
        np_obs = np.array(jax_next_obs)
        np_reward = np.array(jax_reward)
        np_done = np.array(jax_done, dtype=bool)
        
        # SB3 requires a list of dicts for infos (one for each environment in the batch)
        infos = [{} for _ in range(self.num_envs)]
        
        return np_obs, np_reward, np_done, infos

    # --- Dummy methods required by SB3's VecEnv Abstract Base Class ---
    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [None] * self.num_envs
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [None] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs