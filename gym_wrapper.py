import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym

class EnergySimGymWrapper(gym.vector.VectorEnv):
    """
    A hardware-bridging wrapper that translates between PyTorch/NumPy (CPU) 
    and EnergySim/JAX (CPU/GPU).
    """
    def __init__(self, jax_env, num_envs, obs_dim, action_dim, extract_obs_fn, map_actions_fn):
        # 1. Initialize the Standard VectorEnv Base
        self.jax_env = jax_env
        self.num_envs = num_envs
        self.extract_obs_fn = extract_obs_fn
        self.map_actions_fn = map_actions_fn
        
        # 2. Define the mathematical bounds of your MDP
        # Standard PPO outputs actions in the continuous range [-1.0, 1.0]
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Observations can theoretically be any float (temperatures, weather)
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        super().__init__(num_envs, observation_space, action_space)
        
        # Internal state tracking for JAX
        self._env_state = None
        self._key = jax.random.PRNGKey(42)

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to the initial state distribution.
        """
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
            
        self._key, reset_key = jax.random.split(self._key)
        
        # JAX physics step
        self._env_state = self.jax_env.reset(reset_key)
        
        # Extract observations
        t = self._env_state.time_idx[0]
        exo_batch = jax.tree.map(lambda x: x[t], self.jax_env.shared_exo_data)
        
        jax_obs = jax.vmap(self.extract_obs_fn, in_axes=(0, None))(
            self._env_state.sim.state, exo_batch
        )
        
        # Hardware Bridge: Convert JAX array -> NumPy array
        np_obs = np.array(jax_obs)
        infos = {} # Gymnasium expects a dictionary of extra info
        
        return np_obs, infos

    def step(self, actions):
        """
        Takes a NumPy action batch, steps the JAX physics, and returns NumPy arrays.
        """
        # Hardware Bridge: Convert NumPy array -> JAX array
        jax_actions = jnp.array(actions)
        
        # Map bounded [-1, 1] actions to physical watts/setpoints
        phys_actions = self.map_actions_fn(jax_actions, self.num_envs)
        
        # Step the physics
        self._env_state, jax_reward, jax_done, _ = self.jax_env.step(self._env_state, phys_actions)
        
        # Extract the new observations
        t = self._env_state.time_idx[0]
        exo_batch = jax.tree.map(lambda x: x[t], self.jax_env.shared_exo_data)
        jax_next_obs = jax.vmap(self.extract_obs_fn, in_axes=(0, None))(
            self._env_state.sim.state, exo_batch
        )
        
        # Hardware Bridge: Convert back to NumPy
        np_obs = np.array(jax_next_obs)
        np_reward = np.array(jax_reward)
        np_done = np.array(jax_done, dtype=bool)
        
        # Gymnasium Vector API separates done into 'terminated' (physics ended) 
        # and 'truncated' (time limit reached). We can mirror them for simple MDPs.
        terminations = np_done
        truncations = np.zeros_like(np_done, dtype=bool) 
        infos = {}
        
        # --- The Auto-Reset Requirement ---
        # If an environment finishes an episode, the standard RL loop expects it 
        # to immediately reset so no time is wasted. JAX VectorizedEnvs usually 
        # handle this internally inside the `step` function. If EnergySim does 
        # not auto-reset, you must catch `np_done` here and overwrite the state.
        
        return np_obs, np_reward, terminations, truncations, infos