import jax
import jax.numpy as jnp
import equinox as eqx

class VectorReplayBuffer(eqx.Module):
    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_obs: jax.Array
    dones: jax.Array
    pos: jax.Array
    full: jax.Array
    max_steps: int = eqx.field(static=True)
    num_envs: int = eqx.field(static=True)

    def __init__(self, max_steps: int, num_envs: int, obs_dim: int, action_dim: int):
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.obs = jnp.zeros((max_steps, num_envs, obs_dim))
        self.actions = jnp.zeros((max_steps, num_envs, action_dim))
        self.rewards = jnp.zeros((max_steps, num_envs))
        self.next_obs = jnp.zeros((max_steps, num_envs, obs_dim))
        self.dones = jnp.zeros((max_steps, num_envs))
        self.pos = jnp.array(0, dtype=jnp.int32)
        self.full = jnp.array(False, dtype=jnp.bool_)

    @eqx.filter_jit
    def add(self, obs, action, reward, next_obs, done):
        # Cast 'done' to float32 to match the buffer's dtype and prepare for Q-value math
        done_float = done.astype(jnp.float32)
        
        # Insert the batch of 2048 transitions at the current time pointer
        new_obs = jax.lax.dynamic_update_slice(self.obs, obs[None, ...], (self.pos, 0, 0))
        new_actions = jax.lax.dynamic_update_slice(self.actions, action[None, ...], (self.pos, 0, 0))
        new_rewards = jax.lax.dynamic_update_slice(self.rewards, reward[None, ...], (self.pos, 0))
        new_next_obs = jax.lax.dynamic_update_slice(self.next_obs, next_obs[None, ...], (self.pos, 0, 0))
        new_dones = jax.lax.dynamic_update_slice(self.dones, done_float[None, ...], (self.pos, 0))
            
        next_pos = self.pos + 1
        new_full = jnp.logical_or(self.full, next_pos == self.max_steps)
        new_pos = next_pos % self.max_steps
        
        return eqx.tree_at(
            lambda b: (b.obs, b.actions, b.rewards, b.next_obs, b.dones, b.pos, b.full),
            self,
            (new_obs, new_actions, new_rewards, new_next_obs, new_dones, new_pos, new_full)
        )

    @eqx.filter_jit
    def sample(self, key, batch_size):
        # Defines the upper bound for our random sampling
        max_idx = jnp.where(self.full, self.max_steps, self.pos)
        
        k1, k2 = jax.random.split(key)
        # Sample random timesteps and random environments independently
        idx_t = jax.random.randint(k1, (batch_size,), minval=0, maxval=max_idx)
        idx_e = jax.random.randint(k2, (batch_size,), minval=0, maxval=self.num_envs)
        
        return (
            self.obs[idx_t, idx_e],
            self.actions[idx_t, idx_e],
            self.rewards[idx_t, idx_e],
            self.next_obs[idx_t, idx_e],
            self.dones[idx_t, idx_e]
        )