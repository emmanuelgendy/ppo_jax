import jax
import jax.numpy as jnp
import equinox as eqx

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0

class SACActor(eqx.Module):
    trunk: eqx.nn.MLP
    mean_head: eqx.nn.Linear
    log_std_head: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.trunk = eqx.nn.MLP(obs_dim, hidden_dim, hidden_dim, 2, key=k1)
        self.mean_head = eqx.nn.Linear(hidden_dim, action_dim, key=k2)
        self.log_std_head = eqx.nn.Linear(hidden_dim, action_dim, key=k3)

    def __call__(self, obs: jax.Array, key: jax.Array = None):
        features = jax.nn.relu(self.trunk(obs))
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Bound the standard deviation for numerical stability
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        if key is None:
            # Deterministic mode (for evaluation)
            action = jnp.tanh(mean)
            log_prob = None
        else:
            # Stochastic mode (for training and exploration)
            noise = jax.random.normal(key, mean.shape)
            pi = mean + noise * std
            action = jnp.tanh(pi)
            
            # Enforce the Tanh squashing correction on the log probability
            log_prob = -0.5 * (((pi - mean) / std)**2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
            log_prob = jnp.sum(log_prob, axis=-1)
            log_prob -= jnp.sum(jnp.log(1.0 - action**2 + 1e-6), axis=-1)
            
        return action, log_prob

class SACTwinQ(eqx.Module):
    q1: eqx.nn.MLP
    q2: eqx.nn.MLP

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        in_size = obs_dim + action_dim
        # Twin-Q to prevent overestimation bias
        self.q1 = eqx.nn.MLP(in_size, 1, hidden_dim, 2, key=k1)
        self.q2 = eqx.nn.MLP(in_size, 1, hidden_dim, 2, key=k2)

    def __call__(self, obs: jax.Array, action: jax.Array):
        # The critic evaluates the value of a specific state-action pair
        x = jnp.concatenate([obs, action], axis=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)