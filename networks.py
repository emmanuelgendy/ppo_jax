import jax
import jax.numpy as jnp
import equinox as eqx

class PPOPolicy(eqx.Module):
    trunk: eqx.nn.MLP
    actor_mean: eqx.nn.Linear
    critic_head: eqx.nn.Linear
    action_log_std: jax.Array
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, key: jax.Array):
        # JAX requires explicit random keys for initializing weights
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Shared feature extractor (The 'eyes')
        self.trunk = eqx.nn.MLP(
            in_size=obs_dim, 
            out_size=hidden_dim, 
            width_size=hidden_dim, 
            depth=2, 
            key=k1
        )
        
        # Actor: Outputs the mean action (e.g., heat pump control signal)
        self.actor_mean = eqx.nn.Linear(hidden_dim, action_dim, key=k2)
        
        # Critic: Outputs a single scalar value estimating future reward
        self.critic_head = eqx.nn.Linear(hidden_dim, 1, key=k3)
        
        # Trainable parameter dictating exploration variance. Initializes to 0 (std = 1.0)
        self.action_log_std = jnp.zeros(action_dim)
        
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        features = jax.nn.relu(self.trunk(obs))
        
        # Tanh bounds the action between -1.0 and 1.0
        mean_action = jax.nn.tanh(self.actor_mean(features))
        value_estimate = self.critic_head(features)[0]
        
        return mean_action, self.action_log_std, value_estimate