import jax
import jax.numpy as jnp
from typing import NamedTuple

# JAX natively understands NamedTuples and will automatically stack 
# everything inside this into big arrays for us.
class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    value: jax.Array
    log_prob: jax.Array
    done: jax.Array

def create_rollout_function(env, num_envs, rollout_steps, extract_obs_fn, map_actions_fn):
    
    def collect_rollout(policy, initial_env_state, key):
        
        # We define rollout_step INSIDE here so it can "see" the policy variable
        # without us having to pass the policy through the JAX loop's 'carry'
        def rollout_step(carry, _):
            env_state, current_key = carry
            current_key, action_key = jax.random.split(current_key)
            
            # Extract Observations
            t = env_state.time_idx[0]
            exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
            obs = jax.vmap(extract_obs_fn, in_axes=(0, None, None))(
                env_state.sim.state, 
                exo_batch, 
                room_indices # <-- Ensure this variable is defined or pulled from the environment config!
)            
            # Ask Policy
            mean, log_std, value = jax.vmap(policy)(obs)
            std = jnp.exp(log_std)
            
            # Add Noise
            noise = jax.random.normal(action_key, mean.shape)
            action = mean + noise * std
            
            # Calculate Probability
            log_prob = -0.5 * jnp.sum(((action - mean) / std)**2 + 2*log_std + jnp.log(2*jnp.pi), axis=-1)
            
            # Step Environment
            phys_actions = map_actions_fn(action, num_envs)
            next_env_state, reward, done, _ = env.step(env_state, phys_actions)
            
            # Save memory into our JAX-friendly NamedTuple
            transition = Transition(
                obs=obs, 
                action=action, 
                reward=reward, 
                value=value, 
                log_prob=log_prob, 
                done=done.astype(jnp.float32)
            )
            
            # Only the things that actually change go into the next loop
            return (next_env_state, current_key), transition

        initial_carry = (initial_env_state, key)
        
        final_carry, transitions = jax.lax.scan(
            rollout_step, 
            initial_carry, 
            None, 
            length=rollout_steps
        )
        
        final_env_state, final_key = final_carry
        return final_env_state, final_key, transitions
        
    return collect_rollout