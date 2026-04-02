import jax
import jax.numpy as jnp

# A clean data structure to hold the memory of our rollout
class Transition:
    def __init__(self, obs, action, reward, value, log_prob, done):
        self.obs = obs               # What the AI saw (Temperatures)
        self.action = action         # What the AI did (Heat Pump Signal)
        self.reward = reward         # What the AI got (Comfort/Price Score)
        self.value = value           # What the Critic guessed it would get
        self.log_prob = log_prob     # The exact probability of taking that action
        self.done = done             # Did the day/episode end?

def create_rollout_function(env, num_envs, rollout_steps, extract_obs_fn, map_actions_fn):
    """
    This function creates the rollout loop. We wrap it in a function so we can 
    pass in your specific environment (env) and helper functions.
    """
    
    def rollout_step(carry, _):
        """
        This is a single step inside the loop. 
        'carry' holds the current state of the environment and the random key.
        """
        # 1. Unpack the current state
        policy, env_state, key = carry
        key, action_key = jax.random.split(key)
        
        # 2. Extract the Observation (What does the room look like right now?)
        # (This uses your custom extract_obs function)
        t = env_state.time_idx[0]
        exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
        obs = jax.vmap(extract_obs_fn, in_axes=(0, None))(env_state.sim.state, exo_batch)
        
        # 3. Ask the Policy for an action
        # mean is the best guess, std is the exploration width, value is the critic's guess
        mean, log_std, value = jax.vmap(policy)(obs)
        std = jnp.exp(log_std)
        
        # 4. EXPLORATION: Add random noise to the action
        noise = jax.random.normal(action_key, mean.shape)
        action = mean + noise * std
        
        # Calculate the log probability of this exact action occurring
        # (We need this later to calculate the PPO Advantage)
        log_prob = -0.5 * jnp.sum(((action - mean) / std)**2 + 2*log_std + jnp.log(2*jnp.pi), axis=-1)
        
        # 5. Step the Environment forward
        # (This uses your custom map_actions function to scale Watts)
        phys_actions = map_actions_fn(action, num_envs)
        next_env_state, reward, done, _ = env.step(env_state, phys_actions)
        
        # 6. Save this exact moment in time to memory
        transition = Transition(obs, action, reward, value, log_prob, done.astype(jnp.float32))
        
        # Return the new state (to carry into the next loop) and the memory we just saved
        return (policy, next_env_state, key), transition

    def collect_rollout(policy, initial_env_state, key):
        """
        This uses JAX's hyper-fast 'scan' to run the rollout_step 64 times in a row,
        stacking all the transitions into one massive memory array.
        """
        initial_carry = (policy, initial_env_state, key)
        
        # jax.lax.scan is the JAX equivalent of:
        # for i in range(rollout_steps): carry, memory[i] = rollout_step(carry, None)
        final_carry, transitions = jax.lax.scan(
            rollout_step, 
            initial_carry, 
            None, 
            length=rollout_steps
        )
        
        _, final_env_state, final_key = final_carry
        return final_env_state, final_key, transitions
        
    return collect_rollout