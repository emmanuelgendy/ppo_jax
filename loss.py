from email import policy

import jax
import jax.numpy as jnp
import equinox as eqx

def calculate_gae(transitions, last_val, gamma=0.99, gae_lambda=0.95):
    """
    Generalized Advantage Estimation (GAE).
    Calculates how much BETTER an action was compared to what the Critic expected.
    """
    # 1. Calculate the raw difference between what happened and what was expected
    
    # FIX: Reshape last_val to act as a single "timestep" row, 
    # then concatenate it cleanly along the Time axis (axis=0).
    last_val_reshaped = last_val.reshape(1, -1) 
    next_values = jnp.concatenate([transitions.value[1:], last_val_reshaped], axis=0)
    
    # Delta = Immediate Reward + (Discounted Next Value) - Current Value Guess
    deltas = transitions.reward + gamma * next_values * (1.0 - transitions.done) - transitions.value

    # 2. Accumulate these differences backwards through time
    def scan_fn(carry, delta_and_done):
        gae_so_far = carry
        delta, done = delta_and_done
        # If the day ended (done=1), we reset the accumulation
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae_so_far
        return gae, gae
    
    # Run the accumulation backwards
    initial_gae = jnp.zeros(deltas.shape[1])
    _, advantages = jax.lax.scan(scan_fn, initial_gae, (deltas, transitions.done), reverse=True)
    
    # 3. The Target Return is what the Critic *should* have guessed
    returns = advantages + transitions.value
    
    return advantages, returns

def ppo_loss(policy, transitions, advantages, returns, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01):
    """
    The core PPO objective function. We try to MAXIMIZE the advantage, 
    but PPO "clips" the update so the AI doesn't change its mind too wildly at once.
    """
    # Normalize advantages (makes training mathematically stable)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 1. Ask the CURRENT policy what it thinks of the OLD observations
    batch_policy = jax.vmap(jax.vmap(policy))
    mean, log_std, values = batch_policy(transitions.obs)
    std = jnp.exp(log_std)
    
    # Calculate the probability of the old actions under the NEW policy
    new_log_prob = -0.5 * jnp.sum(((transitions.action - mean) / std)**2 + 2*log_std + jnp.log(2*jnp.pi), axis=-1)
    
    # 2. How much did the policy change? (Ratio of new probability to old probability)
    ratio = jnp.exp(new_log_prob - transitions.log_prob)

    # 3. PPO's Magic Trick: The Clipped Surrogate Objective
    # We want to increase the probability of good actions, but not by too much at once.
    p1 = ratio * advantages
    p2 = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    actor_loss = -jnp.mean(jnp.minimum(p1, p2)) # Negative because optimizers MINIMIZE loss

    # 4. Critic Loss: Did the Critic guess the return correctly? (Mean Squared Error)
    critic_loss = vf_coef * jnp.mean(jnp.square(values - returns))

    # 5. Entropy Bonus: Reward the AI for keeping its options open (Exploration)
    entropy = jnp.mean(0.5 + 0.5 * jnp.log(2 * jnp.pi * jnp.square(std)))
    entropy_loss = -ent_coef * entropy

    # Total loss is Actor + Critic - Entropy
    total_loss = actor_loss + critic_loss + entropy_loss

    return total_loss, (actor_loss, critic_loss, entropy)