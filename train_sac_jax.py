import os
import time
import sys
import csv
sys.path.append("/home/emmanuel-gendy/Documents/EnergySim")

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from sac_networks import SACActor, SACTwinQ
from sac_buffer import VectorReplayBuffer

from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, PVConfig
)
from energysim.rl.helpers import extract_obs
from examples.build_my_house import create_2_room_house
import energysim.sim.simulator as sim_module

# --- HYPERPARAMETERS ---
NUM_ENVS = 2048
EPOCHS = 200
STEPS_PER_EPOCH = 64
BATCH_SIZE = 4096
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005 # Polyak averaging constant

def map_actions(norm_actions, n_envs, n_rooms):
    from energysim.core.shared.data_structs import SystemActions
    return SystemActions(
        battery_power_w=norm_actions[:, 0] * 3000.0,
        heat_pump_power_w=(norm_actions[:, 1:1+n_rooms] + 1.0) * 1000.0,
        ac_power_w=jnp.zeros((n_envs, n_rooms)), 
        storage_discharge_w=jnp.zeros((n_envs, n_rooms))
    )

def soft_update(target_model, online_model, tau):
    # Only apply Polyak averaging to the arrays (weights/biases), ignore static functions (like relu)
    return jax.tree_util.tree_map(
        lambda t, o: (1 - tau) * t + tau * o if eqx.is_array(t) else t, 
        target_model, 
        online_model
    )

def train():
    print("--- [JAX-SAC] 1. Setting up Environment ---")
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)
    room_indices = jnp.array(t_config.room_air_indices)

    if hasattr(sim_module, "create_heat_pump"):
        orig_hp = sim_module.create_heat_pump
        sim_module.create_heat_pump = lambda cfg, n: orig_hp(cfg, n_rooms)

    dataset = SimulationDataset(file_path="/home/emmanuel-gendy/Documents/EnergySim/examples/sample_data.csv", dt_seconds=900)
    sim = JAXSimulator(
        dt_seconds=900, t_config=t_config, r_config=RewardConfig(price_weight=1.0, comfort_weight=5.0),
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig(), 
        ts_config=ThermalStorageConfig(), pv_config=PVConfig()
    )
    env = VectorizedEnergyEnv(sim, dataset, num_envs=NUM_ENVS)

    print("--- [JAX-SAC] 2. Initializing Networks & Buffer ---")
    key = jax.random.PRNGKey(42)
    key, actor_key, critic_key, env_key = jax.random.split(key, 4)
    
    obs_dim = n_rooms + 5
    action_dim = 1 + n_rooms
    
    actor = SACActor(obs_dim, action_dim, hidden_dim=64, key=actor_key)
    critic = SACTwinQ(obs_dim, action_dim, hidden_dim=64, key=critic_key)
    critic_target = critic # Initialize target equal to online

    # Automatic Entropy Tuning (Alpha)
    log_alpha = jnp.array(0.0)
    target_entropy = -float(action_dim)

    # Optimizers
    opt_actor = optax.adam(LEARNING_RATE)
    opt_critic = optax.adam(LEARNING_RATE)
    opt_alpha = optax.adam(LEARNING_RATE)
    
    state_actor = opt_actor.init(eqx.filter(actor, eqx.is_array))
    state_critic = opt_critic.init(eqx.filter(critic, eqx.is_array))
    state_alpha = opt_alpha.init(log_alpha)

    # Replay Buffer (Stores 50 steps of 2048 envs = 102,400 transitions)
    buffer = VectorReplayBuffer(max_steps=50, num_envs=NUM_ENVS, obs_dim=obs_dim, action_dim=action_dim)
    env_state = env.reset(env_key)
    bound_extract_obs = lambda state, exo: extract_obs(state, exo, room_indices)

    print("--- [JAX-SAC] 3. Compiling the Update Logic ---")

    @eqx.filter_jit
    def env_step(actor, env_state, buffer, key):
        t = env_state.time_idx[0]
        exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
        obs = jax.vmap(bound_extract_obs, in_axes=(0, None))(env_state.sim.state, exo_batch)
        
        # Sample stochastic actions
        actions, _ = jax.vmap(actor)(obs, jax.random.split(key, NUM_ENVS))
        phys_actions = map_actions(actions, NUM_ENVS, n_rooms)
        
        next_env_state, rewards, dones, _ = env.step(env_state, phys_actions)
        
        t_next = next_env_state.time_idx[0]
        exo_batch_next = jax.tree.map(lambda x: x[t_next], env.shared_exo_data)
        next_obs = jax.vmap(bound_extract_obs, in_axes=(0, None))(next_env_state.sim.state, exo_batch_next)
        
        new_buffer = buffer.add(obs, actions, rewards, next_obs, dones)
        return next_env_state, new_buffer, jnp.mean(rewards)

    @eqx.filter_jit
    def sac_update(actor, critic, critic_target, log_alpha, s_actor, s_critic, s_alpha, buffer, key):
        alpha = jnp.exp(log_alpha)
        obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(key, BATCH_SIZE)
        
        k1, k2 = jax.random.split(key)

        # 1. Update Critic
        def critic_loss_fn(c):
            next_actions, next_log_probs = jax.vmap(actor)(next_obs_b, jax.random.split(k1, BATCH_SIZE))
            next_q1, next_q2 = jax.vmap(critic_target)(next_obs_b, next_actions)
            next_q = jnp.minimum(next_q1, next_q2) - alpha * next_log_probs
            target_q = rew_b + GAMMA * (1.0 - done_b) * next_q
            
            q1, q2 = jax.vmap(c)(obs_b, act_b)
            return jnp.mean((q1 - target_q)**2) + jnp.mean((q2 - target_q)**2)

        c_loss, c_grads = eqx.filter_value_and_grad(critic_loss_fn)(critic)
        c_updates, s_critic = opt_critic.update(c_grads, s_critic, critic)
        critic = eqx.apply_updates(critic, c_updates)

        # 2. Update Actor
        def actor_loss_fn(a):
            actions, log_probs = jax.vmap(a)(obs_b, jax.random.split(k2, BATCH_SIZE))
            q1, q2 = jax.vmap(critic)(obs_b, actions)
            q = jnp.minimum(q1, q2)
            return jnp.mean(alpha * log_probs - q)

        a_loss, a_grads = eqx.filter_value_and_grad(actor_loss_fn)(actor)
        a_updates, s_actor = opt_actor.update(a_grads, s_actor, actor)
        actor = eqx.apply_updates(actor, a_updates)

        # 3. Update Alpha (Temperature)
        def alpha_loss_fn(la):
            _, log_probs = jax.vmap(actor)(obs_b, jax.random.split(k2, BATCH_SIZE))
            return -jnp.mean(jnp.exp(la) * (log_probs + target_entropy))
        
        alpha_loss, alpha_grads = eqx.filter_value_and_grad(alpha_loss_fn)(log_alpha)
        alpha_updates, s_alpha = opt_alpha.update(alpha_grads, s_alpha, log_alpha)
        log_alpha = eqx.apply_updates(log_alpha, alpha_updates)

        # 4. Soft Update Target Network
        critic_target = soft_update(critic_target, critic, TAU)

        return actor, critic, critic_target, log_alpha, s_actor, s_critic, s_alpha

    print("--- [JAX-SAC] 4. Starting Training ---")
    with open("jax_sac_metrics.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Total_Steps", "Wall_Clock_Time", "FPS", "Mean_Reward"])
        
        global_start_time = time.time()
        total_steps = 0
        
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            rewards = []
            
            # Step environments and fill buffer
            for _ in range(STEPS_PER_EPOCH):
                key, step_key = jax.random.split(key)
                env_state, buffer, mean_reward = env_step(actor, env_state, buffer, step_key)
                rewards.append(mean_reward)
                total_steps += NUM_ENVS

            # Update networks
            if epoch > 1: # Wait for buffer to populate
                for _ in range(4): # Perform multiple updates per rollout phase
                    key, update_key = jax.random.split(key)
                    actor, critic, critic_target, log_alpha, state_actor, state_critic, state_alpha = sac_update(
                        actor, critic, critic_target, log_alpha, state_actor, state_critic, state_alpha, buffer, update_key
                    )
            
            step_time = time.time() - epoch_start_time
            fps = (NUM_ENVS * STEPS_PER_EPOCH) / step_time
            wall_clock_time = time.time() - global_start_time
            mean_batch_reward = np.mean(rewards) * 96 # Scale to episodic return
            
            print(f"Epoch {epoch:03d}/{EPOCHS} | FPS: {fps:,.0f} | Expected Return: {mean_batch_reward:.4f}")
            writer.writerow([epoch, total_steps, wall_clock_time, fps, float(mean_batch_reward)])

    eqx.tree_serialise_leaves("jax_sac_model.eqx", actor)
    print("✨ Model saved to jax_sac_model.eqx")

if __name__ == "__main__":
    train()