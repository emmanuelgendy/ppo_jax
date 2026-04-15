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

from networks import PPOPolicy
from rollout import create_rollout_function
from loss import calculate_gae, ppo_loss

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

# os.environ["JAX_PLATFORM_NAME"] = "cpu"

NUM_ENVS = 2048
ROLLOUT_STEPS = 64
EPOCHS = 200 # Set back to 200 for full benchmarking
LEARNING_RATE = 3e-4

def map_actions(norm_actions, n_envs, n_rooms):
    from energysim.core.shared.data_structs import SystemActions
    return SystemActions(
        battery_power_w=norm_actions[:, 0] * 3000.0,
        heat_pump_power_w=(norm_actions[:, 1:1+n_rooms] + 1.0) * 1000.0,
        ac_power_w=jnp.zeros((n_envs, n_rooms)), storage_discharge_w=jnp.zeros((n_envs, n_rooms))
    )

def train():
    print("--- [JAX] 1. Setting up Environment ---")
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

    print("--- [JAX] 2. Initializing Brain & Optimizer ---")
    key = jax.random.PRNGKey(42)
    key, net_key, env_key = jax.random.split(key, 3)
    
    obs_dim = n_rooms + 5
    action_dim = 1 + n_rooms
    policy = PPOPolicy(obs_dim, action_dim, hidden_dim=64, key=net_key)
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))
    env_state = env.reset(env_key)
    
    bound_extract_obs = lambda state, exo: extract_obs(state, exo, room_indices)
    
    collect_rollout = create_rollout_function(
        env, NUM_ENVS, ROLLOUT_STEPS, bound_extract_obs, lambda a, n: map_actions(a, n, n_rooms)
    )

    print("--- [JAX] 3. Compiling the Training Loop ---")
    @eqx.filter_jit
    def update_step(policy, env_state, opt_state, key):
        next_env_state, next_key, transitions = collect_rollout(policy, env_state, key)
        
        EPISODE_LENGTH = 96 
        mean_episodic_return = jnp.mean(transitions.reward) * EPISODE_LENGTH

        # Calculate expected return for logging
        mean_batch_reward = jnp.mean(transitions.reward)
        
        t = next_env_state.time_idx[0]
        exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
        final_obs = jax.vmap(bound_extract_obs, in_axes=(0, None))(next_env_state.sim.state, exo_batch)
        _, _, last_val = jax.vmap(policy)(final_obs)
        
        advantages, returns = calculate_gae(transitions, last_val)
        
        def objective(p):
            loss, metrics = ppo_loss(p, transitions, advantages, returns)
            return loss, metrics
            
        (loss, metrics), grads = eqx.filter_value_and_grad(objective, has_aux=True)(policy)
        updates, new_opt_state = optimizer.update(grads, opt_state, policy)
        new_policy = eqx.apply_updates(policy, updates)
        
        return new_policy, next_env_state, new_opt_state, next_key, metrics, loss, mean_episodic_return

    print("--- [JAX] 4. Starting Training ---")
    with open("jax_metrics.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Total_Steps", "Wall_Clock_Time", "FPS", "Mean_Reward"])
        
        global_start_time = time.time()
        total_steps = 0
        
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            policy, env_state, opt_state, key, metrics, loss, mean_batch_reward = update_step(policy, env_state, opt_state, key)
            
            step_time = time.time() - epoch_start_time
            wall_clock_time = time.time() - global_start_time
            fps = (NUM_ENVS * ROLLOUT_STEPS) / step_time
            total_steps += (NUM_ENVS * ROLLOUT_STEPS)
            
            actor_loss, critic_loss, entropy = metrics
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | FPS: {fps:,.0f} | Reward: {mean_batch_reward:.4f} | Loss: {loss:.4f}")
            writer.writerow([epoch + 1, total_steps, wall_clock_time, fps, float(mean_batch_reward)])

    eqx.tree_serialise_leaves("jax_ppo_model.eqx", policy)
    print("✨ Model saved to jax_ppo_model.eqx")

if __name__ == "__main__":
    train()