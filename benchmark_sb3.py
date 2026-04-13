import time
import jax
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
# Tell Python to look in the original repo for the 'examples' folder
sys.path.append("/home/emmanuel-gendy/Documents/EnergySim")


# Import your JAX environment and components
from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, PVConfig
)
from examples.build_my_house import create_2_room_house

# Import the wrapper and helper functions
from gym_wrapper import EnergySimGymWrapper
from energysim.rl.helpers import extract_obs

# --- HYPERPARAMETERS ---
# Keep these identical to your train.py for a fair fight!
NUM_ENVS = 2048
TOTAL_TIMESTEPS = 2048 * 64 * 200 # (Envs * Rollout_Steps * Epochs)
LEARNING_RATE = 3e-4

# Reuse your mapping function
def map_actions(norm_actions, n_envs, n_rooms):
    import jax.numpy as jnp
    from energysim.core.shared.data_structs import SystemActions
    return SystemActions(
        battery_power_w=norm_actions[:, 0] * 3000.0,
        heat_pump_power_w=(norm_actions[:, 1:1+n_rooms] + 1.0) * 1000.0,
        ac_power_w=jnp.zeros((n_envs, n_rooms)), 
        storage_discharge_w=jnp.zeros((n_envs, n_rooms))
    )

def run_benchmark():
    print("--- 1. Initializing JAX Physics Engine ---")
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)
    
    # Needs to match train.py exactly
    dataset = SimulationDataset(file_path="/home/emmanuel-gendy/Documents/EnergySim/examples/sample_data.csv", dt_seconds=900)    
    sim = JAXSimulator(
        dt_seconds=900, t_config=t_config, r_config=RewardConfig(price_weight=1.0, comfort_weight=5.0),
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig(), 
        ts_config=ThermalStorageConfig(), pv_config=PVConfig()
    )
    jax_env = VectorizedEnergyEnv(sim, dataset, num_envs=NUM_ENVS)
    
    print("--- 2. Building the PyTorch/NumPy Bridge ---")
    obs_dim = n_rooms + 5
    action_dim = 1 + n_rooms
    
    import jax.numpy as jnp
    room_indices = jnp.array(t_config.room_air_indices)
    bound_extract_obs = lambda state, exo: extract_obs(state, exo, room_indices)
    
    # Instantiate the wrapper
    gym_env = EnergySimGymWrapper(
        jax_env=jax_env, 
        num_envs=NUM_ENVS, 
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        extract_obs_fn=bound_extract_obs, 
        map_actions_fn=lambda a, n: map_actions(a, n, n_rooms)
    )

    print("--- 3. Initializing Stable-Baselines3 PPO ---")
    # We configure SB3 to match your JAX PPO architecture
    model = PPO(
        "MlpPolicy", 
        gym_env, 
        learning_rate=LEARNING_RATE,
        n_steps=64, # Rollout steps
        batch_size=2048 * 64, # Train on the whole batch at once
        n_epochs=1, # Number of optimization epochs per rollout
        verbose=1,
        tensorboard_log="./ppo_benchmark_tensorboard/"
    )

    print(f"--- 4. Starting Benchmark: {TOTAL_TIMESTEPS} total steps ---")
    start_time = time.time()
    
    # Train the agent
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    end_time = time.time()
    wall_clock_time = end_time - start_time
    
    print(f"--- Benchmark Complete ---")
    print(f"Total Wall-Clock Time: {wall_clock_time:.2f} seconds")
    print(f"Average FPS: {TOTAL_TIMESTEPS / wall_clock_time:,.0f}")

if __name__ == "__main__":
    run_benchmark()