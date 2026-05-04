import sys
import time
import numpy as np
import jax
sys.path.append("/home/emmanuel-gendy/Documents/EnergySim")

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, PVConfig
)
from examples.build_my_house import create_2_room_house
from gym_wrapper import EnergySimGymWrapper
from energysim.rl.helpers import extract_obs

NUM_ENVS = 2048
EPOCHS = 200
TOTAL_TIMESTEPS = 2048 * 64 * EPOCHS 

def map_actions(norm_actions, n_envs, n_rooms):
    import jax.numpy as jnp
    from energysim.core.shared.data_structs import SystemActions
    return SystemActions(
        battery_power_w=norm_actions[:, 0] * 3000.0,
        heat_pump_power_w=(norm_actions[:, 1:1+n_rooms] + 1.0) * 1000.0,
        ac_power_w=jnp.zeros((n_envs, n_rooms)), 
        storage_discharge_w=jnp.zeros((n_envs, n_rooms))
    )

import time # Ensure this is imported at the top of your file!

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_rewards = []
        self.log_frequency = 64 
        self.start_time = None
    
    def _on_training_start(self) -> None:
        # Start the clock when training begins
        self.start_time = time.time()

    def _on_step(self) -> bool:
        rew = self.locals.get('reward')
        if rew is None:
            rew = self.locals.get('rewards')
            
        if rew is not None:
            self.step_rewards.append(np.mean(rew))
            
        # Manually force the CSV dump every 64 steps
        if len(self.step_rewards) >= self.log_frequency:
            mean_episodic_return = np.mean(self.step_rewards) * 96 
            
            # 1. Log the Y-Axis (Reward)
            self.logger.record("custom/mean_episodic_return", mean_episodic_return)
            
            # 2. ✅ Log the X-Axes manually so the plotter can see them!
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.record("time/time_elapsed", time.time() - self.start_time)
            
            # 3. Force the dump to the CSV
            self.logger.dump(step=self.num_timesteps) 
            self.step_rewards = []
            
        return True

import csv
import time
import os

# ✅ FOOLPROOF LOGGER: Writes directly to hard drive, ignoring SB3's rules
class DirectCSVLoggerCallback(BaseCallback):
    def __init__(self, filename="sac_metrics.csv", verbose=0):
        super().__init__(verbose)
        self.filename = filename
        self.step_rewards = []
        self.start_time = None
        self.log_frequency = 64
        
        # Create CSV and write headers
        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Total_Steps", "Wall_Clock_Time", "Mean_Reward"])
            
    def _on_training_start(self):
        self.start_time = time.time()
        
    def _on_step(self):
        # SAC usually stores in 'reward' (singular)
        rew = self.locals.get("reward")
        if rew is None:
            rew = self.locals.get("rewards")
            
        if rew is not None:
            self.step_rewards.append(np.mean(rew))
            
        # Every 64 steps, forcibly write a row to the CSV
        if len(self.step_rewards) >= self.log_frequency:
            mean_ep_return = np.mean(self.step_rewards) * 96
            wall_clock = time.time() - self.start_time
            
            with open(self.filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, wall_clock, mean_ep_return])
                
            self.step_rewards = []
        return True

# Inside your run_benchmark() function:
def run_benchmark():
    print("--- [SB3 SAC] 1. Initializing JAX Physics Engine ---")
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)
    
    dataset = SimulationDataset(file_path="/home/emmanuel-gendy/Documents/EnergySim/examples/sample_data.csv", dt_seconds=900)    
    sim = JAXSimulator(
        dt_seconds=900, t_config=t_config, r_config=RewardConfig(price_weight=1.0, comfort_weight=5.0),
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig(), 
        ts_config=ThermalStorageConfig(), pv_config=PVConfig()
    )
    jax_env = VectorizedEnergyEnv(sim, dataset, num_envs=NUM_ENVS)
    
    print("--- [SB3 SAC] 2. Building the PyTorch/NumPy Bridge ---")
    obs_dim = n_rooms + 5
    action_dim = 1 + n_rooms
    room_indices = jax.numpy.array(t_config.room_air_indices)
    bound_extract_obs = lambda state, exo: extract_obs(state, exo, room_indices)    
    gym_env = EnergySimGymWrapper(
        jax_env=jax_env, num_envs=NUM_ENVS, obs_dim=obs_dim, action_dim=action_dim, 
        extract_obs_fn=bound_extract_obs, map_actions_fn=lambda a, n: map_actions(a, n, n_rooms)
    )

    print("--- [SB3 SAC] 3. Initializing Stable-Baselines3 SAC ---")
    # SAC doesn't use n_steps. We limit the buffer size so it doesn't crash your RAM with 2048 envs!
    model = SAC("MlpPolicy", gym_env, buffer_size=5_000, learning_rate=3e-4, verbose=1)
    
    print(f"--- [SB3 SAC] 4. Starting Benchmark ---")
    start_time = time.time()
    
    # ✅ Pass the new direct logger
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=DirectCSVLoggerCallback())

    wall_clock_time = time.time() - start_time
    print(f"--- Benchmark Complete ---")
    print(f"Total Wall-Clock Time: {wall_clock_time:.2f} seconds")

if __name__ == "__main__":
    run_benchmark()