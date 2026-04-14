import sys
import time
import numpy as np
import jax
sys.path.append("/home/emmanuel-gendy/Documents/EnergySim")

from stable_baselines3 import PPO
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
LEARNING_RATE = 3e-4

def map_actions(norm_actions, n_envs, n_rooms):
    import jax.numpy as jnp
    from energysim.core.shared.data_structs import SystemActions
    return SystemActions(
        battery_power_w=norm_actions[:, 0] * 3000.0,
        heat_pump_power_w=(norm_actions[:, 1:1+n_rooms] + 1.0) * 1000.0,
        ac_power_w=jnp.zeros((n_envs, n_rooms)), 
        storage_discharge_w=jnp.zeros((n_envs, n_rooms))
    )

# ✅ NEW: Custom Callback to guarantee mathematical parity with JAX
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_rewards = []
    
    def _on_step(self) -> bool:
        # Capture the raw reward from all 2048 environments at every step
        step_rewards = self.locals.get('rewards')
        if step_rewards is not None:
            self.rollout_rewards.append(np.mean(step_rewards))
        return True
        
    def _on_rollout_end(self) -> None:
        if len(self.rollout_rewards) > 0:
            # Match JAX exactly: Average over 64 steps, then scale to 96-step episode
            mean_episodic_return = np.mean(self.rollout_rewards) * 96
            self.logger.record("custom/mean_episodic_return", mean_episodic_return)
            self.rollout_rewards = []

def run_benchmark():
    print("--- [SB3] 1. Initializing JAX Physics Engine ---")
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)
    
    dataset = SimulationDataset(file_path="/home/emmanuel-gendy/Documents/EnergySim/examples/sample_data.csv", dt_seconds=900)    
    sim = JAXSimulator(
        dt_seconds=900, t_config=t_config, r_config=RewardConfig(price_weight=1.0, comfort_weight=5.0),
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig(), 
        ts_config=ThermalStorageConfig(), pv_config=PVConfig()
    )
    jax_env = VectorizedEnergyEnv(sim, dataset, num_envs=NUM_ENVS)
    
    print("--- [SB3] 2. Building the PyTorch/NumPy Bridge ---")
    obs_dim = n_rooms + 5
    action_dim = 1 + n_rooms
    
    import jax.numpy as jnp
    room_indices = jnp.array(t_config.room_air_indices)
    bound_extract_obs = lambda state, exo: extract_obs(state, exo, room_indices)
    
    gym_env = EnergySimGymWrapper(
        jax_env=jax_env, num_envs=NUM_ENVS, obs_dim=obs_dim, action_dim=action_dim, 
        extract_obs_fn=bound_extract_obs, map_actions_fn=lambda a, n: map_actions(a, n, n_rooms)
    )

    print("--- [SB3] 3. Initializing Stable-Baselines3 PPO ---")
    model = PPO(
        "MlpPolicy", gym_env, learning_rate=LEARNING_RATE, n_steps=64, 
        batch_size=2048 * 64, n_epochs=1, verbose=1
    )
    
    new_logger = configure("./sb3_logs", ["stdout", "csv"])
    model.set_logger(new_logger)

    print(f"--- [SB3] 4. Starting Benchmark: {TOTAL_TIMESTEPS} total steps ---")
    start_time = time.time()
    
    # ✅ FIX: Attach the custom logger callback
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=RewardLoggingCallback())
    
    wall_clock_time = time.time() - start_time
    print(f"--- Benchmark Complete ---")
    print(f"Total Wall-Clock Time: {wall_clock_time:.2f} seconds")

if __name__ == "__main__":
    run_benchmark()