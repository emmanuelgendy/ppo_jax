import os
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/emmanuel-gendy/Documents/EnergySim")
sys.path.append("/home/emmanuel-gendy/Documents/EnergySim/examples")
from build_my_house import create_2_room_house

# =================================================================
# 🔥 THE RUNTIME TRANSLATOR PATCH (Zero changes to environment files)
import energysim.utils.objectives as obj
if hasattr(obj, 'f_stage_cost'):
    _original_f_stage_cost = obj.f_stage_cost
    def _cost_translator(*args, **kwargs):
        if 'state' in kwargs: kwargs['current_state'] = kwargs.pop('state')
        if 'action' in kwargs: kwargs['actions'] = kwargs.pop('action')
        if 'dt' in kwargs: kwargs['dt_seconds'] = kwargs.pop('dt')
        if 'next_state' not in kwargs and len(args) < 2: kwargs['next_state'] = None
        return _original_f_stage_cost(*args, **kwargs)
    obj.f_cost_step = _cost_translator
# =================================================================

from networks import PPOPolicy
from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, PVConfig, SystemActions
)
from energysim.rl.helpers import extract_obs
from build_my_house import create_2_room_house
import energysim.sim.simulator as sim_module

def map_ppo_actions_for_eval(action_mean, n_envs, n_rooms):
    # The PPO was NOT trained on the battery. We must zero it out to ignore it.
    # Allow battery to control the bill [-5000W, +5000W]
    bat_w = action_mean[:, 0] * 5000.0 
    
    # FORCE HEAT PUMP TO ZERO - The agent cannot touch it
    hp_w = jnp.zeros((n_envs, n_rooms)) 
    
    return SystemActions(
        battery_power_w=bat_w, 
        heat_pump_power_w=hp_w,
        ac_power_w=jnp.zeros((n_envs, n_rooms)), 
        storage_discharge_w=jnp.zeros((n_envs, n_rooms))
    )

def main():
    print("--- 🔬 Energysim PPO Evaluation ---")
    NUM_ENVS = 2048
    
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)
    room_indices = jnp.array(t_config.room_air_indices)

    if hasattr(sim_module, "create_heat_pump"):
        orig_hp = sim_module.create_heat_pump
        sim_module.create_heat_pump = lambda cfg, n: orig_hp(cfg, n_rooms)

    dataset_path = "/home/emmanuel-gendy/Downloads/sample_data.csv"
    dataset = SimulationDataset(file_path=dataset_path, dt_seconds=900)
    sim = JAXSimulator(
        dt_seconds=900, t_config=t_config, r_config=RewardConfig(price_weight=0.01, comfort_weight=10.0),
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig(), 
        ts_config=ThermalStorageConfig(), pv_config=PVConfig()
    )
    env = VectorizedEnergyEnv(sim, dataset, num_envs=NUM_ENVS)

    print("Loading Trained PPO Model...")
    obs_dim = n_rooms + 5
    action_dim = 1 + n_rooms 
    
    # Initialize a dummy policy structure, then overwrite it with saved weights
    dummy_policy = PPOPolicy(obs_dim, action_dim, 64, jax.random.PRNGKey(0))
    policy = eqx.tree_deserialise_leaves("jax_ppo_model.eqx", dummy_policy)
    
    key = jax.random.PRNGKey(42)
    init_state = env.reset(key)
    bound_extract_obs = lambda state, exo: extract_obs(state, exo, room_indices)

    @eqx.filter_jit
    def run_eval(state):
        def step_fn(curr_state, _):
            t = curr_state.time_idx[0]
            exo_batch = jax.tree.map(lambda x: x[t], env.shared_exo_data)
            
            # 1. Extract observation for the PPO agent
            obs = jax.vmap(bound_extract_obs, in_axes=(0, None))(curr_state.sim.state, exo_batch)
            
            # 2. Get deterministic action (using the mean, ignoring std dev for evaluation)
            action_mean, _, _ = jax.vmap(policy)(obs)
            
            # 3. Map network outputs to physical Watts
            actions = map_ppo_actions_for_eval(action_mean, NUM_ENVS, n_rooms)
            
            # 4. Step environment
            next_state, rewards, done, info = env.step(curr_state, actions)
            
            history = {
                "reward": rewards[0],
                "battery_power": actions.battery_power_w[0], 
                "hp_power": jnp.mean(actions.heat_pump_power_w[0]), # Tracking HP!
                "outdoor_temp": exo_batch.ambient_temp, 
                "indoor_temp": curr_state.sim.state.thermal.T_vector[0],
                "price": exo_batch.price,
                "soc": curr_state.sim.battery.soc[0]  
            }
            return next_state, history
            
        _, full_history = jax.lax.scan(step_fn, state, None, length=672)
        return full_history

    print("Running Simulation...")
    history = run_eval(init_state)
    
    avg_reward = jnp.mean(jnp.sum(history["reward"], axis=0))
    print(f"💰 PPO Average Return: {avg_reward:.2f}")

    # ==========================================
    # 📊 Graph Generation (3-Panel Layout)
    # ==========================================
    battery_powers = jnp.asarray(history["battery_power"])
    hp_powers = jnp.asarray(history["hp_power"])
    outdoor_temps = jnp.asarray(history["outdoor_temp"])
    indoor_temps = jnp.asarray(history["indoor_temp"])
    prices = jnp.asarray(history["price"])
    socs = jnp.asarray(history["soc"])  
    
    if battery_powers.ndim > 1: battery_powers = battery_powers[:, 0]
    if hp_powers.ndim > 1: hp_powers = hp_powers[:, 0]
    if outdoor_temps.ndim > 1: outdoor_temps = outdoor_temps[:, 0]
    if prices.ndim > 1: prices = prices[:, 0]
    if socs.ndim > 1: socs = socs[:, 0]
        
    if indoor_temps.ndim > 1:
        if indoor_temps.shape[1] == 2048: indoor_temps = indoor_temps[:, 0]
        if indoor_temps.ndim > 1: indoor_temps = jnp.mean(indoor_temps, axis=-1)
            
    time_steps = jnp.arange(len(battery_powers))

    fig, (ax_price, ax_batt) = plt.subplots(2, 1, figsize=(12, 12), dpi=150, sharex=True)
    '''
    # --- PANEL 1: Thermal Dynamics ---
    ax_temp.set_ylabel('Temp (°C)', fontweight='bold')
    ax_temp.axhspan(20, 22, color='mediumseagreen', alpha=0.2, label='Comfort (20-22°C)')
    ax_temp.plot(time_steps, outdoor_temps, color='gray', linestyle='--', label='Outdoor')
    ax_temp.plot(time_steps, indoor_temps, color='navy', linewidth=2, label='Indoor')
    ax_temp.legend(loc='upper right')
    ax_temp.grid(True, linestyle=':', alpha=0.6)
    ax_temp.set_title('1. House Thermal Dynamics', fontweight='bold')
    '''
    # --- PANEL 2: Financial Logic ---
    ax_price.set_ylabel('Price ($/kWh)', color='darkorange', fontweight='bold')
    ax_price.plot(time_steps, prices, color='darkorange', linewidth=2, label='Grid Price')
    ax_price.legend(loc='upper right')
    ax_price.grid(True, linestyle=':', alpha=0.6)
    ax_price.set_title('1. Grid Price Fluctuations', fontweight='bold')

    # --- PANEL 3: Physical Dispatch (PPO Output) ---
    ax_batt.set_xlabel('Time Steps (15-min intervals)', fontweight='bold')
    ax_batt.set_ylabel('Power (W)', color='black', fontweight='bold')
    
    ax_batt.plot(time_steps, battery_powers, color='firebrick', label='Battery Dispatch (W)')
    ax_batt.tick_params(axis='y', labelcolor='black')
    
    ax_soc = ax_batt.twinx()
    ax_soc.set_ylabel('State of Charge (SOC)', color='purple', fontweight='bold')
    ax_soc.plot(time_steps, socs, color='purple', linewidth=2.5, label='SOC Level')
    ax_soc.set_ylim(-0.05, 1.05)
    ax_soc.tick_params(axis='y', labelcolor='purple')
    
    l1, lab1 = ax_batt.get_legend_handles_labels()
    l2, lab2 = ax_soc.get_legend_handles_labels()
    ax_batt.legend(l1 + l2, lab1 + lab2, loc='upper right')
    ax_batt.grid(True, linestyle=':', alpha=0.6)
    ax_batt.set_title('2. PPO Agent Physical Response (Continuous Control)', fontweight='bold')

    plt.tight_layout()
    plt.savefig("ppo_evaluation_complete.png")
    plt.show()

if __name__ == "__main__":
    main()