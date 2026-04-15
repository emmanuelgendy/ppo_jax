import os
import sys
import time
import csv
import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

sys.path.append("/home/emmanuel-gendy/Documents/EnergySim")

from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.rl.vector_env import VectorizedEnergyEnv
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig, 
    ThermalStorageConfig, PVConfig
)
from examples.build_my_house import create_2_room_house
from energysim.rl.helpers import extract_obs

# --- HYPERPARAMETERS ---
NUM_ENVS = 2048
ROLLOUT_STEPS = 64
EPOCHS = 200
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

def map_actions(norm_actions, n_envs, n_rooms):
    return SystemActions(
        battery_power_w=norm_actions[:, 0] * 3000.0,
        heat_pump_power_w=(norm_actions[:, 1:1+n_rooms] + 1.0) * 1000.0,
        ac_power_w=jnp.zeros((n_envs, n_rooms)), 
        storage_discharge_w=jnp.zeros((n_envs, n_rooms))
    )

from energysim.core.shared.data_structs import SystemActions

# --- GYMNASIUM WRAPPER (Standard) ---
class CleanRLGymWrapper(gym.vector.VectorEnv):
    def __init__(self, jax_env, num_envs, obs_dim, action_dim, extract_obs_fn, map_actions_fn):
        super().__init__()
        self.jax_env = jax_env
        self.extract_obs_fn = extract_obs_fn
        self.map_actions_fn = map_actions_fn
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))
        self._key = jax.random.PRNGKey(42)

    def reset(self, seed=None, options=None):
        self._key, reset_key = jax.random.split(self._key)
        self._env_state = self.jax_env.reset(reset_key)
        t = self._env_state.time_idx[0]
        exo = jax.tree.map(lambda x: x[t], self.jax_env.shared_exo_data)
        obs = jax.vmap(self.extract_obs_fn, in_axes=(0, None))(self._env_state.sim.state, exo)
        return np.array(obs), {}

    def step(self, actions):
        phys_actions = self.map_actions_fn(jnp.array(actions), self.num_envs)
        self._env_state, reward, done, _ = self.jax_env.step(self._env_state, phys_actions)
        t = self._env_state.time_idx[0]
        exo = jax.tree.map(lambda x: x[t], self.jax_env.shared_exo_data)
        obs = jax.vmap(self.extract_obs_fn, in_axes=(0, None))(self._env_state.sim.state, exo)
        return np.array(obs), np.array(reward), np.array(done, dtype=bool), np.zeros_like(np.array(done, dtype=bool)), {}

# --- CLEAN RL AGENT ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.ReLU(),
            layer_init(nn.Linear(64, 64)), nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.ReLU(),
            layer_init(nn.Linear(64, 64)), nn.ReLU(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = torch.tanh(self.actor_mean(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# --- TRAINING LOOP ---
def run_cleanrl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- [CleanRL] 1. Initializing Engine on {device} ---")
    
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)
    dataset = SimulationDataset(file_path="/home/emmanuel-gendy/Documents/EnergySim/examples/sample_data.csv", dt_seconds=900)    
    sim = JAXSimulator(
        dt_seconds=900, t_config=t_config, r_config=RewardConfig(price_weight=1.0, comfort_weight=5.0),
        b_config=BatteryConfig(), hp_config=HeatPumpConfig(), ac_config=AirConditionerConfig(), 
        ts_config=ThermalStorageConfig(), pv_config=PVConfig()
    )
    jax_env = VectorizedEnergyEnv(sim, dataset, num_envs=NUM_ENVS)
    
    obs_dim = n_rooms + 5
    act_dim = 1 + n_rooms
    room_indices = jnp.array(t_config.room_air_indices)
    bound_extract_obs = lambda state, exo: extract_obs(state, exo, room_indices)
    
    env = CleanRLGymWrapper(jax_env, NUM_ENVS, obs_dim, act_dim, bound_extract_obs, lambda a, n: map_actions(a, n, n_rooms))
    
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Storage setup
    obs = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, obs_dim)).to(device)
    actions = torch.zeros((ROLLOUT_STEPS, NUM_ENVS, act_dim)).to(device)
    logprobs = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((ROLLOUT_STEPS, NUM_ENVS)).to(device)

    print("--- [CleanRL] 2. Starting Training ---")
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)

    with open("cleanrl_metrics.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Total_Steps", "Wall_Clock_Time", "FPS", "Mean_Reward"])
        
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            rollout_rewards = []

            for step in range(0, ROLLOUT_STEPS):
                global_step += NUM_ENVS
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()

                actions[step] = action
                next_obs, reward, done, _, _ = env.step(action.cpu().numpy())
                rollout_rewards.append(np.mean(reward))
                
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # GAE
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(ROLLOUT_STEPS)):
                    if t == ROLLOUT_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + values

            # Update
            b_obs = obs.reshape((-1, obs_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, act_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()

            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((newvalue.view(-1) - b_returns) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            # Metrics
            fps = int((NUM_ENVS * ROLLOUT_STEPS) / (time.time() - epoch_start))
            mean_ep_return = np.mean(rollout_rewards) * 96
            writer.writerow([epoch, global_step, time.time() - start_time, fps, float(mean_ep_return)])
            print(f"Epoch {epoch:03d}/{EPOCHS} | FPS: {fps:,} | Reward: {mean_ep_return:.4f}")

if __name__ == "__main__":
    run_cleanrl()