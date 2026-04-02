# EnergySim JAX PPO

A custom, high-performance Proximal Policy Optimization (PPO) agent implemented in pure JAX and Equinox, designed to control building energy simulations (PhyLFlex project) at massive scale.

This repository features a fully compiled, vector-parallel architecture (Anakin style) that leverages `@eqx.filter_jit` and `jax.vmap` to run thousands of environment steps per second on CPU/GPU.

##  Project Architecture

To keep the reinforcement learning logic transparent and modular, the project is split into four core files:

* **`networks.py`**: Defines the Actor-Critic neural network architecture using Equinox.
* **`rollout.py`**: Handles JAX-compiled environment interactions (`jax.lax.scan`) and memory collection.
* **`loss.py`**: Contains the mathematics for Generalized Advantage Estimation (GAE) and the PPO Clipped Surrogate Objective.
* **`train.py`**: The main execution engine. Ties the components together, compiles the training step, and uses Optax for network updates.

## Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/energysim-jax-ppo.git
cd energysim-jax-ppo

```
### 2. Set Up the Virtual Environment

It is highly recommended to run this project inside an isolated Python virtual environment.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate
```
### 3. Install Dependencies

This project uses a requirements.txt file which installs core machine learning libraries (JAX, Equinox, Optax) and pulls the custom EnergySim simulator directly from its source GitHub repository.

```bash
# Install all required packages
pip install -r requirements.txt
```
### 4. Add the Data File

Because EnergySim does not package its example data sets during installation, you must manually place the sample_data.csv file into the root directory of this project before running.

Ensure your directory looks exactly like this:

```
networks.py
rollout.py
loss.py
train.py
requirements.txt
sample_data.csv (Required)
```
## Running the Training Engine

Once your environment is active and the data file is in place, you can start the training loop:
```bash
python train.py
```

### What to expect:

1. Initialization: The script will initialize the vectorized environments and neural networks.
2. Compilation: The very first epoch will take several seconds as the JAX compiler (@eqx.filter_jit) translates the Python training loop into optimized machine code.
3. Execution: Epoch 2 and onwards will run in milliseconds, printing the Frames Per Second (FPS) and Loss metrics to the terminal.
4. Saving: Upon completion, the trained policy weights will be saved locally as jax_ppo_model.eqx.

## Running Tests

To verify that the JAX compiler, the environment wrapper, and the PPO math are functioning correctly without running a full 200-epoch training session, you can run a quick diagnostic test by overriding the epochs:
```bash
# To test if the pipeline compiles and runs successfully:
sed -i 's/EPOCHS = 200/EPOCHS = 2/g' train.py && python train.py
```
_(Note: Remember to change EPOCHS back to 200 in train.py for actual training runs!)_