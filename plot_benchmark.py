import pandas as pd
import matplotlib.pyplot as plt

def generate_plots():
    print("--- Generating Benchmark Visualizations ---")
    
    try:
        jax_data = pd.read_csv("jax_metrics.csv")
        sb3_data = pd.read_csv("./sb3_logs/progress.csv")
        cleanrl_data = pd.read_csv("cleanrl_metrics.csv")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Ensure all training scripts finished.")
        return

    sb3_reward_col = 'custom/mean_episodic_return'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Sample Efficiency (Math Check) ---
    ax1.plot(jax_data["Total_Steps"], jax_data["Mean_Reward"], label="JAX PPO (Native)", color="#2ca02c", linewidth=2)
    ax1.plot(sb3_data["time/total_timesteps"], sb3_data[sb3_reward_col], label="Stable-Baselines3 (PyTorch)", color="#ff7f0e", linewidth=2)
    ax1.plot(cleanrl_data["Total_Steps"], cleanrl_data["Mean_Reward"], label="CleanRL (PyTorch)", color="#1f77b4", linewidth=2, linestyle="--")
    
    ax1.set_title("Sample Efficiency: Expected Return over Time")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Expected Return (Reward)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Plot 2: Computational Efficiency (Hardware Check) ---
    ax2.plot(jax_data["Wall_Clock_Time"], jax_data["Mean_Reward"], label="JAX PPO (Native)", color="#2ca02c", linewidth=2)
    if 'time/time_elapsed' in sb3_data.columns:
        ax2.plot(sb3_data["time/time_elapsed"], sb3_data[sb3_reward_col], label="Stable-Baselines3 (PyTorch)", color="#ff7f0e", linewidth=2)
    ax2.plot(cleanrl_data["Wall_Clock_Time"], cleanrl_data["Mean_Reward"], label="CleanRL (PyTorch)", color="#1f77b4", linewidth=2, linestyle="--")

    ax2.set_title("Computational Efficiency: Reward vs Wall-Clock Time")
    ax2.set_xlabel("Wall-Clock Time (Seconds)")
    ax2.set_ylabel("Expected Return (Reward)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=300)
    print("✨ Plot saved successfully to benchmark_results.png")

if __name__ == "__main__":
    generate_plots()