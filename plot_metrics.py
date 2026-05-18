import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(csv_path="jax_metrics.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'. Run train.py first.")
        return

    # Load the training data
    df = pd.read_csv(csv_path)

    # Create a figure with two side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Episodic Return (For Uli) ---
    ax1.plot(df["Epoch"], df["Mean_Episodic_Return"], color="#1f77b4", linewidth=2)
    ax1.set_title("Mean Episodic Return over Training")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Return (Accumulated Reward)")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # --- Plot 2: Step Reward ---
    ax2.plot(df["Epoch"], df["Mean_Step_Reward"], color="#ff7f0e", linewidth=2)
    ax2.set_title("Mean Step Reward over Training")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Reward per Step")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Polish and save
    plt.suptitle("PPO Agent Performance (Isolated Component)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save the figure so you can easily drop it into a presentation
    plt.savefig("ppo_training_results.png", dpi=300)
    print("Plot saved as 'ppo_training_results.png'")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    plot_training_results()