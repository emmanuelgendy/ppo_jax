import subprocess
import sys

def run_command(command, description):
    print(f"\n{'='*50}\n🚀 Starting: {description}\n{'='*50}")
    try:
        # We use subprocess.run to execute the scripts in isolated memory spaces
        result = subprocess.run([sys.executable, command], check=True)
        print(f"✅ Finished: {description}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}. Exit code: {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    print("Welcome to the JAX vs PyTorch PPO Benchmarking Suite.")
    
    run_command("train.py", "Native JAX PPO Engine")
    run_command("benchmark_sb3.py", "Stable-Baselines3 PyTorch Engine")
    run_command("plot_benchmark.py", "Generating Visualizations")
    
    print("\n🎉 Benchmark Suite Complete! Open 'benchmark_results.png' to view the results.")