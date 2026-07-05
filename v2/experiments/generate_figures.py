import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

sns.set_theme(style="darkgrid")

def generate_curves(metrics_dir: Path, output_dir: Path):
    csv_file = metrics_dir / "training_metrics.csv"
    if not csv_file.exists():
        return
        
    df = pd.read_csv(csv_file)
    metrics = ["Reward", "Queue", "Delay", "Carbon", "Entropy", "Policy_Loss"]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6), dpi=300)
        sns.lineplot(data=df, x='Episode', y=metric)
        
        plt.title(f"SPGRL Training {metric} Curve")
        plt.xlabel("Episode")
        plt.ylabel(metric)
        plt.tight_layout()
        
        # Save formats
        for ext in ['png', 'pdf', 'svg']:
            plt.savefig(output_dir / f"fig_{metric.lower()}.{ext}", dpi=300, format=ext)
        plt.close()
        
def generate_ablation_plot(ablation_csv: Path, output_dir: Path):
    if not ablation_csv.exists():
        return
        
    df = pd.read_csv(ablation_csv)
    plt.figure(figsize=(12, 6), dpi=300)
    sns.barplot(data=df, x='Ablated_Stream', y='Final_Reward')
    plt.title("SPGRL Stream Ablation Study (Impact on Reward)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f"fig_ablation.{ext}", dpi=300, format=ext)
    plt.close()
    
def generate_benchmark_plot(benchmark_csv: Path, output_dir: Path):
    if not benchmark_csv.exists():
        return
        
    df = pd.read_csv(benchmark_csv)
    plt.figure(figsize=(12, 6), dpi=300)
    sns.boxplot(data=df, x='Agent', y='Reward')
    plt.title("SPGRL vs Baselines (Reward Distribution)")
    plt.tight_layout()
    
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(output_dir / f"fig_benchmark.{ext}", dpi=300, format=ext)
    plt.close()

def main(exp_dir: Path):
    output_dir = exp_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating publication figures in {output_dir}")
    
    # Generate Training Curves
    generate_curves(exp_dir, output_dir)
    
    # Generate Ablation
    ablation_csv = exp_dir / "ablation_results.csv"
    generate_ablation_plot(ablation_csv, output_dir)
    
    # Generate Benchmark
    benchmark_csv = exp_dir / "benchmark_summary.csv"
    generate_benchmark_plot(benchmark_csv, output_dir)
    
    print("Figure generation complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()
    
    main(Path(args.exp_dir))
