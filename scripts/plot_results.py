import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    csv_path = "benchmarks/results_breakdown.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Total Latency vs Sequence Length
    plt.figure(figsize=(10, 6))
    plt.plot(df['seq_len'], df['total_ms'], marker='o', linestyle='-', color='b')
    plt.title('Total Multi-GPU MoE Latency vs Sequence Length')
    plt.xlabel('Sequence Length (Tokens)')
    plt.ylabel('Latency (ms)')
    plt.grid(True)
    plt.savefig('benchmarks/latency_vs_seqlen.png')
    print("Saved benchmarks/latency_vs_seqlen.png")

    # 2. Stacked Bar Chart for Latency Breakdown
    stages = ['routing', 'dispatch', 'nccl_fw', 'expert_compute', 'nccl_bw', 'combine']
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99FF', '#C2C2F0']
    
    plt.figure(figsize=(12, 7))
    bottom = None
    for i, stage in enumerate(stages):
        if bottom is None:
            plt.bar(df['seq_len'].astype(str), df[stage], label=stage.capitalize(), color=colors[i])
            bottom = df[stage].values
        else:
            plt.bar(df['seq_len'].astype(str), df[stage], bottom=bottom, label=stage.capitalize(), color=colors[i])
            bottom += df[stage].values

    plt.title('Stage-wise Latency Breakdown of Distributed MoE')
    plt.xlabel('Sequence Length (Tokens)')
    plt.ylabel('Latency (ms)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('benchmarks/latency_breakdown.png')
    print("Saved benchmarks/latency_breakdown.png")

if __name__ == "__main__":
    plot_results()
