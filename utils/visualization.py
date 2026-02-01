#!/usr/bin/env python3
"""
Comprehensive Analysis Report Generator
Generates comparative visualizations and analysis across all GPU acceleration experiments.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

PROJECT_ROOT = Path(__file__).parent.parent


def load_results(project_name, device):
    results_path = PROJECT_ROOT / f"{project_name}" / "runs" / f"{project_name.split('_')[1]}_{device}" / "results.json"
    if not results_path.exists():
        return None
    with open(results_path, 'r') as f:
        return json.load(f)


def generate_comprehensive_analysis():
    print("Generating comprehensive analysis report...")
    
    # Load all results
    unet_cpu = load_results("project1_unet", "cpu")
    unet_gpu = load_results("project1_unet", "cuda")
    lstm_cpu = load_results("project2_lstm", "cpu")
    lstm_gpu = load_results("project2_lstm", "cuda")
    
    if not all([unet_cpu, unet_gpu, lstm_cpu, lstm_gpu]):
        print("⚠ Warning: Some results are missing. Skipping comprehensive report.")
        return
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Training time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    projects = ['U-Net', 'LSTM']
    cpu_times = [unet_cpu['total_time'], lstm_cpu['total_time']]
    gpu_times = [unet_gpu['total_time'], lstm_gpu['total_time']]
    
    x = np.arange(len(projects))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Total Training Time (s)', fontsize=11, fontweight='bold')
    ax1.set_title('Training Time Comparison: CPU vs GPU', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(projects)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s',
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Speedup comparison
    ax2 = fig.add_subplot(gs[0, 1])
    speedups = [
        unet_cpu['total_time'] / unet_gpu['total_time'],
        lstm_cpu['total_time'] / lstm_gpu['total_time']
    ]
    
    bars = ax2.bar(projects, speedups, color=['#2ecc71', '#f39c12'], alpha=0.8)
    ax2.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax2.set_title('GPU Speedup (CPU time / GPU time)', fontsize=12, fontweight='bold')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}×',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. U-Net metrics evolution
    ax3 = fig.add_subplot(gs[1, 0])
    epochs = list(range(len(unet_cpu['train_losses'])))
    
    ax3.plot(epochs, unet_cpu['train_losses'], 'o-', label='CPU Train', color='#3498db', linewidth=2)
    ax3.plot(epochs, unet_gpu['train_losses'], 's-', label='GPU Train', color='#e74c3c', linewidth=2)
    ax3.plot(epochs, unet_cpu['test_losses'], 'o--', label='CPU Test', color='#3498db', alpha=0.6, linewidth=1.5)
    ax3.plot(epochs, unet_gpu['test_losses'], 's--', label='GPU Test', color='#e74c3c', alpha=0.6, linewidth=1.5)
    
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax3.set_title('U-Net: Loss Evolution', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)
    
    # 4. LSTM metrics evolution
    ax4 = fig.add_subplot(gs[1, 1])
    epochs_lstm = list(range(len(lstm_cpu['train_losses'])))
    
    ax4.plot(epochs_lstm, lstm_cpu['train_losses'], 'o-', label='CPU Train', color='#3498db', linewidth=2)
    ax4.plot(epochs_lstm, lstm_gpu['train_losses'], 's-', label='GPU Train', color='#e74c3c', linewidth=2)
    ax4.plot(epochs_lstm, lstm_cpu['test_losses'], 'o--', label='CPU Test', color='#3498db', alpha=0.6, linewidth=1.5)
    ax4.plot(epochs_lstm, lstm_gpu['test_losses'], 's--', label='GPU Test', color='#e74c3c', alpha=0.6, linewidth=1.5)
    
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax4.set_title('LSTM: Loss Evolution', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)
    
    # 5. Time per epoch comparison
    ax5 = fig.add_subplot(gs[2, 0])
    unet_epoch_times_cpu = unet_cpu['epoch_times']
    unet_epoch_times_gpu = unet_gpu['epoch_times']
    lstm_epoch_times_cpu = lstm_cpu['epoch_times']
    lstm_epoch_times_gpu = lstm_gpu['epoch_times']
    
    ax5.plot(epochs, unet_epoch_times_cpu, 'o-', label='U-Net CPU', color='#3498db', linewidth=2)
    ax5.plot(epochs, unet_epoch_times_gpu, 's-', label='U-Net GPU', color='#e74c3c', linewidth=2)
    ax5.plot(epochs_lstm, lstm_epoch_times_cpu, '^-', label='LSTM CPU', color='#9b59b6', linewidth=2)
    ax5.plot(epochs_lstm, lstm_epoch_times_gpu, 'd-', label='LSTM GPU', color='#f39c12', linewidth=2)
    
    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax5.set_title('Time Per Epoch Comparison', fontsize=12, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(alpha=0.3)
    
    # 6. Performance metrics table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Prepare table data
    table_data = [
        ['Metric', 'U-Net CPU', 'U-Net GPU', 'LSTM CPU', 'LSTM GPU'],
        ['Total Time (s)', f"{unet_cpu['total_time']:.2f}", f"{unet_gpu['total_time']:.2f}",
         f"{lstm_cpu['total_time']:.2f}", f"{lstm_gpu['total_time']:.2f}"],
        ['Avg Time/Epoch (s)', f"{np.mean(unet_epoch_times_cpu):.2f}", f"{np.mean(unet_epoch_times_gpu):.2f}",
         f"{np.mean(lstm_epoch_times_cpu):.2f}", f"{np.mean(lstm_epoch_times_gpu):.2f}"],
        ['Final Train Loss', f"{unet_cpu['train_losses'][-1]:.4f}", f"{unet_gpu['train_losses'][-1]:.4f}",
         f"{lstm_cpu['train_losses'][-1]:.4f}", f"{lstm_gpu['train_losses'][-1]:.4f}"],
        ['Final Test Loss', f"{unet_cpu['test_losses'][-1]:.4f}", f"{unet_gpu['test_losses'][-1]:.4f}",
         f"{lstm_cpu['test_losses'][-1]:.4f}", f"{lstm_gpu['test_losses'][-1]:.4f}"],
        ['Speedup', '1.00×', f"{speedups[0]:.2f}×", '1.00×', f"{speedups[1]:.2f}×"]
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(5):
            if j == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#ffffff' if i % 2 == 0 else '#f8f9fa')
    
    ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('GPU Acceleration Analysis: Comprehensive Report', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = PROJECT_ROOT / "runs" / "comprehensive_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive analysis saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nU-Net:")
    print(f"  CPU Time: {unet_cpu['total_time']:.2f}s | GPU Time: {unet_gpu['total_time']:.2f}s")
    print(f"  Speedup: {speedups[0]:.2f}× ({(1 - unet_gpu['total_time']/unet_cpu['total_time'])*100:.1f}% reduction)")
    print(f"  Final Train Loss: {unet_cpu['train_losses'][-1]:.4f} (CPU) | {unet_gpu['train_losses'][-1]:.4f} (GPU)")
    
    print(f"\nLSTM:")
    print(f"  CPU Time: {lstm_cpu['total_time']:.2f}s | GPU Time: {lstm_gpu['total_time']:.2f}s")
    print(f"  Speedup: {speedups[1]:.2f}× ({(1 - lstm_gpu['total_time']/lstm_cpu['total_time'])*100:.1f}% reduction)")
    print(f"  Final Train Loss: {lstm_cpu['train_losses'][-1]:.4f} (CPU) | {lstm_gpu['train_losses'][-1]:.4f} (GPU)")
    
    print("\n" + "="*70)
    print(f"Overall average speedup: {np.mean(speedups):.2f}×")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_comprehensive_analysis()
