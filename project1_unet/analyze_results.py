import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


class ResultsAnalyzer:
    def __init__(self):
        self.cpu_results = None
        self.gpu_results = None
        self.load_results()
    
    def load_results(self):
        cpu_path = Path('runs/unet_cpu/results.json')
        gpu_path = Path('runs/unet_cuda/results.json')
        
        if cpu_path.exists():
            with open(cpu_path, 'r') as f:
                self.cpu_results = json.load(f)
        else:
            print(f"CPU results not found at {cpu_path}")
        
        if gpu_path.exists():
            with open(gpu_path, 'r') as f:
                self.gpu_results = json.load(f)
        else:
            print(f"GPU results not found at {gpu_path}")
    
    def calculate_speedup(self):
        if not self.cpu_results or not self.gpu_results:
            print("Both CPU and GPU results required for speedup calculation")
            return None
        
        cpu_time = self.cpu_results['total_time']
        gpu_time = self.gpu_results['total_time']
        speedup = cpu_time / gpu_time
        
        print("=" * 70)
        print("RÉSULTATS D'ACCÉLÉRATION GPU - U-NET SEGMENTATION")
        print("=" * 70)
        print(f"\nTemps CPU:      {cpu_time:.2f}s")
        print(f"Temps GPU:      {gpu_time:.2f}s")
        print(f"Speedup:        {speedup:.2f}x")
        print(f"\nGain de temps:  {cpu_time - gpu_time:.2f}s ({(1 - gpu_time/cpu_time)*100:.1f}%)")
        print("\nTemps par epoch:")
        print(f"  CPU: {np.mean(self.cpu_results['epoch_times']):.2f}s ± {np.std(self.cpu_results['epoch_times']):.2f}s")
        print(f"  GPU: {np.mean(self.gpu_results['epoch_times']):.2f}s ± {np.std(self.gpu_results['epoch_times']):.2f}s")
        print("=" * 70)
        
        return speedup
    
    def plot_comparison(self):
        if not self.cpu_results or not self.gpu_results:
            print("Both CPU and GPU results required for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('U-Net Segmentation Analysis: CPU vs GPU', fontsize=16, fontweight='bold')
        
        epochs_cpu = range(len(self.cpu_results['train_losses']))
        epochs_gpu = range(len(self.gpu_results['train_losses']))
        
        axes[0, 0].plot(epochs_cpu, self.cpu_results['train_losses'], 'o-', label='CPU', linewidth=2)
        axes[0, 0].plot(epochs_gpu, self.gpu_results['train_losses'], 's-', label='GPU', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs_cpu, self.cpu_results['train_dice'], 'o-', label='CPU', linewidth=2)
        axes[0, 1].plot(epochs_gpu, self.gpu_results['train_dice'], 's-', label='GPU', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Coefficient')
        axes[0, 1].set_title('Training Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].bar(['CPU', 'GPU'], 
                       [self.cpu_results['total_time'], self.gpu_results['total_time']],
                       color=['#1f77b4', '#ff7f0e'])
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Total Training Time')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        for i, (device, time) in enumerate([('CPU', self.cpu_results['total_time']), 
                                            ('GPU', self.gpu_results['total_time'])]):
            axes[1, 0].text(i, time, f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        axes[1, 1].plot(epochs_cpu, self.cpu_results['epoch_times'], 'o-', label='CPU', linewidth=2)
        axes[1, 1].plot(epochs_gpu, self.gpu_results['epoch_times'], 's-', label='GPU', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Time per Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = Path('runs/unet_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_path}")
        plt.close()
    
    def generate_report(self):
        speedup = self.calculate_speedup()
        
        if speedup:
            self.plot_comparison()
            
            report_path = Path('runs/unet_report.txt')
            with open(report_path, 'w') as f:
                f.write("RAPPORT D'ANALYSE - U-NET SEGMENTATION\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("1. CONFIGURATION\n")
                f.write(f"   Modèle: U-Net\n")
                f.write(f"   Dataset: Synthetic segmentation data\n")
                f.write(f"   Batch size: {self.cpu_results['batch_size']}\n")
                f.write(f"   Epochs: {self.cpu_results['epochs']}\n\n")
                
                f.write("2. RÉSULTATS TEMPORELS\n")
                f.write(f"   Temps total CPU:  {self.cpu_results['total_time']:.2f}s\n")
                f.write(f"   Temps total GPU:  {self.gpu_results['total_time']:.2f}s\n")
                f.write(f"   Speedup GPU:      {speedup:.2f}x\n\n")
                
                f.write("3. PERFORMANCE MODÈLE\n")
                f.write(f"   Dice coefficient final CPU: {self.cpu_results['test_dice'][-1]:.3f}\n")
                f.write(f"   Dice coefficient final GPU: {self.gpu_results['test_dice'][-1]:.3f}\n\n")
                
                f.write("4. ANALYSE\n")
                if speedup > 1:
                    f.write(f"   L'utilisation du GPU offre une accélération de {speedup:.2f}x.\n")
                    f.write(f"   Le gain de temps est de {(1 - self.gpu_results['total_time']/self.cpu_results['total_time'])*100:.1f}%.\n")
                    f.write("   U-Net avec encoder-decoder bénéficie du parallélisme GPU\n")
                    f.write("   grâce aux convolutions et aux skip connections.\n")
                else:
                    f.write(f"   Le GPU est plus lent ({speedup:.2f}x) que le CPU.\n")
                    f.write("   Causes possibles: overhead de transfert, batch size trop petit.\n")
            
            print(f"Report saved to {report_path}")


def main():
    analyzer = ResultsAnalyzer()
    analyzer.generate_report()


if __name__ == '__main__':
    main()
