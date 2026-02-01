# GPU Acceleration Analysis with PyTorch

Comprehensive performance analysis demonstrating **20.65× GPU speedup** for U-Net and **13.14× GPU speedup** for LSTM using PyTorch Profiler on RTX 3070 Ti.

## Key Results

| Model | CPU Time | GPU Time | Speedup | Time Reduction |
|-------|----------|----------|---------|----------------|
| **U-Net** | 1081.55s | 52.38s | **20.65×** | 95.2% |
| **LSTM** | 270.92s | 20.62s | **13.14×** | 92.4% |

## Projects

### 1. U-Net Image Segmentation
- **Architecture**: Encoder-decoder with skip connections (4 down/up blocks)
- **Parameters**: 31M
- **Task**: 3-class semantic segmentation
- **Dataset**: Synthetic 128×128 RGB images (500 train / 100 test)
- **Performance**: Dice coefficient 1.000 (CPU) vs 0.993 (GPU)
- **Batch Size**: 16
- **Key Insight**: Convolution operations achieve >30× speedup due to 2D spatial parallelism

### 2. Bidirectional LSTM Text Classification
- **Architecture**: 2-layer bidirectional LSTM
- **Parameters**: 2.4M (256 hidden, 128 embedding)
- **Task**: 3-class sentiment classification
- **Dataset**: Synthetic character-level sequences (2000 train / 400 test)
- **Performance**: 100% accuracy on both CPU and GPU
- **Batch Size**: 64
- **Key Insight**: LSTM gates benefit from parallel element-wise operations despite sequential dependencies

## Requirements

```bash
torch>=2.0.0
torchvision
numpy
matplotlib
pandas
jupyter
tensorboard
```

## Quick Start

### 1. Setup Environment

```bash
./setup_and_run_all.sh
```

This script will:
- Create a Python virtual environment
- Install all dependencies
- Check GPU availability
- Run training experiments on CPU and GPU
- Generate analysis results
- Launch Jupyter Lab for interactive analysis

### 2. Manual Training

Train individual projects:

```bash
source venv/bin/activate

python project1_unet/train.py --device cpu --epochs 5 --batch-size 16
python project1_unet/train.py --device cuda --epochs 5 --batch-size 16

python project2_lstm/train.py --device cpu --epochs 5 --batch-size 64
python project2_lstm/train.py --device cuda --epochs 5 --batch-size 64
```

### 3. Analyze Results

```bash
python project1_unet/analyze_results.py
python project2_lstm/analyze_results.py
```

### 4. Interactive Analysis

Launch Jupyter notebooks:

```bash
source venv/bin/activate
jupyter lab
```

Open:
- `project1_unet/analysis_notebook.ipynb`
- `project2_lstm/analysis_notebook.ipynb`

## Results Structure

```
runs/
├── unet_cpu/
│   ├── results.json
│   └── *.pt.trace.json
├── unet_cuda/
│   ├── results.json
│   └── *.pt.trace.json
├── lstm_cpu/
│   ├── results.json
│   └── *.pt.trace.json
└── lstm_cuda/
    ├── results.json
    └── *.pt.trace.json
```

## Profiling Features

### PyTorch Profiler Integration
- **Operation-level analysis**: Track CPU/CUDA time for each operation
- **Memory profiling**: Monitor GPU memory allocation and usage
- **Stack traces**: Identify performance bottlenecks in code
- **TensorBoard visualization**: Interactive profiling trace viewer

### Analysis Notebooks
Interactive Jupyter notebooks with:
- **Speedup calculations** with detailed formulas
- **Loss/accuracy evolution** plots (CPU vs GPU)
- **Top 10 operations** by execution time
- **Category breakdowns**: Convolution, LSTM gates, matrix operations, etc.
- **Operation distribution** visualizations (pie charts, bar charts)

### Generated Reports
- Performance summary tables
- Speedup analysis with time reduction percentages
- Model convergence comparison (CPU vs GPU numerical consistency)
- Profiling insights and optimization recommendations

## Hardware Environment

- **GPU**: NVIDIA GeForce RTX 3070 Ti Laptop
- **CUDA**: 12.8
- **PyTorch**: 2.10.0+cu128
- **Python**: 3.11
- **RAM**: 31 GB
- **OS**: Linux (Arch)

## TensorBoard

View detailed profiling traces:

```bash
source venv/bin/activate
tensorboard --logdir=project1_unet/runs/
tensorboard --logdir=project2_lstm/runs/
```

Navigate to http://localhost:6006 → PROFILE tab to explore:
- Operation timeline
- Memory usage over time
- Kernel launch patterns
- CPU-GPU synchronization points

## Project Structure

```
.
├── project1_unet/
│   ├── train.py                    # U-Net training with profiler
│   ├── analyze_results.py          # Generate plots and reports
│   ├── analysis_notebook.ipynb     # Interactive profiling analysis
│   └── runs/
│       ├── unet_cpu/              # CPU training results
│       └── unet_cuda/             # GPU training results + traces
├── project2_lstm/
│   ├── train.py                    # LSTM training with profiler
│   ├── analyze_results.py          # Generate plots and reports
│   ├── analysis_notebook.ipynb     # Interactive profiling analysis
│   └── runs/
│       ├── lstm_cpu/              # CPU training results
│       └── lstm_cuda/             # GPU training results + traces
├── gpu_analysis_sections.tex       # LaTeX report sections
├── gpu_report_standalone.tex       # Complete LaTeX document
├── requirements.txt                # Python dependencies
├── setup_and_run_all.sh           # Automated setup and execution
└── README.md
```

## LaTeX Report

Comprehensive analysis report with:
- Architecture specifications and training configurations
- Performance results tables with speedup calculations
- Training dynamics visualizations (loss curves, accuracy plots)
- Operation-level profiling analysis with insights
- Technical implementation details (PyTorch Profiler code)
- Optimization recommendations for convolution-heavy and sequential models

**Compile the report:**
```bash
pdflatex gpu_report_standalone.tex
pdflatex gpu_report_standalone.tex  # Run twice for ToC
```

Or include sections in your own report:
```latex
\input{gpu_analysis_sections.tex}
```

## Key Insights

### Why U-Net Has Higher Speedup (20.65× vs 13.14×)
1. **Convolution parallelism**: 2D convolutions computed independently across thousands of CUDA cores
2. **Regular memory access**: Predictable patterns optimize GPU cache usage
3. **Larger model** (31M params): More computation per batch amortizes transfer costs
4. **Batch processing**: 16 images processed simultaneously with full spatial parallelism

### LSTM Speedup Limitations
1. **Sequential dependencies**: Each timestep depends on previous hidden state
2. **Cannot parallelize** across sequence length (only across batch dimension)
3. **Irregular memory access**: Complex patterns in gate computations
4. **Still significant**: 13× speedup from batch parallelism and gate-level parallelism

## Notes

- Both models achieve numerical consistency between CPU and GPU (<1% difference)
- First epoch often shows initialization overhead (especially for LSTM)
- Profiling data saved in `.pt.trace.json` files viewable in TensorBoard
- The .gitignore is configured to exclude training data and model checkpoints
- Both projects use synthetic data (no downloads required)
- Profiler traces are saved in JSON format for TensorBoard
- All code is production-ready without unnecessary comments

## Typical Speedups

- **U-Net**: 3-5x speedup (convolution-heavy)
- **LSTM**: 10-15x speedup (sequential operations)

Results vary based on GPU, batch size, and model configuration.
