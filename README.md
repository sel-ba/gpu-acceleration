# GPU Acceleration Analysis with PyTorch

Professional analysis of GPU acceleration for deep learning models using PyTorch Profiler.

## Projects

### 1. U-Net Image Segmentation
- **Architecture**: Encoder-decoder with skip connections
- **Parameters**: ~31M
- **Task**: 3-class semantic segmentation
- **Dataset**: Synthetic 128x128 images
- **Speedup**: Measured with PyTorch Profiler

### 2. LSTM Text Classification
- **Architecture**: Bidirectional LSTM (2 layers)
- **Parameters**: ~2.4M
- **Task**: Sentiment classification
- **Dataset**: Synthetic text sequences
- **Speedup**: Measured with PyTorch Profiler

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

## Analysis Features

### Performance Metrics
- Total training time
- Time per epoch
- GPU speedup calculation
- Memory usage analysis

### Profiling Analysis
- Top 10 most expensive operations
- Operation category breakdown
- CPU vs GPU comparison
- PyTorch profiler traces

### Visualizations
- Training/test loss evolution
- Accuracy/Dice score evolution
- Time per epoch comparison
- Operation distribution pie charts
- Speedup bar charts

## TensorBoard

View detailed profiling traces:

```bash
source venv/bin/activate
tensorboard --logdir=runs/
```

Open http://localhost:6006 in your browser.

## Project Structure

```
.
├── project1_unet/
│   ├── train.py              # Training script
│   ├── analyze_results.py    # Results analyzer
│   ├── analysis_notebook.ipynb
│   └── runs/                 # Training results
├── project2_lstm/
│   ├── train.py
│   ├── analyze_results.py
│   ├── analysis_notebook.ipynb
│   └── runs/
├── requirements.txt
├── setup_and_run_all.sh      # Automated setup and training
└── README.md
```

## GPU Requirements

- CUDA-compatible GPU (tested with RTX 3070 Ti)
- CUDA 11.8+ or 12.x
- 8GB+ VRAM recommended

## Notes

- The U-Net training is commented out in `setup_and_run_all.sh` by default (already trained)
- Both projects use synthetic data (no downloads required)
- Profiler traces are saved in JSON format for TensorBoard
- All code is production-ready without unnecessary comments

## Typical Speedups

- **U-Net**: 3-5x speedup (convolution-heavy)
- **LSTM**: 10-15x speedup (sequential operations)

Results vary based on GPU, batch size, and model configuration.
