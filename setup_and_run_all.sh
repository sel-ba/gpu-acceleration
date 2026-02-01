#!/bin/bash

set -e

PYTHON_VERSION="python3.11"
VENV_DIR="venv"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "  GPU Acceleration Analysis - Complete Setup and Execution"
echo "════════════════════════════════════════════════════════════════"
echo ""

check_python() {
    if command -v $PYTHON_VERSION &> /dev/null; then
        echo "✓ $PYTHON_VERSION found: $($PYTHON_VERSION --version)"
        return 0
    elif command -v python3.14 &> /dev/null; then
        PYTHON_VERSION="python3.14"
        echo "✓ Using python3.14 instead: $(python3.14 --version)"
        return 0
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION="python3"
        echo "⚠ Using default python3: $(python3 --version)"
        return 0
    else
        echo "✗ Python not found. Please install Python 3.11 or 3.14"
        exit 1
    fi
}

setup_venv() {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "1. Setting up virtual environment..."
    echo "────────────────────────────────────────────────────────────────"
    
    if [ -d "$VENV_DIR" ]; then
        echo "⚠ Virtual environment already exists. Removing..."
        rm -rf "$VENV_DIR"
    fi
    
    $PYTHON_VERSION -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
    
    source "$VENV_DIR/bin/activate"
    echo "✓ Virtual environment activated"
    
    pip install --upgrade pip setuptools wheel
    echo "✓ pip upgraded"
}

install_dependencies() {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "2. Installing dependencies..."
    echo "────────────────────────────────────────────────────────────────"
    
    pip install -r requirements.txt
    echo "✓ All dependencies installed"
    
    pip install jupyter ipykernel
    echo "✓ Jupyter installed"
    
    python -m ipykernel install --user --name=gpu-acceleration
    echo "✓ Jupyter kernel registered"
}

check_gpu() {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "3. Checking GPU availability..."
    echo "────────────────────────────────────────────────────────────────"
    
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
}

run_unet_experiments() {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "4. Running U-Net experiments..."
    echo "────────────────────────────────────────────────────────────────"
    
    cd "$PROJECT_DIR/project1_unet"
    
    echo "→ Training U-Net on CPU..."
    python train.py --device cpu --epochs 5 --batch-size 16
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "→ Training U-Net on GPU..."
        python train.py --device cuda --epochs 5 --batch-size 16
        
        echo "→ Analyzing U-Net results..."
        python analyze_results.py
        
        echo "→ Clearing GPU memory..."
        python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
        sleep 2
    else
        echo "⚠ GPU not available, skipping GPU training for U-Net"
    fi
    
    cd "$PROJECT_DIR"
}

run_lstm_experiments() {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "5. Running LSTM experiments..."
    echo "────────────────────────────────────────────────────────────────"
    
    cd "$PROJECT_DIR/project2_lstm"
    
    echo "→ Training LSTM on CPU..."
    python train.py --device cpu --epochs 5 --batch-size 64
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "→ Training LSTM on GPU..."
        python train.py --device cuda --epochs 5 --batch-size 64
        
        echo "→ Analyzing LSTM results..."
        python analyze_results.py
        
        echo "→ Clearing GPU memory..."
        python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    else
        echo "⚠ GPU not available, skipping GPU training for LSTM"
    fi
    
    cd "$PROJECT_DIR"
}

launch_notebooks() {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "6. Launching analysis notebooks..."
    echo "────────────────────────────────────────────────────────────────"
    
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "Opening Jupyter Lab with analysis notebooks..."
        echo ""
        echo "Notebooks available:"
        echo "  - project1_unet/analysis_notebook.ipynb"
        echo "  - project2_bert/analysis_notebook.ipynb"
        echo ""
        echo "Press Ctrl+C to stop Jupyter Lab"
        echo ""
        jupyter lab --no-browser
    else
        echo "⚠ GPU not available. Run notebooks manually after GPU training."
        echo "   jupyter lab"
    fi
}

print_summary() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  EXECUTION COMPLETE"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Results saved in:"
    echo "  📁 runs/unet_cpu/results.json"
    echo "  📁 runs/unet_cuda/results.json"
    echo "  📁 runs/bert_cpu/results.json"
    echo "  📁 runs/bert_cuda/results.json"
    echo ""
    echo "Analysis visualizations:"
    echo "  📊 project1_unet/runs/unet_analysis.png"
    echo "  📊 project2_lstm/runs/lstm_analysis.png"
    echo ""
    echo "Analysis notebooks:"
    echo "  📓 project1_unet/analysis_notebook.ipynb"
    echo "  📓 project2_bert/analysis_notebook.ipynb"
    echo ""
    echo "View TensorBoard traces:"
    echo "  tensorboard --logdir=runs"
    echo ""
    echo "Run Jupyter notebooks:"
    echo "  source venv/bin/activate"
    echo "  jupyter lab"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
}

main() {
    check_python
    setup_venv
    install_dependencies
    check_gpu
    
    read -p "Start training experiments? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_unet_experiments
        run_lstm_experiments
        
        echo ""
        read -p "Launch Jupyter Lab to view analysis notebooks? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            launch_notebooks
        fi
    fi
    
    print_summary
}

if [ "$1" == "--skip-training" ]; then
    check_python
    setup_venv
    install_dependencies
    launch_notebooks
else
    main
fi
