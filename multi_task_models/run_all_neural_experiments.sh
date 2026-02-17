#!/bin/bash
# Run comprehensive neural network experiment sweep
#
# Usage:
#   ./run_all_neural_experiments.sh              # Exhaustive mode (default, ~51 experiments)
#   MODE=quick ./run_all_neural_experiments.sh   # Quick mode (~13 experiments)
#   MODE=comprehensive ./run_all_neural_experiments.sh  # Comprehensive mode (~156 experiments)
#
# Optional environment variables:
#   MODE: "quick", "comprehensive", or "exhaustive" (default)
#   FEATURE_LIST: Path to custom feature list file

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Neural Network Multi-Task Experiment Sweep"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "$REPO_DIR/.venv" ]; then
    echo "ERROR: Virtual environment not found at $REPO_DIR/.venv"
    echo "Please create it first:"
    echo "  cd $REPO_DIR"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$REPO_DIR/.venv/bin/activate"

# Set default mode
MODE=${MODE:-exhaustive}
echo "Mode: $MODE"

# Check if feature list is specified
if [ -n "$FEATURE_LIST" ]; then
    echo "Feature list: $FEATURE_LIST"
    if [ ! -f "$FEATURE_LIST" ]; then
        echo "WARNING: Feature list file not found: $FEATURE_LIST"
        echo "Will use default feature selection"
        unset FEATURE_LIST
    fi
fi

echo ""
echo "=========================================="
echo "Step 1: Running Smoke Tests"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"
python test_neural_components.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Smoke tests failed!"
    echo "Please fix errors before running experiments."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Running Experiment Sweep"
echo "=========================================="
echo ""

# Get experiment count
if [ "$MODE" = "quick" ]; then
    EXP_COUNT=13
    EST_TIME="30 minutes"
elif [ "$MODE" = "comprehensive" ]; then
    EXP_COUNT=156
    EST_TIME="8 hours"
else
    EXP_COUNT=51
    EST_TIME="2 hours"
fi

echo "Total experiments: ~$EXP_COUNT"
echo "Estimated time: ~$EST_TIME"
echo ""
echo "Results will be saved to: $REPO_DIR/results/neural_sweep/"
echo "MLflow tracking: hea_neural_multitask experiment"
echo ""

# Ask for confirmation unless SKIP_CONFIRM is set
if [ -z "$SKIP_CONFIRM" ]; then
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Run sweep
echo ""
echo "Starting sweep..."
echo ""

START_TIME=$(date +%s)

export MODE
if [ -n "$FEATURE_LIST" ]; then
    export FEATURE_LIST
fi

python run_neural_sweep.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Experiment sweep failed!"
    exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "Step 3: Analyzing Results"
echo "=========================================="
echo ""

python analyze_neural_results.py

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Result analysis failed"
    echo "Results are still saved in results/neural_sweep/"
fi

echo ""
echo "=========================================="
echo "SWEEP COMPLETED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to: $REPO_DIR/results/neural_sweep/"
echo "  - Individual JSON files per experiment"
echo "  - summary.json: All results aggregated"
echo "  - analysis.txt: Statistical analysis"
echo ""
echo "To view in MLflow:"
echo "  cd $REPO_DIR"
echo "  mlflow ui"
echo "  Open http://localhost:5000"
echo "  Navigate to 'hea_neural_multitask' experiment"
echo ""
echo "To compare with tree models:"
echo "  cd $SCRIPT_DIR"
echo "  python compare_results.py"
echo ""
