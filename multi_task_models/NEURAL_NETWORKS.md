# Neural Network Multi-Task Models

This directory contains a comprehensive neural network framework for multi-task disease prediction, with extensive experimentation infrastructure.

## Overview

The neural network implementation supports:
- **5 Architecture Types**: MLP, ResNet, Attention, Wide & Deep, TabNet
- **Multiple Loss Functions**: Weighted BCE, Focal Loss, Uncertainty Weighting
- **Advanced Training**: Early stopping, learning rate scheduling, mixed precision
- **Comprehensive Experiments**: Automated hyperparameter sweeps with MLflow logging
- **Per-Task Evaluation**: Detailed metrics for each disease

## Architecture Types

### 1. Multi-Task MLP
Basic feed-forward network with shared layers and task-specific heads.
- **Variants**: Shallow/Deep (2-4 layers), Narrow/Wide (64-512 units)
- **Use case**: Baseline, fast training
- **Example**: `[256, 128]` hidden dims → 2-layer network

### 2. ResNet-Style
Residual connections for deeper networks without vanishing gradients.
- **Variants**: 2-4 residual blocks
- **Use case**: Very deep models, gradient stability
- **Architecture**: Input → ResBlock × N → Task Heads

### 3. Attention-Based
Task-specific feature weighting via attention mechanism.
- **Variants**: 256 or 512 hidden dim
- **Use case**: When different features are important per disease
- **Architecture**: Encoder → Per-Task Attention → Task Heads

### 4. Wide & Deep
Combines linear (wide) and deep paths.
- **Variants**: Medium/Large deep path
- **Use case**: Capturing both memorization and generalization
- **Architecture**: [Wide Path | Deep Path] → Concatenate → Task Heads

### 5. TabNet (Experimental)
Sequential attention for feature selection.
- **Use case**: Interpretable feature importance
- **Note**: Requires `pytorch-tabnet` package

## Files

### Core Components

- **`losses.py`**: Loss functions
  - `FocalLoss`: Handles class imbalance with focusing parameter
  - `MultiTaskLoss`: Manages per-task losses with masking and weighting

- **`neural_architectures.py`**: Model definitions
  - All 5 architecture implementations
  - Factory function `create_model()` for config-based instantiation

- **`neural_data.py`**: PyTorch datasets and dataloaders
  - `MultiTaskDataset`: Handles features + 8 targets with masking
  - `prepare_dataloaders()`: Creates train/val loaders with optional SMOTE

- **`neural_trainer.py`**: Training infrastructure
  - `MultiTaskTrainer`: Handles training loop, validation, early stopping
  - Mixed precision support (AMP)
  - Learning rate scheduling

### Experiment Infrastructure

- **`neural_experiments_config.py`**: Experiment grid definitions
  - Architecture variants (13 configs)
  - Regularization variants (5 configs)
  - Optimization variants (5 configs)
  - Imbalance handling variants (4 configs)
  - Grid builder with "quick", "comprehensive", "exhaustive" modes

- **`run_neural_sweep.py`**: Main experiment runner
  - Loads data and features
  - Runs all experiments in grid
  - Logs to MLflow
  - Saves results to JSON

- **`analyze_neural_results.py`**: Result analysis
  - Top models by metric
  - Architecture comparison
  - Per-task performance breakdown
  - Hyperparameter impact analysis

- **`test_neural_components.py`**: Smoke tests
  - Unit tests for all components
  - End-to-end pipeline test

### Utilities

- **`compare_results.py`**: Compare neural networks vs tree models
- **`NEURAL_NETWORKS.md`**: This documentation file

## Experiment Modes

### Quick Mode (~13 experiments, ~30 minutes)
Tests all architectures with baseline hyperparameters.

```bash
MODE=quick ./run_all_neural_experiments.sh
```

### Comprehensive Mode (~156 experiments, ~8 hours)
Tests all architectures with key hyperparameter variations.

```bash
MODE=comprehensive ./run_all_neural_experiments.sh
```

### Exhaustive Mode (~51 experiments, ~2 hours, **default**)
Staged approach:
1. All architectures with baseline
2. Top 5 architectures × all regularization
3. Top 3 architectures × all optimization
4. Top 2 architectures × all imbalance handling

```bash
./run_all_neural_experiments.sh
```

## Usage

### 1. Run Smoke Tests

```bash
cd /Users/cirogam/Desktop/hea_hackathon/hea_competition/multi_task_models
source ../.venv/bin/activate
python test_neural_components.py
```

### 2. Run Experiment Sweep

Using the convenience script (recommended):

```bash
cd /Users/cirogam/Desktop/hea_hackathon/hea_competition/multi_task_models
./run_all_neural_experiments.sh
```

Or manually with custom features:

```bash
cd /Users/cirogam/Desktop/hea_hackathon/hea_competition/multi_task_models
source ../.venv/bin/activate

# Use union features (default)
python run_neural_sweep.py

# Use custom feature list
FEATURE_LIST=../model/custom_features.txt python run_neural_sweep.py

# Quick mode
MODE=quick python run_neural_sweep.py
```

### 3. Analyze Results

```bash
source ../.venv/bin/activate
python analyze_neural_results.py
```

### 4. Compare with Tree Models

```bash
source ../.venv/bin/activate
python compare_results.py
```

## Output

All experiments save to `results/neural_sweep/`:
- Individual JSON files per experiment
- `summary.json`: All results aggregated
- `analysis.txt`: Statistical analysis

MLflow tracking:
- Experiment name: `hea_neural_multitask`
- View: `mlflow ui` (see main README)

## Key Configuration Parameters

### Architecture
- `type`: `"mlp"`, `"resnet"`, `"attention"`, `"wide_deep"`, `"tabnet"`
- `hidden_dims`: List of layer sizes (MLP, Wide&Deep)
- `n_blocks`: Number of residual blocks (ResNet)
- `hidden_dim`: Shared hidden dimension (ResNet, Attention)

### Regularization
- `dropout`: 0.1 - 0.4 (probability of dropping neurons)
- `weight_decay`: 1e-5 - 1e-3 (L2 regularization)
- `batch_norm`: Boolean (use batch normalization)

### Optimization
- `lr`: 1e-4 - 1e-3 (learning rate)
- `batch_size`: 256 - 1024
- `optimizer`: `"adam"`, `"adamw"`, `"sgd"`

### Imbalance Handling
- `loss`: `"weighted_bce"`, `"focal"`, `"focal_weighted"`
- `use_smote`: Boolean (apply SMOTE oversampling)

## Training Details

- **Train/Val Split**: Temporal (train: waves 3-9, val: waves 10-12)
- **Early Stopping**: Patience of 15 epochs
- **Learning Rate**: ReduceLROnPlateau scheduler (factor=0.5, patience=5)
- **Gradient Clipping**: Max norm 1.0
- **Mixed Precision**: Enabled on GPU (AMP)
- **Device**: Auto-detect (CUDA if available, else CPU)

## Metrics

Per-task metrics:
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **F2-score**: Weighted F-score (favors recall)

Aggregate metrics:
- **Mean**: Average across all 8 diseases
- **Std**: Standard deviation across diseases

## Feature Selection

The sweep script uses features in this priority order:

1. **`FEATURE_LIST` environment variable**: Explicit feature list file
2. **Union features**: `model/union_features.txt` (101 features from top single-task importances)
3. **Discovered importances**: Latest importance file from `model/`
4. **Fallback**: Top 100 features by column order

## Performance Expectations

Based on tree model baselines:

| Disease | LightGBM (101 feat) | Neural Target |
|---------|---------------------|---------------|
| Diabetes | 0.815 | > 0.82 |
| Heart Disease | 0.810 | > 0.81 |
| Stroke | 0.844 | > 0.85 |
| Hypertension | 0.775 | > 0.78 |
| Arthritis | 0.761 | > 0.76 |
| Cancer | 0.723 | > 0.73 |
| Lung Disease | 0.806 | > 0.81 |
| Psych Condition | 0.732 | > 0.74 |

**Mean ROC-AUC Target**: > 0.78

## Next Steps

After running experiments:

1. **Review MLflow**: Check training curves, compare runs
2. **Analyze Results**: Run `analyze_neural_results.py` for statistical breakdown
3. **Compare Models**: Run `compare_results.py` to see neural vs tree performance
4. **Iterate**: If results are suboptimal, consider:
   - More features (increase from 101)
   - Deeper architectures (add to config)
   - Longer training (increase `max_epochs`)
   - Different loss functions
   - Ensemble methods (combine top models)

## Troubleshooting

**Out of memory**:
- Reduce `batch_size` in optimization config
- Reduce model size (smaller `hidden_dims`)
- Disable mixed precision (`use_amp=False`)

**Poor convergence**:
- Increase `patience` in trainer
- Try different learning rates
- Check for NaN losses (reduce LR if found)

**SMOTE errors**:
- Disable SMOTE if dataset is too small
- Ensure at least 2 classes present per task

**TabNet issues**:
- Fallback to MLP is automatic if TabNet unavailable
- Install: `pip install pytorch-tabnet`

## References

- Focal Loss: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- Multi-Task Learning: [Ruder, 2017](https://arxiv.org/abs/1706.05098)
- TabNet: [Arik & Pfister, 2019](https://arxiv.org/abs/1908.07442)
