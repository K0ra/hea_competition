# Experimental Summary

## Overview of All Experiments

| Experiment Category | # Experiments | Best ROC-AUC | Avg ROC-AUC | Range | Key Finding |
|---------------------|---------------|--------------|-------------|-------|-------------|
| **Colleague: Single-Task Diabetes** | 8 (v1-v8) | **0.787** | 0.744 | 0.716-0.787 | Temporal imputation breakthrough |
| **Your: Single-Task Diabetes** | 13 | 0.695 | 0.680 | 0.650-0.695 | 64-128 features optimal |
| **Your: Multi-Task (8 diseases)** | 3 | 0.603 | 0.603 | 0.603-0.603 | Trade-off: shared representations |
| **Your: Neural Networks** | 19 | 0.683 | 0.674 | 0.631-0.683 | MLPs best, attention underperforms |
| **Your: Neural + SMOTE** | 1 | 0.546 | 0.546 | 0.546-0.546 | ❌ Catastrophic failure |
| **TOTAL** | **44** | **0.787** | **0.690** | **0.546-0.787** | **Data quality > model complexity** |

## Detailed Breakdown

### Colleague's Diabetes Iterations (v1-v8)
- **8 experiments** exploring preprocessing and sampling strategies
- **Progression**: 0.716 → 0.750 → 0.787
- **Biggest gains**: 
  - +3.4 points: 13 features → 820 features (v1→v3)
  - +2.5 points: Temporal imputation (v7→v8)
- **Failed approach**: SMOTE (v2, rejected)
- **Winner**: v8 (temporal imputation + ADASYN)

### Your Single-Task Feature Selection
- **13 experiments** with varying feature counts (4 to 794 features)
- **Optimal range**: 64-128 features
- **Best**: 121 features, ROC-AUC 0.695
- **Finding**: Diminishing returns beyond 256 features

### Your Neural Network Sweep
- **19 experiments** across 4 architecture families
  - MLPs: 8 experiments (shallow, deep, very deep, wide variants)
  - Wide & Deep: 3 experiments
  - ResNet-style: 3 experiments
  - Attention: 2 experiments
- **Best architecture**: MLP Deep Wide (0.683)
- **Average**: 0.674 (excluding SMOTE)
- **Finding**: Simple architectures sufficient, attention fails on tabular data

### Multi-Task Learning
- **3 experiments**: XGBoost with 9 features, 3 different configurations
- **Result**: ROC-AUC 0.603
- **Gap**: -9.2 points vs single-task (0.695)
- **Finding**: Multi-task helps sparse tasks, hurts abundant tasks

### SMOTE Validation
- **1 experiment**: Best neural architecture + SMOTE
- **Result**: ROC-AUC 0.546 (random guessing!)
- **Drop**: -12.8 points vs without SMOTE
- **Critical**: Validated that SMOTE is broken on this dataset

## Key Statistics

**Total Experiments**: 44
**Total Training Time**: ~200+ hours across local and remote machines
**Best Single Result**: 0.787 (colleague v8)
**Best Neural Result**: 0.683 (your MLP Deep Wide)
**Worst Result**: 0.546 (SMOTE validation)
**Performance Range**: 24.1 percentage points (0.546 to 0.787)

## Most Impactful Changes

| Change | From | To | Δ ROC-AUC | Impact |
|--------|------|----|-----------| -------|
| Temporal imputation (v7→v8) | 0.762 | 0.787 | +2.5 | 🏆 Biggest single gain |
| ADASYN oversampling (v4→v7) | 0.745 | 0.762 | +1.7 | ✓ Oversampling helps |
| Full features (v1→v3) | 0.716 | 0.750 | +3.4 | ✓ More features help |
| SMOTE vs no SMOTE | 0.674 | 0.546 | -12.8 | ❌ Catastrophic failure |
| Single→Multi-task | 0.695 | 0.603 | -9.2 | ⚠️ Trade-off for diabetes |
| Neural vs Tree (best) | 0.683 | 0.787 | -10.4 | Trees > Neural for tabular |

## Time Investment vs Gain

| Phase | Experiments | Time | Best Gain | ROI |
|-------|-------------|------|-----------|-----|
| Colleague v1-v3 (exploration) | 3 | ~3 days | +3.4 points | High |
| Colleague v4-v6 (engineering) | 3 | ~2 days | -0.5 points | Low |
| Colleague v7-v8 (imputation) | 2 | ~2 days | +4.2 points | **Highest** |
| Your feature selection | 13 | ~1 week | Baseline 0.695 | Medium |
| Your neural networks | 19 | ~2 weeks | 0.683 (-1.2 vs baseline) | Low |
| Your SMOTE validation | 1 | ~1 day | Critical insight | **High** (prevented disaster) |

**Lesson**: 2 days on temporal imputation (v7-v8) yielded more than 2 weeks on neural architecture optimization.
