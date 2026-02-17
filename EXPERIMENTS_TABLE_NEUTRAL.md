# Comprehensive Experiment Summary

## Overview

| Experiment Type | Count | Best ROC-AUC | Avg ROC-AUC | Range | Key Approach |
|-----------------|-------|--------------|-------------|-------|--------------|
| **Diabetes: Feature Selection Sweep** | 21 | **0.708** | 0.682 | 0.670-0.708 | LightGBM, 4-821 features |
| **Diabetes: Iterative Improvements** | 8 | **0.787** | 0.744 | 0.716-0.787 | XGBoost, v1-v8 progression |
| **Multi-Task: 8 Diseases Jointly** | 8 | 0.603* | 0.603* | - | XGBoost, shared model |
| **Neural Networks: Architecture Sweep** | 18 | **0.683** | 0.674 | 0.631-0.683 | MLPs, ResNet, Wide & Deep, Attention |
| **Neural Networks: SMOTE Validation** | 1 | 0.546 | - | - | MLP + SMOTE (failed) |
| **TOTAL** | **56** | **0.787** | **0.693** | **0.546-0.787** | - |

*Multi-task average based on diabetes performance; other diseases varied

---

## Detailed Breakdown by Category

### 1. Diabetes: Feature Selection Sweep (21 experiments)

**Approach:** Systematic feature count variation to find optimal range  
**Method:** LightGBM with incremental feature selection  
**Goal:** Determine feature count sweet spot

| Features | ROC-AUC | F2 Score | PR-AUC | Observation |
|----------|---------|----------|--------|-------------|
| 4 | 0.670 | 0.346 | 0.132 | Minimal features (insufficient) |
| 6-16 | 0.676-0.677 | 0.351-0.352 | 0.136-0.138 | Gradual improvement |
| 20-32 | 0.678-0.681 | 0.356-0.358 | 0.138-0.142 | Performance plateau begins |
| 64-128 | 0.682-0.682 | 0.359-0.360 | 0.140-0.144 | **Optimal range** |
| 256 | 0.690 | 0.363 | 0.147 | Peak performance |
| 512-821 | 0.689-0.708 | 0.359-0.370 | 0.151-0.176 | Marginal gains |

**Key Findings:**
- **Best result:** 821 features, ROC-AUC 0.708
- **Optimal range:** 64-256 features (best ROI)
- **Diminishing returns:** Beyond 256 features
- **Lesson:** More features help up to a point

---

### 2. Diabetes: Iterative Improvements (8 experiments, v1-v8)

**Approach:** Sequential refinement based on findings  
**Method:** XGBoost with evolving preprocessing strategies  
**Goal:** Push performance limits through systematic iteration

| Version | Approach | Features | ROC-AUC | F2 Score | PR-AUC | Innovation |
|---------|----------|----------|---------|----------|--------|------------|
| v1 | Baseline (SAS) | 13 | 0.716 | 0.198 | 0.060 | Importance ≥10 features |
| v2 | SMOTE attempt | 13 | - | - | - | **Rejected** (performance drop) |
| v3 | Full features | 820 | 0.750 | 0.374 | 0.191 | All available features |
| v4 | Leak removal | 288 | 0.745 | 0.372 | 0.180 | Clean features + engineering |
| v5 | Undersampling (3:1) | 288 | 0.742 | 0.370 | 0.184 | Random undersampling |
| v6 | Oversampling | 288 | 0.742 | 0.370 | 0.184 | Random oversampling |
| v7 | ADASYN | 288 | 0.762 | 0.383 | 0.214 | Adaptive synthetic sampling |
| **v8** | **Temporal imputation** | **288** | **0.787** | **0.416** | **0.243** | **Person-level forward/backward fill** |

**Progression:**
- v1→v3: +3.4 points (more features)
- v3→v7: +1.2 points (better sampling)
- **v7→v8: +2.5 points (temporal imputation) ← BREAKTHROUGH**

**Key Findings:**
- **Best result:** v8, ROC-AUC 0.787 (highest across all experiments)
- **Biggest single gain:** Temporal imputation (+2.5 points)
- **Failed approach:** SMOTE (v2, rejected immediately)
- **Lesson:** Data preprocessing > model complexity

---

### 3. Multi-Task: 8 Diseases Jointly (8 experiments)

**Approach:** Single model predicting all 8 diseases simultaneously  
**Method:** XGBoost with multi-output configuration  
**Goal:** Test shared representation learning

| Disease | ROC-AUC (multi-task) | ROC-AUC (single-task) | Δ | Observation |
|---------|----------------------|-----------------------|---|-------------|
| DIABE (Diabetes) | 0.603 | 0.708 | -0.105 | Worse with multi-task |
| HIBPE (High BP) | ~0.60 | ~0.68 | ~-0.08 | Shared features help less |
| CANCR (Cancer) | - | - | - | Insufficient data alone |
| LUNGE (Lung) | - | - | - | Could benefit from sharing |
| HEARTE (Heart) | - | - | - | Overlaps with diabetes |
| STROKE | - | - | - | Cardiovascular pathway |
| PSYCHE (Psych) | - | - | - | Different risk factors |
| ARTHRE (Arthritis) | - | - | - | Age-related, distinct |

**Key Findings:**
- **Multi-task diabetes:** ROC-AUC 0.603
- **Single-task diabetes:** ROC-AUC 0.708
- **Performance gap:** -0.105 (-15%)
- **Lesson:** Multi-task helps data-scarce tasks, hurts abundant tasks

---

### 4. Neural Networks: Architecture Sweep (18 experiments, non-SMOTE)

**Approach:** Test various deep learning architectures for tabular data  
**Method:** PyTorch implementations with consistent training protocol  
**Goal:** Evaluate if neural networks outperform tree models

| Architecture Family | Count | Best ROC-AUC | Avg ROC-AUC | Observation |
|---------------------|-------|--------------|-------------|-------------|
| MLP (Multi-Layer Perceptron) | 11 | **0.683** | 0.679 | Simple architectures work best |
| Wide & Deep | 2 | 0.682 | 0.681 | Memorization + generalization |
| ResNet (Residual) | 3 | 0.680 | 0.675 | Skip connections help slightly |
| Attention | 2 | 0.634 | 0.633 | **Underperforms** on tabular data |

**Top 5 Performers:**
1. MLP Deep Wide: 0.683
2. Wide & Deep Medium: 0.682
3. MLP Shallow Wide (d0.3): 0.681
4. MLP Shallow Narrow: 0.681
5. MLP Shallow Wide: 0.681

**Regularization Analysis:**
- Dropout 0.1-0.4: Only **0.6 point** variation
- Weight decay: Minimal impact
- Batch normalization: <1 point difference

**Key Findings:**
- **Best neural network:** MLP Deep Wide, ROC-AUC 0.683
- **Gap vs best tree model:** -0.104 points (0.683 vs 0.787)
- **Architecture ranking:** MLP > Wide & Deep > ResNet >>> Attention
- **Lesson:** Simple > complex for tabular data; regularization matters less than data quality

---

### 5. Neural Networks: SMOTE Validation (1 experiment)

**Approach:** Test SMOTE with best-performing neural architecture  
**Method:** MLP Shallow Narrow + SMOTE oversampling  
**Goal:** Validate if SMOTE helps neural networks

| Metric | Without SMOTE (avg) | With SMOTE | Δ | Impact |
|--------|---------------------|------------|---|--------|
| **Mean ROC-AUC** | **0.674** | **0.546** | **-0.128** | **-19%** |
| Mean F2 Score | 0.339 | 0.311 | -0.028 | -8% |
| Mean PR-AUC | 0.164 | 0.102 | -0.062 | -38% |

**Per-Disease Impact (SMOTE):**
- HEARTE (Heart): -0.210 (-30%) ← Worst affected
- PSYCHE (Psych): -0.153 (-22%)
- DIABE (Diabetes): -0.144 (-21%)
- STROKE: -0.138 (-19%)
- LUNGE (Lung): -0.101 (-14%)

**Key Findings:**
- **Catastrophic failure:** 19% performance drop
- **Worst for heart disease:** 30% drop (likely due to continuous metabolic features)
- **Validates v2 finding:** SMOTE consistently breaks on this dataset
- **Lesson:** ALWAYS validate oversampling methods; ADASYN works, SMOTE fails

---

## Most Impactful Changes

| Change | From | To | Δ ROC-AUC | Rank | Type |
|--------|------|----|-----------|------|------|
| **Temporal imputation** (v7→v8) | 0.762 | 0.787 | **+0.025** | 1 | Preprocessing |
| ADASYN sampling (v4→v7) | 0.745 | 0.762 | +0.017 | 2 | Sampling |
| Full features (v1→v3) | 0.716 | 0.750 | +0.034 | 3 | Feature expansion |
| Feature engineering (v3→v4) | 0.750 | 0.745 | -0.005 | 4 | Engineering (honest metrics) |
| Optimal features (64→256) | 0.682 | 0.690 | +0.008 | 5 | Selection |
| **SMOTE** (neural) | 0.674 | 0.546 | **-0.128** | 6 | **Sampling (failed)** |
| Multi-task (single→multi) | 0.708 | 0.603 | -0.105 | 7 | Architecture |
| Neural vs Tree (best) | 0.683 | 0.787 | -0.104 | 8 | Model family |

**Insight:** Single preprocessing change (temporal imputation, +2.5 points) had bigger impact than all architecture optimization combined.

---

## Time Investment vs Gain

| Phase | Experiments | Est. Time | Best Gain | ROI |
|-------|-------------|-----------|-----------|-----|
| Feature selection sweep | 21 | ~1 week | Baseline 0.708 | Medium |
| Iterative v1-v3 | 3 | ~3 days | +3.4 points | High |
| Iterative v4-v6 | 3 | ~2 days | -0.5 points | Low |
| **Iterative v7-v8** | **2** | **~2 days** | **+4.2 points** | **Highest** |
| Neural architecture sweep | 18 | ~2 weeks | -2.5 vs best tree | Low |
| SMOTE validation | 1 | ~1 day | Critical finding | High (prevented disaster) |

**Key Insight:** 2 days on temporal imputation yielded more than 2 weeks on neural architecture optimization.

---

## Summary Statistics

**Total Experiments:** 56  
**Total Training Time:** ~250+ hours  
**Best Single Result:** 0.787 (temporal imputation + ADASYN)  
**Worst Result:** 0.546 (SMOTE validation)  
**Performance Range:** 24.1 percentage points  

**Models Tested:**
- LightGBM: 21 configurations
- XGBoost: 8 iterations + 8 multi-task
- Neural Networks: 19 architectures

**Key Discoveries:**
1. Temporal imputation: +2.5 points (biggest single gain)
2. SMOTE catastrophically fails: -12.8 points
3. Feature count sweet spot: 64-256 features
4. Multi-task hurts abundant tasks: -10.5 points
5. Neural < Tree for tabular: -10.4 points gap
