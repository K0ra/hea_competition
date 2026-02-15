# Multi-Task Models

Multi-task learning for all 8 health conditions simultaneously, addressing class imbalance by sharing representations across related medical conditions.

## Rationale

The individual disease models (e.g., diabetes) suffer from severe class imbalance (e.g., 14.6:1 negative to positive ratio). Multi-task learning can help by:

1. **More positive signals**: Training on all 8 diseases provides ~8x more "disease" events to learn from
2. **Shared risk factors**: Age, BMI, lifestyle factors are relevant across multiple conditions
3. **Natural comorbidities**: Diabetes → cardiovascular disease, high BP → stroke, etc.
4. **Transfer learning**: More prevalent conditions help models learn representations that benefit rarer conditions

## Approach

**Phase 1 (Current)**: Multi-output tree models (XGBoost, LightGBM)
- Train one classifier per disease, but in same script with shared features
- Compute per-task class weights based on prevalence
- Evaluate all tasks simultaneously
- Log to MLflow under single run

**Phase 2 (If Phase 1 shows promise)**: Neural network multi-task
- Shared dense layers learn common health patterns
- Task-specific heads for each disease
- Weighted BCE loss per task

## Data

- **Source**: `data/rand_hrs_model_merged.parquet` (452K rows, 830 columns)
- **Targets**: 8 binary targets (one per disease)
  - `target_DIABE` (Diabetes): 125K samples, 8.5K positive (14.6:1)
  - `target_HIBPE` (High blood pressure): 77K samples, 14K positive (5.4:1)
  - `target_CANCR` (Cancer): 132K samples, 7.9K positive (16.8:1)
  - `target_LUNGE` (Lung disease): 140K samples, 5.3K positive (26.4:1)
  - `target_HEARTE` (Heart disease): 123K samples, 11K positive (11.0:1)
  - `target_STROKE` (Stroke): 141K samples, 5.9K positive (24.2:1)
  - `target_PSYCHE` (Psychiatric): 131K samples, 6.6K positive (19.9:1)
  - `target_ARTHRE` (Arthritis): 73K samples, 14K positive (5.2:1)
- **Split**: Temporal — train on waves 3-9, test on waves 10-12

## Usage

### 0. Create union feature set (recommended for multi-disease learning)

Since we only have comprehensive importances for diabetes, create a union feature set:

```bash
python multi_task_models/create_union_features.py
```

This extracts the top 100 features from diabetes single-task importances and saves them to `model/union_features.txt`. These features represent the most important predictors that can be shared across all diseases.

### 1. Smoke tests (run first)

```bash
python multi_task_models/test_multi_task.py
```

Verifies data loading, feature extraction, train/val split, weight computation, evaluation, and mini XGBoost training.

### 2. Multi-output XGBoost

```bash
# Recommended: Use union feature set (100 features)
FEATURE_LIST=model/union_features.txt python multi_task_models/run_multi_xgboost.py

# Alternative: Use all features
python multi_task_models/run_multi_xgboost.py

# With importance-based selection (using diabetes importances)
MIN_IMPORTANCE=5 python multi_task_models/run_multi_xgboost.py
TOP_N_FEATURES=50 python multi_task_models/run_multi_xgboost.py

# Custom hyperparameters
XGBOOST_MAX_DEPTH=8 XGBOOST_LEARNING_RATE=0.1 XGBOOST_N_ESTIMATORS=1000 \
  FEATURE_LIST=model/union_features.txt python multi_task_models/run_multi_xgboost.py
```

**Output**:
- Prints per-task metrics (ROC-AUC, PR-AUC, F2) and aggregate mean
- Logs to MLflow experiment `hea_multitask`
- Saves results to `model/multi_xgb_results_<timestamp>.json`
- Saves per-task feature importances to `model/multi_xgb_<disease>_importances_<timestamp>.csv`

### 3. Multi-output LightGBM

```bash
# Recommended: Use union feature set (100 features)
FEATURE_LIST=model/union_features.txt python multi_task_models/run_multi_lgbm.py

# Alternative: With importance-based selection
MIN_IMPORTANCE=5 python multi_task_models/run_multi_lgbm.py

# Custom hyperparameters
LGBM_MAX_DEPTH=8 LGBM_LEARNING_RATE=0.1 LGBM_N_ESTIMATORS=200 \
  FEATURE_LIST=model/union_features.txt python multi_task_models/run_multi_lgbm.py
```

**Output**: Same structure as XGBoost (logs to `hea_multitask` experiment).

### 4. Compare with single-task models

```bash
python multi_task_models/compare_results.py
```

**Output**: Comparison table showing single-task vs multi-task performance for each disease, with improvement percentages.

**Note**: Requires single-task baselines to exist (run `deep_models/run_xgboost.py` for each disease first, or at least for diabetes).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `data/rand_hrs_model_merged.parquet` | Path to data file |
| `FEATURE_LIST` | (none) | Path to text file with feature names (one per line) |
| `IMPORTANCES_PATH` | (auto-discover) | Path to importances CSV for feature selection |
| `TOP_N_FEATURES` | (none) | Use top N features by importance |
| `MIN_IMPORTANCE` | (none) | Use features with importance > threshold |
| `XGBOOST_MAX_DEPTH` | `6` | XGBoost max tree depth |
| `XGBOOST_LEARNING_RATE` | `0.05` | XGBoost learning rate |
| `XGBOOST_N_ESTIMATORS` | `500` | XGBoost number of trees |
| `LGBM_MAX_DEPTH` | `6` | LightGBM max tree depth |
| `LGBM_LEARNING_RATE` | `0.05` | LightGBM learning rate |
| `LGBM_N_ESTIMATORS` | `100` | LightGBM number of trees |

## Files

- `utils.py` — Shared utilities (data loading, feature selection, evaluation, train/val split)
- `test_multi_task.py` — Smoke tests for all components
- `create_union_features.py` — Create union feature set from available importances
- `run_multi_xgboost.py` — Multi-output XGBoost trainer
- `run_multi_lgbm.py` — Multi-output LightGBM trainer
- `compare_results.py` — Compare multi-task vs single-task performance

## Expected Outcomes

**Success indicators**:
1. Multi-task model trains successfully on all 8 diseases
2. Mean ROC-AUC across all diseases ≥ single-task baseline
3. Rare diseases (STROKE, PSYCHE, CANCR) show improvement
4. Reduced variance in performance across diseases

**If successful**:
- Proceed to Phase 2 (neural networks with shared + task-specific layers)
- Consider ensemble: multi-task + single-task predictions
- Explore task-specific feature selection within multi-task framework

**If not successful**:
- Analyze which diseases interfere with each other
- Try grouping related diseases (cardiovascular vs metabolic vs other)
- Fall back to improved single-task models with SMOTE/focal loss from experiment sweep

## Architecture (Planned Neural Network)

```
Input features (n_features)
    ↓
Dense(512, ReLU) + Dropout(0.3)     ← Shared representation
    ↓
Dense(256, ReLU) + Dropout(0.3)     ← Shared representation
    ↓
Dense(128, ReLU) + Dropout(0.2)     ← Shared representation
    ↓
┌───────────┬───────────┬─────────┐
│ DIABE     │ HIBPE     │ ...     │  ← Task-specific heads (8 total)
│ Dense(32) │ Dense(32) │ Dense(32)│
│ Dense(1)  │ Dense(1)  │ Dense(1)│
│ Sigmoid   │ Sigmoid   │ Sigmoid │
└───────────┴───────────┴─────────┘
  8 binary outputs
```

**Loss**: Weighted BCE per task, averaged across tasks

## MLflow Organization

- **Experiment**: `hea_multitask`
- **Run naming**: `multi_{model}_{n_features}feat_{timestamp}`
- **Metrics logged**:
  - Per-task: `{DISEASE}_ROC_AUC`, `{DISEASE}_PR_AUC`, `{DISEASE}_F2`, `{DISEASE}_class_weight`, `{DISEASE}_train_time`
  - Aggregate: `mean_ROC_AUC`, `mean_PR_AUC`, `mean_F2`, `total_duration_sec`
- **Parameters logged**: `model_type`, `n_tasks`, `n_features`, hyperparameters, feature selection method

View results: `mlflow ui` (browse to `hea_multitask` experiment)

## Comparison with Single-Task

To establish a baseline for comparison, run single-task models first:

```bash
# Run single-task XGBoost for diabetes
MIN_IMPORTANCE=5 python deep_models/run_xgboost.py diabetes

# Then run multi-task
MIN_IMPORTANCE=5 python multi_task_models/run_multi_xgboost.py

# Compare
python multi_task_models/compare_results.py
```

The comparison script will show:
- Per-disease ROC-AUC and PR-AUC for both approaches
- Improvement percentages
- Count of diseases where multi-task outperforms single-task
- Mean improvement across all diseases
