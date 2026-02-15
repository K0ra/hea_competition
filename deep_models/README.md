# Deep models

Experiments with more powerful models for binary classification (diabetes, then other conditions). Use this folder for PoCs locally before scaling on a stronger machine. Runs are **config-driven**: pass a disease name or a config path so the same script works for any condition.

## Data and target

- **Data:** The mixed dataset `data/rand_hrs_model_merged.parquet` (model table + family-aggregated features). Override with `DATA_PATH`.
- **Target:** Taken from the disease config: `target_<target_suffix>` (e.g. `target_DIABE` for diabetes). Same incident-by-t+2 definition as the main pipeline.
- **Split:** Temporal — train on waves 3–9, test on waves 10–12 (no future leakage).

## XGBoost (config-driven)

Run from the **repository root** (hea_competition) by passing a **disease name** or **config path**:

```bash
python deep_models/run_xgboost.py diabetes
python deep_models/run_xgboost.py diseases/diabetes/config.yaml
```

For another condition, add a folder under `diseases/<name>/` with `config.yaml` (including `target_suffix`), run the main pipeline once to get `model/<name>_*_importances.csv`, then:

```bash
python deep_models/run_xgboost.py <name>
```

Uses the **top 8 features** by default from that condition’s importances file. To use more features:

```bash
TOP_N_FEATURES=20 python deep_models/run_xgboost.py diabetes
```

To use **all features with importance greater than 5**:

```bash
MIN_IMPORTANCE=5 python deep_models/run_xgboost.py diabetes
```

When `MIN_IMPORTANCE` is set, `TOP_N_FEATURES` is ignored.

### How importances are chosen

1. **Config** — If the disease config has `importances_path: path/to/file.csv`, that file is used.
2. **Env** — Else if `IMPORTANCES_PATH` is set and the file exists, that file is used.
3. **Discovery** — Else the latest `model/<disease>_*_importances.csv` (by modification time) is used. Run the main pipeline (e.g. `python scripts/run_disease.py diabetes`) first to generate it.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | (none) | If set, used as config path or disease name (same as first argument). |
| `DATA_PATH` | `data/rand_hrs_model_merged.parquet` | Parquet file with features and `target_*` columns. |
| `IMPORTANCES_PATH` | (see above) | Override path to importances CSV. |
| `TOP_N_FEATURES` | `8` | Number of top features (used only when `MIN_IMPORTANCE` is not set). |
| `MIN_IMPORTANCE` | (not set) | If set, use all features with importance **greater than** this value. Overrides `TOP_N_FEATURES`. |

### Config

Each disease config (e.g. `diseases/diabetes/config.yaml`) must have at least:

- **target_suffix** — e.g. `DIABE`; the target column is `target_DIABE`.

Optional:

- **importances_path** — Path (relative to repo root) to an importances CSV for this condition.

### Output

The script prints ROC-AUC, PR-AUC, F2-score (with best threshold), classification report, and XGBoost feature importances. Runs are logged to **MLflow** (params, metrics, config, importances) so you can compare runs in the MLflow UI. A summary JSON and an importances CSV are written to `model/` for each run.

## Experiment Sweep (comprehensive)

Run a full sweep testing multiple feature sets, model types, hyperparameters, and SMOTE variants. Each experiment logs to MLflow and appends to `model/experiment_results_<disease>_<timestamp>.json`.

### Smoke tests (run first)

```bash
python deep_models/test_sweep_components.py
```

Verifies data loading, temporal features, model training (XGBoost, CatBoost, LightGBM, MLP), and SMOTE. All tests should pass before running the full sweep.

### Run sweep

From the **repository root**:

```bash
# Quick mode: ~6-12 experiments (3 feature sets, 3 models, no SMOTE)
EXPERIMENTS_SUBSET=quick python deep_models/run_experiment_sweep.py diabetes

# Medium mode (default): ~40-48 experiments (4 feature sets, 5 models, SMOTE on/off)
python deep_models/run_experiment_sweep.py diabetes

# Full mode: ~100-120 experiments (6 feature sets, 8 models, SMOTE on/off)
EXPERIMENTS_SUBSET=full python deep_models/run_experiment_sweep.py diabetes
```

### Feature sets tested

1. **top8** — Top 8 features by importance
2. **top50** — Top 50 features by importance
3. **min_imp_5** — All features with importance > 5 (~121 features for diabetes)
4. **all** — All 821 features
5. **temporal_only** — Only temporal features (deltas and pct_change)
6. **top50_plus_temporal** — Top 50 + temporal features

### Models tested

- **XGBoost**: baseline, aggressive (max_depth=10, lr=0.1), regularized (subsample, colsample)
- **CatBoost**: default (auto_class_weights), aggressive (depth=10, lr=0.1)
- **LightGBM**: default (from main pipeline)
- **Neural Network**: simple MLP (128-64), deeper MLP (256-128-64)

### Temporal features

The sweep script automatically builds temporal features from key health indicators (BMI, WEIGHT, SHLT, HIBPE, HEARTE, DIAB, CONDE) by computing:

- `{feature}_delta` — Change from previous wave (per person)
- `{feature}_pct_change` — Percent change from previous wave

These capture rate-of-change and trends over time.

### SMOTE

Each model is tested with and without SMOTE oversampling (minority class upsampled to 1:3 ratio on training set only).

### Output

**During sweep**: Progress printed for each experiment with ROC-AUC, PR-AUC, F2-score, and duration.

**After sweep**:
- Summary table (top 10 experiments by ROC-AUC)
- Full results in `model/experiment_results_<disease>_<timestamp>.json`
- Each experiment individually logged to MLflow

View all runs: `mlflow ui`

## Adding more models

Add new scripts under `deep_models/` (e.g. other classifiers, neural nets) that accept the same config path or disease name and read the same mixed parquet. Keep the same temporal split and target for comparability.
