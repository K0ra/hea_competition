# Diabetes (DIABE)

This folder contains **config and docs only** for the diabetes model. No disease-specific scripts; the shared runner is `scripts/run_disease.py`.

- **config.yaml** – Target suffix, optional `feature_columns`, model type (default **lgbm**), **sample_fraction** / **sample_rows**, and **mlflow_experiment**. If `feature_columns` is omitted, all non-target columns from the model table are used.
- **REDUNDANT_FEATURES.md** – Notes on redundant variables; useful when interpreting importances.

**Run** (from repo root, after steps 3–5):

```bash
python scripts/run_disease.py diabetes
# or
python scripts/run_disease.py diseases/diabetes/config.yaml
```

**Output:** `model/diabetes_*.json`, `model/diabetes_*_importances.csv`, and MLflow experiment `hea_diabetes`.
