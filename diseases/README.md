# Diseases (one folder per disease, config only)

Each disease has its own folder (e.g. `diabetes/`) containing **config and optional docs only**—no scripts. The shared runner is **`scripts/run_disease.py`**. Pass either a disease name (it loads `diseases/<name>/config.yaml`) or a config path:

```bash
python scripts/run_disease.py <name>
python scripts/run_disease.py diseases/diabetes/config.yaml
```

**Add a disease:** Create `diseases/<name>/config.yaml` with at least `target_suffix`, `model`, and optionally `feature_columns`, `mlflow_experiment`, `sample_fraction`, `sample_rows`. Then run `python scripts/run_disease.py <name>` (or the config path).

**Current:** `diabetes/` — see [diabetes/README.md](diabetes/README.md) and [diabetes/config.yaml](diabetes/config.yaml).
