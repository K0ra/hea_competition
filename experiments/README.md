# Experiment runs (legacy)

**Run records have moved to [../model/](../model/).** Scripts write to `model/` (e.g. `model/baseline_run.json`). This folder is kept for reference.

Each run is documented as a **JSON file** with:

- **task**: target, metrics
- **data**: source, split (train/val waves), features, preprocessing
- **model**: type and hyperparameters
- **results**: validation metrics (F2, PR-AUC, ROC-AUC, best threshold)

## Files

- `baseline_run.json` – Setup and results for the first baseline (logistic regression, temporal split, minimal features).

## Interpreting results

See [../docs/metrics_guide.md](../docs/metrics_guide.md) for what F2, PR-AUC, and ROC-AUC mean and how to read the baseline numbers.

## Adding new runs

Re-run `step06_train_baseline.py` to overwrite `baseline_run.json`, or copy the script and write a new JSON (e.g. `run_002.json`) with a different `run_id` and model/split to compare.
