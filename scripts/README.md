# Pipeline scripts

**Single entry point:** from the repository root, run:

```bash
python scripts/run_pipeline.py
```

This runs steps 3 → 4 → 5 → 6 in sequence. Use `RUN_FROM_STEP=4` (or 5, 6) to start from a later step.

## Order (what the pipeline does)

1. **step03_reshape_to_long.py** – Load RAND HRS sample, reshape wide to long; writes `data/rand_hrs_long_sample.parquet`.
2. **step04_build_target.py** – Build incident-by-t+2 targets for all 8 conditions (see `config/conditions.yaml`); writes `data/rand_hrs_long_with_target.parquet`, `data/rand_hrs_model_sample.parquet` (one table with columns `target_DIABE`, …, `target_ARTHRE`).
3. **step05_build_features.py** – Build minimal feature set; writes `data/X_minimal.parquet`, `data/y_conditions.parquet` (8 target columns), `data/y_minimal.parquet` (diabetes-only, same rows as X), `data/feature_manifest.csv`.
4. **step06_train_baseline.py** – Train one logistic regression per condition (temporal split); writes `model/baseline_run_<suffix>.json` for each condition and `model/baseline_run.json` (diabetes, backward compat).

The RAND HRS SAS file is expected under **`data/randhrs1992_2022v1_SAS/`** (see main README; `data/` is gitignored). You can run individual step scripts (e.g. `python scripts/step05_build_features.py`) to re-run only that step.
