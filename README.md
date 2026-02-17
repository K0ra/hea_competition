# hea_competition
Hea Hackathon: "AI in Search of Hidden Health Signals"

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline (modeling data and baseline)

Run the full pipeline from the **repository root** (hea_competition) with a single command:

```bash
python scripts/run_pipeline.py
```

This runs, in order: reshape to long (3) → build 8 targets (4) → build X/y (5) → train 8 baseline models (6). To start from a later step (e.g. after changing features), set `RUN_FROM_STEP=5` (or 4, 6) in the environment.

See [scripts/README.md](scripts/README.md) for outputs and optional per-step scripts.

**Targets**: The pipeline builds **8 incident-by-t+2** targets (see [target_definition.md](target_definition.md) and **`config/conditions.yaml`**): Diabetes, High blood pressure, Cancer, Lung disease, Heart disease, Stroke, Psychiatric, Arthritis. Step 6 trains one baseline model per condition and writes `model/baseline_run_<suffix>.json`; `model/baseline_run.json` is the diabetes run for backward compatibility.

**Data**: Place the RAND HRS dataset under **`data/randhrs1992_2022v1_SAS/`** (the folder containing `randhrs1992_2022v1.sas7bdat`). The `data/` directory is in `.gitignore` so raw data and pipeline outputs are not committed. **Model run outputs** go to **`model/`**. To see how many respondents the SAS file has and what share of the data the pipeline uses, run **`python scripts/count_sas_rows.py`** (uses metadata only, no full load).

**Data quality report:** After running the pipeline (or after step 5), run `python scripts/report_data_quality.py` to generate `model/data_quality_report.json` (sample sizes, missingness per feature, target balance per condition, feature stats).

---

## One-disease workflow (config per disease, shared script)

**Layout:** `diseases/` has one folder per disease (e.g. `diseases/diabetes/`). Each folder contains **config only** (e.g. `config.yaml`) and optional docs—no disease-specific scripts. A **shared script** runs any disease from its config.

- **Run one disease** (after pipeline steps 3–5 have built `data/rand_hrs_model_sample.parquet`):

```bash
python scripts/run_disease.py diabetes
# or pass the config file directly:
python scripts/run_disease.py diseases/diabetes/config.yaml
```

**Optional — merged table with family features:** After step 4, run `python scripts/merge_fam_into_model.py` to produce `data/rand_hrs_model_merged.parquet` (model table + `fam_aggregated_filtered.pkl` joined on `id`). Then train on it with: `MODEL_TABLE=rand_hrs_model_merged.parquet python scripts/run_disease.py diabetes`.

The shared runner takes either a **disease name** (loads `diseases/<name>/config.yaml`) or a **config path**. It uses all non-target columns as features (or `feature_columns` in config to restrict), trains, writes `model/<disease>_*.json` and `model/<disease>_*_importances.csv`, and logs to MLflow. To add another disease, add `diseases/<name>/config.yaml` and run `python scripts/run_disease.py <name>` or `python scripts/run_disease.py diseases/<name>/config.yaml`.

---

## Iterating to improve results (experiments + MLflow)

Experiments vary **data size**, **feature set**, and **model type**. Each run is logged to **MLflow** (params, metrics, duration, config artifact) so you can reproduce and compare.

**1. Install and run one experiment (Phase 1 baseline)**  
From repo root:

```bash
pip install -r requirements.txt   # includes mlflow
python scripts/run_experiment.py --config config/experiments/phase1_baseline.yaml
```

This runs the full pipeline (steps 3→5) with the config’s `sample_rows` and `feature_columns`, trains 8 models (LR by default), and logs to MLflow. You’ll see an estimated time and a warning if `sample_rows` > 50k.

**2. Run other phases**  
- Phase 2 (more data, 50k rows):  
  `python scripts/run_experiment.py --config config/experiments/phase2_50k.yaml`  
- Phase 4 (RandomForest):  
  `python scripts/run_experiment.py --config config/experiments/phase4_rf.yaml`  
  Or reuse existing data (no re-run of steps 3–5):  
  `python scripts/run_experiment.py --config config/experiments/phase4_rf.yaml --skip-data`

**3. Sample batch (feature-set comparison)**  
To validate reportability and compare which feature set works best **per illness** and per model type, run a small batch on the same 50k data. Each run logs `feature_set_id` and per-condition metrics so you can see which run wins for each of the 8 conditions.

From repo root (venv activated):

```bash
# Step 1 – Build 50k data once (or reuse if you already ran phase2_50k)
python scripts/run_experiment.py --config config/experiments/phase2_50k.yaml

# Step 2 – Same data, different feature sets / models (--skip-data)
python scripts/run_experiment.py --config config/experiments/sample_feat_baseline_lr.yaml --skip-data
python scripts/run_experiment.py --config config/experiments/sample_feat_set_A_lr.yaml --skip-data
python scripts/run_experiment.py --config config/experiments/sample_feat_baseline_lgbm.yaml --skip-data
```

Then run `mlflow ui`, open experiment `hea_8conditions`, and add columns for `feature_set_id`, `model_type`, and metrics like `cond/DIABE/F2_score`, `cond/LUNGE/F2_score`, etc. Sort/filter to see which run (feature set) is best for each illness and each model type. Use that to assign tailored feature sets per condition before launching a larger batch.

**4. Compare runs in the MLflow UI**  
From repo root:

```bash
mlflow ui
```

Open the URL shown (e.g. http://127.0.0.1:5000). Select experiment `hea_8conditions` and compare runs (params, metrics, duration). To see score variations over time per illness: use **Compare** (select multiple runs) → **Charts**; add charts with X = start time (or run name) and Y = a metric (e.g. `cond/DIABE/F2_score`, `cond/LUNGE/PR_AUC`). One chart per condition shows trends across runs.

**5. Local vs remote**  
- **Local:** Runs are stored under `mlruns/` by default.  
- **Remote:** Set `MLFLOW_TRACKING_URI` to your tracking server before running the same command (e.g. `export MLFLOW_TRACKING_URI=http://your-server:5000`). No code change; run the same `run_experiment.py` on a remote machine and point it at the same URI to see all runs in one place.

**Note:** Each experiment overwrites `config/pipeline_params.yaml` with that experiment’s `sample_rows` and `feature_columns`. The exact config used is saved as an MLflow artifact for reproducibility.
