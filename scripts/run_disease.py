"""
Shared runner for one-disease models. Takes a config file path or a disease name.

Run from repo root after steps 3-5 built data/rand_hrs_model_sample.parquet:
  python scripts/run_disease.py diseases/diabetes/config.yaml
  python scripts/run_disease.py diabetes   # loads diseases/diabetes/config.yaml

Each disease folder under diseases/ contains only config (e.g. config.yaml), no scripts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
_repo_dir = _script_dir.parent
os.chdir(_repo_dir)

import pandas as pd
import mlflow

from shared.train_eval import MODELS_REQUIRE_COMPLETE_CASES, train_eval_one  # noqa: E402

REPO_DIR = _repo_dir
DATA_DIR = REPO_DIR / "data"
MODEL_DIR = REPO_DIR / "model"
MODEL_TABLE_PATH = DATA_DIR / "rand_hrs_model_sample.parquet"


def run(config_path: Path, disease: str) -> None:
    """Run training using the given config file; disease name used for output/MLflow."""
    if not config_path.is_file():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    target_suffix = config.get("target_suffix", "DIABE")
    target_col = f"target_{target_suffix}"
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "lr")
    if os.environ.get("MODEL_TYPE"):
        model_type = os.environ["MODEL_TYPE"]
    model_params = dict(model_cfg.get("params", {}))
    mlflow_experiment = config.get("mlflow_experiment", f"hea_{disease}")

    if not MODEL_TABLE_PATH.is_file():
        print(
            f"ERROR: Model table not found: {MODEL_TABLE_PATH}. "
            "Run pipeline steps 3-5 first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_parquet(MODEL_TABLE_PATH)
    if target_col not in df.columns:
        print(
            f"ERROR: Target column {target_col} not in model table.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Optional: subset rows by fraction of dataset or by absolute count
    n_before = len(df)
    sample_fraction = config.get("sample_fraction")
    sample_rows = config.get("sample_rows")
    if sample_fraction is not None and isinstance(sample_fraction, (int, float)):
        frac = float(sample_fraction)
        if 0 < frac <= 1:
            n_use = max(1, int(n_before * frac))
            seed = config.get("sample_seed", 42)
            df = df.sample(n=n_use, random_state=seed)
            print(
                f"  Using sample_fraction={frac} ({n_use} rows from {n_before})"
            )
    elif sample_rows is not None and isinstance(sample_rows, int) and sample_rows > 0:
        df = df.head(sample_rows)
        print(f"  Using sample_rows={sample_rows} (from {n_before} total rows)")

    # Features: all non-target columns (optionally override with config feature_columns)
    if config.get("feature_columns"):
        feature_columns = [c for c in config["feature_columns"] if c in df.columns]
        missing = [c for c in config["feature_columns"] if c not in df.columns]
        if missing:
            print(f"  WARNING: Config feature_columns not in data (skipped): {missing}")
    else:
        feature_columns = [
            c for c in df.columns
            if c != "id" and not c.startswith("target_")
        ]
    if "wave" not in feature_columns:
        feature_columns = ["wave"] + [c for c in feature_columns if c != "wave"]
    if not feature_columns:
        print("ERROR: No feature columns in model table.", file=sys.stderr)
        sys.exit(1)

    print(f"{disease} ({target_suffix}): {len(feature_columns)} features (from data)")
    print(f"  Model: {model_type}")

    X = df[feature_columns].copy()
    y = df[target_col]

    # Fail fast: ensure we have enough rows to train
    if model_type in MODELS_REQUIRE_COMPLETE_CASES:
        complete = X[feature_columns].notna().all(axis=1)
        mask = y.notna() & complete
        n_complete = int(mask.sum())
        if n_complete == 0:
            print(
                f"ERROR: No rows with non-missing target and all {len(feature_columns)} "
                "features non-missing. Use model type 'lgbm' or reduce features.",
                file=sys.stderr,
            )
            sys.exit(1)
        wave = X.loc[mask, "wave"].values
        n_train = int(((wave >= 3) & (wave <= 9)).sum())
        if n_train == 0:
            print(
                "ERROR: No complete-case rows in train waves 3-9. Reduce features or use lgbm.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  Complete cases: {n_complete}, in train waves 3-9: {n_train}")
    else:
        # LGBM etc.: only need non-missing target; features may have missing
        mask = y.notna()
        wave = X.loc[mask, "wave"].values
        n_train = int(((wave >= 3) & (wave <= 9)).sum())
        if n_train == 0:
            print(
                "ERROR: No rows with non-missing target in train waves 3-9.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"  Rows with non-missing target in train waves 3-9: {n_train} "
            "(features may have missing; model handles them)"
        )

    n_features = len(feature_columns)
    time_id = datetime.now().strftime("%Y%m%d_%H%M")
    run_name_slug = f"{model_type}_{n_features}feat_{n_train}train_{time_id}"

    start_sec = time.time()
    results, fitted_model = train_eval_one(
        X, y, feature_columns, model_type, model_params
    )
    duration_sec = time.time() - start_sec

    print(
        f"  F2: {results['F2_score']}, PR-AUC: {results['PR_AUC']}, "
        f"ROC-AUC: {results['ROC_AUC']}"
    )
    print(f"  Duration: {duration_sec:.1f}s")

    out_record = {
        "disease": disease,
        "target": target_col,
        "target_suffix": target_suffix,
        "feature_columns": feature_columns,
        "model_type": model_type,
        "model_params": model_params,
        "results": results,
        "duration_sec": round(duration_sec, 1),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / f"{disease}_{run_name_slug}.json"
    with open(out_path, "w") as f:
        json.dump(out_record, f, indent=2)
    print(f"  Saved: {out_path}")

    # Save feature importances when the model provides them (lgbm, rf)
    importances_path = None
    if fitted_model is not None and hasattr(
        fitted_model, "feature_importances_"
    ):
        imp = fitted_model.feature_importances_
        importances_df = pd.DataFrame(
            {"feature": feature_columns, "importance": imp}
        ).sort_values("importance", ascending=False)
        importances_path = MODEL_DIR / f"{disease}_{run_name_slug}_importances.csv"
        importances_df.to_csv(importances_path, index=False)
        print(f"  Saved importances: {importances_path}")
        n_nonzero = int((imp > 0).sum())
        print(f"  Features with non-zero importance: {n_nonzero} / {len(feature_columns)}")

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"{disease}_{run_name_slug}"):
        mlflow.log_param("disease", disease)
        mlflow.log_param("target_suffix", target_suffix)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("feature_list", ",".join(feature_columns))
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(
            {f"model_{k}": str(v) for k, v in model_params.items()}
        )
        for k, v in results.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("duration_sec", out_record["duration_sec"])
        mlflow.log_artifact(str(config_path), artifact_path="config")
        if importances_path is not None and importances_path.is_file():
            mlflow.log_artifact(str(importances_path), artifact_path="importances")

    print("  MLflow run logged. View with: mlflow ui")


def main() -> None:
    arg = os.environ.get("CONFIG") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not arg:
        print("Usage: python scripts/run_disease.py <config path or disease name>", file=sys.stderr)
        print("   e.g. python scripts/run_disease.py diseases/diabetes/config.yaml", file=sys.stderr)
        print("   e.g. python scripts/run_disease.py diabetes", file=sys.stderr)
        sys.exit(1)
    arg = arg.strip()

    # Config path: either explicit path (contains / or ends with .yaml/.yml) or disease name
    if "/" in arg or arg.endswith(".yaml") or arg.endswith(".yml"):
        config_path = (REPO_DIR / arg).resolve()
        # Disease name = parent dir of config (e.g. diseases/diabetes/config.yaml -> diabetes)
        disease = config_path.parent.name
    else:
        disease = arg.lower()
        config_path = REPO_DIR / "diseases" / disease / "config.yaml"

    run(config_path, disease)


if __name__ == "__main__":
    main()
