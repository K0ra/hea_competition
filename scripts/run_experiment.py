"""
Run one experiment: apply config (data size, features, model), run pipeline 3->5,
train 8 models, log setup and results to MLflow. Reproducible via config file.

Run from repo root:
  python scripts/run_experiment.py --config config/experiments/phase1_baseline.yaml

Prints time estimate; logs params, metrics, duration, config artifact.
Set MLFLOW_TRACKING_URI for remote tracking.
If an NVIDIA GPU is detected (nvidia-smi), LightGBM experiments use device='gpu'
(LightGBM must be built with GPU support; see LightGBM GPU docs).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Ensure scripts/ on path and repo root cwd before other imports
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
_repo_dir = _script_dir.parent
os.chdir(_repo_dir)

import pandas as pd
import mlflow
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from conditions_config import load_conditions

REPO_DIR = _repo_dir
DATA_DIR = REPO_DIR / "data"
MODEL_DIR = REPO_DIR / "model"
CONFIG_DIR = REPO_DIR / "config"
PIPELINE_PARAMS_YAML = CONFIG_DIR / "pipeline_params.yaml"
X_PATH = DATA_DIR / "X_minimal.parquet"
Y_CONDITIONS_PATH = DATA_DIR / "y_conditions.parquet"
TRAIN_WAVES = (3, 9)
VAL_WAVES = (10, 12)

# Rough time estimates (min) for full pipeline 3->6: 10k / 50k / 100k rows
TIME_ESTIMATE_MIN = {10_000: (1, 4), 50_000: (4, 10), 100_000: (8, 25)}

# Minimum free RAM (MB) to assume for OS/other when allocating workers
MIN_FREE_RAM_MB = 1500
# Per-worker multiplier: each worker gets a copy of X,y (loky); allow some overhead
RAM_PER_WORKER_MULTIPLIER = 1.8

_GPU_AVAILABLE: bool | None = None


def _gpu_available() -> bool:
    """True if an NVIDIA GPU is available (nvidia-smi succeeds). Cached."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
        )
        _GPU_AVAILABLE = r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def _get_safe_n_parallel(
    n_tasks: int,
    data_mb: float,
    explicit: int | None,
) -> tuple[int, str]:
    """
    Compute safe number of parallel condition-workers from machine and data.

    - Uses CPU count (leave 1 core for OS).
    - If psutil available, caps by available RAM so n_workers * data_size fits.
    - explicit: if set (e.g. from CLI), use it (capped by n_tasks); else auto.
    """
    n_cpus = os.cpu_count() or 8
    # Leave one core for OS and main process
    n_from_cpu = min(n_tasks, max(1, n_cpus - 1))
    n_parallel = n_from_cpu
    reason = f"{n_cpus} CPUs -> using {n_parallel} workers (1 core reserved)"

    if explicit is not None:
        n_parallel = max(1, min(explicit, n_tasks))
        reason = f"user request (capped to {n_parallel})"
        return n_parallel, reason

    try:
        import psutil
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        total_mb = mem.total / (1024 * 1024)
        # Each worker gets a copy of X,y (loky); need headroom
        need_per_worker_mb = data_mb * RAM_PER_WORKER_MULTIPLIER
        max_by_ram = int((available_mb - MIN_FREE_RAM_MB) / need_per_worker_mb) if need_per_worker_mb > 0 else n_tasks
        max_by_ram = max(1, min(max_by_ram, n_tasks))
        if max_by_ram < n_parallel:
            n_parallel = max_by_ram
            reason = f"{n_cpus} CPUs, {total_mb:.0f}MB RAM ({available_mb:.0f}MB free) -> {n_parallel} workers (RAM-safe)"
        else:
            reason = f"{n_cpus} CPUs, {total_mb:.0f}MB RAM -> {n_parallel} workers (1 core reserved)"
    except ImportError:
        pass  # no psutil: rely on CPU only

    n_parallel = max(1, min(n_parallel, n_tasks))
    return n_parallel, reason


def _estimate_min(sample_rows: int) -> tuple[int, int]:
    low, high = TIME_ESTIMATE_MIN.get(sample_rows, (2, 10))
    for k, v in sorted(TIME_ESTIMATE_MIN.items(), reverse=True):
        if sample_rows >= k:
            return v
    return (1, 5)


def _write_pipeline_params(sample_rows: int, feature_columns: list[str]) -> None:
    data = {"sample_rows": sample_rows, "feature_columns": feature_columns}
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_PARAMS_YAML, "w") as f:
        yaml.safe_dump(
            data, f, default_flow_style=False, sort_keys=False
        )


def _run_pipeline_steps_3_to_5() -> None:
    import step03_reshape_to_long as step03
    import step04_build_target as step04
    import step05_build_features as step05
    step03.main()
    step04.main()
    step05.main()


def _train_eval_one(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    model_type: str,
    model_params: dict,
) -> dict:
    # Models that cannot handle missing: use complete cases only (no imputation)
    if model_type in ("lr", "rf"):
        complete = X[feature_columns].notna().all(axis=1)
        mask = y.notna() & complete
        X = X.loc[mask]
        y = y.loc[mask]
    else:
        # e.g. lgbm: keep all rows with non-missing target; pass X with NaN
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

    y = y.astype(int)
    wave = X["wave"].values
    train_mask = (wave >= TRAIN_WAVES[0]) & (wave <= TRAIN_WAVES[1])
    val_mask = (wave >= VAL_WAVES[0]) & (wave <= VAL_WAVES[1])
    X_train = X.loc[train_mask][feature_columns]
    X_val = X.loc[val_mask][feature_columns]
    y_train = y[train_mask].values
    y_val = y[val_mask].values

    if model_type == "lr":
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_va_s = scaler.transform(X_val)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=model_params.get("max_iter", 1000),
            random_state=model_params.get("random_state", 42),
        )
        model.fit(X_tr_s, y_train)
        y_prob = model.predict_proba(X_va_s)[:, 1]
    elif model_type == "rf":
        # RF does not support NaN; we already restricted to complete cases above
        X_tr_s = X_train.values
        X_va_s = X_val.values
        model = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 50),
            max_depth=model_params.get("max_depth", 6),
            class_weight="balanced",
            random_state=model_params.get("random_state", 42),
            n_jobs=model_params.get("n_jobs", -1),
        )
        model.fit(X_tr_s, y_train)
        y_prob = model.predict_proba(X_va_s)[:, 1]
    elif model_type == "lgbm":
        import lightgbm as lgb
        # LightGBM handles missing natively; uses all rows (with non-missing target)
        X_tr_s = X_train
        X_va_s = X_val
        device = model_params.get("device")
        if device is None and _gpu_available():
            device = "gpu"
        elif device is None:
            device = "cpu"
        params = {
            "objective": "binary",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": model_params.get("num_leaves", 31),
            "learning_rate": model_params.get("learning_rate", 0.05),
            "n_estimators": model_params.get("n_estimators", 100),
            "random_state": model_params.get("random_state", 42),
            "class_weight": "balanced",
            "n_jobs": model_params.get("n_jobs", -1),
            "device": device,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr_s,
            y_train,
            eval_set=[(X_va_s, y_val)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        y_prob = model.predict_proba(X_va_s)[:, 1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    y_pred = (y_prob >= 0.5).astype(int)

    f2 = fbeta_score(y_val, y_pred, beta=2)
    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)
    prec, rec, thresh = precision_recall_curve(y_val, y_prob)
    f2_scores = [fbeta_score(y_val, (y_prob >= t).astype(int), beta=2) for t in thresh]
    best_idx = max(range(len(f2_scores)), key=lambda i: f2_scores[i])
    best_thresh = float(thresh[best_idx])

    return {
        "F2_score": round(f2, 4),
        "PR_AUC": round(pr_auc, 4),
        "ROC_AUC": round(roc_auc, 4),
        "best_F2_threshold": round(best_thresh, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment and log to MLflow")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_DIR / "config" / "experiments" / "phase1_baseline.yaml",
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip pipeline steps 3-5; use existing X/y (config must match)",
    )
    parser.add_argument(
        "--n-parallel-conditions",
        type=int,
        default=None,
        help="Train this many conditions in parallel (default: min(8, CPU count)). Use 1 for sequential.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        exp_config = yaml.safe_load(f)
    name = exp_config.get("name", config_path.stem)
    feature_set_id = exp_config.get("feature_set_id", name)
    sample_rows = int(exp_config.get("sample_rows", 10000))
    feature_columns = list(exp_config.get("feature_columns", []))
    model_cfg = exp_config.get("model", {})
    model_type = model_cfg.get("type", "lr")
    model_params = dict(model_cfg.get("params", {}))
    if model_type == "lgbm" and "device" not in model_params and _gpu_available():
        model_params["device"] = "gpu"
        print("  GPU detected -> LightGBM will use device='gpu'")

    if not feature_columns:
        print("ERROR: feature_columns required in config", file=sys.stderr)
        sys.exit(1)

    # Write pipeline params so steps 3/5/6 use this experiment's data and features
    _write_pipeline_params(sample_rows, feature_columns)

    low, high = _estimate_min(sample_rows)
    print(f"Experiment: {name}")
    print(
        f"  sample_rows={sample_rows}, model={model_type}, "
        f"features={len(feature_columns)}"
    )
    print(f"  Estimated time: {low}-{high} min (full pipeline 3->6)")
    if sample_rows > 50_000:
        print(
            "  WARNING: Large sample; ensure enough RAM. Consider remote machine."
        )
    print()

    mlflow.set_experiment("hea_8conditions")
    with mlflow.start_run(run_name=name):
        start_sec = time.time()

        mlflow.log_param("experiment_name", name)
        mlflow.log_param("feature_set_id", str(feature_set_id))
        mlflow.log_param("sample_rows", sample_rows)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_features", len(feature_columns))
        feature_list_str = ",".join(feature_columns)
        mlflow.log_param(
            "feature_list",
            feature_list_str if len(feature_list_str) <= 450 else feature_list_str[:447] + "...",
        )
        mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})
        mlflow.log_artifact(str(config_path), artifact_path="config")
        # Log feature list so each run is self-describing for comparison
        feature_list_path = MODEL_DIR / "feature_list.txt"
        feature_list_path.write_text("\n".join(feature_columns), encoding="utf-8")
        mlflow.log_artifact(str(feature_list_path), artifact_path="config")
        feature_list_path.unlink(missing_ok=True)

        if not args.skip_data:
            _run_pipeline_steps_3_to_5()
        else:
            if not X_PATH.is_file() or not Y_CONDITIONS_PATH.is_file():
                print(
                    "ERROR: --skip-data but X/y not found. Run without --skip-data first."
                )
                sys.exit(1)

        # Load X, y
        print("Loading X and y_conditions ...")
        X = pd.read_parquet(X_PATH)
        y_all = pd.read_parquet(Y_CONDITIONS_PATH)
        print(f"  X shape: {X.shape}, y_conditions shape: {y_all.shape}")
        fc = [c for c in feature_columns if c in X.columns]
        if len(fc) != len(feature_columns):
            print("ERROR: Some feature_columns missing in X", file=sys.stderr)
            sys.exit(1)

        conditions = load_conditions()
        # Tasks: (suffix, cname) for conditions that have data
        tasks = []
        for suffix, cname in conditions:
            col = f"target_{suffix}"
            if col not in y_all.columns:
                continue
            y = y_all[col]
            if y.notna().sum() == 0:
                continue
            tasks.append((suffix, cname))

        data_mb = (X.memory_usage(deep=True).sum() + y_all.memory_usage(deep=True).sum()) / (1024 * 1024)
        n_parallel, parallel_reason = _get_safe_n_parallel(
            len(tasks), data_mb, args.n_parallel_conditions
        )
        print(f"  Parallelism: {n_parallel} workers ({parallel_reason})")
        mlflow.log_param("n_parallel_conditions", n_parallel)
        # When running multiple conditions in parallel, use n_jobs=1 per model to avoid oversubscription
        params_for_training = dict(model_params)
        params_for_training["n_jobs"] = 1 if n_parallel > 1 else -1

        if n_parallel <= 1:
            # Sequential: each model uses all cores (n_jobs=-1 for RF/LGBM)
            results_per_condition = {}
            for suffix, cname in tasks:
                col = f"target_{suffix}"
                y = y_all[col]
                mask = y.notna()
                X_c = X.loc[mask].copy()
                y_c = y.loc[mask].astype(int)
                print(f"  Training {suffix} ({cname}) ...")
                res = _train_eval_one(X_c, y_c, fc, model_type, params_for_training)
                results_per_condition[suffix] = res
                for k, v in res.items():
                    mlflow.log_metric(f"cond/{suffix}/{k}", v)
        else:
            # Parallel: one process per condition, each with n_jobs=1 to avoid oversubscription
            print(f"  Training {len(tasks)} conditions in parallel (n_jobs={n_parallel}) ...")

            def run_one(suffix: str, cname: str) -> tuple[str, dict]:
                col = f"target_{suffix}"
                y = y_all[col]
                mask = y.notna()
                X_c = X.loc[mask].copy()
                y_c = y.loc[mask].astype(int)
                res = _train_eval_one(X_c, y_c, fc, model_type, params_for_training)
                return (suffix, res)

            pairs = Parallel(n_jobs=n_parallel, backend="loky")(
                delayed(run_one)(suffix, cname) for suffix, cname in tasks
            )
            results_per_condition = dict(pairs)
            for suffix, res in results_per_condition.items():
                for k, v in res.items():
                    mlflow.log_metric(f"cond/{suffix}/{k}", v)

        # Per-condition summary table (Artifacts) for easy viewing by condition
        if results_per_condition:
            summary_rows = [
                {"condition": suffix, **res}
                for suffix, res in results_per_condition.items()
            ]
            summary_df = pd.DataFrame(summary_rows)
            try:
                mlflow.log_table(summary_df, "conditions_summary.json")
            except AttributeError:
                summary_path = MODEL_DIR / "conditions_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                try:
                    mlflow.log_artifact(str(summary_path))
                finally:
                    summary_path.unlink(missing_ok=True)

        # Aggregate metrics (mean over conditions)
        if results_per_condition:
            for key in ("F2_score", "PR_AUC", "ROC_AUC"):
                vals = [r[key] for r in results_per_condition.values()]
                mlflow.log_metric(f"mean_{key}", sum(vals) / len(vals))

        duration_sec = time.time() - start_sec
        mlflow.log_metric("duration_sec", round(duration_sec, 1))
        print(f"\nRun finished in {duration_sec:.1f} s")

        # Data quality report and artifact
        try:
            import report_data_quality as dq
            dq.main()
            dq_path = MODEL_DIR / "data_quality_report.json"
            if dq_path.is_file():
                mlflow.log_artifact(str(dq_path))
        except Exception as e:
            print(f"  Data quality report skipped: {e}")

    print("  MLflow run logged. View with: mlflow ui")


if __name__ == "__main__":
    main()
