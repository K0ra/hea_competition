"""
XGBoost binary classification on the mixed dataset, driven by disease config.

Loads diseases/<name>/config.yaml (or the given config path) for target_suffix,
optionally importances_path. Uses data/rand_hrs_model_merged.parquet,
temporal split (train waves 3-9, test 10-12). Feature selection: TOP_N_FEATURES
or MIN_IMPORTANCE.

Run from repo root:
  python deep_models/run_xgboost.py diabetes
  python deep_models/run_xgboost.py diseases/diabetes/config.yaml
  MIN_IMPORTANCE=5 python deep_models/run_xgboost.py diabetes
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import fbeta_score
from xgboost import XGBClassifier

REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_DIR / "model"
DEFAULT_DATA_PATH = REPO_DIR / "data" / "rand_hrs_model_merged.parquet"
TRAIN_WAVES = (3, 9)
VAL_WAVES = (10, 12)
RANDOM_STATE = 42


def _resolve_importances_path(
    config: dict, disease: str, env_path: str | None
) -> Path | None:
    """Return path to importances CSV, or None if not found."""
    # 1) Config
    cfg_path = config.get("importances_path")
    if cfg_path:
        p = (REPO_DIR / cfg_path).resolve()
        if p.is_file():
            return p
        return None
    # 2) Env
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file():
            return p
    # 3) Latest model/{disease}_*_importances.csv
    model_dir = REPO_DIR / "model"
    if model_dir.is_dir():
        candidates = sorted(
            model_dir.glob(f"{disease}_*_importances.csv"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def run(config_path: Path, disease: str) -> None:
    """Run XGBoost for one disease from its config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    target_suffix = config.get("target_suffix")
    if not target_suffix:
        print(
            "ERROR: Config must have target_suffix (e.g. DIABE).",
            file=sys.stderr,
        )
        sys.exit(1)
    target_col = f"target_{target_suffix}"
    mlflow_experiment = config.get(
        "mlflow_experiment", f"hea_deep_{disease}"
    )

    data_path = Path(
        os.environ.get("DATA_PATH", str(DEFAULT_DATA_PATH))
    ).resolve()
    env_importances = os.environ.get("IMPORTANCES_PATH", "").strip() or None
    importances_path = _resolve_importances_path(
        config, disease, env_importances
    )
    top_n = int(os.environ.get("TOP_N_FEATURES", "8"))
    min_importance_str = os.environ.get("MIN_IMPORTANCE", "").strip()
    min_importance = float(min_importance_str) if min_importance_str else None

    if not data_path.is_file():
        print(f"ERROR: Data not found: {data_path}", file=sys.stderr)
        sys.exit(1)
    if importances_path is None:
        print(
            "ERROR: No importances file found. Set IMPORTANCES_PATH or add "
            "importances_path to config, or run the main pipeline first to "
            f"generate model/{disease}_*_importances.csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print(f"Run XGBoost — {disease} ({target_suffix})")
    print("=" * 60)
    print(f"  Config: {config_path}")
    print(f"  Data: {data_path}")
    print(f"  Importances: {importances_path}")

    start_sec = time.time()
    print("\n" + "=" * 60)
    print("Load data and feature list")
    print("=" * 60)
    df = pd.read_parquet(data_path)
    print(f"  Parquet shape: {df.shape}")

    if target_col not in df.columns:
        print(
            f"ERROR: Target column '{target_col}' not in data.",
            file=sys.stderr,
        )
        sys.exit(1)

    mask = df[target_col].notna()
    df = df.loc[mask].copy()
    df[target_col] = df[target_col].astype(int)
    print(f"  Rows with non-null target: {len(df)}")

    imp_df = pd.read_csv(importances_path)
    if "feature" not in imp_df.columns or "importance" not in imp_df.columns:
        print(
            "ERROR: Importances CSV must have columns 'feature' and "
            "'importance'.",
            file=sys.stderr,
        )
        sys.exit(1)
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    if min_importance is not None:
        top_features_raw = imp_df.loc[
            imp_df["importance"] > min_importance, "feature"
        ].tolist()
        print(
            f"  Features with importance > {min_importance}: "
            f"{len(top_features_raw)}"
        )
    else:
        top_features_raw = imp_df.head(top_n)["feature"].tolist()

    feature_columns = ["wave"]
    missing = []
    for f in top_features_raw:
        if f in df.columns:
            feature_columns.append(f)
        else:
            missing.append(f)
    if missing:
        print(f"  WARNING: Features not in data (skipped): {missing}")
    if len(feature_columns) <= 1:
        print(
            "ERROR: None of the selected features are in the data.",
            file=sys.stderr,
        )
        sys.exit(1)
    n_feat = len(feature_columns) - 1
    if min_importance is not None:
        print(
            f"  Using {n_feat} features (importance > {min_importance}) + wave"
        )
    else:
        print(f"  Using {n_feat} top features + wave")

    X = df[feature_columns].copy()
    y = df[target_col]

    wave = X["wave"].values
    train_mask = (wave >= TRAIN_WAVES[0]) & (wave <= TRAIN_WAVES[1])
    val_mask = (wave >= VAL_WAVES[0]) & (wave <= VAL_WAVES[1])
    if train_mask.sum() == 0:
        print(
            f"ERROR: No training samples in waves "
            f"{TRAIN_WAVES[0]}-{TRAIN_WAVES[1]}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if val_mask.sum() == 0:
        print(
            f"ERROR: No validation samples in waves "
            f"{VAL_WAVES[0]}-{VAL_WAVES[1]}.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_train = X.loc[train_mask][feature_columns]
    X_val = X.loc[val_mask][feature_columns]
    y_train = y.loc[train_mask].values
    y_val = y.loc[val_mask].values

    print(
        f"  Train: {X_train.shape[0]} rows "
        f"(waves {TRAIN_WAVES[0]}-{TRAIN_WAVES[1]})"
    )
    print(
        f"  Test:  {X_val.shape[0]} rows "
        f"(waves {VAL_WAVES[0]}-{VAL_WAVES[1]})"
    )
    print(f"  Train positive rate: {y_train.mean():.4f}")
    print(f"  Test positive rate:  {y_val.mean():.4f}")

    print("\n" + "=" * 60)
    print("Train XGBoost")
    print("=" * 60)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    print(f"  scale_pos_weight (neg/pos): {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=False,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    y_prob = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, y_prob)
    precision_arr, recall_arr, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall_arr, precision_arr)

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f2 = 0.0
    best_thresh = 0.5
    for t in thresholds:
        y_p = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_val, y_p, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = t
    y_pred = (y_prob >= best_thresh).astype(int)

    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'ROC-AUC':<20} {roc_auc:.4f}")
    print(f"{'PR-AUC':<20} {pr_auc:.4f}")
    print(f"{'F2-Score':<20} {best_f2:.4f}")
    print(f"{'Threshold (F2)':<20} {best_thresh:.2f}")

    name = disease.replace("_", " ").title()
    print(f"\nClassification Report (threshold={best_thresh:.2f}):")
    report = classification_report(
        y_val,
        y_pred,
        target_names=[f"No {name}", f"Develops {name}"],
    )
    print(report)

    print("\nXGBoost feature importances (gain):")
    for feat, imp in sorted(
        zip(feature_columns, model.feature_importances_),
        key=lambda x: -x[1],
    ):
        print(f"  {feat:<25} {imp:.4f}")

    duration_sec = time.time() - start_sec
    n_train = int(len(y_train))
    time_id = datetime.now().strftime("%Y%m%d_%H%M")
    run_slug = f"xgboost_{len(feature_columns)}feat_{n_train}train_{time_id}"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_record = {
        "disease": disease,
        "target_suffix": target_suffix,
        "target_col": target_col,
        "feature_columns": feature_columns,
        "n_train": n_train,
        "n_val": int(len(y_val)),
        "model_type": "xgboost",
        "params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "top_n": top_n,
            "min_importance": min_importance,
        },
        "metrics": {
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "F2_score": round(best_f2, 4),
            "best_F2_threshold": round(best_thresh, 3),
        },
        "duration_sec": round(duration_sec, 1),
    }
    out_path = MODEL_DIR / f"{disease}_{run_slug}.json"
    with open(out_path, "w") as f:
        json.dump(out_record, f, indent=2)
    print(f"\n  Saved: {out_path}")

    importances_df = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances_out = MODEL_DIR / f"{disease}_{run_slug}_importances.csv"
    importances_df.to_csv(importances_out, index=False)
    print(f"  Saved importances: {importances_out}")

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"{disease}_{run_slug}"):
        mlflow.log_param("disease", disease)
        mlflow.log_param("target_suffix", target_suffix)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("feature_list", ",".join(feature_columns))
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("model_n_estimators", 500)
        mlflow.log_param("model_max_depth", 6)
        mlflow.log_param("model_learning_rate", 0.05)
        if min_importance is not None:
            mlflow.log_param("min_importance", min_importance)
        else:
            mlflow.log_param("top_n", top_n)
        mlflow.log_metric("ROC_AUC", roc_auc)
        mlflow.log_metric("PR_AUC", pr_auc)
        mlflow.log_metric("F2_score", best_f2)
        mlflow.log_metric("best_F2_threshold", best_thresh)
        mlflow.log_metric("duration_sec", round(duration_sec, 1))
        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_artifact(
            str(importances_out), artifact_path="importances"
        )
    print("  MLflow run logged. View with: mlflow ui")

    print("\nDone.")


def main() -> None:
    arg = (
        os.environ.get("CONFIG")
        or (sys.argv[1] if len(sys.argv) > 1 else None)
    )
    if not arg:
        print(
            "Usage: python deep_models/run_xgboost.py "
            "<config path or disease>",
            file=sys.stderr,
        )
        print(
            "   e.g. python deep_models/run_xgboost.py diabetes",
            file=sys.stderr,
        )
        print(
            "   e.g. python deep_models/run_xgboost.py "
            "diseases/diabetes/config.yaml",
            file=sys.stderr,
        )
        sys.exit(1)
    arg = arg.strip()

    if "/" in arg or arg.endswith(".yaml") or arg.endswith(".yml"):
        config_path = (REPO_DIR / arg).resolve()
        disease = config_path.parent.name
    else:
        disease = arg.lower()
        config_path = REPO_DIR / "diseases" / disease / "config.yaml"

    if not config_path.is_file():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    run(config_path, disease)


if __name__ == "__main__":
    main()
