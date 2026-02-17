"""
Comprehensive experiment sweep: test models, features, hyperparameters, SMOTE.

Each experiment logs to MLflow and appends to
model/experiment_results_<disease>.json. Prints a summary table at the end.

Run from repo root:
  python deep_models/run_experiment_sweep.py diabetes
  EXPERIMENTS_SUBSET=quick python deep_models/run_experiment_sweep.py diabetes
  EXPERIMENTS_SUBSET=full python deep_models/run_experiment_sweep.py diabetes
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
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_DIR / "model"
DEFAULT_DATA_PATH = REPO_DIR / "data" / "rand_hrs_model_merged.parquet"
TRAIN_WAVES = (3, 9)
VAL_WAVES = (10, 12)
RANDOM_STATE = 42

# Features for temporal engineering (key health indicators)
TEMPORAL_FEATURES_BASE = [
    "BMI",
    "WEIGHT",
    "SHLT",
    "HIBPE",
    "HEARTE",
    "DIAB",
    "CONDE",
]


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal delta and pct_change columns for key features."""
    df = df.sort_values(["id", "wave"]).reset_index(drop=True)
    for col in TEMPORAL_FEATURES_BASE:
        if col not in df.columns:
            continue
        df[f"{col}_delta"] = df.groupby("id")[col].diff()
        df[f"{col}_pct_change"] = df.groupby("id")[col].pct_change()
    return df


def get_feature_set(
    set_name: str, df: pd.DataFrame, imp_df: pd.DataFrame
) -> list[str]:
    """Return feature columns for a given feature set name."""
    temporal_cols = [
        c for c in df.columns if c.endswith("_delta") or c.endswith("_pct_change")
    ]

    if set_name == "top8":
        features = imp_df.head(8)["feature"].tolist()
    elif set_name == "top50":
        features = imp_df.head(50)["feature"].tolist()
    elif set_name == "min_imp_5":
        features = imp_df[imp_df["importance"] > 5]["feature"].tolist()
    elif set_name == "all":
        features = [
            c for c in df.columns
            if c not in ["id", "wave"] and not c.startswith("target_")
        ]
        features = [
            c for c in features
            if not c.endswith("_delta") and not c.endswith("_pct_change")
        ]
    elif set_name == "temporal_only":
        features = temporal_cols
    elif set_name == "top50_plus_temporal":
        top50 = imp_df.head(50)["feature"].tolist()
        features = [f for f in top50 if f in df.columns] + temporal_cols
    else:
        raise ValueError(f"Unknown feature set: {set_name}")

    return ["wave"] + [
        f for f in features if f in df.columns and f != "wave"
    ]


def train_model(
    model_config: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    use_smote: bool,
) -> tuple[object, dict]:
    """Train one model; return (model, metrics_dict)."""
    model_type = model_config["type"]
    params = model_config["params"]

    if use_smote and len(np.unique(y_train)) > 1:
        smote = SMOTE(sampling_strategy=0.33, random_state=RANDOM_STATE)
        try:
            X_train_arr = X_train.values
            X_train_res, y_train_res = smote.fit_resample(X_train_arr, y_train)
            X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
            y_train = y_train_res
        except Exception:
            pass  # SMOTE can fail on very small data; continue without it

    if model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            **params, random_state=RANDOM_STATE, verbose=False
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    elif model_type == "lgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            **params, random_state=RANDOM_STATE, verbosity=-1
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
    elif model_type == "mlp":
        model = MLPClassifier(**params, random_state=RANDOM_STATE)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train.values)
        X_va_s = scaler.transform(X_val.values)
        model.fit(X_tr_s, y_train)
        y_prob = model.predict_proba(X_va_s)[:, 1]
        # Early return for MLP since we scaled
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
        return model, {
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "F2_score": best_f2,
            "best_F2_threshold": best_thresh,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
    y_prob = model.predict_proba(X_val_arr)[:, 1]
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

    metrics = {
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "F2_score": round(best_f2, 4),
        "best_F2_threshold": round(best_thresh, 3),
    }
    return model, metrics


def run_one_experiment(
    exp_config: dict,
    disease: str,
    target_col: str,
    df: pd.DataFrame,
    imp_df: pd.DataFrame,
    mlflow_experiment: str,
) -> dict:
    """Run one experiment and return results dict."""
    exp_name = exp_config["name"]
    feature_set = exp_config["feature_set"]
    model_config = exp_config["model"]
    use_smote = exp_config.get("smote", False)

    feature_columns = get_feature_set(feature_set, df, imp_df)
    if len(feature_columns) <= 1:
        return {
            "name": exp_name,
            "error": "No features available",
            "ROC_AUC": 0,
            "PR_AUC": 0,
            "F2_score": 0,
        }

    X = df[feature_columns].copy()
    y = df[target_col].values

    wave = X["wave"].values
    train_mask = (wave >= TRAIN_WAVES[0]) & (wave <= TRAIN_WAVES[1])
    val_mask = (wave >= VAL_WAVES[0]) & (wave <= VAL_WAVES[1])

    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    if len(y_train) == 0 or len(y_val) == 0:
        return {
            "name": exp_name,
            "error": "No train or val samples",
            "ROC_AUC": 0,
            "PR_AUC": 0,
            "F2_score": 0,
        }

    start_sec = time.time()
    try:
        model, metrics = train_model(
            model_config, X_train, y_train, X_val, y_val, use_smote
        )
        duration_sec = time.time() - start_sec
    except Exception as e:
        return {
            "name": exp_name,
            "error": str(e),
            "ROC_AUC": 0,
            "PR_AUC": 0,
            "F2_score": 0,
        }

    result = {
        "name": exp_name,
        "disease": disease,
        "target_col": target_col,
        "feature_set": feature_set,
        "n_features": len(feature_columns),
        "model_type": model_config["type"],
        "model_params": model_config["params"],
        "smote": use_smote,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "duration_sec": round(duration_sec, 1),
        **metrics,
    }

    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_slug = (
        f"{model_config['type']}_{len(feature_columns)}feat_"
        f"{len(y_train)}train_{time_id}"
    )

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"{disease}_{run_name_slug}"):
        mlflow.log_param("disease", disease)
        mlflow.log_param("experiment_name", exp_name)
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("model_type", model_config["type"])
        mlflow.log_param("smote", use_smote)
        for k, v in model_config["params"].items():
            mlflow.log_param(f"model_{k}", v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("duration_sec", round(duration_sec, 1))

    return result


def main() -> None:
    arg = (
        os.environ.get("CONFIG")
        or (sys.argv[1] if len(sys.argv) > 1 else None)
    )
    if not arg:
        print(
            "Usage: python deep_models/run_experiment_sweep.py "
            "<config path or disease>",
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

    with open(config_path) as f:
        config = yaml.safe_load(f)
    target_suffix = config.get("target_suffix")
    if not target_suffix:
        print("ERROR: Config must have target_suffix.", file=sys.stderr)
        sys.exit(1)
    target_col = f"target_{target_suffix}"
    mlflow_experiment = config.get("mlflow_experiment", f"hea_deep_{disease}")

    data_path = Path(
        os.environ.get("DATA_PATH", str(DEFAULT_DATA_PATH))
    ).resolve()
    if not data_path.is_file():
        print(f"ERROR: Data not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Load importances (latest from model/)
    model_dir = REPO_DIR / "model"
    candidates = sorted(
        model_dir.glob(f"{disease}_*_importances.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(
            f"ERROR: No importances file found in model/. "
            f"Run main pipeline or deep_models/run_xgboost.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    importances_path = candidates[0]

    print("=" * 60)
    print(f"Experiment Sweep — {disease} ({target_suffix})")
    print("=" * 60)
    print(f"  Config: {config_path}")
    print(f"  Data: {data_path}")
    print(f"  Importances: {importances_path}")

    df = pd.read_parquet(data_path)
    mask = df[target_col].notna()
    df = df.loc[mask].copy()
    df[target_col] = df[target_col].astype(int)
    print(f"  Rows with non-null target: {len(df)}")

    print("\nBuilding temporal features...")
    df = build_temporal_features(df)
    temporal_cols = [
        c for c in df.columns
        if c.endswith("_delta") or c.endswith("_pct_change")
    ]
    print(f"  Added {len(temporal_cols)} temporal columns")

    imp_df = pd.read_csv(importances_path)
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(
        drop=True
    )

    # Define experiment grid
    subset = os.environ.get("EXPERIMENTS_SUBSET", "medium").lower()

    if subset == "quick":
        feature_sets = ["top8", "top50", "top50_plus_temporal"]
        model_configs = [
            {
                "name": "xgb_baseline",
                "type": "xgboost",
                "params": {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "scale_pos_weight": 14.6,
                    "tree_method": "hist",
                },
            },
            {
                "name": "catboost_default",
                "type": "catboost",
                "params": {
                    "iterations": 300,
                    "depth": 6,
                    "learning_rate": 0.05,
                    "auto_class_weights": "Balanced",
                },
            },
            {
                "name": "lgbm_default",
                "type": "lgbm",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "class_weight": "balanced",
                },
            },
        ]
        smote_variants = [False]
    elif subset == "full":
        feature_sets = [
            "top8",
            "top50",
            "min_imp_5",
            "all",
            "temporal_only",
            "top50_plus_temporal",
        ]
        model_configs = [
            {
                "name": "xgb_baseline",
                "type": "xgboost",
                "params": {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "scale_pos_weight": 14.6,
                    "tree_method": "hist",
                },
            },
            {
                "name": "xgb_aggressive",
                "type": "xgboost",
                "params": {
                    "n_estimators": 1000,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "scale_pos_weight": 14.6,
                    "gamma": 2,
                    "tree_method": "hist",
                },
            },
            {
                "name": "xgb_regularized",
                "type": "xgboost",
                "params": {
                    "n_estimators": 1000,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "scale_pos_weight": 14.6,
                    "min_child_weight": 3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "tree_method": "hist",
                },
            },
            {
                "name": "catboost_default",
                "type": "catboost",
                "params": {
                    "iterations": 500,
                    "depth": 8,
                    "learning_rate": 0.05,
                    "auto_class_weights": "Balanced",
                },
            },
            {
                "name": "catboost_aggressive",
                "type": "catboost",
                "params": {
                    "iterations": 1000,
                    "depth": 10,
                    "learning_rate": 0.1,
                    "auto_class_weights": "Balanced",
                },
            },
            {
                "name": "lgbm_default",
                "type": "lgbm",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "mlp_simple",
                "type": "mlp",
                "params": {
                    "hidden_layer_sizes": (128, 64),
                    "max_iter": 200,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                },
            },
            {
                "name": "mlp_deep",
                "type": "mlp",
                "params": {
                    "hidden_layer_sizes": (256, 128, 64),
                    "max_iter": 200,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                },
            },
        ]
        smote_variants = [False, True]
    else:  # medium (default)
        feature_sets = ["top8", "top50", "min_imp_5", "top50_plus_temporal"]
        model_configs = [
            {
                "name": "xgb_baseline",
                "type": "xgboost",
                "params": {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "scale_pos_weight": 14.6,
                    "tree_method": "hist",
                },
            },
            {
                "name": "xgb_aggressive",
                "type": "xgboost",
                "params": {
                    "n_estimators": 1000,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "scale_pos_weight": 14.6,
                    "gamma": 2,
                    "tree_method": "hist",
                },
            },
            {
                "name": "catboost_default",
                "type": "catboost",
                "params": {
                    "iterations": 500,
                    "depth": 8,
                    "learning_rate": 0.05,
                    "auto_class_weights": "Balanced",
                },
            },
            {
                "name": "lgbm_default",
                "type": "lgbm",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "mlp_simple",
                "type": "mlp",
                "params": {
                    "hidden_layer_sizes": (128, 64),
                    "max_iter": 200,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                },
            },
        ]
        smote_variants = [False, True]

    experiments = []
    for fs in feature_sets:
        for mc in model_configs:
            for sm in smote_variants:
                smote_str = "_smote" if sm else ""
                experiments.append(
                    {
                        "name": f"{mc['name']}_{fs}{smote_str}",
                        "feature_set": fs,
                        "model": mc,
                        "smote": sm,
                    }
                )

    print(f"\nRunning {len(experiments)} experiments ({subset} mode)...")
    print("=" * 60)

    results = []
    for i, exp in enumerate(experiments, 1):
        result = run_one_experiment(
            exp, disease, target_col, df, imp_df, mlflow_experiment
        )
        results.append(result)
        print(
            f"[{i}/{len(experiments)}] {result['name']}: "
            f"ROC={result.get('ROC_AUC', 0):.4f}, "
            f"PR={result.get('PR_AUC', 0):.4f}, "
            f"F2={result.get('F2_score', 0):.4f} "
            f"({result.get('duration_sec', 0):.1f}s)"
        )

    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = MODEL_DIR / f"experiment_results_{disease}_{time_id}.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    print("\n" + "=" * 60)
    print("Top 10 Experiments (by ROC-AUC)")
    print("=" * 60)
    sorted_results = sorted(
        [r for r in results if "error" not in r],
        key=lambda x: x.get("ROC_AUC", 0),
        reverse=True,
    )[:10]

    print(
        f"{'Rank':<6}{'Experiment':<45}{'ROC-AUC':<10}{'PR-AUC':<10}"
        f"{'F2':<10}{'Time':<10}"
    )
    print("-" * 90)
    for rank, r in enumerate(sorted_results, 1):
        print(
            f"{rank:<6}{r['name'][:44]:<45}{r['ROC_AUC']:<10.4f}"
            f"{r['PR_AUC']:<10.4f}{r['F2_score']:<10.4f}"
            f"{r['duration_sec']:<10.1f}"
        )

    print("\nDone. View all runs in MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()
