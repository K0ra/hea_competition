"""
Multi-output XGBoost for all 8 health conditions.

Trains separate XGBClassifier for each condition with shared features,
logs all results to MLflow under single run.

Run from repo root:
  python multi_task_models/run_multi_xgboost.py
  MIN_IMPORTANCE=5 python multi_task_models/run_multi_xgboost.py
  TOP_N_FEATURES=50 python multi_task_models/run_multi_xgboost.py
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
from xgboost import XGBClassifier

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "multi_task_models"))

from utils import (
    DISEASE_SUFFIXES,
    MODEL_DIR,
    TARGET_COLS,
    compute_per_task_weights,
    discover_importances,
    evaluate_multi_task,
    get_feature_columns,
    load_data,
    prepare_train_val_split,
)

RANDOM_STATE = 42


def main() -> None:
    """Run multi-task XGBoost for all 8 conditions."""
    print("=" * 70)
    print("Multi-Task XGBoost — All 8 Health Conditions")
    print("=" * 70)
    
    # Load data
    data_path_str = os.environ.get("DATA_PATH", "")
    data_path = (
        Path(data_path_str).resolve()
        if data_path_str
        else REPO_DIR / "data" / "rand_hrs_model_merged.parquet"
    )
    print(f"Loading data: {data_path}")
    df = load_data(data_path)
    print(f"  Shape: {df.shape}")
    
    # Feature selection
    feature_list_str = os.environ.get("FEATURE_LIST", "").strip()
    feature_list_path = (
        Path(feature_list_str).resolve() if feature_list_str else None
    )
    
    importances_path_str = os.environ.get("IMPORTANCES_PATH", "").strip()
    if importances_path_str:
        importances_path = Path(importances_path_str).resolve()
    else:
        importances_path = discover_importances("diabetes")
    
    top_n_str = os.environ.get("TOP_N_FEATURES", "").strip()
    top_n = int(top_n_str) if top_n_str else None
    min_importance_str = os.environ.get("MIN_IMPORTANCE", "").strip()
    min_importance = (
        float(min_importance_str) if min_importance_str else None
    )
    
    if feature_list_path and feature_list_path.is_file():
        print(f"Using feature list: {feature_list_path}")
    elif importances_path and importances_path.is_file():
        print(f"Using importances: {importances_path}")
        if min_importance is not None:
            print(f"  Filter: importance > {min_importance}")
        elif top_n is not None:
            print(f"  Filter: top {top_n} features")
    else:
        print("Using all features (no importances/feature list found)")
        importances_path = None
        feature_list_path = None
    
    feature_columns = get_feature_columns(
        df, importances_path, top_n, min_importance, feature_list_path
    )
    print(f"  Selected {len(feature_columns)} features (including wave)")
    
    # Filter to targets that exist
    present_targets = [t for t in TARGET_COLS if t in df.columns]
    print(f"\nTargets: {len(present_targets)}/8")
    for target_col in present_targets:
        n_samples = df[target_col].notna().sum()
        n_pos = (df[target_col] == 1).sum()
        print(f"  {target_col}: {n_samples} samples, {n_pos} positive")
    
    # Prepare train/val splits for all tasks
    print("\nPreparing temporal train/val splits (waves 3-9 / 10-12)...")
    X_train_dict, y_train_dict, X_val_dict, y_val_dict = (
        prepare_train_val_split(df, feature_columns, present_targets)
    )
    
    if len(X_train_dict) == 0:
        print("ERROR: No tasks with sufficient training data", file=sys.stderr)
        sys.exit(1)
    
    print(f"  {len(X_train_dict)} tasks ready for training")
    
    # Compute per-task class weights
    weights = compute_per_task_weights(y_train_dict)
    print("\nPer-task class weights (scale_pos_weight):")
    for suffix, weight in weights.items():
        print(f"  {suffix}: {weight:.2f}")
    
    # XGBoost hyperparameters from env or defaults
    max_depth = int(os.environ.get("XGBOOST_MAX_DEPTH", "6"))
    learning_rate = float(os.environ.get("XGBOOST_LEARNING_RATE", "0.05"))
    n_estimators = int(os.environ.get("XGBOOST_N_ESTIMATORS", "500"))
    
    print(f"\nXGBoost params: max_depth={max_depth}, lr={learning_rate}, "
          f"n_est={n_estimators}")
    
    # Train one model per task
    print("\n" + "=" * 70)
    print("Training models...")
    print("=" * 70)
    
    models = {}
    y_pred_val_dict = {}
    y_true_val_dict = {}
    train_times = {}
    
    start_total = time.time()
    
    for target_col in X_train_dict.keys():
        suffix = DISEASE_SUFFIXES[target_col]
        print(f"\n[{suffix}] Training...")
        
        X_train = X_train_dict[target_col]
        y_train = y_train_dict[target_col]
        X_val = X_val_dict[target_col]
        y_val = y_val_dict[target_col]
        
        start_t = time.time()
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=weights[suffix],
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        duration = time.time() - start_t
        train_times[suffix] = duration
        
        y_pred = model.predict_proba(X_val)[:, 1]
        models[target_col] = model
        y_pred_val_dict[suffix] = y_pred
        y_true_val_dict[suffix] = y_val
        
        print(f"  Done in {duration:.1f}s")
    
    total_duration = time.time() - start_total
    print(f"\nTotal training time: {total_duration:.1f}s")
    
    # Evaluate all tasks
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    results = evaluate_multi_task(y_true_val_dict, y_pred_val_dict)
    
    print(
        f"{'Disease':<12} {'ROC-AUC':<10} {'PR-AUC':<10} {'F2':<10} "
        f"{'n_pos':<8} {'n_samples':<10}"
    )
    print("-" * 70)
    for suffix, metrics in results.items():
        print(
            f"{suffix:<12} {metrics['ROC_AUC']:<10.4f} "
            f"{metrics['PR_AUC']:<10.4f} {metrics['F2_score']:<10.4f} "
            f"{metrics['n_positive']:<8} {metrics['n_samples']:<10}"
        )
    
    # Aggregate metrics
    mean_roc = np.mean([m["ROC_AUC"] for m in results.values()])
    mean_pr = np.mean([m["PR_AUC"] for m in results.values()])
    mean_f2 = np.mean([m["F2_score"] for m in results.values()])
    
    print("-" * 70)
    print(
        f"{'MEAN':<12} {mean_roc:<10.4f} {mean_pr:<10.4f} "
        f"{mean_f2:<10.4f}"
    )
    
    # MLflow logging
    print("\n" + "=" * 70)
    print("Logging to MLflow...")
    print("=" * 70)
    
    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"multi_xgb_{len(feature_columns)}feat_{time_id}"
    
    mlflow.set_experiment("hea_multitask")
    with mlflow.start_run(run_name=run_name):
        # Model-level params
        mlflow.log_param("model_type", "multi_output_xgboost")
        mlflow.log_param("n_tasks", len(results))
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("feature_selection", "importances" if importances_path else "all")
        if min_importance is not None:
            mlflow.log_param("min_importance", min_importance)
        if top_n is not None:
            mlflow.log_param("top_n_features", top_n)
        
        # Per-task metrics
        for suffix, metrics in results.items():
            mlflow.log_metric(f"{suffix}_ROC_AUC", metrics["ROC_AUC"])
            mlflow.log_metric(f"{suffix}_PR_AUC", metrics["PR_AUC"])
            mlflow.log_metric(f"{suffix}_F2", metrics["F2_score"])
            mlflow.log_metric(
                f"{suffix}_class_weight", weights.get(suffix, 1.0)
            )
            mlflow.log_metric(f"{suffix}_train_time", train_times[suffix])
        
        # Aggregate metrics
        mlflow.log_metric("mean_ROC_AUC", mean_roc)
        mlflow.log_metric("mean_PR_AUC", mean_pr)
        mlflow.log_metric("mean_F2", mean_f2)
        mlflow.log_metric("total_duration_sec", total_duration)
    
    print(f"  Logged run: {run_name}")
    
    # Save results to JSON
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results_path = MODEL_DIR / f"multi_xgb_results_{time_id}.json"
    
    output = {
        "timestamp": time_id,
        "model_type": "multi_output_xgboost",
        "n_tasks": len(results),
        "n_features": len(feature_columns),
        "hyperparameters": {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
        },
        "per_task_results": {
            suffix: {
                **metrics,
                "class_weight": weights.get(suffix, 1.0),
                "train_time": train_times.get(suffix, 0.0),
            }
            for suffix, metrics in results.items()
        },
        "aggregate_metrics": {
            "mean_ROC_AUC": mean_roc,
            "mean_PR_AUC": mean_pr,
            "mean_F2": mean_f2,
        },
        "total_duration_sec": total_duration,
    }
    
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"  Saved results: {results_path}")
    
    # Save per-task feature importances
    for target_col, model in models.items():
        suffix = DISEASE_SUFFIXES[target_col]
        imp = model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": feature_columns,
            "importance": imp,
        })
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_path = MODEL_DIR / f"multi_xgb_{suffix}_importances_{time_id}.csv"
        imp_df.to_csv(imp_path, index=False)
    
    print(f"  Saved {len(models)} importance CSVs")
    
    print("\n" + "=" * 70)
    print("Done. View results: mlflow ui")
    print("=" * 70)


if __name__ == "__main__":
    main()
