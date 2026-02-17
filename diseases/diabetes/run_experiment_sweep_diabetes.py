#!/usr/bin/env python3
"""
Diabetes Tree-Based and Simple MLP Experiment Sweep
====================================================

Tests LightGBM, XGBoost, CatBoost, and sklearn MLP on diabetes prediction.
Loads preprocessed train/test parquet files directly.

Usage:
    python diseases/diabetes/run_experiment_sweep_diabetes.py
    EXPERIMENTS_SUBSET=quick python diseases/diabetes/run_experiment_sweep_diabetes.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
TRAIN_DATA_PATH = SCRIPT_DIR / "diabetes_train.parquet"
TEST_DATA_PATH = SCRIPT_DIR / "diabetes_test.parquet"
RESULTS_DIR = SCRIPT_DIR / "results" / "experiment_sweep"
RANDOM_STATE = 42
TARGET_COL = "target_DIABE"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# LightGBM configurations
LGBM_CONFIGS = [
    {
        "type": "lgbm",
        "name": "lgbm_default",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 7,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    {
        "type": "lgbm",
        "name": "lgbm_deep",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 10,
            "num_leaves": 63,
            "min_child_samples": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
]

# XGBoost configurations
XGB_CONFIGS = [
    {
        "type": "xgboost",
        "name": "xgb_default",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 7,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
        },
    },
]

# CatBoost configurations
CATBOOST_CONFIGS = [
    {
        "type": "catboost",
        "name": "catboost_default",
        "params": {
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 7,
            "l2_leaf_reg": 3,
            "subsample": 0.8,
        },
    },
]

# MLP configurations
MLP_CONFIGS = [
    {
        "type": "mlp",
        "name": "mlp_small",
        "params": {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": 256,
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
    },
    {
        "type": "mlp",
        "name": "mlp_large",
        "params": {
            "hidden_layer_sizes": (256, 128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": 512,
            "learning_rate_init": 0.0003,
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
    },
]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets."""
    if not TRAIN_DATA_PATH.is_file():
        raise FileNotFoundError(f"Train data not found: {TRAIN_DATA_PATH}")
    if not TEST_DATA_PATH.is_file():
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")
    
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    
    return train_df, test_df


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_subset: str = "all",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    """
    Prepare features and targets.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        feature_subset: "all", "top100", "top200", or "statistical"
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names
    """
    # Get all feature columns
    all_features = [
        c for c in train_df.columns
        if c != TARGET_COL and not c.startswith("target_") and c != "id"
    ]
    
    # Select feature subset
    if feature_subset == "all":
        feature_cols = all_features
    elif feature_subset == "top100":
        # Use first 100 features (arbitrary but consistent)
        feature_cols = all_features[:100]
    elif feature_subset == "top200":
        feature_cols = all_features[:200]
    elif feature_subset == "statistical":
        # Select statistical/aggregated features if they exist
        stat_features = [
            f for f in all_features
            if any(keyword in f.lower() for keyword in [
                "mean", "max", "min", "std", "count", "sum", "median"
            ])
        ]
        feature_cols = stat_features if stat_features else all_features[:100]
    else:
        raise ValueError(f"Unknown feature subset: {feature_subset}")
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].values
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].values
    
    # Fill NaN with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train, y_train, X_test, y_test, feature_cols


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    model_config: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    use_smote: bool,
    class_weight: float,
) -> tuple[object, dict]:
    """Train one model and return metrics."""
    model_type = model_config["type"]
    params = model_config["params"].copy()
    
    # Apply SMOTE if requested
    if use_smote and len(np.unique(y_train)) > 1:
        try:
            smote = SMOTE(sampling_strategy=0.33, random_state=RANDOM_STATE)
            X_train_res, y_train_res = smote.fit_resample(X_train.values, y_train)
            X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
            y_train = y_train_res
        except Exception as e:
            print(f"    SMOTE failed: {e}")
    
    # Train model
    if model_type == "lgbm":
        import lightgbm as lgb
        params["scale_pos_weight"] = class_weight
        model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, verbosity=-1)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        y_prob = model.predict_proba(X_test.values)[:, 1]
        
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        params["scale_pos_weight"] = class_weight
        model = XGBClassifier(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
        y_prob = model.predict_proba(X_test.values)[:, 1]
        
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        params["scale_pos_weight"] = class_weight
        model = CatBoostClassifier(**params, random_state=RANDOM_STATE, verbose=False)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        y_prob = model.predict_proba(X_test.values)[:, 1]
        
    elif model_type == "mlp":
        model = MLPClassifier(**params, random_state=RANDOM_STATE)
        # Scale features for MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_test_scaled = scaler.transform(X_test.values)
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compute metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_arr, precision_arr)
    
    # Find best F2 threshold
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f2 = 0.0
    best_thresh = 0.5
    for t in thresholds:
        y_pred_bin = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_test, y_pred_bin, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = t
    
    metrics = {
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "F2_score": round(best_f2, 4),
        "best_threshold": round(best_thresh, 3),
        "n_samples": len(y_test),
        "n_positive": int(np.sum(y_test == 1)),
    }
    
    return model, metrics


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_one_experiment(
    exp_config: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_weight: float,
    output_dir: Path,
) -> dict:
    """Run a single experiment."""
    exp_name = exp_config["name"]
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Prepare features
    feature_subset = exp_config.get("feature_subset", "all")
    use_smote = exp_config.get("use_smote", False)
    
    print(f"  Feature subset: {feature_subset}")
    print(f"  SMOTE: {use_smote}")
    
    X_train, y_train, X_test, y_test, feature_cols = prepare_features(
        train_df, test_df, feature_subset
    )
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(y_train)} (positive: {y_train.sum()})")
    print(f"  Test samples: {len(y_test)} (positive: {y_test.sum()})")
    
    # Train model
    try:
        model, metrics = train_model(
            exp_config["model"],
            X_train,
            y_train,
            X_test,
            y_test,
            use_smote,
            class_weight,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "experiment_name": exp_name,
            "status": "failed",
            "error": str(e),
        }
    
    elapsed_time = time.time() - start_time
    
    # Save results
    result_dict = {
        "experiment_name": exp_name,
        "model_type": exp_config["model"]["type"],
        "model_params": exp_config["model"]["params"],
        "feature_subset": feature_subset,
        "n_features": len(feature_cols),
        "use_smote": use_smote,
        "class_weight": class_weight,
        "training_time_sec": elapsed_time,
        "metrics": metrics,
    }
    
    json_path = output_dir / f"{exp_name}.json"
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"\n  Results: ROC={metrics['ROC_AUC']:.4f}, "
          f"PR={metrics['PR_AUC']:.4f}, F2={metrics['F2_score']:.4f}")
    print(f"  Time: {elapsed_time:.1f}s")
    print(f"  Saved to: {json_path}")
    
    return result_dict


def build_experiment_grid(mode: str = "full") -> list[dict]:
    """Build experiment grid based on mode."""
    experiments = []
    
    # Feature subsets to test
    if mode == "quick":
        feature_subsets = ["all"]
        model_configs = LGBM_CONFIGS[:1] + XGB_CONFIGS[:1]
        smote_variants = [False]
    else:  # full
        feature_subsets = ["all", "top200", "top100"]
        model_configs = LGBM_CONFIGS + XGB_CONFIGS + CATBOOST_CONFIGS + MLP_CONFIGS
        smote_variants = [False, True]
    
    # Build grid
    for model_config in model_configs:
        for feature_subset in feature_subsets:
            for use_smote in smote_variants:
                exp_name = f"{model_config['name']}_{feature_subset}"
                if use_smote:
                    exp_name += "_smote"
                
                experiments.append({
                    "name": exp_name,
                    "model": model_config,
                    "feature_subset": feature_subset,
                    "use_smote": use_smote,
                })
    
    return experiments


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("DIABETES TREE-BASED MODEL EXPERIMENT SWEEP")
    print("="*80)
    print(f"Train data: {TRAIN_DATA_PATH}")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"Results dir: {RESULTS_DIR}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_data()
    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Compute class weight
    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    print(f"\nTrain positive rate: {y_train.mean():.3f}")
    print(f"Test positive rate: {y_test.mean():.3f}")
    print(f"Class weight: {class_weight:.2f}")
    
    # Build experiment grid
    mode = os.environ.get("EXPERIMENTS_SUBSET", "full")
    experiments = build_experiment_grid(mode)
    print(f"\nExperiment mode: {mode}")
    print(f"Total experiments: {len(experiments)}")
    
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    print(f"\n{'='*80}")
    print("STARTING EXPERIMENTS")
    print(f"{'='*80}")
    
    results = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp_config['name']}")
        
        try:
            result = run_one_experiment(
                exp_config, train_df, test_df, class_weight, RESULTS_DIR
            )
            if result.get("status") != "failed":
                results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"COMPLETED {len(results)}/{len(experiments)} EXPERIMENTS")
    print(f"{'='*80}")
    
    if results:
        sorted_results = sorted(
            results,
            key=lambda x: x["metrics"]["F2_score"],
            reverse=True
        )
        print("\nTop 10 by F2 Score:")
        print("-"*80)
        for i, r in enumerate(sorted_results[:10], 1):
            m = r["metrics"]
            print(f"{i}. {r['experiment_name']:<50} "
                  f"F2={m['F2_score']:.4f} ROC={m['ROC_AUC']:.4f}")
        
        # Save summary
        summary_path = RESULTS_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(sorted_results, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
    
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
