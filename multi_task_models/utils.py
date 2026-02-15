"""Shared utilities for multi-task learning."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
MODEL_DIR = REPO_DIR / "model"

# All 8 target columns
TARGET_COLS = [
    "target_DIABE",
    "target_HIBPE",
    "target_CANCR",
    "target_LUNGE",
    "target_HEARTE",
    "target_STROKE",
    "target_PSYCHE",
    "target_ARTHRE",
]

# Disease suffix mapping
DISEASE_SUFFIXES = {
    "target_DIABE": "DIABE",
    "target_HIBPE": "HIBPE",
    "target_CANCR": "CANCR",
    "target_LUNGE": "LUNGE",
    "target_HEARTE": "HEARTE",
    "target_STROKE": "STROKE",
    "target_PSYCHE": "PSYCHE",
    "target_ARTHRE": "ARTHRE",
}

TRAIN_WAVES = (3, 9)
VAL_WAVES = (10, 12)


def load_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load the merged parquet with all features and targets."""
    if data_path is None:
        data_path = DATA_DIR / "rand_hrs_model_merged.parquet"
    if not data_path.is_file():
        raise FileNotFoundError(f"Data not found: {data_path}")
    return pd.read_parquet(data_path)


def get_feature_columns(
    df: pd.DataFrame,
    importances_path: Path | None = None,
    top_n: int | None = None,
    min_importance: float | None = None,
    feature_list_path: Path | None = None,
) -> list[str]:
    """
    Get feature columns based on importance filtering or explicit list.
    
    Priority:
    1. If feature_list_path provided, use that (one feature per line)
    2. Else if importances_path provided, filter by top_n or min_importance
    3. Otherwise, use all non-target, non-id, non-wave columns
    """
    exclude_cols = set(TARGET_COLS + ["id", "wave"])
    all_features = [c for c in df.columns if c not in exclude_cols]
    
    # Priority 1: Explicit feature list file
    if feature_list_path is not None:
        if not feature_list_path.is_file():
            raise FileNotFoundError(
                f"Feature list not found: {feature_list_path}"
            )
        with open(feature_list_path) as f:
            features = [line.strip() for line in f if line.strip()]
        # Filter to features that exist in df
        features = [f for f in features if f in df.columns and f != "wave"]
        return ["wave"] + features
    
    # Priority 2: Importances-based filtering
    if importances_path is not None:
        if not importances_path.is_file():
            raise FileNotFoundError(
                f"Importances not found: {importances_path}"
            )
        
        imp_df = pd.read_csv(importances_path)
        imp_df = imp_df.sort_values("importance", ascending=False)
        
        if min_importance is not None:
            features = imp_df[imp_df["importance"] > min_importance][
                "feature"
            ].tolist()
        elif top_n is not None:
            features = imp_df.head(top_n)["feature"].tolist()
        else:
            features = imp_df["feature"].tolist()
        
        # Filter to features that exist in df
        features = [f for f in features if f in df.columns and f != "wave"]
        return ["wave"] + features
    
    # Priority 3: Use all available features
    return ["wave"] + all_features


def compute_per_task_weights(
    y_train_dict: dict[str, np.ndarray]
) -> dict[str, float]:
    """
    Compute class weights for each disease based on prevalence.
    
    Returns dict mapping disease suffix to scale_pos_weight value.
    """
    weights = {}
    for target_col, y in y_train_dict.items():
        suffix = DISEASE_SUFFIXES.get(target_col, target_col)
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        if n_positive > 0:
            # Inverse frequency weighting (same as scale_pos_weight)
            weights[suffix] = n_negative / n_positive
        else:
            weights[suffix] = 1.0
    return weights


def evaluate_multi_task(
    y_true_dict: dict[str, np.ndarray],
    y_pred_dict: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """
    Compute ROC-AUC, PR-AUC, F2 for each task.
    
    Args:
        y_true_dict: Map from disease suffix to true labels
        y_pred_dict: Map from disease suffix to predicted probabilities
    
    Returns:
        Dict mapping disease suffix to metrics dict
    """
    results = {}
    for disease, y_true in y_true_dict.items():
        y_pred = y_pred_dict[disease]
        
        # Skip if no positive or negative samples
        if len(np.unique(y_true)) < 2:
            results[disease] = {
                "ROC_AUC": 0.0,
                "PR_AUC": 0.0,
                "F2_score": 0.0,
                "best_threshold": 0.5,
                "n_samples": len(y_true),
                "n_positive": int(np.sum(y_true == 1)),
            }
            continue
        
        # Per-task metrics
        roc_auc = roc_auc_score(y_true, y_pred)
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall_arr, precision_arr)
        
        # Best F2 threshold
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f2 = 0.0
        best_thresh = 0.5
        for t in thresholds:
            y_bin = (y_pred >= t).astype(int)
            f2 = fbeta_score(y_true, y_bin, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_thresh = t
        
        results[disease] = {
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "F2_score": round(best_f2, 4),
            "best_threshold": round(best_thresh, 3),
            "n_samples": len(y_true),
            "n_positive": int(np.sum(y_true == 1)),
        }
    return results


def prepare_train_val_split(
    df: pd.DataFrame, feature_columns: list[str], target_cols: list[str]
) -> tuple[dict, dict, dict, dict]:
    """
    Prepare train/val splits for all tasks with temporal split.
    
    Returns:
        X_train_dict, y_train_dict, X_val_dict, y_val_dict
        Each dict maps target_col to arrays/dataframes
    """
    wave = df["wave"].values
    train_mask = (wave >= TRAIN_WAVES[0]) & (wave <= TRAIN_WAVES[1])
    val_mask = (wave >= VAL_WAVES[0]) & (wave <= VAL_WAVES[1])
    
    X_train_dict = {}
    y_train_dict = {}
    X_val_dict = {}
    y_val_dict = {}
    
    for target_col in target_cols:
        if target_col not in df.columns:
            continue
        
        # Filter to rows with non-null target for this task
        task_mask = df[target_col].notna()
        task_train_mask = task_mask & train_mask
        task_val_mask = task_mask & val_mask
        
        if task_train_mask.sum() == 0 or task_val_mask.sum() == 0:
            continue
        
        # Exclude 'wave' from features (it's only for splitting)
        feat_cols_no_wave = [c for c in feature_columns if c != "wave"]
        X_train = df.loc[task_train_mask, feat_cols_no_wave].copy()
        y_train = df.loc[task_train_mask, target_col].astype(int).values
        X_val = df.loc[task_val_mask, feat_cols_no_wave].copy()
        y_val = df.loc[task_val_mask, target_col].astype(int).values
        
        # Fill NaN values with 0 (neural networks can't handle NaN)
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        
        # Scale features for neural networks (tree models don't need this)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_val_scaled = scaler.transform(X_val.values)
        X_train = pd.DataFrame(
            X_train_scaled, index=X_train.index, columns=X_train.columns
        )
        X_val = pd.DataFrame(
            X_val_scaled, index=X_val.index, columns=X_val.columns
        )
        
        X_train_dict[target_col] = X_train
        y_train_dict[target_col] = y_train
        X_val_dict[target_col] = X_val
        y_val_dict[target_col] = y_val
    
    return X_train_dict, y_train_dict, X_val_dict, y_val_dict


def discover_importances(disease: str = "diabetes") -> Path | None:
    """Discover latest importances file for a disease."""
    model_dir = REPO_DIR / "model"
    if not model_dir.is_dir():
        return None
    candidates = sorted(
        model_dir.glob(f"{disease}_*_importances.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
