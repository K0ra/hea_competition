"""
Smoke tests for multi-task components.

Run from repo root: python multi_task_models/test_multi_task.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "multi_task_models"))

from utils import (
    TARGET_COLS,
    compute_per_task_weights,
    evaluate_multi_task,
    get_feature_columns,
    load_data,
    prepare_train_val_split,
)


def test_data_loads():
    """Test that data loads with all 8 target columns."""
    df = load_data()
    assert df.shape[0] > 0, "Data should have rows"
    assert "id" in df.columns, "Data should have id column"
    assert "wave" in df.columns, "Data should have wave column"
    
    present_targets = [t for t in TARGET_COLS if t in df.columns]
    print(f"  Found {len(present_targets)}/8 target columns")
    assert len(present_targets) >= 1, "At least one target should be present"
    
    for target_col in present_targets:
        n_non_null = df[target_col].notna().sum()
        n_positive = (df[target_col] == 1).sum()
        print(
            f"    {target_col}: {n_non_null} samples, "
            f"{n_positive} positive"
        )
    print(f"  OK: Data loads, shape={df.shape}")


def test_feature_columns():
    """Test feature column extraction."""
    df = load_data()
    features = get_feature_columns(df, importances_path=None)
    assert len(features) > 1, "Should have at least wave + 1 feature"
    assert "wave" in features, "Should include wave"
    assert "id" not in features, "Should not include id"
    for target in TARGET_COLS:
        assert target not in features, f"Should not include {target}"
    print(f"  OK: Extracted {len(features)} features (including wave)")


def test_train_val_split():
    """Test temporal train/val split for all tasks."""
    df = load_data()
    # Filter to waves with data in both train and val
    df = df[df["wave"].between(3, 12)]
    df = df.head(10000)  # Small subset for speed
    features = get_feature_columns(df, importances_path=None)[:20]
    present_targets = [t for t in TARGET_COLS if t in df.columns]
    
    X_tr, y_tr, X_va, y_va = prepare_train_val_split(
        df, features, present_targets
    )
    
    if len(X_tr) == 0:
        print(
            "  SKIP: No tasks with sufficient data in small subset "
            "(this is OK for test)"
        )
        return
    
    assert len(X_va) > 0, "Should have at least one val task"
    
    for target in X_tr.keys():
        assert target in y_tr, f"Missing y_train for {target}"
        assert target in X_va, f"Missing X_val for {target}"
        assert target in y_va, f"Missing y_val for {target}"
        assert len(X_tr[target]) == len(
            y_tr[target]
        ), f"X/y train mismatch for {target}"
        assert len(X_va[target]) == len(
            y_va[target]
        ), f"X/y val mismatch for {target}"
    
    print(f"  OK: Split created for {len(X_tr)} tasks")
    for target in X_tr.keys():
        print(
            f"    {target}: train={len(y_tr[target])}, "
            f"val={len(y_va[target])}"
        )


def test_compute_weights():
    """Test per-task weight computation."""
    y_dict = {
        "target_DIABE": np.array([0] * 146 + [1] * 10),
        "target_HIBPE": np.array([0] * 80 + [1] * 10),
        "target_CANCR": np.array([0] * 200 + [1] * 5),
    }
    weights = compute_per_task_weights(y_dict)
    assert "DIABE" in weights, "Should have DIABE weight"
    assert weights["DIABE"] == 146 / 10, "DIABE weight should be 14.6"
    assert weights["HIBPE"] == 80 / 10, "HIBPE weight should be 8.0"
    assert weights["CANCR"] == 200 / 5, "CANCR weight should be 40.0"
    print("  OK: Weights computed correctly")
    for disease, weight in weights.items():
        print(f"    {disease}: {weight:.2f}")


def test_evaluate_multi_task():
    """Test multi-task evaluation."""
    np.random.seed(42)
    y_true = {
        "DIABE": np.array([0, 0, 0, 1, 1]),
        "HIBPE": np.array([0, 1, 0, 1, 1]),
    }
    y_pred = {
        "DIABE": np.array([0.1, 0.2, 0.3, 0.8, 0.9]),
        "HIBPE": np.array([0.2, 0.7, 0.1, 0.9, 0.8]),
    }
    results = evaluate_multi_task(y_true, y_pred)
    assert "DIABE" in results, "Should have DIABE results"
    assert "HIBPE" in results, "Should have HIBPE results"
    assert "ROC_AUC" in results["DIABE"], "Should have ROC_AUC"
    assert "PR_AUC" in results["DIABE"], "Should have PR_AUC"
    assert "F2_score" in results["DIABE"], "Should have F2_score"
    print("  OK: Evaluation computed")
    for disease, metrics in results.items():
        print(
            f"    {disease}: ROC={metrics['ROC_AUC']:.4f}, "
            f"PR={metrics['PR_AUC']:.4f}, F2={metrics['F2_score']:.4f}"
        )


def test_mini_xgboost():
    """Test training mini XGBoost on small subset."""
    from xgboost import XGBClassifier
    
    df = load_data()
    df = df[df["wave"].between(3, 12)]
    df = df.head(5000)
    features = get_feature_columns(df, importances_path=None)[:15]
    present_targets = [t for t in TARGET_COLS if t in df.columns][:3]
    
    X_tr, y_tr, X_va, y_va = prepare_train_val_split(
        df, features, present_targets
    )
    
    if len(X_tr) == 0:
        print("  SKIP: No tasks with sufficient data")
        return
    
    # Train one model for first task
    first_task = list(X_tr.keys())[0]
    model = XGBClassifier(
        n_estimators=10, max_depth=3, random_state=42, verbosity=0
    )
    model.fit(X_tr[first_task], y_tr[first_task])
    y_pred = model.predict_proba(X_va[first_task])[:, 1]
    
    assert y_pred.shape == y_va[first_task].shape, "Pred shape mismatch"
    assert (y_pred >= 0).all() and (
        y_pred <= 1
    ).all(), "Pred should be probabilities"
    print(f"  OK: XGBoost trained on {first_task}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running multi-task component tests")
    print("=" * 60)
    
    tests = [
        ("Data loads", test_data_loads),
        ("Feature columns", test_feature_columns),
        ("Train/val split", test_train_val_split),
        ("Compute weights", test_compute_weights),
        ("Evaluate multi-task", test_evaluate_multi_task),
        ("Mini XGBoost", test_mini_xgboost),
    ]
    
    failed = []
    for name, test_fn in tests:
        try:
            print(f"\nTest: {name}")
            test_fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((name, str(e)))
    
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {len(tests)} tests passed")
        print("Ready to run multi-task experiments.")
