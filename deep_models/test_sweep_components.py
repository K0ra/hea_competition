"""
Smoke tests for experiment sweep components.

Run from repo root: python deep_models/test_sweep_components.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_DIR / "data" / "rand_hrs_model_merged.parquet"


def test_data_loads():
    """Test that merged parquet loads correctly."""
    if not DATA_PATH.is_file():
        print(f"  SKIP: {DATA_PATH} not found")
        return
    df = pd.read_parquet(DATA_PATH)
    assert df.shape[0] > 0, "Data should have rows"
    assert "id" in df.columns, "Data should have id column"
    assert "wave" in df.columns, "Data should have wave column"
    assert "target_DIABE" in df.columns, "Data should have target_DIABE"
    print(f"  OK: Data loads, shape={df.shape}")


def test_temporal_features():
    """Test temporal feature computation on synthetic data."""
    df = pd.DataFrame({
        "id": [1, 1, 1, 2, 2],
        "wave": [3, 4, 5, 3, 4],
        "BMI": [25.0, 26.0, 27.5, 30.0, 29.0],
        "WEIGHT": [150, 155, 160, 200, 198],
    })
    df = df.sort_values(["id", "wave"]).reset_index(drop=True)

    for col in ["BMI", "WEIGHT"]:
        df[f"{col}_delta"] = df.groupby("id")[col].diff()
        df[f"{col}_pct_change"] = df.groupby("id")[col].pct_change()

    # First wave per person should have NaN deltas
    assert pd.isna(df.loc[0, "BMI_delta"])
    assert pd.isna(df.loc[3, "BMI_delta"])
    # Second wave should have deltas
    assert df.loc[1, "BMI_delta"] == 1.0
    assert df.loc[2, "BMI_delta"] == 1.5
    assert df.loc[4, "WEIGHT_delta"] == -2.0
    print("  OK: Temporal features computed correctly")


def test_xgboost_trains():
    """Test XGBoost training on small subset."""
    from xgboost import XGBClassifier

    if not DATA_PATH.is_file():
        print("  SKIP: Data not found")
        return

    df = pd.read_parquet(DATA_PATH)
    df = df[df["target_DIABE"].notna()].head(1000)
    X = df[["wave", "BMI", "WEIGHT"]].fillna(0)
    y = df["target_DIABE"].astype(int)

    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    pred = model.predict_proba(X)
    assert pred.shape == (len(X), 2), "Should predict 2 classes"
    print("  OK: XGBoost trains successfully")


def test_catboost_trains():
    """Test CatBoost training on small subset."""
    from catboost import CatBoostClassifier

    if not DATA_PATH.is_file():
        print("  SKIP: Data not found")
        return

    df = pd.read_parquet(DATA_PATH)
    df = df[df["target_DIABE"].notna()].head(1000)
    X = df[["wave", "BMI", "WEIGHT"]].fillna(0)
    y = df["target_DIABE"].astype(int)

    model = CatBoostClassifier(
        iterations=10, depth=3, random_state=42, verbose=False
    )
    model.fit(X, y)
    pred = model.predict_proba(X)
    assert pred.shape == (len(X), 2), "Should predict 2 classes"
    print("  OK: CatBoost trains successfully")


def test_lgbm_trains():
    """Test LightGBM training on small subset."""
    import lightgbm as lgb

    if not DATA_PATH.is_file():
        print("  SKIP: Data not found")
        return

    df = pd.read_parquet(DATA_PATH)
    df = df[df["target_DIABE"].notna()].head(1000)
    X = df[["wave", "BMI", "WEIGHT"]].fillna(0)
    y = df["target_DIABE"].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=10, max_depth=3, random_state=42, verbosity=-1
    )
    model.fit(X, y)
    pred = model.predict_proba(X)
    assert pred.shape == (len(X), 2), "Should predict 2 classes"
    print("  OK: LightGBM trains successfully")


def test_mlp_trains():
    """Test MLPClassifier training on small subset."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if not DATA_PATH.is_file():
        print("  SKIP: Data not found")
        return

    df = pd.read_parquet(DATA_PATH)
    df = df[df["target_DIABE"].notna()].head(1000)
    X = df[["wave", "BMI", "WEIGHT"]].fillna(0).values
    y = df["target_DIABE"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
    )
    model.fit(X_scaled, y)
    pred = model.predict_proba(X_scaled)
    assert pred.shape == (len(X), 2), "Should predict 2 classes"
    print("  OK: MLPClassifier trains successfully")


def test_smote():
    """Test SMOTE oversampling."""
    from imblearn.over_sampling import SMOTE

    X = np.array([[1, 2], [2, 3], [3, 4]] * 100 + [[10, 20]] * 10)
    y = np.array([0] * 300 + [1] * 10)

    smote = SMOTE(sampling_strategy=0.33, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    assert len(y_res) > len(y), "SMOTE should add samples"
    pos_before = (y == 1).sum()
    pos_after = (y_res == 1).sum()
    assert pos_after > pos_before, "SMOTE should increase minority class"
    print(
        f"  OK: SMOTE works (minority: {pos_before} -> {pos_after})"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Running sweep component smoke tests")
    print("=" * 60)

    tests = [
        ("Data loads", test_data_loads),
        ("Temporal features", test_temporal_features),
        ("XGBoost", test_xgboost_trains),
        ("CatBoost", test_catboost_trains),
        ("LightGBM", test_lgbm_trains),
        ("MLP", test_mlp_trains),
        ("SMOTE", test_smote),
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
        print("Ready to run experiment sweep.")
