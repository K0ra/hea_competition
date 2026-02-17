"""
Step 5: Build feature matrix X and targets y for modeling.

Uses only columns at wave t (no leakage). Keeps all rows; X may contain
missing values. Models that cannot handle missing (e.g. sklearn LR/RF) use
complete cases at train time; tree models with native missing support
(e.g. LightGBM) can use all rows.

Run from repo root: python -m feature_engineering.steps.step05_build_features
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from pathlib import Path

from feature_engineering.conditions_config import load_conditions, target_columns
from feature_engineering.pipeline_config import get_feature_columns

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
MODEL_PARQUET = DATA_DIR / "rand_hrs_model_sample.parquet"
OUT_X_PATH = DATA_DIR / "X_minimal.parquet"
OUT_Y_PATH = DATA_DIR / "y_conditions.parquet"
OUT_Y_MINIMAL_PATH = DATA_DIR / "y_minimal.parquet"
OUT_MANIFEST = DATA_DIR / "feature_manifest.csv"


def main() -> None:
    if not MODEL_PARQUET.is_file():
        print(f"ERROR: {MODEL_PARQUET} not found. Run step04_build_target first.")
        sys.exit(1)

    feature_columns = get_feature_columns()
    tcol_all = target_columns()
    print(f"Loading {MODEL_PARQUET} ...")
    df = pd.read_parquet(str(MODEL_PARQUET))
    print(f"  Shape: {df.shape}")

    tcol = [c for c in tcol_all if c in df.columns]
    if not tcol:
        print("ERROR: No target columns found in model table. Run step04_build_target first.")
        sys.exit(1)
    if len(tcol) < len(tcol_all):
        print(f"  Using target columns present in data: {tcol}")
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        print(f"ERROR: Missing feature columns: {missing}")
        sys.exit(1)

    # Keep all rows; do not drop or impute. Missing handled at train time per model.
    use = feature_columns + tcol + ["id"]
    df = df[use].copy()
    n_rows = len(df)
    n_complete = df[feature_columns].notna().all(axis=1).sum()
    print(f"  Rows: {n_rows} (complete cases: {n_complete}, with missing: {n_rows - n_complete})")

    X = df[feature_columns].copy()
    y_all = df[tcol].copy()

    print("\n--- Feature matrix and targets ---")
    print(f"  X shape: {X.shape}")
    print(f"  y (8 conditions) shape: {y_all.shape}")
    for c in tcol:
        valid = y_all[c].notna().sum()
        rate = (y_all[c] == 1).sum() / valid if valid else 0
        print(f"    {c}: non-missing={valid}, event rate={rate:.2%}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(str(OUT_X_PATH), index=False)
    y_all.to_parquet(str(OUT_Y_PATH), index=False)
    print(f"\nSaved: {OUT_X_PATH}, {OUT_Y_PATH}")

    # Backward compatibility: diabetes-only target when available, else first condition
    if "target_DIABE" in y_all.columns:
        y_minimal = y_all[["target_DIABE"]].rename(columns={"target_DIABE": "target"})
    else:
        y_minimal = y_all[[tcol[0]]].rename(columns={tcol[0]: "target"})
    y_minimal.to_parquet(str(OUT_Y_MINIMAL_PATH), index=False)
    print(f"Saved (single target, same rows as X): {OUT_Y_MINIMAL_PATH}")

    manifest = pd.DataFrame({
        "feature": feature_columns,
        "role": "input",
        "source": "RAND HRS long, wave t",
    })
    manifest.to_csv(str(OUT_MANIFEST), index=False)
    print(f"Saved: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
