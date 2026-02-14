"""
Data quality report on pipeline outputs.

Reads data/rand_hrs_long_sample.parquet, data/rand_hrs_model_sample.parquet,
and optionally X/y; computes sample sizes, missingness, target balance,
and basic feature stats. Writes model/data_quality_report.json.

Run from repo root after the pipeline (or after step 5): python scripts/report_data_quality.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

from pipeline_config import get_feature_columns

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data"
MODEL_DIR = REPO_DIR / "model"
LONG_PARQUET = DATA_DIR / "rand_hrs_long_sample.parquet"
MODEL_PARQUET = DATA_DIR / "rand_hrs_model_sample.parquet"
X_PARQUET = DATA_DIR / "X_minimal.parquet"
Y_PARQUET = DATA_DIR / "y_conditions.parquet"
OUT_REPORT = MODEL_DIR / "data_quality_report.json"


def main() -> None:
    report = {}

    # Long table
    if LONG_PARQUET.is_file():
        df_long = pd.read_parquet(LONG_PARQUET)
        report["long"] = {
            "path": str(LONG_PARQUET),
            "rows": int(len(df_long)),
            "unique_ids": int(df_long["id"].nunique()) if "id" in df_long.columns else None,
            "waves": sorted(df_long["wave"].unique().tolist()) if "wave" in df_long.columns else None,
            "rows_per_wave": df_long.groupby("wave").size().to_dict() if "wave" in df_long.columns else None,
        }
    else:
        report["long"] = {"path": str(LONG_PARQUET), "found": False}

    # Model table (before step 5 dropna)
    if not MODEL_PARQUET.is_file():
        print(f"ERROR: {MODEL_PARQUET} not found. Run pipeline through step 4 first.", file=sys.stderr)
        sys.exit(1)

    df_model = pd.read_parquet(MODEL_PARQUET)
    report["model_table"] = {
        "path": str(MODEL_PARQUET),
        "rows": int(len(df_model)),
    }

    # Missingness per feature (same as step05)
    feature_columns = get_feature_columns()
    missing = {}
    for col in feature_columns:
        if col in df_model.columns:
            n = df_model[col].isna().sum()
            missing[col] = {"count": int(n), "pct": round(100 * n / len(df_model), 2)}
        else:
            missing[col] = "column_not_present"
    report["missingness_per_feature"] = missing

    # Step 5: no global drop; complete cases used only at train time for LR/RF
    present = [c for c in feature_columns if c in df_model.columns]
    n_rows = len(df_model)
    n_complete = df_model.dropna(subset=present).shape[0] if present else 0
    report["step5"] = {
        "rows_total": n_rows,
        "complete_cases": n_complete,
        "rows_with_any_missing_feature": n_rows - n_complete,
    }

    # Target balance per condition
    tcol = [c for c in df_model.columns if c.startswith("target_")]
    report["target_balance"] = {}
    for col in tcol:
        valid = df_model[col].notna().sum()
        events = (df_model[col] == 1).sum()
        report["target_balance"][col] = {
            "non_missing": int(valid),
            "events": int(events),
            "event_rate_pct": round(100 * events / valid, 2) if valid else None,
        }

    # Feature sanity (min/max/mean for numeric features)
    report["feature_stats"] = {}
    for col in feature_columns:
        if col not in df_model.columns:
            continue
        s = df_model[col].dropna()
        if s.dtype.kind in ("i", "u", "f"):
            report["feature_stats"][col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": round(float(s.mean()), 4),
            }

    # X/y if present (final training set size)
    if X_PARQUET.is_file():
        X = pd.read_parquet(X_PARQUET)
        report["X_final"] = {"path": str(X_PARQUET), "rows": int(len(X))}
    if Y_PARQUET.is_file():
        y = pd.read_parquet(Y_PARQUET)
        report["y_conditions"] = {"path": str(Y_PARQUET), "rows": int(len(y))}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
