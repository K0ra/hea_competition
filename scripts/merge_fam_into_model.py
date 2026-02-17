"""
Merge fam_aggregated_filtered (one row per person) into the model table.

Reads rand_hrs_model_sample.parquet and fam_aggregated_filtered.pkl,
left-joins on id, writes rand_hrs_model_merged.parquet.

Run from repo root after step04: python scripts/merge_fam_into_model.py

To use the merged table for training:
  MODEL_TABLE=rand_hrs_model_merged.parquet python scripts/run_disease.py <disease>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
MODEL_PARQUET = DATA_DIR / "rand_hrs_model_sample.parquet"
FAM_PKL = DATA_DIR / "fam_aggregated_filtered.pkl"
OUT_PARQUET = DATA_DIR / "rand_hrs_model_merged.parquet"


def merge_fam_into_model_df(
    df_model: pd.DataFrame, df_fam: pd.DataFrame
) -> pd.DataFrame:
    """Left-join df_fam (one row per id) onto df_model (id, wave, ...)."""
    df_fam = df_fam.copy()
    if "id" not in df_fam.columns and "HHIDPN" in df_fam.columns:
        df_fam = df_fam.rename(columns={"HHIDPN": "id"})
    if "id" not in df_fam.columns:
        raise ValueError(
            "Family table has no 'id' or 'HHIDPN' column. "
            f"Columns: {list(df_fam.columns[:20])}"
        )
    id_dtype = df_model["id"].dtype
    if df_fam["id"].dtype != id_dtype:
        df_fam["id"] = df_fam["id"].astype(id_dtype)
    model_cols = set(df_model.columns) - {"id"}
    fam_cols = set(df_fam.columns) - {"id"}
    overlap = model_cols & fam_cols
    if overlap:
        df_merged = df_model.merge(
            df_fam, on="id", how="left", suffixes=("", "_fam")
        )
    else:
        df_merged = df_model.merge(df_fam, on="id", how="left")
    return df_merged


def main() -> None:
    if not MODEL_PARQUET.is_file():
        print(
            f"ERROR: Model table not found: {MODEL_PARQUET}",
            file=sys.stderr,
        )
        print("  Run step04_build_target.py first.", file=sys.stderr)
        sys.exit(1)
    if not FAM_PKL.is_file():
        print(f"ERROR: Family table not found: {FAM_PKL}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {MODEL_PARQUET} ...")
    df_model = pd.read_parquet(MODEL_PARQUET)
    print(f"  Shape: {df_model.shape}")

    print(f"Loading {FAM_PKL} ...")
    with open(FAM_PKL, "rb") as f:
        df_fam = pd.read_pickle(f)
    print(f"  Shape: {df_fam.shape}")

    try:
        df_merged = merge_fam_into_model_df(df_model, df_fam)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    model_cols = set(df_model.columns) - {"id"}
    fam_cols = set(df_fam.columns) - {"id"}
    overlap = model_cols & fam_cols
    if overlap:
        print(
            "  Overlapping columns (fam side suffixed with '_fam'):",
            sorted(overlap),
        )

    print(f"Merged shape: {df_merged.shape}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved: {OUT_PARQUET}")
    print(
        "  Use with: MODEL_TABLE=rand_hrs_model_merged.parquet "
        "python scripts/run_disease.py <disease>"
    )


if __name__ == "__main__":
    main()
