"""
Step 3: Load a small sample of RAND HRS and reshape from wide to long format.

Wide: one row per respondent (columns R1DIABE, R2DIABE, ...).
Long: one row per (respondent, wave) with columns id, wave, DIABE, BMI, ...

Run from repo root: python scripts/step03_reshape_to_long.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pyreadstat

from pipeline_config import get_sample_rows

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAS_FILE = os.path.normpath(
    os.path.join(REPO_DIR, "data", "randhrs1992_2022v1_SAS", "randhrs1992_2022v1.sas7bdat")
)
OUT_DIR = os.path.join(REPO_DIR, "data")
WAVES_MIN, WAVES_MAX = 3, 14  # waves to keep (target + features available)


def get_columns_for_wave(all_columns: list[str], wave: int) -> list[str]:
    """Return column names that belong to this wave (R1..., R2..., R10..., etc.)."""
    prefix_short = f"R{wave}"
    cols = []
    for c in all_columns:
        if not c.startswith("R"):
            continue
        if wave <= 9:
            # R1xxx, R2xxx, ... but not R10xxx, R11xxx
            ok = len(c) <= len(prefix_short) or not c[len(prefix_short)].isdigit()
            if c.startswith(prefix_short) and ok:
                cols.append(c)
        else:
            # R10xxx, R11xxx, ...
            if c.startswith(prefix_short):
                cols.append(c)
    return cols


def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide (one row per respondent) to long (one row per respondent-wave)."""
    id_col = df_wide["HHIDPN"].astype("Int64")
    all_cols = list(df_wide.columns)
    long_parts = []

    for w in range(1, 17):
        wave_cols = get_columns_for_wave(all_cols, w)
        if not wave_cols:
            continue
        prefix = f"R{w}"
        sub = df_wide[wave_cols].copy()
        # Strip wave prefix to get concept name
        sub.columns = [c[len(prefix):] for c in wave_cols]
        sub["id"] = id_col.values
        sub["wave"] = w
        long_parts.append(sub)

    long = pd.concat(long_parts, ignore_index=True)
    return long


def main() -> None:
    if not os.path.isfile(SAS_FILE):
        print(f"ERROR: SAS file not found: {SAS_FILE}", file=sys.stderr)
        sys.exit(1)

    sample_rows = get_sample_rows()
    if sample_rows == 0:
        print(f"Loading all rows from {SAS_FILE} (SAMPLE_ROWS=0) ...")
        df_wide, _ = pyreadstat.read_sas7bdat(SAS_FILE)
    else:
        print(f"Loading first {sample_rows} rows from {SAS_FILE} ...")
        df_wide, _ = pyreadstat.read_sas7bdat(SAS_FILE, row_limit=sample_rows)
    print(f"  Wide shape: {df_wide.shape}")

    print("Reshaping to long (one row per respondent-wave) ...")
    df_long = wide_to_long(df_wide)
    print(f"  Long shape (all waves): {df_long.shape}")

    df_long = df_long[
        (df_long["wave"] >= WAVES_MIN) & (df_long["wave"] <= WAVES_MAX)
    ]
    print(f"  Long shape (waves {WAVES_MIN}-{WAVES_MAX}): {df_long.shape}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "rand_hrs_long_sample.parquet")
    df_long.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    # Quick summary
    print("\n--- Summary ---")
    print(f"  Unique ids: {df_long['id'].nunique()}")
    print(f"  Waves: {sorted(df_long['wave'].unique().tolist())}")
    print(f"  Columns (first 12): {list(df_long.columns[:12])}")
    if "DIABE" in df_long.columns:
        non_miss = df_long["DIABE"].notna() & (df_long["DIABE"].isin([0, 1]))
        print(f"  DIABE non-missing (0/1): {non_miss.sum()} rows")


if __name__ == "__main__":
    main()
