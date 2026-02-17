"""
Step 4: Build incident-by-t+2 targets for all 8 conditions on the long data.

Uses config/conditions.yaml. For each condition: at-risk = condition=0 at wave t;
event = condition=1 at t+1 or t+2. Output: one model table with columns
target_DIABE, ..., target_ARTHRE.

Run from repo root: python -m feature_engineering.steps.step04_build_target
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from pathlib import Path

from feature_engineering.conditions_config import load_conditions

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
LONG_PARQUET = DATA_DIR / "rand_hrs_long_sample.parquet"
OUT_PARQUET = DATA_DIR / "rand_hrs_long_with_target.parquet"
# Build target only for waves where t+2 in data (waves 3-14), so t in 3..12
WAVE_T_MAX = 12


def build_target_one(
    df_long: pd.DataFrame, condition_suffix: str
) -> pd.DataFrame | None:
    """
    Build incident-by-t+2 target for one condition.

    Returns a DataFrame with columns id, wave, target_<suffix>, or None if
    the condition column is missing in the long data.
    """
    col = condition_suffix
    if col not in df_long.columns:
        return None
    df = df_long[df_long["wave"] <= WAVE_T_MAX][["id", "wave"]].copy()

    cond = df_long[["id", "wave", col]].copy()
    cond_t1 = cond.rename(columns={"wave": "wave_t1", col: f"{col}_t1"})
    cond_t1["wave"] = cond_t1["wave_t1"] - 1
    cond_t1 = cond_t1[["id", "wave", f"{col}_t1"]]
    cond_t2 = cond.rename(columns={"wave": "wave_t2", col: f"{col}_t2"})
    cond_t2["wave"] = cond_t2["wave_t2"] - 2
    cond_t2 = cond_t2[["id", "wave", f"{col}_t2"]]

    df = df.merge(cond_t1, on=["id", "wave"], how="left")
    df = df.merge(cond_t2, on=["id", "wave"], how="left")
    # Current wave value for at-risk
    df = df.merge(
        df_long[["id", "wave", col]].rename(columns={col: f"{col}_curr"}),
        on=["id", "wave"],
        how="left",
    )

    at_risk = (df[f"{col}_curr"] == 0) & (df[f"{col}_curr"].notna())
    event = (df[f"{col}_t1"] == 1) | (df[f"{col}_t2"] == 1)
    both_zero = (
        (df[f"{col}_t1"] == 0)
        & (df[f"{col}_t2"] == 0)
        & df[f"{col}_t1"].notna()
        & df[f"{col}_t2"].notna()
    )

    target = pd.Series(float("nan"), index=df.index)
    target[event] = 1
    target[both_zero] = 0
    target_col = f"target_{condition_suffix}"
    df[target_col] = target
    df.loc[~at_risk, target_col] = float("nan")
    return df[["id", "wave", target_col]]


def main() -> None:
    if not LONG_PARQUET.is_file():
        print(
            f"ERROR: Long parquet not found: {LONG_PARQUET}",
            file=sys.stderr,
        )
        print("Run step03_reshape_to_long first.", file=sys.stderr)
        sys.exit(1)

    conditions = load_conditions()
    print(f"Conditions from config: {[s for s, _ in conditions]}")

    print(f"Loading {LONG_PARQUET} ...")
    df_long = pd.read_parquet(str(LONG_PARQUET))
    print(f"  Shape: {df_long.shape}")

    # Base: waves 3..12 with all columns from long
    df = df_long[df_long["wave"] <= WAVE_T_MAX].copy()
    print(f"  Rows (waves 3..{WAVE_T_MAX}): {len(df)}")

    for suffix, name in conditions:
        tdf = build_target_one(df_long, suffix)
        if tdf is None:
            print(f"  Skipping target_{suffix} ({name}): column not in long data")
            continue
        print(f"  Building target_{suffix} ({name}) ...")
        df = df.merge(tdf, on=["id", "wave"], how="left")

    # Summary per condition
    print("\n--- Target summary (per condition) ---")
    for suffix, name in conditions:
        tc = f"target_{suffix}"
        if tc not in df.columns:
            continue
        with_t = df[tc].notna().sum()
        events = (df[tc] == 1).sum()
        rate = events / with_t if with_t else 0
        print(f"  {tc}: non-missing={with_t}, events={events}, rate={rate:.2%}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(OUT_PARQUET), index=False)
    print(f"\nSaved: {OUT_PARQUET}")

    # Model table: same table (one row per id-wave, 8 target columns)
    out_model = DATA_DIR / "rand_hrs_model_sample.parquet"
    df.to_parquet(str(out_model), index=False)
    print(f"Saved (model table with 8 target columns): {out_model}")
    print(f"  Model table rows: {len(df)}")


if __name__ == "__main__":
    main()
