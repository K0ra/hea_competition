"""
Quick test that the fam merge logic is correct (synthetic data).

Run from repo root: python scripts/test_merge_fam.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# run from repo root; script dir in path
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from merge_fam_into_model import merge_fam_into_model_df  # noqa: E402


def test_merge_preserves_rows_and_joins_fam() -> None:
    # Model table: two people, two waves each -> 4 rows
    df_model = pd.DataFrame(
        {
            "id": [101, 101, 102, 102],
            "wave": [3, 4, 3, 4],
            "target_DIABE": [0.0, 1.0, 0.0, 0.0],
            "feat_a": [10, 11, 20, 21],
        }
    )
    df_model["id"] = df_model["id"].astype("Int64")

    # Fam table: one row per id, one extra id not in model (103)
    df_fam = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "fam_x": [100.0, 200.0, 300.0],
            "fam_y": [1, 2, 3],
        }
    )
    df_fam["id"] = df_fam["id"].astype("Int64")

    merged = merge_fam_into_model_df(df_model, df_fam)

    # Left join: same number of rows as model
    assert len(merged) == len(df_model), "merge must preserve model row count"
    assert list(merged["id"]) == [101, 101, 102, 102]
    assert list(merged["wave"]) == [3, 4, 3, 4]

    # Fam columns present
    assert "fam_x" in merged.columns and "fam_y" in merged.columns

    # Same id gets same fam values for every wave
    assert list(merged["fam_x"]) == [100.0, 100.0, 200.0, 200.0]
    assert list(merged["fam_y"]) == [1, 1, 2, 2]

    # Original model columns unchanged
    assert list(merged["feat_a"]) == [10, 11, 20, 21]
    assert list(merged["target_DIABE"]) == [0.0, 1.0, 0.0, 0.0]


def test_merge_with_hhidpn_rename() -> None:
    df_model = pd.DataFrame({"id": [1], "wave": [3], "x": [10]})
    df_model["id"] = df_model["id"].astype("Int64")
    df_fam = pd.DataFrame({"HHIDPN": [1], "fam_z": [99]})

    merged = merge_fam_into_model_df(df_model, df_fam)

    assert len(merged) == 1
    assert merged["fam_z"].iloc[0] == 99
    assert "id" in merged.columns and merged["id"].iloc[0] == 1


def test_merge_with_overlapping_columns_uses_suffixes() -> None:
    df_model = pd.DataFrame(
        {"id": [1, 2], "wave": [3, 3], "score": [0.1, 0.2]}
    )
    df_model["id"] = df_model["id"].astype("Int64")
    df_fam = pd.DataFrame({"id": [1, 2], "score": [1.0, 2.0]})
    df_fam["id"] = df_fam["id"].astype("Int64")

    merged = merge_fam_into_model_df(df_model, df_fam)

    assert len(merged) == 2
    # Model keeps "score", fam gets "score_fam"
    assert "score" in merged.columns and "score_fam" in merged.columns
    assert list(merged["score"]) == [0.1, 0.2]
    assert list(merged["score_fam"]) == [1.0, 2.0]


if __name__ == "__main__":
    test_merge_preserves_rows_and_joins_fam()
    print("  OK: merge preserves rows and joins fam by id")
    test_merge_with_hhidpn_rename()
    print("  OK: HHIDPN renamed to id")
    test_merge_with_overlapping_columns_uses_suffixes()
    print("  OK: overlapping columns get _fam suffix")
    print("All merge tests passed.")
