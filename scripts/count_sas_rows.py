"""
Report total row count of the RAND HRS SAS file (wide = one row per respondent).

Uses pyreadstat metadata when available to avoid loading the full file.
Run from repo root: python scripts/count_sas_rows.py
"""
from __future__ import annotations

import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAS_FILE = os.path.normpath(
    os.path.join(REPO_DIR, "data", "randhrs1992_2022v1_SAS", "randhrs1992_2022v1.sas7bdat")
)


def main() -> None:
    if not os.path.isfile(SAS_FILE):
        print(f"ERROR: SAS file not found: {SAS_FILE}", file=sys.stderr)
        print("  Place the RAND HRS SAS data in data/randhrs1992_2022v1_SAS/", file=sys.stderr)
        sys.exit(1)

    import pyreadstat

    # Try metadata-only first (fast; no full read)
    _, meta = pyreadstat.read_sas7bdat(SAS_FILE, metadataonly=True)
    total = getattr(meta, "number_rows", None) or getattr(meta, "nrows", None)

    if total is None:
        print("  Metadata does not include row count; loading full file (this may take a while)...")
        df_wide, _ = pyreadstat.read_sas7bdat(SAS_FILE)
        total = len(df_wide)

    total = int(total)
    # Current default from config/pipeline_params.yaml; step03 uses min(sample_rows, file size)
    current_config = 50_000
    actual_loaded = min(current_config, total)
    pct = 100.0 * actual_loaded / total if total else 0

    print(f"\n--- RAND HRS SAS file (wide: one row per respondent) ---")
    print(f"  File: {SAS_FILE}")
    print(f"  Total rows (respondents): {total:,}")
    print(f"\n  Current pipeline config: sample_rows = {current_config:,} (config/pipeline_params.yaml)")
    print(f"  So step 3 loads: min(50_000, file size) = {actual_loaded:,} respondents = {pct:.1f}% of full.")

    # Recommendation
    print("\n--- Recommendation ---")
    if actual_loaded >= total:
        print("  You are already using 100% of the dataset (config sample_rows >= file size).")
        print("  No change needed. Use SAMPLE_ROWS=0 in step 3 if you want to make 'full data' explicit.")
    elif pct >= 80:
        print("  You are using most of the data. For final/production runs, use 100%:")
        print("  Set SAMPLE_ROWS=0 when running step 3 (or set sample_rows in config to at least the total above).")
    elif pct >= 50:
        print("  For final results and stable feature importances, use at least 80–100% of the data.")
        print(f"  Suggested: sample_rows >= {int(0.8 * total):,} (80%), or SAMPLE_ROWS=0 for 100%.")
    else:
        print("  For credible model and importance estimates, use at least 80% of the dataset.")
        print(f"  Suggested: sample_rows >= {int(0.8 * total):,} (80%), or SAMPLE_ROWS=0 for 100%.")
    print()


if __name__ == "__main__":
    main()
