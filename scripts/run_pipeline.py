"""
Run the full pipeline: reshape long (3) -> build 8 targets (4) -> build X/y (5)
-> train 8 baselines (6). Single entry point.

Run from repo root: python scripts/run_pipeline.py

To start from a later step, set RUN_FROM_STEP=4 (or 5, 6) in the environment.
"""
from __future__ import annotations

import os
import sys

# Ensure scripts/ is on path so step04/05/06 can import conditions_config.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Change to repo root so paths in steps resolve correctly
_repo_dir = os.path.dirname(_script_dir)
os.chdir(_repo_dir)


def main() -> None:
    run_from = int(os.environ.get("RUN_FROM_STEP", "3"))

    if run_from <= 3:
        import step03_reshape_to_long as step03
        step03.main()

    if run_from <= 4:
        import step04_build_target as step04
        step04.main()

    if run_from <= 5:
        import step05_build_features as step05
        step05.main()

    if run_from <= 6:
        import step06_train_baseline as step06
        step06.main()

    print("\n--- Pipeline finished ---")


if __name__ == "__main__":
    main()
