"""
Step 6: Train baseline models for all 8 conditions and report F2, PR-AUC, ROC-AUC.

For each condition: temporal split (train waves 3-9, val 10-12), LogisticRegression
with class_weight='balanced'. Writes model/baseline_run_<suffix>.json per condition
and model/baseline_run.json (diabetes-only, backward compat).

Run from repo root: python scripts/step06_train_baseline.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from conditions_config import load_conditions, target_columns
from pipeline_config import get_feature_columns

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")
MODEL_DIR = os.path.join(REPO_DIR, "model")
X_PATH = os.path.join(DATA_DIR, "X_minimal.parquet")
Y_CONDITIONS_PATH = os.path.join(DATA_DIR, "y_conditions.parquet")

TRAIN_WAVES = (3, 9)
VAL_WAVES = (10, 12)


def train_one_condition(
    X: pd.DataFrame,
    y: pd.Series,
    condition_suffix: str,
    condition_name: str,
    feature_columns: list[str],
) -> dict:
    """Train one LR model for one condition; return run_record dict."""
    wave = X["wave"].values
    train_mask = (wave >= TRAIN_WAVES[0]) & (wave <= TRAIN_WAVES[1])
    val_mask = (wave >= VAL_WAVES[0]) & (wave <= VAL_WAVES[1])

    X_train, X_val = X.loc[train_mask], X.loc[val_mask]
    y_train = y[train_mask].values
    y_val = y[val_mask].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_val_s)
    y_prob = model.predict_proba(X_val_s)[:, 1]

    f2 = fbeta_score(y_val, y_pred, beta=2)
    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)

    prec, rec, thresh = precision_recall_curve(y_val, y_prob)
    f2_scores = [
        fbeta_score(y_val, (y_prob >= t).astype(int), beta=2) for t in thresh
    ]
    best_idx = max(range(len(f2_scores)), key=lambda i: f2_scores[i])
    best_thresh = float(thresh[best_idx])
    f2_at_best = float(f2_scores[best_idx])

    run_record = {
        "run_id": f"baseline_001_{condition_suffix}",
        "description": f"Logistic regression, temporal split, minimal features ({condition_name})",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "task": {
            "target": f"incident_{condition_suffix.lower()}_by_t2",
            "target_definition": "target_definition.md",
            "evaluation_metrics": ["F2", "PR-AUC", "ROC-AUC"],
        },
        "data": {
            "source": "data/X_minimal.parquet, data/y_conditions.parquet",
            "feature_manifest": "data/feature_manifest.csv",
            "n_samples_total": int(len(X)),
            "train": {
                "wave_range": list(TRAIN_WAVES),
                "n_samples": int(train_mask.sum()),
            },
            "validation": {
                "wave_range": list(VAL_WAVES),
                "n_samples": int(val_mask.sum()),
            },
            "split_type": "temporal",
            "features": feature_columns,
            "preprocessing": {
                "missing": "complete cases only (no imputation); per-condition mask for target",
                "scaling": "StandardScaler (fit on train, transform train and val)",
            },
        },
        "model": {
            "type": "sklearn.linear_model.LogisticRegression",
            "hyperparameters": {
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
            },
        },
        "results": {
            "validation": {
                "F2_score": round(f2, 4),
                "PR_AUC": round(pr_auc, 4),
                "ROC_AUC": round(roc_auc, 4),
                "best_F2_threshold": round(best_thresh, 3),
                "F2_at_best_threshold": round(f2_at_best, 4),
            },
        },
    }
    return run_record


def main() -> None:
    if not os.path.isfile(X_PATH) or not os.path.isfile(Y_CONDITIONS_PATH):
        print(
            f"ERROR: Need {X_PATH} and {Y_CONDITIONS_PATH} (run step05_build_features.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    conditions = load_conditions()
    tcol = target_columns()
    feature_columns = get_feature_columns()

    print("Loading X and y_conditions ...")
    X = pd.read_parquet(X_PATH)
    y_all = pd.read_parquet(Y_CONDITIONS_PATH)
    print(f"  X shape: {X.shape}, y_conditions shape: {y_all.shape}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for suffix, name in conditions:
        col = f"target_{suffix}"
        if col not in y_all.columns:
            print(f"  Skip {col}: not in y_conditions", file=sys.stderr)
            continue
        y = y_all[col]
        # Use complete cases only (LR cannot handle missing features; we do not impute)
        complete = X[feature_columns].notna().all(axis=1)
        mask = y.notna() & complete
        if mask.sum() == 0:
            print(f"  Skip {col}: no non-missing target in complete cases")
            continue
        X_c = X.loc[mask].copy()
        y_c = y.loc[mask].astype(int)

        print(f"\n--- {name} ({suffix}) ---")
        print(f"  Complete cases (non-missing target + all features): {mask.sum()}")
        print(f"  Event rate: {y_c.mean():.2%}")

        run_record = train_one_condition(X_c, y_c, suffix, name, feature_columns)

        print(f"  F2: {run_record['results']['validation']['F2_score']:.4f}, "
              f"PR-AUC: {run_record['results']['validation']['PR_AUC']:.4f}, "
              f"ROC-AUC: {run_record['results']['validation']['ROC_AUC']:.4f}")

        out_path = os.path.join(MODEL_DIR, f"baseline_run_{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(run_record, f, indent=2)
        print(f"  Saved: {out_path}")

    # Backward compatibility: diabetes-only run as baseline_run.json
    diabe_record_path = os.path.join(MODEL_DIR, "baseline_run_DIABE.json")
    out_baseline = os.path.join(MODEL_DIR, "baseline_run.json")
    if os.path.isfile(diabe_record_path):
        with open(diabe_record_path) as f:
            rec = json.load(f)
        rec["run_id"] = "baseline_001"
        rec["task"]["target"] = "incident_diabetes_by_t2"
        rec["description"] = "Logistic regression, temporal split, minimal features (Diabetes)"
        with open(out_baseline, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"\nSaved (backward compat): {out_baseline}")


if __name__ == "__main__":
    main()
