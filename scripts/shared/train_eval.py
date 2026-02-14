"""
Shared train/eval for one condition: temporal split, train one model, return metrics.
Used by run_disease.py and run_experiment.py.
"""
from __future__ import annotations

import subprocess

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

TRAIN_WAVES = (3, 9)
VAL_WAVES = (10, 12)

# Models that require all features non-missing (no native missing support).
# Tree models like LightGBM handle missing natively; only target must be non-missing.
MODELS_REQUIRE_COMPLETE_CASES = ("lr", "rf")


def _gpu_available() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def train_eval_one(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    model_type: str,
    model_params: dict,
) -> tuple[dict, object]:
    """
    Train one model for one target; temporal split (waves 3-9 train, 10-12 val).
    Returns (metrics_dict, fitted_model). Caller can use model.feature_importances_
    for tree models (lgbm, rf).
    LR/RF require complete cases (all features non-missing). LGBM handles missing
    natively: only rows with non-missing target are used; features may have NaNs.
    """
    if model_type in MODELS_REQUIRE_COMPLETE_CASES:
        complete = X[feature_columns].notna().all(axis=1)
        mask = y.notna() & complete
        if mask.sum() == 0:
            raise ValueError(
                f"No rows with non-missing target and all {len(feature_columns)} "
                "features non-missing. Reduce features or use model type 'lgbm' to "
                "support missing features."
            )
        X = X.loc[mask]
        y = y.loc[mask]
    else:
        # Tree models (e.g. lgbm): only require non-missing target; features may be missing
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

    y = y.astype(int)
    wave = X["wave"].values
    train_mask = (
        (wave >= TRAIN_WAVES[0]) & (wave <= TRAIN_WAVES[1])
    )
    val_mask = (wave >= VAL_WAVES[0]) & (wave <= VAL_WAVES[1])
    if train_mask.sum() == 0:
        raise ValueError(
            f"No training samples in waves {TRAIN_WAVES[0]}-{TRAIN_WAVES[1]}. "
            "Reduce features or check feature missingness so complete cases exist "
            "in the train wave range."
        )
    X_train = X.loc[train_mask][feature_columns]
    X_val = X.loc[val_mask][feature_columns]
    y_train = y[train_mask].values
    y_val = y[val_mask].values

    if model_type == "lr":
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_va_s = scaler.transform(X_val)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=model_params.get("max_iter", 1000),
            random_state=model_params.get("random_state", 42),
        )
        model.fit(X_tr_s, y_train)
        y_prob = model.predict_proba(X_va_s)[:, 1]
    elif model_type == "rf":
        X_tr_s = X_train.values
        X_va_s = X_val.values
        model = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 50),
            max_depth=model_params.get("max_depth", 6),
            class_weight="balanced",
            random_state=model_params.get("random_state", 42),
            n_jobs=model_params.get("n_jobs", -1),
        )
        model.fit(X_tr_s, y_train)
        y_prob = model.predict_proba(X_va_s)[:, 1]
    elif model_type == "lgbm":
        import lightgbm as lgb

        X_tr_s = X_train
        X_va_s = X_val
        device = model_params.get("device")
        if device is None and _gpu_available():
            device = "gpu"
        elif device is None:
            device = "cpu"
        params = {
            "objective": "binary",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": model_params.get("num_leaves", 31),
            "learning_rate": model_params.get("learning_rate", 0.05),
            "n_estimators": model_params.get("n_estimators", 100),
            "random_state": model_params.get("random_state", 42),
            "class_weight": "balanced",
            "n_jobs": model_params.get("n_jobs", -1),
            "device": device,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr_s,
            y_train,
            eval_set=[(X_va_s, y_val)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        y_prob = model.predict_proba(X_va_s)[:, 1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    y_pred = (y_prob >= 0.5).astype(int)
    f2 = fbeta_score(y_val, y_pred, beta=2)
    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)
    prec, rec, thresh = precision_recall_curve(y_val, y_prob)
    f2_scores = [
        fbeta_score(y_val, (y_prob >= t).astype(int), beta=2)
        for t in thresh
    ]
    best_idx = max(range(len(f2_scores)), key=lambda i: f2_scores[i])
    best_thresh = float(thresh[best_idx])

    results = {
        "F2_score": round(f2, 4),
        "PR_AUC": round(pr_auc, 4),
        "ROC_AUC": round(roc_auc, 4),
        "best_F2_threshold": round(best_thresh, 3),
    }
    return results, model
