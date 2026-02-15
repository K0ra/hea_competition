"""
Diabetes prediction using XGBoost on RAND HRS 1992-2022 data.

Pipeline:
1. Read pre-processed parquet (long format, one row per person-wave)
2. Remove diabetes-related leaky columns
3. Engineer lag/delta features (health trajectories)
4. Drop high-missingness columns
5. Train XGBoost with class weighting for imbalance
6. Tune threshold on validation set
7. Report F2-score, PR-AUC, ROC-AUC on held-out test set
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    fbeta_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    classification_report,
)
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PARQUET_PATH = os.environ.get(
    "PARQUET_PATH",
    str(Path(__file__).resolve().parent / "data" / "rand_hrs_model_merged.parquet"),
)

RANDOM_STATE = 42
TARGET_COL = "target_DIABE"

# Columns to exclude from features (targets, IDs)
TARGET_COLS = [
    "target_DIABE", "target_HIBPE", "target_CANCR", "target_LUNGE",
    "target_HEARTE", "target_STROKE", "target_PSYCHE", "target_ARTHRE",
]
ID_COLS = ["id", "wave"]

# Diabetes-related columns that leak the target
LEAKY_COLS = [
    "DIAB", "DIABE", "DIABE2", "DIABF", "DIABQ", "DIABS",
    "DBORLMED", "DBINSULN", "DBDIAGYR", "DBSTAGE",
]

# Key health features for lag/delta engineering
LAG_FEATURES = [
    "BMI", "WEIGHT", "SHLT", "CESD", "COG27", "COGTOT", "MSTOT",
    "IMRC", "DLRC", "SER7", "BWC86", "VOCAB",
    "DRINKD", "DRINKN", "DRINK",
    "HIBPE", "HEARTE", "STROKE", "PSYCHE", "ARTHRE",
    "ADL5A", "IADL5A", "MOBILA", "GROSSA", "FINEA",
    "HOSP", "DOCTOR", "DOCTIM",
    "IEARN", "WTHH",
]

MAX_MISSING_PCT = 0.60  # Drop columns with >80% missing

# ---------------------------------------------------------------------------
# Step 1: Load parquet
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Loading parquet")
print("=" * 60)

df = pd.read_parquet(PARQUET_PATH)

# Downcast to save memory
float_cols = df.select_dtypes(include=["float64"]).columns
df[float_cols] = df[float_cols].astype("float32")

print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

# ---------------------------------------------------------------------------
# Step 2: Remove leaky columns
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Removing diabetes-related leaky columns")
print("=" * 60)

leaky_found = [c for c in LEAKY_COLS if c in df.columns]
print(f"Removing {len(leaky_found)} leaky columns: {leaky_found}")
df = df.drop(columns=leaky_found)

# ---------------------------------------------------------------------------
# Step 3: Feature engineering -- lag/delta features
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Engineering lag/delta features")
print("=" * 60)

# Sort by person and wave for correct lag computation
df.sort_values(["id", "wave"], inplace=True)
df.reset_index(drop=True, inplace=True)

lag_cols_available = [c for c in LAG_FEATURES if c in df.columns]
print(f"Computing lags for {len(lag_cols_available)} features...")

new_cols_count = 0
for col in lag_cols_available:
    # Lag-1: previous wave value
    lag_col = f"{col}_lag1"
    df[lag_col] = df.groupby("id")[col].shift(1)
    new_cols_count += 1

    # Delta: change from previous wave
    delta_col = f"{col}_delta"
    df[delta_col] = df[col] - df[lag_col]
    new_cols_count += 1

    # Lag-2: two waves ago
    lag2_col = f"{col}_lag2"
    df[lag2_col] = df.groupby("id")[col].shift(2)
    new_cols_count += 1

# Rolling mean (3-wave window) for key continuous features
rolling_features = ["BMI", "WEIGHT", "SHLT", "CESD", "COG27", "DRINKD", "WTHH"]
rolling_available = [c for c in rolling_features if c in df.columns]
for col in rolling_available:
    roll_col = f"{col}_roll3mean"
    df[roll_col] = df.groupby("id")[col].transform(
        lambda s: s.rolling(3, min_periods=2).mean()
    )
    new_cols_count += 1

# Acceleration: delta of delta (is BMI increasing faster?)
accel_features = ["BMI", "WEIGHT", "SHLT", "CESD", "COG27"]
accel_available = [c for c in accel_features if f"{c}_delta" in df.columns]
for col in accel_available:
    accel_col = f"{col}_accel"
    df[accel_col] = df.groupby("id")[f"{col}_delta"].shift(0) - df.groupby("id")[f"{col}_delta"].shift(1)
    new_cols_count += 1

# Interaction features
if "BMI" in df.columns and "HIBPE" in df.columns:
    df["BMI_x_HIBPE"] = df["BMI"] * df["HIBPE"]
    new_cols_count += 1
if "BMI" in df.columns and "AGEY_B" in df.columns:
    df["BMI_x_AGE"] = df["BMI"] * df["AGEY_B"]
    new_cols_count += 1
if "SHLT" in df.columns and "CESD" in df.columns:
    df["SHLT_x_CESD"] = df["SHLT"] * df["CESD"]
    new_cols_count += 1
if "DRINK" in df.columns and "BMI" in df.columns:
    df["DRINK_x_BMI"] = df["DRINK"] * df["BMI"]
    new_cols_count += 1

# Cumulative condition count (comorbidity burden)
condition_cols = [c for c in ["HIBPE", "CANCRE", "LUNGE", "HEARTE", "STROKE", "PSYCHE", "ARTHRE"] if c in df.columns]
if condition_cols:
    df["comorbidity_count"] = df[condition_cols].sum(axis=1)
    new_cols_count += 1

# Waves since first observation (exposure time proxy)
df["waves_observed"] = df.groupby("id")["wave"].transform(lambda s: s - s.min())
new_cols_count += 1

print(f"Created {new_cols_count} new features")
print(f"Total columns now: {df.shape[1]}")

# ---------------------------------------------------------------------------
# Step 4: 
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Step 4: Dropping columns with >{MAX_MISSING_PCT*100:.0f}% missing")
print("=" * 60)

# Only check feature columns (not targets/IDs)
exclude_set = set(TARGET_COLS + ID_COLS)
feat_candidates = [c for c in df.columns if c not in exclude_set]

missing_pct = df[feat_candidates].isnull().mean()
high_missing = missing_pct[missing_pct > MAX_MISSING_PCT].index.tolist()
print(f"Dropping {len(high_missing)} columns with >{MAX_MISSING_PCT*100:.0f}% missing")
if high_missing:
    print(f"  Examples: {high_missing[:10]}{'...' if len(high_missing) > 10 else ''}")
df = df.drop(columns=high_missing)

# ---------------------------------------------------------------------------
# Step 5: Impute missing values per person (temporal ffill/bfill + median)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Imputing NaN (per-person ffill/bfill -> person median -> global median)")
print("=" * 60)

exclude_from_impute = set(TARGET_COLS + ID_COLS)
numeric_cols = [c for c in df.columns if c not in exclude_from_impute and df[c].dtype in ["float64", "float32", "int64", "int32"]]

missing_before = df[numeric_cols].isnull().sum().sum()
# Already sorted by [id, wave] in Step 3

# Pass 1: Forward-fill then backward-fill per person (uses nearest wave values)
grouped = df.groupby("id")[numeric_cols]
df[numeric_cols] = grouped.ffill().bfill()

# Pass 2: Per-person median (fills columns where person has no data at all in nearby waves)
person_medians = grouped.transform("median")
df[numeric_cols] = df[numeric_cols].fillna(person_medians)

# Pass 3: Global median (last resort for persons with zero observations in a column)
global_medians = df[numeric_cols].median()
df[numeric_cols] = df[numeric_cols].fillna(global_medians)

missing_after = df[numeric_cols].isnull().sum().sum()
print(f"Missing values: {missing_before:,} -> {missing_after:,}")
print(f"Imputed {missing_before - missing_after:,} values")

# ---------------------------------------------------------------------------
# Step 6: Prepare features and target
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Preparing features and target")
print("=" * 60)

df = df.dropna(subset=[TARGET_COL])
print(f"Rows after dropping NaN target: {df.shape[0]}")

y = df[TARGET_COL].astype(int)
exclude_cols = set(TARGET_COLS + ID_COLS)
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols]

print(f"Features: {len(feature_cols)}")
print(f"\nTarget distribution:")
print(y.value_counts().to_string())
print(f"Positive rate: {y.mean():.4f}")

# ---------------------------------------------------------------------------
# Step 7: Train / Val / Test split (60/20/20)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 7: Train/Val/Test split (60/20/20)")
print("=" * 60)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25,
    random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"Train: {X_train.shape[0]} rows (pos: {y_train.sum()}, neg: {(y_train==0).sum()})")
print(f"Val:   {X_val.shape[0]} rows (pos: {y_val.sum()}, neg: {(y_val==0).sum()})")
print(f"Test:  {X_test.shape[0]} rows (pos: {y_test.sum()}, neg: {(y_test==0).sum()})")

# ---------------------------------------------------------------------------
# Step 8: ADASYN Oversample minority class in training data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 8: ADASYN Oversampling minority class (training only)")
print("=" * 60)

adasyn = ADASYN(sampling_strategy=2/3, random_state=RANDOM_STATE, n_neighbors=5)
X_train_ada, y_train_ada = adasyn.fit_resample(X_train, y_train)

print(f"After ADASYN oversampling:")
print(f"  Train: {X_train_ada.shape[0]} rows (pos: {y_train_ada.sum()}, neg: {(y_train_ada==0).sum()})")
print(f"  Ratio: {(y_train_ada==0).sum() / y_train_ada.sum():.1f}:1 (neg:pos)")
print(f"  Val/Test: unchanged (original distribution)")

X_train = X_train_ada
y_train = y_train_ada

# ---------------------------------------------------------------------------
# Step 9: Train XGBoost
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 9: Training XGBoost")
print("=" * 60)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    early_stopping_rounds=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
    enable_categorical=False,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# ---------------------------------------------------------------------------
# Step 10: Threshold tuning (val) + Evaluation (test)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 10: Threshold tuning (val) + Evaluation (test)")
print("=" * 60)

# Tune threshold on validation set
y_val_proba = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.95, 0.01)
best_f2_val = 0
best_thresh = 0.5
for t in thresholds:
    y_pred_t = (y_val_proba >= t).astype(int)
    f2 = fbeta_score(y_val, y_pred_t, beta=2, zero_division=0)
    if f2 > best_f2_val:
        best_f2_val = f2
        best_thresh = t

print(f"Best threshold (from val): {best_thresh:.2f} (val F2={best_f2_val:.4f})")

# Final evaluation on test set
y_test_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_test_proba)

precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc = auc(recall_arr, precision_arr)

y_test_pred = (y_test_proba >= best_thresh).astype(int)
f2_score_val = fbeta_score(y_test, y_test_pred, beta=2)

print(f"\n{'Metric':<20} {'Value':<10}")
print("-" * 30)
print(f"{'ROC-AUC':<20} {roc_auc:.4f}")
print(f"{'PR-AUC':<20} {pr_auc:.4f}")
print(f"{'F2-Score':<20} {f2_score_val:.4f}")
print(f"{'Threshold (F2)':<20} {best_thresh:.2f}")

print(f"\nClassification Report (threshold={best_thresh:.2f}):")
print(classification_report(y_test, y_test_pred, target_names=["No diabetes", "Develops diabetes"]))

# Top 30 feature importances
print("\nTop 30 XGBoost Feature Importances (gain):")
importances = model.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
for feat, imp in feat_imp[:30]:
    print(f"  {feat:<30} {imp:.4f}")

print("\nDone.")
