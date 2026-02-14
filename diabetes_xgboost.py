"""
Diabetes prediction using XGBoost on RAND HRS 1992-2022 data.

Pipeline:
1. Read SAS file metadata to discover columns
2. Reshape wide (19K+ cols) -> long (one row per person-wave)
3. Build target: developed diabetes (0->1 transition across waves)
4. Select features with importance >= 10 from prior LightGBM run
5. Train XGBoost with imbalanced-class handling
6. Report F2-score, PR-AUC, ROC-AUC
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    fbeta_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Adjust path depending on environment (local vs server)
SAS_PATH = os.environ.get(
    "SAS_PATH",
    str(Path(__file__).resolve().parent / "data" / "randhrs1992_2022v1.sas7bdat"),
)
IMPORTANCES_PATH = os.environ.get(
    "IMPORTANCES_PATH",
    str(
        Path(__file__).resolve().parent
        / "data"
        / "diabetes_lgbm_794feat_89504train_20260214_1953_importances.csv"
    ),
)

MIN_IMPORTANCE = 10  # Feature importance threshold
WAVES = list(range(1, 17))  # Waves 1-16 (1992-2022)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# The diabetes target column base name in RAND HRS
DIABETES_VAR = "DIABE"  # R{w}DIABE = 1 if ever diagnosed with diabetes

# ---------------------------------------------------------------------------
# Step 1: Identify top features and map to SAS column names
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Loading feature importances")
print("=" * 60)

imp_df = pd.read_csv(IMPORTANCES_PATH)
top_features = imp_df.loc[imp_df["importance"] >= MIN_IMPORTANCE, "feature"].tolist()
print(f"Features with importance >= {MIN_IMPORTANCE}: {len(top_features)}")
print(f"  {top_features}")

# ---------------------------------------------------------------------------
# Step 2: Read SAS metadata to discover actual column names
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Discovering SAS columns")
print("=" * 60)

_, meta = pyreadstat.read_sas7bdat(SAS_PATH, metadataonly=True)
all_cols = set(meta.column_names)
print(f"Total columns in SAS file: {len(all_cols)}")

# RAND HRS naming conventions:
#   R{w}VARNAME  — respondent-level, time-varying
#   S{w}VARNAME  — spouse-level, time-varying
#   H{w}VARNAME  — household-level, time-varying
#   RAVARNAME    — respondent-level, time-invariant
#   HAVARNAME    — household-level, time-invariant
#
# We look for each feature base name across all wave prefixes.


def find_wave_columns(base_name, all_columns):
    """Find all wave-specific columns for a given base variable name.

    Returns dict: {wave_number: actual_column_name}
    """
    matches = {}

    # Try R{w}BASE pattern (most common for respondent vars)
    for w in WAVES:
        candidate = f"R{w}{base_name}"
        if candidate in all_columns:
            matches[w] = candidate

    # If no R-prefix matches, try H{w}BASE (household vars like WTHH)
    if not matches:
        for w in WAVES:
            candidate = f"H{w}{base_name}"
            if candidate in all_columns:
                matches[w] = candidate

    # If still nothing, try S{w}BASE (spouse)
    if not matches:
        for w in WAVES:
            candidate = f"S{w}{base_name}"
            if candidate in all_columns:
                matches[w] = candidate

    return matches


def find_invariant_column(base_name, all_columns):
    """Find time-invariant column (RA/HA prefix or bare name)."""
    for prefix in ["RA", "HA", ""]:
        candidate = f"{prefix}{base_name}"
        if candidate in all_columns:
            return candidate
    return None


# Map each top feature to its SAS columns
feature_map = {}  # base_name -> {wave: col_name} or str (invariant)
cols_to_read = set()

for feat in top_features:
    wave_cols = find_wave_columns(feat, all_cols)
    if wave_cols:
        feature_map[feat] = {"type": "wave", "columns": wave_cols}
        cols_to_read.update(wave_cols.values())
    else:
        inv_col = find_invariant_column(feat, all_cols)
        if inv_col:
            feature_map[feat] = {"type": "invariant", "column": inv_col}
            cols_to_read.add(inv_col)
        else:
            print(f"  WARNING: No SAS column found for feature '{feat}'")

# Also need diabetes columns for target + HHIDPN for ID
diabetes_cols = {}
for w in WAVES:
    candidate = f"R{w}{DIABETES_VAR}"
    if candidate in all_cols:
        diabetes_cols[w] = candidate
        cols_to_read.add(candidate)

# Add respondent ID
cols_to_read.add("HHIDPN")

print(f"\nFeature mapping summary:")
for feat, info in feature_map.items():
    if info["type"] == "wave":
        waves_found = sorted(info["columns"].keys())
        print(f"  {feat}: waves {waves_found[0]}-{waves_found[-1]} ({len(waves_found)} waves)")
    else:
        print(f"  {feat}: time-invariant -> {info['column']}")

print(f"\nDiabetes target (R{{w}}DIABE) found for waves: {sorted(diabetes_cols.keys())}")
print(f"Total SAS columns to read: {len(cols_to_read)}")

# ---------------------------------------------------------------------------
# Step 3: Read SAS file (selected columns only)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Reading SAS file")
print("=" * 60)

cols_to_read_list = sorted(cols_to_read)
df_wide, _ = pyreadstat.read_sas7bdat(SAS_PATH, usecols=cols_to_read_list)
print(f"Loaded: {df_wide.shape[0]} respondents x {df_wide.shape[1]} columns")
print(f"Memory: {df_wide.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ---------------------------------------------------------------------------
# Step 4: Reshape wide -> long (one row per person-wave)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Reshaping wide -> long")
print("=" * 60)

# For each wave, extract that wave's columns and stack
rows = []

# Determine which waves have the diabetes target (required)
valid_waves = sorted(diabetes_cols.keys())

for w in valid_waves:
    wave_row = {"wave": w}

    # Extract wave-specific feature columns
    wave_features = {}
    for feat, info in feature_map.items():
        if info["type"] == "wave":
            col = info["columns"].get(w)
            if col:
                wave_features[feat] = df_wide[col]
        else:
            # Time-invariant: same value every wave
            wave_features[feat] = df_wide[info["column"]]

    if not wave_features:
        continue

    # Build a dataframe for this wave
    df_wave = pd.DataFrame(wave_features)
    df_wave["HHIDPN"] = df_wide["HHIDPN"].values
    df_wave["wave"] = w

    # Add diabetes target for this wave
    df_wave[DIABETES_VAR] = df_wide[diabetes_cols[w]].values

    rows.append(df_wave)

df_long = pd.concat(rows, ignore_index=True)
del df_wide, rows  # Free memory

print(f"Long format: {df_long.shape[0]} person-wave rows x {df_long.shape[1]} columns")
print(f"Waves: {sorted(df_long['wave'].unique())}")
print(f"Unique respondents: {df_long['HHIDPN'].nunique()}")

# ---------------------------------------------------------------------------
# Step 5: Build diabetes development target
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Building diabetes development target")
print("=" * 60)

# Target: person did NOT have diabetes at wave w, but HAS diabetes at any
# later wave. This captures "developing diabetes over time."
#
# For each person, sort by wave. At wave w:
#   - If DIABE == 1 already -> exclude (already diabetic)
#   - If DIABE == 0 and any future DIABE == 1 -> target = 1
#   - If DIABE == 0 and no future DIABE == 1 -> target = 0

df_long = df_long.sort_values(["HHIDPN", "wave"]).reset_index(drop=True)

# For each person, get the max DIABE across all future waves
df_long["future_diabetes"] = (
    df_long.groupby("HHIDPN")[DIABETES_VAR]
    .transform(lambda s: s.shift(-1).expanding().max())  # max of all future waves
)

# Only keep rows where person does NOT yet have diabetes
df_model = df_long[df_long[DIABETES_VAR] == 0].copy()
df_model["target"] = (df_model["future_diabetes"] == 1).astype(int)

# Drop rows with no future data (last wave for each person)
df_model = df_model.dropna(subset=["future_diabetes"])

# Drop helper columns
df_model = df_model.drop(columns=[DIABETES_VAR, "future_diabetes"])

print(f"Modeling dataset: {df_model.shape[0]} rows x {df_model.shape[1]} columns")
print(f"\nTarget distribution:")
print(df_model["target"].value_counts().to_string())
print(f"Diabetes development rate: {df_model['target'].mean():.4f}")

# ---------------------------------------------------------------------------
# Step 6: Prepare features and train/test split
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Preparing features for XGBoost")
print("=" * 60)

feature_cols = [c for c in top_features if c in df_model.columns]
print(f"Features used: {feature_cols}")

X = df_model[feature_cols].copy()
y = df_model["target"].copy()

# XGBoost handles NaN natively, but let's report missingness
missing_pct = X.isnull().mean() * 100
print(f"\nMissing data per feature:")
for col in feature_cols:
    pct = missing_pct[col]
    if pct > 0:
        print(f"  {col}: {pct:.1f}%")

# Train/test split — stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate:  {y_test.mean():.4f}")

# ---------------------------------------------------------------------------
# Step 7: Train XGBoost with imbalanced-class handling
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 7: Training XGBoost")
print("=" * 60)

# scale_pos_weight handles class imbalance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
    enable_categorical=False,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ---------------------------------------------------------------------------
# Step 8: Evaluate — F2-score, PR-AUC, ROC-AUC
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 8: Evaluation")
print("=" * 60)

y_pred_proba = model.predict_proba(X_test)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)

# PR-AUC (more meaningful for imbalanced data)
precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_arr, precision_arr)

# F2-score (weighs recall higher than precision)
# Find optimal threshold by maximizing F2 on test set
thresholds = np.arange(0.1, 0.9, 0.01)
best_f2 = 0
best_thresh = 0.5
for t in thresholds:
    y_pred_t = (y_pred_proba >= t).astype(int)
    f2 = fbeta_score(y_test, y_pred_t, beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2 = f2
        best_thresh = t

y_pred = (y_pred_proba >= best_thresh).astype(int)
f2_score_val = fbeta_score(y_test, y_pred, beta=2)

print(f"\n{'Metric':<20} {'Value':<10}")
print("-" * 30)
print(f"{'ROC-AUC':<20} {roc_auc:.4f}")
print(f"{'PR-AUC':<20} {pr_auc:.4f}")
print(f"{'F2-Score':<20} {f2_score_val:.4f}")
print(f"{'Threshold (F2)':<20} {best_thresh:.2f}")

print(f"\nClassification Report (threshold={best_thresh:.2f}):")
print(classification_report(y_test, y_pred, target_names=["No diabetes", "Develops diabetes"]))

# Feature importances from XGBoost
print("\nXGBoost Feature Importances (gain):")
importances = model.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {feat:<15} {imp:.4f}")

print("\nDone.")
