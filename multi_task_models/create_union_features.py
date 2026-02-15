"""
Create union of top features from all available disease importances.

Strategy:
1. Use diabetes single-task importances (most comprehensive) as base
2. Take top 100 features from diabetes
3. This gives us a good starting feature set for multi-task learning

Run from repo root:
  python multi_task_models/create_union_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_DIR / "model"
DATA_DIR = REPO_DIR / "data"


def find_latest_importances(disease: str) -> Path | None:
    """Find latest single-task importances file for a disease."""
    candidates = sorted(
        MODEL_DIR.glob(f"{disease}_*_importances.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    # Filter out multi-task files
    candidates = [c for c in candidates if not c.name.startswith("multi_")]
    # Prefer larger files (more features)
    if candidates:
        return max(candidates, key=lambda x: x.stat().st_size)
    return None


def get_top_features(
    importances_path: Path, n: int, min_importance: float | None = None
) -> list[str]:
    """Get top N features from importances file."""
    df = pd.read_csv(importances_path)
    df = df.sort_values("importance", ascending=False)
    
    if min_importance is not None:
        df = df[df["importance"] >= min_importance]
    
    return df.head(n)["feature"].tolist()


def main() -> None:
    """Create union feature set."""
    print("=" * 70)
    print("Creating Union Feature Set for Multi-Task Learning")
    print("=" * 70)
    
    # Find available single-task importances
    diseases = ["diabetes", "hibpe", "cancr", "lunge", "hearte", 
                "stroke", "psyche", "arthre"]
    
    available_importances = {}
    for disease in diseases:
        imp_path = find_latest_importances(disease)
        if imp_path:
            available_importances[disease] = imp_path
            print(f"  Found: {disease} -> {imp_path.name}")
    
    if not available_importances:
        print("\nERROR: No single-task importances found!")
        print("Please run single-task models first:")
        print("  python scripts/run_disease.py diabetes")
        sys.exit(1)
    
    print(f"\nFound importances for {len(available_importances)} disease(s)")
    
    # Strategy: Use the most comprehensive importances file as base
    # (Likely diabetes with 821 features)
    base_disease = max(
        available_importances.items(),
        key=lambda x: x[1].stat().st_size
    )[0]
    base_path = available_importances[base_disease]
    
    print(f"\nUsing {base_disease} as base (largest file)")
    
    # Get top features from base disease
    top_n = 100  # Reasonable compromise between 9 and 821
    features = get_top_features(base_path, n=top_n)
    
    print(f"  Selected top {len(features)} features from {base_disease}")
    
    # If we have importances for other diseases, add their top features too
    if len(available_importances) > 1:
        print("\nAdding top features from other diseases...")
        feature_set = set(features)
        
        for disease, imp_path in available_importances.items():
            if disease == base_disease:
                continue
            
            top_disease = get_top_features(imp_path, n=30)
            new_features = [f for f in top_disease if f not in feature_set]
            feature_set.update(new_features)
            print(f"  {disease}: added {len(new_features)} new features")
        
        features = sorted(feature_set)
        print(f"\nTotal union: {len(features)} features")
    
    # Save to file
    output_path = MODEL_DIR / "union_features.txt"
    with open(output_path, "w") as f:
        for feature in features:
            f.write(f"{feature}\n")
    
    print(f"\nSaved: {output_path}")
    print(f"  {len(features)} features")
    
    # Also save a CSV with details
    # Load all importances and average them
    feature_importances = {}
    for disease, imp_path in available_importances.items():
        df = pd.read_csv(imp_path)
        for _, row in df.iterrows():
            feat = row["feature"]
            if feat in features:
                if feat not in feature_importances:
                    feature_importances[feat] = []
                feature_importances[feat].append(row["importance"])
    
    # Create summary
    summary = []
    for feat in features:
        if feat in feature_importances:
            imps = feature_importances[feat]
            summary.append({
                "feature": feat,
                "mean_importance": sum(imps) / len(imps),
                "n_diseases": len(imps),
            })
        else:
            summary.append({
                "feature": feat,
                "mean_importance": 0,
                "n_diseases": 0,
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values("mean_importance", ascending=False)
    
    summary_path = MODEL_DIR / "union_features_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path}")
    
    print("\n" + "=" * 70)
    print("Usage:")
    print("=" * 70)
    print("  # Use this feature set for multi-task training:")
    print("  FEATURE_LIST=model/union_features.txt \\")
    print("    python multi_task_models/run_multi_xgboost.py")
    print("\n  # Or with MIN_IMPORTANCE threshold:")
    print("  MIN_IMPORTANCE=5 python multi_task_models/run_multi_xgboost.py")


if __name__ == "__main__":
    main()
