"""Analyze neural network experiment results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent


def load_all_results(results_dir: Path) -> list[dict]:
    """Load all experiment results from JSON files."""
    results = []
    
    for json_file in results_dir.glob("*.json"):
        if json_file.name == "summary.json":
            continue
        
        try:
            with open(json_file) as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    return results


def analyze_by_architecture(results: list[dict]) -> pd.DataFrame:
    """Analyze results by architecture type."""
    rows = []
    
    for result in results:
        if "aggregate_metrics" not in result:
            continue
        
        arch_type = result["architecture"]["type"]
        arch_name = result["architecture"]["name"]
        metrics = result["aggregate_metrics"]
        
        rows.append({
            "architecture": arch_type,
            "name": arch_name,
            "mean_roc_auc": metrics["mean_roc_auc"],
            "mean_pr_auc": metrics["mean_pr_auc"],
            "mean_f2_score": metrics["mean_f2_score"],
            "std_roc_auc": metrics["std_roc_auc"],
            "epochs_trained": result["epochs_trained"],
            "training_time_sec": result["training_time_sec"],
        })
    
    df = pd.DataFrame(rows)
    
    # Aggregate by architecture type
    grouped = df.groupby("architecture").agg({
        "mean_roc_auc": ["mean", "std", "max"],
        "mean_pr_auc": ["mean", "std", "max"],
        "mean_f2_score": ["mean", "std", "max"],
        "training_time_sec": "mean",
    }).round(4)
    
    return grouped


def analyze_by_hyperparams(results: list[dict]) -> pd.DataFrame:
    """Analyze impact of key hyperparameters."""
    rows = []
    
    for result in results:
        if "aggregate_metrics" not in result:
            continue
        
        reg = result["regularization"]
        opt = result["optimization"]
        imb = result["imbalance"]
        metrics = result["aggregate_metrics"]
        
        rows.append({
            "dropout": reg["dropout"],
            "weight_decay": reg["weight_decay"],
            "batch_norm": reg["batch_norm"],
            "learning_rate": opt["lr"],
            "batch_size": opt["batch_size"],
            "optimizer": opt["optimizer"],
            "loss_type": imb["loss"],
            "use_smote": imb["use_smote"],
            "mean_roc_auc": metrics["mean_roc_auc"],
            "mean_pr_auc": metrics["mean_pr_auc"],
            "mean_f2_score": metrics["mean_f2_score"],
        })
    
    df = pd.DataFrame(rows)
    return df


def find_best_models(results: list[dict], top_k: int = 10) -> pd.DataFrame:
    """Find top performing models."""
    rows = []
    
    for result in results:
        if "aggregate_metrics" not in result:
            continue
        
        metrics = result["aggregate_metrics"]
        
        rows.append({
            "experiment_name": result["experiment_name"],
            "architecture": result["architecture"]["name"],
            "mean_roc_auc": metrics["mean_roc_auc"],
            "mean_pr_auc": metrics["mean_pr_auc"],
            "mean_f2_score": metrics["mean_f2_score"],
            "std_roc_auc": metrics["std_roc_auc"],
            "epochs": result["epochs_trained"],
            "time_sec": result["training_time_sec"],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_roc_auc", ascending=False).head(top_k)
    
    return df


def analyze_per_task_performance(results: list[dict]) -> pd.DataFrame:
    """Analyze performance by disease."""
    from utils import DISEASE_SUFFIXES
    
    task_data = {suffix: [] for suffix in DISEASE_SUFFIXES.values()}
    
    for result in results:
        if "per_task_metrics" not in result:
            continue
        
        for task, metrics in result["per_task_metrics"].items():
            task_data[task].append({
                "experiment": result["experiment_name"],
                "roc_auc": metrics["ROC_AUC"],
                "pr_auc": metrics["PR_AUC"],
                "f2_score": metrics["F2_score"],
            })
    
    # Find best model per task
    summary = []
    for task, exps in task_data.items():
        if not exps:
            continue
        
        best = max(exps, key=lambda x: x["roc_auc"])
        mean_roc = np.mean([e["roc_auc"] for e in exps])
        std_roc = np.std([e["roc_auc"] for e in exps])
        
        summary.append({
            "task": task,
            "best_roc_auc": best["roc_auc"],
            "best_experiment": best["experiment"],
            "mean_roc_auc": mean_roc,
            "std_roc_auc": std_roc,
        })
    
    df = pd.DataFrame(summary)
    df = df.sort_values("best_roc_auc", ascending=False)
    
    return df


def main():
    """Run analysis on neural network results."""
    print("\n" + "=" * 70)
    print("NEURAL NETWORK EXPERIMENT ANALYSIS")
    print("=" * 70)
    
    results_dir = REPO_DIR / "results" / "neural_sweep"
    
    if not results_dir.exists():
        print(f"\nERROR: Results directory not found: {results_dir}")
        print("Run experiments first with run_neural_sweep.py")
        sys.exit(1)
    
    print(f"\nLoading results from: {results_dir}")
    results = load_all_results(results_dir)
    print(f"Loaded {len(results)} experiment results")
    
    if not results:
        print("No results to analyze")
        sys.exit(1)
    
    # Analysis 1: Best models
    print("\n" + "=" * 70)
    print("TOP 10 MODELS BY MEAN ROC-AUC")
    print("=" * 70)
    best_models = find_best_models(results, top_k=10)
    print(best_models.to_string(index=False))
    
    # Analysis 2: Architecture comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE BY ARCHITECTURE TYPE")
    print("=" * 70)
    arch_analysis = analyze_by_architecture(results)
    print(arch_analysis)
    
    # Analysis 3: Per-task performance
    print("\n" + "=" * 70)
    print("PERFORMANCE BY DISEASE (TASK)")
    print("=" * 70)
    task_analysis = analyze_per_task_performance(results)
    print(task_analysis.to_string(index=False))
    
    # Analysis 4: Hyperparameter impact
    print("\n" + "=" * 70)
    print("HYPERPARAMETER ANALYSIS")
    print("=" * 70)
    hyperparam_df = analyze_by_hyperparams(results)
    
    # Dropout impact
    if len(hyperparam_df["dropout"].unique()) > 1:
        dropout_impact = hyperparam_df.groupby("dropout")[
            "mean_roc_auc"
        ].agg(["mean", "std", "count"])
        print("\nDropout Impact:")
        print(dropout_impact)
    
    # Optimizer impact
    if len(hyperparam_df["optimizer"].unique()) > 1:
        opt_impact = hyperparam_df.groupby("optimizer")[
            "mean_roc_auc"
        ].agg(["mean", "std", "count"])
        print("\nOptimizer Impact:")
        print(opt_impact)
    
    # Loss type impact
    if len(hyperparam_df["loss_type"].unique()) > 1:
        loss_impact = hyperparam_df.groupby("loss_type")[
            "mean_roc_auc"
        ].agg(["mean", "std", "count"])
        print("\nLoss Type Impact:")
        print(loss_impact)
    
    # SMOTE impact
    if len(hyperparam_df["use_smote"].unique()) > 1:
        smote_impact = hyperparam_df.groupby("use_smote")[
            "mean_roc_auc"
        ].agg(["mean", "std", "count"])
        print("\nSMOTE Impact:")
        print(smote_impact)
    
    # Save analysis
    output_path = results_dir / "analysis.txt"
    with open(output_path, "w") as f:
        f.write("NEURAL NETWORK EXPERIMENT ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("TOP 10 MODELS\n")
        f.write("-" * 70 + "\n")
        f.write(best_models.to_string(index=False))
        f.write("\n\n")
        
        f.write("ARCHITECTURE COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(str(arch_analysis))
        f.write("\n\n")
        
        f.write("PER-TASK PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(task_analysis.to_string(index=False))
        f.write("\n")
    
    print(f"\nAnalysis saved to: {output_path}")
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
