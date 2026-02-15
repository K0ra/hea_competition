"""
Compare multi-task vs single-task model performance across all model types.

Queries MLflow to find latest runs for each approach (tree models and
neural networks) and creates comparison tables.

Run from repo root:
  python multi_task_models/compare_results.py
  python multi_task_models/compare_results.py --include-neural
  python multi_task_models/compare_results.py --neural-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "multi_task_models"))

from utils import DISEASE_SUFFIXES

# Map disease suffixes to single-task experiment names
SINGLE_TASK_EXPERIMENTS = {
    "DIABE": "hea_deep_diabetes",
    "HIBPE": "hea_deep_hibpe",
    "CANCR": "hea_deep_cancr",
    "LUNGE": "hea_deep_lunge",
    "HEARTE": "hea_deep_hearte",
    "STROKE": "hea_deep_stroke",
    "PSYCHE": "hea_deep_psyche",
    "ARTHRE": "hea_deep_arthre",
}

MULTITASK_EXPERIMENT = "hea_multitask"
NEURAL_MULTITASK_EXPERIMENT = "hea_neural_multitask"


def get_latest_multitask_run() -> dict | None:
    """Get latest multi-task run from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MULTITASK_EXPERIMENT)
        if experiment is None:
            return None
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        
        if not runs:
            return None
        
        run = runs[0]
        metrics = run.data.metrics
        params = run.data.params
        
        # Extract per-disease metrics
        results = {}
        for suffix in DISEASE_SUFFIXES.values():
            roc_key = f"{suffix}_ROC_AUC"
            pr_key = f"{suffix}_PR_AUC"
            f2_key = f"{suffix}_F2"
            
            if roc_key in metrics:
                results[suffix] = {
                    "ROC_AUC": metrics[roc_key],
                    "PR_AUC": metrics[pr_key],
                    "F2_score": metrics[f2_key],
                }
        
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "model_type": params.get("model_type", "unknown"),
            "n_features": int(params.get("n_features", 0)),
            "results": results,
            "mean_ROC_AUC": metrics.get("mean_ROC_AUC", 0.0),
            "mean_PR_AUC": metrics.get("mean_PR_AUC", 0.0),
            "mean_F2": metrics.get("mean_F2", 0.0),
        }
    except Exception as e:
        print(f"Error getting multi-task run: {e}", file=sys.stderr)
        return None


def get_single_task_runs() -> dict[str, dict]:
    """Get latest single-task run for each disease from MLflow."""
    client = mlflow.tracking.MlflowClient()
    results = {}
    
    for suffix, exp_name in SINGLE_TASK_EXPERIMENTS.items():
        try:
            experiment = client.get_experiment_by_name(exp_name)
            if experiment is None:
                continue
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            
            if not runs:
                continue
            
            run = runs[0]
            metrics = run.data.metrics
            
            results[suffix] = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "ROC_AUC": metrics.get("ROC_AUC", 0.0),
                "PR_AUC": metrics.get("PR_AUC", 0.0),
                "F2_score": metrics.get("F2_score", 0.0),
            }
        except Exception as e:
            print(
                f"Warning: Could not get single-task run for {suffix}: {e}",
                file=sys.stderr,
            )
    
    return results


def get_best_neural_multitask_run() -> dict | None:
    """Get best neural multi-task run (by mean ROC-AUC) from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(
            NEURAL_MULTITASK_EXPERIMENT
        )
        if experiment is None:
            return None
        
        # Get all runs and find best by mean_roc_auc
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.mean_roc_auc DESC"],
            max_results=1,
        )
        
        if not runs:
            return None
        
        run = runs[0]
        metrics = run.data.metrics
        params = run.data.params
        
        # Extract per-disease metrics
        results = {}
        for suffix in DISEASE_SUFFIXES.values():
            roc_key = f"{suffix}_roc_auc"
            pr_key = f"{suffix}_pr_auc"
            f2_key = f"{suffix}_f2_score"
            
            if roc_key in metrics:
                results[suffix] = {
                    "ROC_AUC": metrics[roc_key],
                    "PR_AUC": metrics[pr_key],
                    "F2_score": metrics[f2_key],
                }
        
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "architecture": params.get("architecture", "unknown"),
            "n_features": int(params.get("n_features", 0)),
            "n_params": int(params.get("n_params", 0)),
            "results": results,
            "mean_roc_auc": metrics.get("mean_roc_auc", 0.0),
            "mean_pr_auc": metrics.get("mean_pr_auc", 0.0),
            "mean_f2_score": metrics.get("mean_f2_score", 0.0),
        }
    except Exception as e:
        print(f"Error getting neural multi-task run: {e}", file=sys.stderr)
        return None


def main() -> None:
    """Compare multi-task vs single-task results."""
    parser = argparse.ArgumentParser(
        description="Compare multi-task vs single-task model performance"
    )
    parser.add_argument(
        "--multitask_run",
        type=str,
        help="Specific multi-task run ID (default: latest)",
    )
    parser.add_argument(
        "--include-neural",
        action="store_true",
        help="Include neural network results in comparison",
    )
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="Only compare neural networks (skip tree models)",
    )
    args = parser.parse_args()
    
    print("=" * 90)
    print("Multi-Task vs Single-Task Comparison")
    print("=" * 90)
    
    # Determine which comparisons to show
    show_tree = not args.neural_only
    show_neural = args.include_neural or args.neural_only
    
    multitask = None
    neural_multitask = None
    single_task = {}
    
    # Get tree model results
    if show_tree:
        print("\n[TREE MODELS]")
        print("Fetching multi-task tree model results...")
        multitask = get_latest_multitask_run()
        if multitask is None:
            print(
                "WARNING: No tree multi-task runs found. "
                "Run multi_task_models/run_multi_xgboost.py first.",
                file=sys.stderr,
            )
        else:
            print(f"  Latest run: {multitask['run_name']}")
            print(f"  Model: {multitask['model_type']}")
            print(f"  Features: {multitask['n_features']}")
        
        # Get single-task results
        print("\nFetching single-task tree model results...")
        single_task = get_single_task_runs()
        if not single_task:
            print(
                "WARNING: No single-task runs found. "
                "Run deep_models/run_xgboost.py for each disease first.",
                file=sys.stderr,
            )
        else:
            print(f"  Found {len(single_task)} single-task runs")
    
    # Get neural network results
    if show_neural:
        print("\n[NEURAL NETWORKS]")
        print("Fetching best neural multi-task results...")
        neural_multitask = get_best_neural_multitask_run()
        if neural_multitask is None:
            print(
                "WARNING: No neural multi-task runs found. "
                "Run multi_task_models/run_neural_sweep.py first.",
                file=sys.stderr,
            )
        else:
            print(f"  Best run: {neural_multitask['run_name']}")
            print(f"  Architecture: {neural_multitask['architecture']}")
            print(f"  Features: {neural_multitask['n_features']}")
            print(
                f"  Parameters: {neural_multitask['n_params']:,}"
            )
            print(
                f"  Mean ROC-AUC: {neural_multitask['mean_roc_auc']:.4f}"
            )
    
    # Build comparison table(s)
    
    # Tree models comparison
    if show_tree and multitask is not None:
        print("\n" + "=" * 90)
        print("Tree Models: Multi-Task vs Single-Task")
        print("=" * 90)
        
        rows = []
        for suffix in DISEASE_SUFFIXES.values():
            multi_results = multitask["results"].get(suffix)
            single_results = single_task.get(suffix)
            
            if multi_results is None:
                continue
            
            row = {"Disease": suffix}
            
            # Single-task metrics
            if single_results:
                row["Single_ROC"] = single_results["ROC_AUC"]
                row["Single_PR"] = single_results["PR_AUC"]
            else:
                row["Single_ROC"] = None
                row["Single_PR"] = None
            
            # Multi-task metrics
            row["Multi_ROC"] = multi_results["ROC_AUC"]
            row["Multi_PR"] = multi_results["PR_AUC"]
            
            # Improvement
            if single_results:
                roc_diff = (
                    multi_results["ROC_AUC"] - single_results["ROC_AUC"]
                )
                pr_diff = (
                    multi_results["PR_AUC"] - single_results["PR_AUC"]
                )
                row["ROC_Diff"] = roc_diff
                row["PR_Diff"] = pr_diff
                row["ROC_Pct"] = (
                    (roc_diff / single_results["ROC_AUC"] * 100)
                    if single_results["ROC_AUC"] > 0
                    else 0
                )
                row["PR_Pct"] = (
                    (pr_diff / single_results["PR_AUC"] * 100)
                    if single_results["PR_AUC"] > 0
                    else 0
                )
            else:
                row["ROC_Diff"] = None
                row["PR_Diff"] = None
                row["ROC_Pct"] = None
                row["PR_Pct"] = None
            
            rows.append(row)
    
        df = pd.DataFrame(rows)
        
        # Print table
        print(
            f"\n{'Disease':<10} | {'Single-Task':<20} | "
            f"{'Multi-Task':<20} | {'Improvement':<20}"
        )
        print(
            f"{'':10} | {'ROC':<9} {'PR':<9} | "
            f"{'ROC':<9} {'PR':<9} | {'ROC %':<9} {'PR %':<9}"
        )
        print("-" * 90)
        
        for _, row in df.iterrows():
            single_roc = (
                f"{row['Single_ROC']:.4f}"
                if row["Single_ROC"] is not None
                else "N/A      "
            )
            single_pr = (
                f"{row['Single_PR']:.4f}"
                if row["Single_PR"] is not None
                else "N/A      "
            )
            multi_roc = f"{row['Multi_ROC']:.4f}"
            multi_pr = f"{row['Multi_PR']:.4f}"
            roc_pct = (
                f"{row['ROC_Pct']:+.1f}%    "
                if row["ROC_Pct"] is not None
                else "N/A      "
            )
            pr_pct = (
                f"{row['PR_Pct']:+.1f}%    "
                if row["PR_Pct"] is not None
                else "N/A      "
            )
            
            print(
                f"{row['Disease']:<10} | {single_roc} {single_pr} | "
                f"{multi_roc} {multi_pr} | {roc_pct} {pr_pct}"
            )
        
        # Summary statistics
        if df["ROC_Pct"].notna().any():
            mean_roc_improvement = df["ROC_Pct"].mean()
            mean_pr_improvement = df["PR_Pct"].mean()
            
            print("-" * 90)
            print(
                f"{'MEAN':<10} | {'':<20} | {'':<20} | "
                f"{mean_roc_improvement:+.1f}%     "
                f"{mean_pr_improvement:+.1f}%"
            )
            
            # Count improvements
            roc_better = (df["ROC_Diff"] > 0).sum()
            pr_better = (df["PR_Diff"] > 0).sum()
            total = len(df[df["ROC_Diff"].notna()])
            
            print("\n" + "=" * 90)
            print("Tree Models Summary")
            print("=" * 90)
            print(f"Diseases where multi-task is better:")
            print(f"  ROC-AUC: {roc_better}/{total}")
            print(f"  PR-AUC: {pr_better}/{total}")
            print(f"\nMean improvement:")
            print(f"  ROC-AUC: {mean_roc_improvement:+.2f}%")
            print(f"  PR-AUC: {mean_pr_improvement:+.2f}%")
    
    # Neural networks comparison
    if show_neural and neural_multitask is not None:
        print("\n" + "=" * 90)
        print("Neural Networks: Best Multi-Task Architecture")
        print("=" * 90)
        
        print(
            f"\n{'Disease':<10} | {'Neural Multi-Task':<20} | "
            f"{'vs Tree Multi-Task':<20}"
        )
        print(
            f"{'':10} | {'ROC':<9} {'PR':<9} | "
            f"{'ROC Diff':<9} {'PR Diff':<9}"
        )
        print("-" * 90)
        
        neural_rows = []
        for suffix in DISEASE_SUFFIXES.values():
            neural_results = neural_multitask["results"].get(suffix)
            if neural_results is None:
                continue
            
            row = {
                "Disease": suffix,
                "Neural_ROC": neural_results["ROC_AUC"],
                "Neural_PR": neural_results["PR_AUC"],
            }
            
            # Compare to tree multi-task if available
            if multitask and suffix in multitask["results"]:
                tree_results = multitask["results"][suffix]
                row["Tree_ROC"] = tree_results["ROC_AUC"]
                row["Tree_PR"] = tree_results["PR_AUC"]
                row["ROC_Diff"] = (
                    neural_results["ROC_AUC"] - tree_results["ROC_AUC"]
                )
                row["PR_Diff"] = (
                    neural_results["PR_AUC"] - tree_results["PR_AUC"]
                )
            else:
                row["Tree_ROC"] = None
                row["Tree_PR"] = None
                row["ROC_Diff"] = None
                row["PR_Diff"] = None
            
            neural_rows.append(row)
        
        neural_df = pd.DataFrame(neural_rows)
        
        for _, row in neural_df.iterrows():
            neural_roc = f"{row['Neural_ROC']:.4f}"
            neural_pr = f"{row['Neural_PR']:.4f}"
            
            if row["ROC_Diff"] is not None:
                roc_diff_str = f"{row['ROC_Diff']:+.4f}"
                pr_diff_str = f"{row['PR_Diff']:+.4f}"
            else:
                roc_diff_str = "N/A      "
                pr_diff_str = "N/A      "
            
            print(
                f"{row['Disease']:<10} | {neural_roc} {neural_pr} | "
                f"{roc_diff_str} {pr_diff_str}"
            )
        
        # Neural summary
        print("\n" + "=" * 90)
        print("Neural Networks Summary")
        print("=" * 90)
        print(
            f"Mean ROC-AUC: {neural_multitask['mean_roc_auc']:.4f}"
        )
        print(
            f"Mean PR-AUC: {neural_multitask['mean_pr_auc']:.4f}"
        )
        
        if multitask is not None:
            neural_vs_tree_roc = (
                neural_multitask['mean_roc_auc'] -
                multitask['mean_ROC_AUC']
            )
            neural_vs_tree_pr = (
                neural_multitask['mean_pr_auc'] -
                multitask['mean_PR_AUC']
            )
            print(
                f"\nNeural vs Tree Multi-Task:"
            )
            print(
                f"  ROC-AUC: {neural_vs_tree_roc:+.4f}"
            )
            print(
                f"  PR-AUC: {neural_vs_tree_pr:+.4f}"
            )
    
    print("\n" + "=" * 90)
    print("MLflow Run IDs")
    print("=" * 90)
    
    if multitask:
        print(f"Tree Multi-task: {multitask['run_id']}")
    if neural_multitask:
        print(
            f"Neural Multi-task: {neural_multitask['run_id']} "
            f"({neural_multitask['architecture']})"
        )
    if single_task:
        print("\nSingle-task (tree):")
        for suffix, results in single_task.items():
            print(f"  {suffix:10}: {results['run_id']}")


if __name__ == "__main__":
    main()
