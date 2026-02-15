"""Run comprehensive neural network experiment sweep."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "multi_task_models"))

from losses import MultiTaskLoss
from neural_architectures import create_model
from neural_data import prepare_dataloaders
from neural_experiments_config import build_experiment_grid
from neural_trainer import MultiTaskTrainer
from utils import (
    DISEASE_SUFFIXES,
    TARGET_COLS,
    compute_per_task_weights,
    discover_importances,
    get_feature_columns,
    load_data,
    prepare_train_val_split,
)


def run_one_experiment(
    exp_config: dict,
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> dict:
    """
    Run a single neural network experiment.
    
    Args:
        exp_config: Experiment configuration dict
        df: Full dataset
        feature_cols: List of feature column names
        output_dir: Output directory for artifacts
    
    Returns:
        Dict with experiment results
    """
    exp_name = exp_config["name"]
    print(f"\n{'='*70}")
    print(f"Starting experiment: {exp_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Extract configs
    arch_config = exp_config["architecture"]
    reg_config = exp_config["regularization"]
    opt_config = exp_config["optimization"]
    imb_config = exp_config["imbalance"]
    
    # Prepare data for all tasks
    task_names = list(DISEASE_SUFFIXES.values())
    
    # Call prepare_train_val_split ONCE with all target columns
    X_train_dict, y_train_dict, X_val_dict, y_val_dict = (
        prepare_train_val_split(df, feature_cols, TARGET_COLS)
    )
    
    # Compute per-task class weights
    class_weights = compute_per_task_weights(y_train_dict)
    print("\nClass weights (scale_pos_weight):")
    for task, weight in class_weights.items():
        print(f"  {task}: {weight:.2f}")
    
    # Prepare dataloaders
    from neural_data import prepare_dataloaders
    
    try:
        train_loader, val_loaders = prepare_dataloaders(
            X_train_dict,
            y_train_dict,
            X_val_dict,
            y_val_dict,
            task_names,
            batch_size=opt_config["batch_size"],
            use_smote=imb_config["use_smote"],
        )
    except Exception as e:
        print(f"ERROR: Failed to prepare dataloaders: {e}")
        return {
            "experiment_name": exp_name,
            "status": "failed",
            "error": str(e),
        }
    
    # Get number of features (wave is excluded in prepare_train_val_split)
    n_features = len(feature_cols) - 1  # Subtract 1 for 'wave'
    
    # Create model
    model = create_model(arch_config, n_features, task_names, reg_config)
    print(f"\nModel: {arch_config['type']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    # Map task suffixes back to target columns for weights
    class_weights_mapped = {}
    for target_col, suffix in DISEASE_SUFFIXES.items():
        if target_col in class_weights:
            class_weights_mapped[suffix] = class_weights[target_col]
    
    loss_fn = MultiTaskLoss(
        task_names,
        loss_type=imb_config["loss"],
        class_weights=class_weights_mapped,
    )
    
    # Create optimizer
    if opt_config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=reg_config["weight_decay"],
        )
    elif opt_config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=reg_config["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=reg_config["weight_decay"],
            momentum=0.9,
        )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model,
        loss_fn,
        optimizer,
        scheduler,
        device=device,
        patience=15,
        use_amp=(device == "cuda"),
    )
    
    # Train
    print("\nTraining...")
    result = trainer.fit(
        train_loader,
        val_loaders,
        max_epochs=100,
        verbose=True,
    )
    
    elapsed_time = time.time() - start_time
    
    # Extract metrics
    best_metrics = result["best_metrics"]
    
    # Aggregate metrics
    roc_aucs = [m["ROC_AUC"] for m in best_metrics.values()]
    pr_aucs = [m["PR_AUC"] for m in best_metrics.values()]
    f2_scores = [m["F2_score"] for m in best_metrics.values()]
    
    aggregate = {
        "mean_roc_auc": np.mean(roc_aucs),
        "mean_pr_auc": np.mean(pr_aucs),
        "mean_f2_score": np.mean(f2_scores),
        "std_roc_auc": np.std(roc_aucs),
        "std_pr_auc": np.std(pr_aucs),
        "std_f2_score": np.std(f2_scores),
    }
    
    # Log to MLflow
    mlflow.set_experiment("hea_neural_multitask")
    
    with mlflow.start_run(run_name=exp_name):
        # Log params
        mlflow.log_param("architecture", arch_config["type"])
        mlflow.log_param("arch_config", json.dumps(arch_config))
        mlflow.log_param("dropout", reg_config["dropout"])
        mlflow.log_param("weight_decay", reg_config["weight_decay"])
        mlflow.log_param("batch_norm", reg_config["batch_norm"])
        mlflow.log_param("learning_rate", opt_config["lr"])
        mlflow.log_param("batch_size", opt_config["batch_size"])
        mlflow.log_param("optimizer", opt_config["optimizer"])
        mlflow.log_param("loss_type", imb_config["loss"])
        mlflow.log_param("use_smote", imb_config["use_smote"])
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("n_params", sum(p.numel() for p in model.parameters()))
        mlflow.log_param("device", device)
        
        # Log metrics
        mlflow.log_metric("epochs_trained", result["epochs_trained"])
        mlflow.log_metric("training_time_sec", result["total_time"])
        mlflow.log_metric("mean_roc_auc", aggregate["mean_roc_auc"])
        mlflow.log_metric("mean_pr_auc", aggregate["mean_pr_auc"])
        mlflow.log_metric("mean_f2_score", aggregate["mean_f2_score"])
        mlflow.log_metric("std_roc_auc", aggregate["std_roc_auc"])
        
        # Per-task metrics
        for task, metrics in best_metrics.items():
            mlflow.log_metric(f"{task}_roc_auc", metrics["ROC_AUC"])
            mlflow.log_metric(f"{task}_pr_auc", metrics["PR_AUC"])
            mlflow.log_metric(f"{task}_f2_score", metrics["F2_score"])
    
    # Save results to JSON
    result_dict = {
        "experiment_name": exp_name,
        "architecture": arch_config,
        "regularization": reg_config,
        "optimization": opt_config,
        "imbalance": imb_config,
        "epochs_trained": result["epochs_trained"],
        "training_time_sec": result["total_time"],
        "total_time_sec": elapsed_time,
        "aggregate_metrics": aggregate,
        "per_task_metrics": best_metrics,
    }
    
    json_path = output_dir / f"{exp_name}.json"
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    print(f"Mean ROC-AUC: {aggregate['mean_roc_auc']:.4f}")
    print(f"Total time: {elapsed_time:.1f}s")
    
    return result_dict


def main():
    """Run neural network experiment sweep."""
    print("\n" + "=" * 70)
    print("NEURAL NETWORK MULTI-TASK EXPERIMENT SWEEP")
    print("=" * 70)
    
    # Get mode from environment
    mode = os.environ.get("MODE", "quick")
    print(f"\nMode: {mode}")
    
    # Load data
    model_path = REPO_DIR / "data" / "rand_hrs_model_merged.parquet"
    print(f"Loading data from: {model_path}")
    df = load_data(model_path)
    print(f"Loaded {len(df)} samples")
    
    # Get feature columns
    feature_list_path = os.environ.get("FEATURE_LIST")
    if feature_list_path:
        feature_list_path = Path(feature_list_path)
        print(f"Using feature list: {feature_list_path}")
        feature_cols = get_feature_columns(
            df, feature_list_path=feature_list_path
        )
    else:
        # Use union features
        union_path = REPO_DIR / "model" / "union_features.txt"
        if union_path.exists():
            print(f"Using union features: {union_path}")
            feature_cols = get_feature_columns(
                df, feature_list_path=union_path
            )
        else:
            # Fallback: discover importances
            print("Discovering importance files...")
            imp_path = discover_importances(REPO_DIR / "model")
            if imp_path:
                print(f"Using importances: {imp_path}")
                feature_cols = get_feature_columns(df, importances_path=imp_path, top_n=100)
            else:
                print("WARNING: No importances found, using top 100 features")
                feature_cols = get_feature_columns(df, top_n=100)
    
    print(f"Using {len(feature_cols)} features (including wave)")
    
    # Build experiment grid
    experiments = build_experiment_grid(mode)
    print(f"\nTotal experiments: {len(experiments)}")
    
    # Output directory
    output_dir = REPO_DIR / "results" / "neural_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    all_results = []
    for i, exp_config in enumerate(experiments):
        print(f"\n{'#'*70}")
        print(f"Experiment {i+1}/{len(experiments)}")
        print(f"{'#'*70}")
        
        try:
            result = run_one_experiment(exp_config, df, feature_cols, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR in experiment {exp_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "experiment_name": exp_config["name"],
                "status": "failed",
                "error": str(e),
            })
    
    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SWEEP COMPLETED")
    print("=" * 70)
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    
    # Print top 5 by mean ROC-AUC
    successful = [r for r in all_results if "aggregate_metrics" in r]
    if successful:
        top5 = sorted(
            successful,
            key=lambda x: x["aggregate_metrics"]["mean_roc_auc"],
            reverse=True,
        )[:5]
        print("\nTop 5 experiments by mean ROC-AUC:")
        for i, r in enumerate(top5, 1):
            print(
                f"  {i}. {r['experiment_name']}: "
                f"{r['aggregate_metrics']['mean_roc_auc']:.4f}"
            )


if __name__ == "__main__":
    main()
