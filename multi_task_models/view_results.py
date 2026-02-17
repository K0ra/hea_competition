#!/usr/bin/env python3
"""
Simple script to view neural network experiment results.
Shows F2, ROC-AUC, and PR-AUC scores for all completed experiments.
"""
import json
import glob
import os
from pathlib import Path

def main():
    # Get results directory
    results_dir = Path(__file__).parent.parent / "results" / "neural_sweep"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Load all results
    results = []
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append({
                    'name': json_file.stem,
                    'roc_auc': data['aggregate_metrics']['mean_roc_auc'],
                    'pr_auc': data['aggregate_metrics']['mean_pr_auc'],
                    'f2_score': data['aggregate_metrics']['mean_f2_score'],
                    'epochs': data['epochs_trained'],
                    'time': data['training_time_sec']
                })
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
    
    if not results:
        print("No results found yet. Experiments may still be running.")
        return
    
    # Sort by ROC-AUC (descending)
    results.sort(key=lambda x: x['roc_auc'], reverse=True)
    
    # Display results
    print("\n" + "="*100)
    print(f"NEURAL NETWORK EXPERIMENT RESULTS ({len(results)}/51 completed)")
    print("="*100)
    print(f"{'Rank':<5} {'Experiment':<40} {'ROC-AUC':<10} {'PR-AUC':<10} {'F2':<10} {'Epochs':<8} {'Time(s)':<8}")
    print("-"*100)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<5} {result['name']:<40} {result['roc_auc']:<10.4f} "
              f"{result['pr_auc']:<10.4f} {result['f2_score']:<10.4f} "
              f"{result['epochs']:<8} {result['time']:<8.1f}")
    
    print("="*100)
    
    # Show top 3
    print("\n🏆 TOP 3 EXPERIMENTS:")
    print("-"*100)
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {result['name']}")
        print(f"     ROC-AUC: {result['roc_auc']:.4f} | PR-AUC: {result['pr_auc']:.4f} | F2: {result['f2_score']:.4f}")
    print()

if __name__ == "__main__":
    main()
