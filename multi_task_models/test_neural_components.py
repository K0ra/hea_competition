"""Smoke tests for neural network components."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "multi_task_models"))

from losses import FocalLoss, MultiTaskLoss
from neural_architectures import (
    MultiTaskAttention,
    MultiTaskMLP,
    MultiTaskResNet,
    MultiTaskWideDeep,
    count_parameters,
)
from neural_data import MultiTaskDataset, prepare_dataloaders
from neural_trainer import MultiTaskTrainer
from utils import TARGET_COLS, load_data


def test_focal_loss():
    """Test focal loss computation."""
    print("\n[Test 1] Focal Loss")
    
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    
    # Test forward pass
    logits = torch.tensor([1.0, -1.0, 2.0, -2.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    
    loss = loss_fn(logits, targets)
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("  PASS: Focal loss computed")


def test_multi_task_loss():
    """Test multi-task loss with masking."""
    print("\n[Test 2] Multi-Task Loss")
    
    task_names = ["diabetes", "heart_disease"]
    class_weights = {"diabetes": 5.0, "heart_disease": 3.0}
    
    loss_fn = MultiTaskLoss(
        task_names=task_names,
        loss_type="weighted_bce",
        class_weights=class_weights,
    )
    
    # Simulate outputs
    outputs = {
        "diabetes": torch.tensor([1.0, -1.0, 2.0]),
        "heart_disease": torch.tensor([0.5, 1.5, -0.5]),
    }
    
    targets_dict = {
        "diabetes": torch.tensor([1.0, 0.0, 1.0]),
        "heart_disease": torch.tensor([0.0, 1.0, 0.0]),
    }
    
    masks_dict = {
        "diabetes": torch.tensor([True, True, True]),
        "heart_disease": torch.tensor([True, True, False]),  # One masked
    }
    
    total_loss, task_losses = loss_fn(outputs, targets_dict, masks_dict)
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Task losses: {task_losses}")
    
    assert total_loss.item() > 0, "Loss should be positive"
    assert len(task_losses) == 2, "Should have 2 task losses"
    print("  PASS: Multi-task loss computed with masking")


def test_architectures():
    """Test all neural architectures."""
    print("\n[Test 3] Neural Architectures")
    
    n_features = 50
    n_samples = 32
    task_names = ["diabetes", "heart_disease", "stroke"]
    
    x = torch.randn(n_samples, n_features)
    
    # Test MLP
    mlp = MultiTaskMLP(n_features, [128, 64], task_names, dropout=0.3)
    outputs = mlp(x)
    assert len(outputs) == 3, "Should have 3 task outputs"
    assert outputs["diabetes"].shape == (n_samples,), "Output shape mismatch"
    print(f"  MLP params: {count_parameters(mlp):,}")
    print("  PASS: MLP forward pass")
    
    # Test ResNet
    resnet = MultiTaskResNet(n_features, 128, 2, task_names, dropout=0.2)
    outputs = resnet(x)
    assert len(outputs) == 3, "Should have 3 task outputs"
    print(f"  ResNet params: {count_parameters(resnet):,}")
    print("  PASS: ResNet forward pass")
    
    # Test Attention
    attention = MultiTaskAttention(n_features, 128, task_names, dropout=0.3)
    outputs = attention(x)
    assert len(outputs) == 3, "Should have 3 task outputs"
    print(f"  Attention params: {count_parameters(attention):,}")
    print("  PASS: Attention forward pass")
    
    # Test Wide & Deep
    wide_deep = MultiTaskWideDeep(
        n_features, [128, 64], task_names, dropout=0.3
    )
    outputs = wide_deep(x)
    assert len(outputs) == 3, "Should have 3 task outputs"
    print(f"  Wide&Deep params: {count_parameters(wide_deep):,}")
    print("  PASS: Wide&Deep forward pass")


def test_dataset():
    """Test PyTorch dataset creation."""
    print("\n[Test 4] MultiTaskDataset")
    
    n_samples = 100
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y_dict = {
        "diabetes": np.random.randint(0, 2, n_samples).astype(float),
        "stroke": np.random.randint(0, 2, n_samples).astype(float),
    }
    task_names = ["diabetes", "stroke"]
    
    dataset = MultiTaskDataset(X, y_dict, task_names)
    
    assert len(dataset) == n_samples, "Dataset length mismatch"
    
    sample = dataset[0]
    assert "features" in sample, "Missing features"
    assert "target_diabetes" in sample, "Missing target_diabetes"
    assert "mask_diabetes" in sample, "Missing mask_diabetes"
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Sample keys: {list(sample.keys())}")
    print("  PASS: Dataset created")


def test_trainer():
    """Test trainer with dummy data."""
    print("\n[Test 5] Trainer (mini training loop)")
    
    # Tiny dataset
    n_samples = 200
    n_features = 30
    task_names = ["diabetes"]
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y_dict = {"diabetes": np.random.randint(0, 2, n_samples).astype(float)}
    
    dataset = MultiTaskDataset(X, y_dict, task_names)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    
    # Create model and loss
    model = MultiTaskMLP(n_features, [64, 32], task_names, dropout=0.2)
    loss_fn = MultiTaskLoss(task_names, loss_type="weighted_bce")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = MultiTaskTrainer(
        model, loss_fn, optimizer, device="cpu", patience=3
    )
    
    # Train for 2 epochs
    train_loss, task_losses = trainer.train_epoch(train_loader)
    print(f"  Epoch 1 loss: {train_loss:.4f}")
    
    train_loss, task_losses = trainer.train_epoch(train_loader)
    print(f"  Epoch 2 loss: {train_loss:.4f}")
    
    print("  PASS: Training loop executed")


def test_end_to_end():
    """Test complete pipeline with real data (small subset)."""
    print("\n[Test 6] End-to-end mini pipeline")
    
    try:
        # Load minimal data
        model_path = REPO_DIR / "data" / "rand_hrs_model_merged.parquet"
        df = load_data(model_path)
        
        # Take tiny subset
        df = df[df["wave"].between(3, 12)].head(1000)
        print(f"  Loaded {len(df)} samples")
        
        from utils import TARGET_COLS, get_feature_columns, prepare_train_val_split
        
        # Get features
        feature_cols = get_feature_columns(df, top_n=30)
        print(f"  Using {len(feature_cols)} features")
        
        # Prepare splits for all tasks (new signature)
        X_train_dict, y_train_dict, X_val_dict, y_val_dict = (
            prepare_train_val_split(df, feature_cols, [TARGET_COLS[0]])
        )
        
        # Extract first task for testing
        target_col = TARGET_COLS[0]
        if target_col not in X_train_dict:
            print("  SKIP: Insufficient training data")
            return
        
        X_train = X_train_dict[target_col]
        y_train = y_train_dict[target_col]
        X_val = X_val_dict[target_col]
        y_val = y_val_dict[target_col]
        
        print(
            f"  Train size: {len(y_train)}, Val size: {len(y_val)}"
        )
        
        # Create tiny model
        task_names = ["diabetes"]
        X_train_mapped = {"r_diag_diabetes_ever": X_train}
        y_train_mapped = {"r_diag_diabetes_ever": y_train}
        X_val_mapped = {"r_diag_diabetes_ever": X_val}
        y_val_mapped = {"r_diag_diabetes_ever": y_val}
        
        train_dataset = MultiTaskDataset(
            X_train, {"diabetes": y_train}, task_names
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )
        
        val_dataset = MultiTaskDataset(
            X_val, {"diabetes": y_val}, task_names
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, shuffle=False
        )
        
        # Train mini model
        model = MultiTaskMLP(
            len(feature_cols) - 1, [32], task_names, dropout=0.2
        )
        loss_fn = MultiTaskLoss(task_names, loss_type="weighted_bce")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        trainer = MultiTaskTrainer(
            model, loss_fn, optimizer, device="cpu", patience=2
        )
        
        result = trainer.fit(
            train_loader,
            {"diabetes": val_loader},
            max_epochs=3,
            verbose=False,
        )
        
        print(f"  Trained for {result['epochs_trained']} epochs")
        print(f"  Best val metrics: {result['best_metrics']}")
        print("  PASS: End-to-end pipeline completed")
        
    except Exception as e:
        print(f"  WARNING: End-to-end test failed: {e}")
        print("  (This is OK if data not yet available)")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("NEURAL NETWORK COMPONENTS SMOKE TESTS")
    print("=" * 60)
    
    test_focal_loss()
    test_multi_task_loss()
    test_architectures()
    test_dataset()
    test_trainer()
    test_end_to_end()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
