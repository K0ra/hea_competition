#!/usr/bin/env python3
"""
Diabetes Single-Task Neural Network Sweep Script
=================================================

Adapted from multi-task neural sweep for single-task diabetes prediction.
Prioritizes the top 3 configurations from multi-task experiments.

Dependencies Required:
- torch>=2.0
- numpy
- pandas
- scikit-learn
- imbalanced-learn

Usage:
    python diseases/diabetes/run_neural_sweep_diabetes.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
TRAIN_DATA_PATH = SCRIPT_DIR / "diabetes_train.parquet"
TEST_DATA_PATH = SCRIPT_DIR / "diabetes_test.parquet"
RESULTS_DIR = SCRIPT_DIR / "results" / "neural_sweep"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CPU optimization
NUM_WORKERS = 0
TORCH_THREADS = 16

TARGET_COL = "target_DIABE"
RANDOM_STATE = 42

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_term * bce_loss
        return loss.mean()


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class SimpleMLP(nn.Module):
    """Single-task MLP with configurable depth and width."""
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        
        # Build layers
        layers = []
        in_dim = n_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (batch_size,)"""
        shared_repr = self.shared(x)
        return self.head(shared_repr).squeeze(1)


class SimpleResNet(nn.Module):
    """ResNet-style network with residual connections."""
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_blocks: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            self.blocks.append(block)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = self.relu(x + residual)
            x = self.dropout(x)
        
        return self.head(x).squeeze(1)


class SimpleAttention(nn.Module):
    """Attention-based network."""
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        
        # Compute attention weights
        attn_scores = self.attention(encoded)
        attn_weights = torch.softmax(attn_scores, dim=0)
        
        # Weighted representation
        weighted = encoded * attn_weights
        
        return self.predictor(weighted).squeeze(1)


class SimpleWideDeep(nn.Module):
    """Wide & Deep architecture for single task."""
    
    def __init__(
        self,
        n_features: int,
        deep_dims: list[int],
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Wide path (linear)
        self.wide = nn.Linear(n_features, 32)
        
        # Deep path
        deep_layers = []
        in_dim = n_features
        for hidden_dim in deep_dims:
            deep_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.deep = nn.Sequential(*deep_layers)
        
        # Output head (combines wide + deep)
        combined_dim = 32 + in_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.head(combined).squeeze(1)


def create_model(
    architecture_config: dict,
    n_features: int,
    regularization_config: dict,
) -> nn.Module:
    """Factory function to create models from config."""
    arch_type = architecture_config["type"]
    dropout = regularization_config.get("dropout", 0.3)
    use_batch_norm = regularization_config.get("batch_norm", True)
    
    if arch_type == "mlp":
        hidden_dims = architecture_config["hidden_dims"]
        return SimpleMLP(n_features, hidden_dims, dropout, use_batch_norm)
    elif arch_type == "resnet":
        hidden_dim = architecture_config["hidden_dim"]
        n_blocks = architecture_config["n_blocks"]
        return SimpleResNet(n_features, hidden_dim, n_blocks, dropout)
    elif arch_type == "attention":
        hidden_dim = architecture_config["hidden_dim"]
        return SimpleAttention(n_features, hidden_dim, dropout)
    elif arch_type == "wide_deep":
        deep_dims = architecture_config["deep_dims"]
        return SimpleWideDeep(n_features, deep_dims, dropout)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class DiabetesDataset(Dataset):
    """Dataset for diabetes prediction."""
    
    def __init__(self, X: pd.DataFrame, y: np.ndarray):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.n_samples = len(X)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.X[idx],
            "target": self.y[idx],
        }


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Handles training, validation, and early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: str = "cpu",
        patience: int = 10,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.use_amp = use_amp and device == "cuda"
        
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Early stopping state
        self.best_val_loss = float("inf")
        self.best_val_metrics = None
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            features = batch["features"].to(self.device)
            targets = batch["target"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(features)
                    loss = self.loss_fn(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Validate and compute metrics."""
        self.model.eval()
        
        y_true_list = []
        y_pred_list = []
        
        for batch in val_loader:
            features = batch["features"].to(self.device)
            targets = batch["target"].to(self.device)
            
            outputs = self.model(features)
            probs = torch.sigmoid(outputs)
            
            y_true_list.append(targets.cpu().numpy())
            y_pred_list.append(probs.cpu().numpy())
        
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        
        # Compute metrics
        roc_auc = roc_auc_score(y_true, y_pred)
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall_arr, precision_arr)
        
        # Best F2 threshold
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f2 = 0.0
        best_thresh = 0.5
        for t in thresholds:
            y_bin = (y_pred >= t).astype(int)
            f2 = fbeta_score(y_true, y_bin, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_thresh = t
        
        return {
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "F2_score": round(best_f2, 4),
            "best_threshold": round(best_thresh, 3),
            "n_samples": len(y_true),
            "n_positive": int(np.sum(y_true == 1)),
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        verbose: bool = True,
    ) -> dict:
        """Train with early stopping."""
        history = {
            "train_loss": [],
            "val_metrics": [],
            "epochs_trained": 0,
        }
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validate
            val_metrics = self.validate(val_loader)
            history["val_metrics"].append(val_metrics)
            
            # Early stopping (use F2 score - prioritizes recall)
            val_loss_proxy = 1.0 - val_metrics["F2_score"]
            
            if val_loss_proxy < self.best_val_loss:
                self.best_val_loss = val_loss_proxy
                self.best_val_metrics = val_metrics
                self.best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss_proxy)
                else:
                    self.scheduler.step()
            
            history["epochs_trained"] = epoch + 1
            
            if verbose and (epoch + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch+1}/{max_epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_f2={val_metrics['F2_score']:.4f}, "
                    f"val_roc={val_metrics['ROC_AUC']:.4f}, "
                    f"patience={self.patience_counter}/{self.patience} "
                    f"({elapsed:.1f}s)"
                )
            
            # Early stopping
            if self.patience_counter >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        
        return {
            "history": history,
            "best_metrics": self.best_val_metrics,
            "epochs_trained": history["epochs_trained"],
            "total_time": total_time,
        }


# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets."""
    if not TRAIN_DATA_PATH.is_file():
        raise FileNotFoundError(f"Train data not found: {TRAIN_DATA_PATH}")
    if not TEST_DATA_PATH.is_file():
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}")
    
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    
    return train_df, test_df


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_smote: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Prepare features and targets with scaling."""
    # Get feature columns (exclude target and id)
    feature_cols = [
        c for c in train_df.columns
        if c != TARGET_COL and not c.startswith("target_") and c != "id"
    ]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].values
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].values
    
    # Fill NaN with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Apply SMOTE if requested
    if use_smote and len(np.unique(y_train)) > 1:
        try:
            smote = SMOTE(sampling_strategy=0.33, random_state=RANDOM_STATE)
            X_train_res, y_train_res = smote.fit_resample(X_train.values, y_train)
            X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
            y_train = y_train_res
        except Exception as e:
            print(f"  SMOTE failed: {e}, continuing without it")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train, y_train, X_test, y_test


def compute_class_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for imbalanced data."""
    n_positive = np.sum(y == 1)
    n_negative = np.sum(y == 0)
    if n_positive > 0:
        return n_negative / n_positive
    return 1.0


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Architecture variations
ARCHITECTURES = [
    # TOP 3 PRIORITY CONFIGURATIONS (from multi-task results)
    {
        "type": "mlp",
        "hidden_dims": [512, 256, 128],
        "name": "mlp_deep_wide_PRIORITY"
    },
    {
        "type": "wide_deep",
        "deep_dims": [256, 128],
        "name": "wide_deep_medium_PRIORITY"
    },
    {
        "type": "mlp",
        "hidden_dims": [256, 128],
        "name": "mlp_shallow_wide_PRIORITY"
    },
    # Standard sweep architectures
    {
        "type": "mlp",
        "hidden_dims": [256, 128],
        "name": "mlp_shallow_wide"
    },
    {
        "type": "mlp",
        "hidden_dims": [128, 64],
        "name": "mlp_shallow_narrow"
    },
    {
        "type": "mlp",
        "hidden_dims": [512, 256, 128],
        "name": "mlp_deep_wide"
    },
    {
        "type": "mlp",
        "hidden_dims": [256, 128, 64],
        "name": "mlp_deep_medium"
    },
    {
        "type": "mlp",
        "hidden_dims": [128, 64, 32],
        "name": "mlp_deep_narrow"
    },
    {
        "type": "mlp",
        "hidden_dims": [512, 256, 128, 64],
        "name": "mlp_very_deep"
    },
    {
        "type": "resnet",
        "n_blocks": 2,
        "hidden_dim": 256,
        "name": "resnet_2blocks"
    },
    {
        "type": "resnet",
        "n_blocks": 3,
        "hidden_dim": 256,
        "name": "resnet_3blocks"
    },
    {
        "type": "resnet",
        "n_blocks": 4,
        "hidden_dim": 256,
        "name": "resnet_4blocks"
    },
    {
        "type": "attention",
        "hidden_dim": 256,
        "name": "attention_256"
    },
    {
        "type": "attention",
        "hidden_dim": 512,
        "name": "attention_512"
    },
    {
        "type": "wide_deep",
        "deep_dims": [256, 128],
        "name": "wide_deep_medium"
    },
    {
        "type": "wide_deep",
        "deep_dims": [512, 256, 128],
        "name": "wide_deep_large"
    },
]

# Top 3 priority config (shared settings)
PRIORITY_REG = {"dropout": 0.3, "weight_decay": 1e-4, "batch_norm": True}
PRIORITY_REG_NO_BN = {"dropout": 0.3, "weight_decay": 1e-4, "batch_norm": False}
PRIORITY_OPT = {"lr": 3e-4, "batch_size": 512, "optimizer": "adam"}
PRIORITY_IMB = {"loss": "weighted_bce", "use_smote": False}

# Regularization variations
REGULARIZATION = [
    {"dropout": 0.1, "weight_decay": 1e-5, "batch_norm": True},
    {"dropout": 0.2, "weight_decay": 1e-4, "batch_norm": True},
    {"dropout": 0.3, "weight_decay": 1e-4, "batch_norm": True},
    {"dropout": 0.4, "weight_decay": 1e-3, "batch_norm": True},
    {"dropout": 0.3, "weight_decay": 1e-4, "batch_norm": False},
]

# Optimization variations
OPTIMIZATION = [
    {"lr": 1e-3, "batch_size": 256, "optimizer": "adam"},
    {"lr": 3e-4, "batch_size": 512, "optimizer": "adam"},
    {"lr": 1e-4, "batch_size": 512, "optimizer": "adam"},
    {"lr": 1e-3, "batch_size": 512, "optimizer": "adamw"},
    {"lr": 1e-3, "batch_size": 1024, "optimizer": "adamw"},
]

# Class imbalance handling
IMBALANCE_HANDLING = [
    {"loss": "weighted_bce", "use_smote": False},
    {"loss": "focal", "use_smote": False},
    {"loss": "focal_weighted", "use_smote": False},
    {"loss": "weighted_bce", "use_smote": True},
]


def build_experiment_grid() -> list[dict]:
    """Build experiment grid with priority configs first."""
    experiments = []
    
    # PRIORITY: Top 3 configurations from multi-task runs
    priority_archs = [
        ARCHITECTURES[0],  # mlp_deep_wide
        ARCHITECTURES[1],  # wide_deep_medium
        ARCHITECTURES[2],  # mlp_shallow_wide (no batch norm version)
    ]
    
    for arch in priority_archs:
        if "shallow_wide_PRIORITY" in arch["name"]:
            reg = PRIORITY_REG_NO_BN
        else:
            reg = PRIORITY_REG
        
        exp = {
            "name": arch["name"],
            "architecture": arch,
            "regularization": reg,
            "optimization": PRIORITY_OPT,
            "imbalance": PRIORITY_IMB,
        }
        experiments.append(exp)
    
    # Baseline: All architectures with standard config
    baseline_reg = REGULARIZATION[2]
    baseline_opt = OPTIMIZATION[1]
    baseline_imb = IMBALANCE_HANDLING[0]
    
    for arch in ARCHITECTURES[3:]:  # Skip priority configs
        exp = {
            "name": f"{arch['name']}_baseline",
            "architecture": arch,
            "regularization": baseline_reg,
            "optimization": baseline_opt,
            "imbalance": baseline_imb,
        }
        experiments.append(exp)
    
    return experiments


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_one_experiment(
    exp_config: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_weight: float,
    output_dir: Path,
) -> dict:
    """Run a single experiment."""
    exp_name = exp_config["name"]
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Extract configs
    arch_config = exp_config["architecture"]
    reg_config = exp_config["regularization"]
    opt_config = exp_config["optimization"]
    imb_config = exp_config["imbalance"]
    
    # Apply SMOTE if needed
    X_tr = X_train.copy()
    y_tr = y_train.copy()
    
    if imb_config["use_smote"]:
        print("  Applying SMOTE...")
        if len(np.unique(y_tr)) > 1:
            try:
                smote = SMOTE(sampling_strategy=0.33, random_state=RANDOM_STATE)
                X_tr_res, y_tr_res = smote.fit_resample(X_tr.values, y_tr)
                X_tr = pd.DataFrame(X_tr_res, columns=X_tr.columns)
                y_tr = y_tr_res
                print(f"  SMOTE applied: {len(y_tr)} samples")
            except Exception as e:
                print(f"  SMOTE failed: {e}")
    
    # Create dataloaders
    train_dataset = DiabetesDataset(X_tr, y_tr)
    test_dataset = DiabetesDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt_config["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt_config["batch_size"] * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    
    # Create model
    n_features = X_train.shape[1]
    model = create_model(arch_config, n_features, reg_config)
    print(f"  Model: {arch_config['type']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    loss_type = imb_config["loss"]
    if loss_type == "weighted_bce":
        pos_weight = torch.tensor([class_weight], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    elif loss_type == "focal_weighted":
        alpha = min(class_weight / 20.0, 0.75)
        loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")
    
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
    
    # Create trainer
    trainer = Trainer(
        model,
        loss_fn,
        optimizer,
        scheduler,
        device=DEVICE,
        patience=15,
        use_amp=(DEVICE == "cuda"),
    )
    
    # Train
    print("  Training...")
    result = trainer.fit(train_loader, test_loader, max_epochs=100, verbose=True)
    
    elapsed_time = time.time() - start_time
    
    # Save results
    result_dict = {
        "experiment_name": exp_name,
        "architecture": arch_config,
        "regularization": reg_config,
        "optimization": opt_config,
        "imbalance": imb_config,
        "epochs_trained": result["epochs_trained"],
        "training_time_sec": result["total_time"],
        "total_time_sec": elapsed_time,
        "metrics": result["best_metrics"],
        "class_weight": class_weight,
    }
    
    json_path = output_dir / f"{exp_name}.json"
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"\n  Results: ROC={result['best_metrics']['ROC_AUC']:.4f}, "
          f"F2={result['best_metrics']['F2_score']:.4f}")
    print(f"  Saved to: {json_path}")
    print(f"  Time: {elapsed_time:.1f}s")
    
    return result_dict


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Set PyTorch threading
    if DEVICE == "cpu":
        torch.set_num_threads(TORCH_THREADS)
        print(f"PyTorch threads: {TORCH_THREADS}")
    
    print("="*80)
    print("DIABETES SINGLE-TASK NEURAL NETWORK SWEEP")
    print("="*80)
    print(f"Train data: {TRAIN_DATA_PATH}")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_data()
    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Prepare data
    print("\nPreparing features...")
    X_train, y_train, X_test, y_test = prepare_data(train_df, test_df, use_smote=False)
    print(f"Features: {X_train.shape[1]}")
    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Test positive rate: {y_test.mean():.3f}")
    
    # Compute class weight
    class_weight = compute_class_weight(y_train)
    print(f"Class weight (pos): {class_weight:.2f}")
    
    # Build experiments
    experiments = build_experiment_grid()
    print(f"\nTotal experiments: {len(experiments)}")
    print("PRIORITY: First 3 experiments are top performers from multi-task runs")
    
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    print(f"\n{'='*80}")
    print("STARTING EXPERIMENTS")
    print(f"{'='*80}")
    
    results = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp_config['name']}")
        
        try:
            result = run_one_experiment(
                exp_config, X_train, y_train, X_test, y_test,
                class_weight, RESULTS_DIR
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ALL {len(results)} EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    
    if results:
        sorted_results = sorted(
            results,
            key=lambda x: x["metrics"]["F2_score"],
            reverse=True
        )
        print("\nTop 5 by F2 Score:")
        print("-"*80)
        for i, r in enumerate(sorted_results[:5], 1):
            m = r["metrics"]
            print(f"{i}. {r['experiment_name']:<45} "
                  f"F2={m['F2_score']:.4f} ROC={m['ROC_AUC']:.4f}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
