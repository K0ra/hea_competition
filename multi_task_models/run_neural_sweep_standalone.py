#!/usr/bin/env python3
"""
Standalone Neural Network Experiment Sweep Script
=================================================

Self-contained script for running neural network experiments 11-51 in reverse order.
Can be copied to any remote server with minimal dependencies.

Dependencies Required:
- torch>=2.0
- numpy
- pandas  
- scikit-learn
- imbalanced-learn

Usage:
    1. Update DATASET_PATH below to point to your data file
    2. Run: python run_neural_sweep_standalone.py
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
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================================

# Dataset will be in the same directory as this script
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "rand_hrs_model_merged.parquet"
RESULTS_DIR = "./results/neural_sweep"  # Results save location
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# CPU OPTIMIZATION SETTINGS - HPC SERVER COMPATIBLE
# ============================================================================
# Optimized for: Intel Xeon Gold 6338 (16 cores, 60GB RAM, CPU-only)
# 
# NUM_WORKERS: Number of parallel data loading processes
# - Set to 0 to avoid UCX/InfiniBand segmentation faults on HPC servers
# - This disables parallel data loading but keeps the main optimization
#
# TORCH_THREADS: Number of threads for PyTorch operations  
# - THIS IS THE KEY OPTIMIZATION: 2x speedup (16 threads vs default 8)
# - Uses all CPU cores for neural network computations
# - Your setting: 16 threads (all logical cores)
#
# EXPECTED PERFORMANCE:
# - With TORCH_THREADS=16 + NUM_WORKERS=0: ~5-6 hours (2.0x speedup)
# - Original settings (8 threads, 0 workers): ~10 hours
# ============================================================================

NUM_WORKERS = 0  # DataLoader workers (disabled for HPC/UCX compatibility)
TORCH_THREADS = 16  # PyTorch computation threads - THE MAIN OPTIMIZATION!

# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_COLS = [
    "target_DIABE",
    "target_HIBPE",
    "target_CANCR",
    "target_LUNGE",
    "target_HEARTE",
    "target_STROKE",
    "target_PSYCHE",
    "target_ARTHRE",
]

DISEASE_SUFFIXES = {
    "target_DIABE": "DIABE",
    "target_HIBPE": "HIBPE",
    "target_CANCR": "CANCR",
    "target_LUNGE": "LUNGE",
    "target_HEARTE": "HEARTE",
    "target_STROKE": "STROKE",
    "target_PSYCHE": "PSYCHE",
    "target_ARTHRE": "ARTHRE",
}

TRAIN_WAVES = (3, 9)
VAL_WAVES = (10, 12)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw logits (before sigmoid), shape (batch_size,)
            targets: Binary targets, shape (batch_size,)
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_term * bce_loss
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with per-task weighting and masking.
    
    Handles:
    - Per-task class imbalance weights
    - Missing target masking
    - Multiple loss types (BCE, focal)
    """
    
    def __init__(
        self,
        task_names: list[str],
        loss_type: str = "weighted_bce",
        class_weights: dict[str, float] | None = None,
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.task_names = task_names
        self.loss_type = loss_type
        self.class_weights = class_weights or {}
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Create per-task loss functions
        self.loss_fns = {}
        for task in task_names:
            weight = self.class_weights.get(task, 1.0)
            if loss_type == "weighted_bce":
                pos_weight = torch.tensor([weight], dtype=torch.float32)
                self.loss_fns[task] = nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight, reduction="mean"
                )
            elif loss_type == "focal":
                self.loss_fns[task] = FocalLoss(gamma=2.0, alpha=0.25)
            elif loss_type == "focal_weighted":
                # Focal loss with class weight scaling
                self.loss_fns[task] = FocalLoss(
                    gamma=2.0, alpha=min(weight / 20.0, 0.75)
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Learnable task weights for uncertainty weighting
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(
                torch.zeros(len(task_names), dtype=torch.float32)
            )
    
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets_dict: dict[str, torch.Tensor],
        masks_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute multi-task loss with masking.
        
        Args:
            outputs: Dict mapping task name to logits (batch_size,)
            targets_dict: Dict mapping task name to binary targets
            masks_dict: Dict mapping task name to valid sample masks
        
        Returns:
            total_loss: Aggregated loss
            task_losses: Dict of per-task loss values for logging
        """
        total_loss = 0.0
        task_losses = {}
        n_active_tasks = 0
        
        for i, task in enumerate(self.task_names):
            if task not in outputs or task not in targets_dict:
                continue
            
            mask = masks_dict.get(task, None)
            if mask is None or mask.sum() == 0:
                continue
            
            # Apply mask
            logits = outputs[task][mask]
            targets = targets_dict[task][mask]
            
            if len(logits) == 0:
                continue
            
            # Compute task loss
            loss_fn = self.loss_fns[task]
            if self.loss_type in ["weighted_bce"]:
                # Move pos_weight to same device
                loss_fn.pos_weight = loss_fn.pos_weight.to(logits.device)
            
            task_loss = loss_fn(logits, targets)
            task_losses[task] = task_loss.item()
            
            # Apply uncertainty weighting if enabled
            if self.use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * task_loss + self.log_vars[i]
                total_loss += weighted_loss
            else:
                total_loss += task_loss
            
            n_active_tasks += 1
        
        # Normalize by number of active tasks
        if n_active_tasks > 0:
            if not self.use_uncertainty_weighting:
                total_loss = total_loss / n_active_tasks
        else:
            total_loss = torch.tensor(0.0, requires_grad=True)
        
        return total_loss, task_losses


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class MultiTaskMLP(nn.Module):
    """
    Basic multi-layer perceptron with shared + task-specific layers.
    
    Architecture:
        Input → Shared Dense Layers → Task-Specific Heads → 8 Outputs
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int],
        task_names: list[str],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.task_names = task_names
        self.n_features = n_features
        
        # Shared layers
        shared_layers = []
        in_dim = n_features
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                shared_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task in task_names:
            head = nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1),
            )
            self.task_heads[task] = head
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Input features (batch_size, n_features)
        
        Returns:
            Dict mapping task name to logits (batch_size, 1)
        """
        shared_repr = self.shared(x)
        outputs = {}
        for task in self.task_names:
            outputs[task] = self.task_heads[task](shared_repr).squeeze(1)
        return outputs


class MultiTaskResNet(nn.Module):
    """
    ResNet-style multi-task network with residual connections.
    
    Good for deeper networks to avoid vanishing gradients.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_blocks: int,
        task_names: list[str],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.task_names = task_names
        
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
        
        # Task heads
        self.task_heads = nn.ModuleDict()
        for task in task_names:
            head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.task_heads[task] = head
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = self.relu(x + residual)  # Skip connection
            x = self.dropout(x)
        
        outputs = {}
        for task in self.task_names:
            outputs[task] = self.task_heads[task](x).squeeze(1)
        return outputs


class MultiTaskAttention(nn.Module):
    """
    Attention-based multi-task network.
    
    Uses attention to weight features differently per task.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        task_names: list[str],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.task_names = task_names
        
        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Per-task attention and prediction
        self.task_modules = nn.ModuleDict()
        for task in task_names:
            task_module = nn.ModuleDict({
                "attention": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_dim // 2, 1),
                ),
                "predictor": nn.Sequential(
                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 1),
                ),
            })
            self.task_modules[task] = task_module
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Shared encoding
        encoded = self.encoder(x)  # (batch, hidden_dim)
        
        outputs = {}
        for task in self.task_names:
            # Compute attention weights
            attn_scores = self.task_modules[task]["attention"](
                encoded
            )  # (batch, 1)
            attn_weights = torch.softmax(
                attn_scores, dim=0
            )  # Normalize across batch
            
            # Weighted representation
            weighted = encoded * attn_weights  # Element-wise
            
            # Prediction
            outputs[task] = self.task_modules[task]["predictor"](
                weighted
            ).squeeze(1)
        
        return outputs


class MultiTaskWideDeep(nn.Module):
    """
    Wide & Deep architecture for multi-task learning.
    
    Combines linear (wide) and deep paths.
    """
    
    def __init__(
        self,
        n_features: int,
        deep_dims: list[int],
        task_names: list[str],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.task_names = task_names
        
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
        
        # Task heads (input: wide + deep)
        combined_dim = 32 + in_dim
        self.task_heads = nn.ModuleDict()
        for task in task_names:
            head = nn.Sequential(
                nn.Linear(combined_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            )
            self.task_heads[task] = head
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        
        outputs = {}
        for task in self.task_names:
            outputs[task] = self.task_heads[task](combined).squeeze(1)
        return outputs


def create_model(
    architecture_config: dict,
    n_features: int,
    task_names: list[str],
    regularization_config: dict,
) -> nn.Module:
    """
    Factory function to create models from config.
    
    Args:
        architecture_config: Dict with type and architecture params
        n_features: Number of input features
        task_names: List of task names (disease suffixes)
        regularization_config: Dict with dropout, batch_norm settings
    
    Returns:
        nn.Module instance
    """
    arch_type = architecture_config["type"]
    dropout = regularization_config.get("dropout", 0.3)
    use_batch_norm = regularization_config.get("batch_norm", True)
    
    if arch_type == "mlp":
        hidden_dims = architecture_config["hidden_dims"]
        return MultiTaskMLP(
            n_features, hidden_dims, task_names, dropout, use_batch_norm
        )
    elif arch_type == "resnet":
        hidden_dim = architecture_config["hidden_dim"]
        n_blocks = architecture_config["n_blocks"]
        return MultiTaskResNet(
            n_features, hidden_dim, n_blocks, task_names, dropout
        )
    elif arch_type == "attention":
        hidden_dim = architecture_config["hidden_dim"]
        return MultiTaskAttention(
            n_features, hidden_dim, task_names, dropout
        )
    elif arch_type == "wide_deep":
        deep_dims = architecture_config["deep_dims"]
        return MultiTaskWideDeep(
            n_features, deep_dims, task_names, dropout
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task learning with per-task target masking.
    
    Each sample contains features and 8 targets (one per disease).
    Masks indicate which targets are valid (non-null) for each sample.
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y_dict: dict[str, np.ndarray],
        task_names: list[str],
    ):
        """
        Args:
            X: Features dataframe (n_samples, n_features)
            y_dict: Dict mapping task name to target arrays
            task_names: List of task names (disease suffixes)
        """
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.n_samples = len(X)
        self.task_names = task_names
        
        # Build target tensor and mask tensor
        self.targets = {}
        self.masks = {}
        
        for task in task_names:
            if task in y_dict:
                y = y_dict[task]
                # y should already be aligned with X (same length)
                if len(y) == self.n_samples:
                    # Create mask based on NaN values (NaN = not available for this task)
                    mask = ~np.isnan(y)
                    self.targets[task] = torch.tensor(
                        np.nan_to_num(y, nan=0.0), dtype=torch.float32
                    )
                    self.masks[task] = torch.tensor(mask, dtype=torch.bool)
                else:
                    raise ValueError(
                        f"Task {task}: length mismatch - "
                        f"y has {len(y)} samples but X has {self.n_samples}"
                    )
            else:
                # Task not available for this dataset
                self.targets[task] = torch.zeros(
                    self.n_samples, dtype=torch.float32
                )
                self.masks[task] = torch.zeros(
                    self.n_samples, dtype=torch.bool
                )
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict:
        """Return dict with features, targets, and masks."""
        item = {"features": self.X[idx]}
        
        for task in self.task_names:
            item[f"target_{task}"] = self.targets[task][idx]
            item[f"mask_{task}"] = self.masks[task][idx]
        
        return item


def prepare_dataloaders(
    X_train_dict: dict[str, pd.DataFrame],
    y_train_dict: dict[str, np.ndarray],
    X_val_dict: dict[str, pd.DataFrame],
    y_val_dict: dict[str, np.ndarray],
    task_names: list[str],
    batch_size: int = 512,
    use_smote: bool = False,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """
    Create training and validation dataloaders for multi-task learning.
    
    Args:
        X_train_dict: Dict mapping task to training features
        y_train_dict: Dict mapping task to training targets
        X_val_dict: Dict mapping task to validation features
        y_val_dict: Dict mapping task to validation targets
        task_names: List of task names (disease suffixes)
        batch_size: Batch size for training
        use_smote: Whether to apply SMOTE oversampling
    
    Returns:
        train_loader: DataLoader for training (all tasks combined)
        val_loaders: Dict mapping task name to validation DataLoader
    """
    # For training, we need to combine all tasks into one dataset
    # Take union of all training samples across tasks
    all_train_indices = set()
    task_to_indices = {}
    
    for task_col, X_train in X_train_dict.items():
        indices = X_train.index.tolist()
        task_to_indices[task_col] = set(indices)
        all_train_indices.update(indices)
    
    all_train_indices = sorted(all_train_indices)
    
    # Get full feature matrix for training
    first_task = list(X_train_dict.keys())[0]
    all_features = X_train_dict[first_task].columns.tolist()
    
    # Build combined training dataset
    # Need to get features for all indices (from original df)
    # Since X_train_dict already has filtered dfs, reconstruct full
    # Use the first task's dataframe structure
    X_train_full = pd.concat(
        [X_train_dict[t] for t in X_train_dict.keys()]
    ).drop_duplicates()
    X_train_full = X_train_full.loc[all_train_indices]
    
    # Apply SMOTE if requested (per task, then merge)
    if use_smote:
        # Apply SMOTE independently per task
        smote_applied = {}
        for task_col, y_train in y_train_dict.items():
            X_task = X_train_dict[task_col]
            if len(np.unique(y_train)) < 2:
                continue
            
            try:
                smote = SMOTE(sampling_strategy=0.33, random_state=42)
                X_res, y_res = smote.fit_resample(X_task.values, y_train)
                smote_applied[task_col] = (X_res, y_res)
            except Exception:
                pass  # SMOTE can fail; skip if it does
        
        # If SMOTE applied, update dictionaries
        if smote_applied:
            for task_col, (X_res, y_res) in smote_applied.items():
                X_train_dict[task_col] = pd.DataFrame(
                    X_res, columns=X_train_dict[task_col].columns
                )
                y_train_dict[task_col] = y_res
            
            # Rebuild combined dataset
            all_train_indices = set()
            for X_train in X_train_dict.values():
                all_train_indices.update(X_train.index.tolist())
            all_train_indices = sorted(all_train_indices)
            X_train_full = pd.concat(
                [X_train_dict[t] for t in X_train_dict.keys()]
            ).drop_duplicates()
    
    # Build y_train dict with suffixes as keys
    # CRITICAL: Align each task's targets with X_train_full indices
    y_train_mapped = {}
    for task_col, y in y_train_dict.items():
        suffix = DISEASE_SUFFIXES[task_col]
        X_task = X_train_dict[task_col]
        
        # Create DataFrame with task's targets
        y_task_df = pd.DataFrame(
            {"target": y}, index=X_task.index
        )
        
        # Align with X_train_full (reindex will add NaN for missing)
        y_full = y_task_df.reindex(X_train_full.index)["target"]
        
        y_train_mapped[suffix] = y_full.values
    
    train_dataset = MultiTaskDataset(X_train_full, y_train_mapped, task_names)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Disabled for HPC/UCX compatibility
    )
    
    # Validation loaders (one per task)
    val_loaders = {}
    for task_col, X_val in X_val_dict.items():
        suffix = DISEASE_SUFFIXES[task_col]
        y_val = y_val_dict[task_col]
        
        y_val_dict_single = {suffix: y_val}
        val_dataset = MultiTaskDataset(X_val, y_val_dict_single, [suffix])
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,  # Disabled for HPC/UCX compatibility
        )
        val_loaders[suffix] = val_loader
    
    return train_loader, val_loaders


# ============================================================================
# TRAINER
# ============================================================================

class MultiTaskTrainer:
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
        self.loss_fn = loss_fn.to(device)
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
    
    def train_epoch(
        self, train_loader: DataLoader
    ) -> tuple[float, dict[str, float]]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        task_losses_sum = {task: 0.0 for task in self.model.task_names}
        n_batches = 0
        
        for batch in train_loader:
            features = batch["features"].to(self.device)
            
            # Per-task targets and masks
            targets_dict = {}
            masks_dict = {}
            for task in self.model.task_names:
                targets_dict[task] = batch[f"target_{task}"].to(self.device)
                masks_dict[task] = batch[f"mask_{task}"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(features)
                    loss, task_losses = self.loss_fn(
                        outputs, targets_dict, masks_dict
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss, task_losses = self.loss_fn(
                    outputs, targets_dict, masks_dict
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            for task, task_loss in task_losses.items():
                task_losses_sum[task] += task_loss
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        avg_task_losses = {
            task: loss / n_batches for task, loss in task_losses_sum.items()
        }
        return avg_loss, avg_task_losses
    
    @torch.no_grad()
    def validate(
        self, val_loaders: dict[str, DataLoader]
    ) -> dict[str, dict[str, float]]:
        """
        Validate on per-task validation sets.
        
        Returns dict mapping disease suffix to metrics.
        """
        self.model.eval()
        
        y_true_dict = {}
        y_pred_dict = {}
        
        for task, val_loader in val_loaders.items():
            y_true_list = []
            y_pred_list = []
            
            for batch in val_loader:
                features = batch["features"].to(self.device)
                targets = batch[f"target_{task}"].to(self.device)
                masks = batch[f"mask_{task}"].to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                logits = outputs[task]
                probs = torch.sigmoid(logits)
                
                # Only keep valid samples
                y_true_list.append(targets[masks].cpu().numpy())
                y_pred_list.append(probs[masks].cpu().numpy())
            
            if y_true_list:
                y_true_dict[task] = np.concatenate(y_true_list)
                y_pred_dict[task] = np.concatenate(y_pred_list)
        
        # Compute metrics using existing utility
        metrics = evaluate_multi_task(y_true_dict, y_pred_dict)
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loaders: dict[str, DataLoader],
        max_epochs: int = 100,
        verbose: bool = True,
    ) -> dict:
        """
        Train with early stopping.
        
        Returns dict with training history and best metrics.
        """
        history = {
            "train_loss": [],
            "val_metrics": [],
            "epochs_trained": 0,
        }
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            # Train
            train_loss, task_losses = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validate
            val_metrics = self.validate(val_loaders)
            history["val_metrics"].append(val_metrics)
            
            # Compute mean validation ROC for early stopping
            mean_val_roc = np.mean(
                [m["ROC_AUC"] for m in val_metrics.values()]
            )
            val_loss_proxy = 1.0 - mean_val_roc  # Lower is better
            
            # Early stopping check
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
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss_proxy)
                else:
                    self.scheduler.step()
            
            history["epochs_trained"] = epoch + 1
            
            if verbose and (epoch + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch+1}/{max_epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"mean_val_roc={mean_val_roc:.4f}, "
                    f"patience={self.patience_counter}/{self.patience} "
                    f"({elapsed:.1f}s)"
                )
            
            # Early stopping
            if self.patience_counter >= self.patience:
                if verbose:
                    print(
                        f"  Early stopping at epoch {epoch+1} "
                        f"(best epoch: {epoch+1-self.patience})"
                    )
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

def load_data(data_path: Path) -> pd.DataFrame:
    """Load the dataset from parquet file."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Data not found: {data_path}")
    return pd.read_parquet(data_path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Get feature columns (all non-target, non-id, non-wave columns).
    """
    exclude_cols = set(TARGET_COLS + ["id", "wave"])
    all_features = [c for c in df.columns if c not in exclude_cols]
    return ["wave"] + all_features


def compute_per_task_weights(
    y_train_dict: dict[str, np.ndarray]
) -> dict[str, float]:
    """
    Compute class weights for each disease based on prevalence.
    
    Returns dict mapping disease suffix to scale_pos_weight value.
    """
    weights = {}
    for target_col, y in y_train_dict.items():
        suffix = DISEASE_SUFFIXES.get(target_col, target_col)
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        if n_positive > 0:
            # Inverse frequency weighting (same as scale_pos_weight)
            weights[suffix] = n_negative / n_positive
        else:
            weights[suffix] = 1.0
    return weights


def evaluate_multi_task(
    y_true_dict: dict[str, np.ndarray],
    y_pred_dict: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """
    Compute ROC-AUC, PR-AUC, F2 for each task.
    
    Args:
        y_true_dict: Map from disease suffix to true labels
        y_pred_dict: Map from disease suffix to predicted probabilities
    
    Returns:
        Dict mapping disease suffix to metrics dict
    """
    results = {}
    for disease, y_true in y_true_dict.items():
        y_pred = y_pred_dict[disease]
        
        # Skip if no positive or negative samples
        if len(np.unique(y_true)) < 2:
            results[disease] = {
                "ROC_AUC": 0.0,
                "PR_AUC": 0.0,
                "F2_score": 0.0,
                "best_threshold": 0.5,
                "n_samples": len(y_true),
                "n_positive": int(np.sum(y_true == 1)),
            }
            continue
        
        # Per-task metrics
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
        
        results[disease] = {
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "F2_score": round(best_f2, 4),
            "best_threshold": round(best_thresh, 3),
            "n_samples": len(y_true),
            "n_positive": int(np.sum(y_true == 1)),
        }
    return results


def prepare_train_val_split(
    df: pd.DataFrame, feature_columns: list[str], target_cols: list[str]
) -> tuple[dict, dict, dict, dict]:
    """
    Prepare train/val splits for all tasks with temporal split.
    
    Returns:
        X_train_dict, y_train_dict, X_val_dict, y_val_dict
        Each dict maps target_col to arrays/dataframes
    """
    wave = df["wave"].values
    train_mask = (wave >= TRAIN_WAVES[0]) & (wave <= TRAIN_WAVES[1])
    val_mask = (wave >= VAL_WAVES[0]) & (wave <= VAL_WAVES[1])
    
    X_train_dict = {}
    y_train_dict = {}
    X_val_dict = {}
    y_val_dict = {}
    
    for target_col in target_cols:
        if target_col not in df.columns:
            continue
        
        # Filter to rows with non-null target for this task
        task_mask = df[target_col].notna()
        task_train_mask = task_mask & train_mask
        task_val_mask = task_mask & val_mask
        
        if task_train_mask.sum() == 0 or task_val_mask.sum() == 0:
            continue
        
        # Exclude 'wave' from features (it's only for splitting)
        feat_cols_no_wave = [c for c in feature_columns if c != "wave"]
        X_train = df.loc[task_train_mask, feat_cols_no_wave].copy()
        y_train = df.loc[task_train_mask, target_col].astype(int).values
        X_val = df.loc[task_val_mask, feat_cols_no_wave].copy()
        y_val = df.loc[task_val_mask, target_col].astype(int).values
        
        # Fill NaN values with 0 (neural networks can't handle NaN)
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        
        # Scale features for neural networks (tree models don't need this)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_val_scaled = scaler.transform(X_val.values)
        X_train = pd.DataFrame(
            X_train_scaled, index=X_train.index, columns=X_train.columns
        )
        X_val = pd.DataFrame(
            X_val_scaled, index=X_val.index, columns=X_val.columns
        )
        
        X_train_dict[target_col] = X_train
        y_train_dict[target_col] = y_train
        X_val_dict[target_col] = X_val
        y_val_dict[target_col] = y_val
    
    return X_train_dict, y_train_dict, X_val_dict, y_val_dict


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Architecture variations
ARCHITECTURES = [
    # MLPs (shallow to deep, narrow to wide)
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
    # ResNet-style
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
    # Attention
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
    # Wide & Deep
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


def build_experiment_grid(mode: str = "exhaustive") -> list[dict]:
    """
    Build experiment grid based on mode.
    
    Args:
        mode: "quick", "comprehensive", or "exhaustive"
    
    Returns:
        List of experiment config dicts
    """
    experiments = []
    
    if mode == "quick":
        # Test all architectures with baseline config
        baseline_reg = REGULARIZATION[2]  # dropout=0.3
        baseline_opt = OPTIMIZATION[1]  # lr=3e-4, bs=512, adam
        baseline_imb = IMBALANCE_HANDLING[0]  # weighted_bce, no smote
        
        for arch in ARCHITECTURES:
            exp = {
                "name": f"{arch['name']}_baseline",
                "architecture": arch,
                "regularization": baseline_reg,
                "optimization": baseline_opt,
                "imbalance": baseline_imb,
            }
            experiments.append(exp)
    
    elif mode == "comprehensive":
        # Test all architectures with 3 regularization configs
        for arch in ARCHITECTURES:
            for reg in [REGULARIZATION[1], REGULARIZATION[2], REGULARIZATION[3]]:
                for opt in [OPTIMIZATION[1], OPTIMIZATION[3]]:
                    for imb in [IMBALANCE_HANDLING[0], IMBALANCE_HANDLING[3]]:
                        exp = {
                            "name": (
                                f"{arch['name']}_d{reg['dropout']}_"
                                f"lr{opt['lr']}_"
                                f"{imb['loss']}"
                                f"{'_smote' if imb['use_smote'] else ''}"
                            ),
                            "architecture": arch,
                            "regularization": reg,
                            "optimization": opt,
                            "imbalance": imb,
                        }
                        experiments.append(exp)
    
    else:  # exhaustive
        # Test all combinations for top architectures
        # Stage 1: All architectures with baseline
        baseline_reg = REGULARIZATION[2]
        baseline_opt = OPTIMIZATION[1]
        baseline_imb = IMBALANCE_HANDLING[0]
        
        for arch in ARCHITECTURES:
            exp = {
                "name": f"{arch['name']}_baseline",
                "architecture": arch,
                "regularization": baseline_reg,
                "optimization": baseline_opt,
                "imbalance": baseline_imb,
            }
            experiments.append(exp)
        
        # Stage 2: Top 5 architectures with all regularization
        top_archs = ARCHITECTURES[:5]
        for arch in top_archs:
            for reg in REGULARIZATION:
                if reg == baseline_reg:
                    continue  # Skip, already tested
                exp = {
                    "name": (
                        f"{arch['name']}_d{reg['dropout']}_"
                        f"wd{reg['weight_decay']:.0e}_"
                        f"{'bn' if reg['batch_norm'] else 'nobn'}"
                    ),
                    "architecture": arch,
                    "regularization": reg,
                    "optimization": baseline_opt,
                    "imbalance": baseline_imb,
                }
                experiments.append(exp)
        
        # Stage 3: Top 3 architectures with all optimization
        top3_archs = ARCHITECTURES[:3]
        for arch in top3_archs:
            for opt in OPTIMIZATION:
                if opt == baseline_opt:
                    continue  # Skip, already tested
                exp = {
                    "name": (
                        f"{arch['name']}_lr{opt['lr']:.0e}_"
                        f"bs{opt['batch_size']}_"
                        f"{opt['optimizer']}"
                    ),
                    "architecture": arch,
                    "regularization": baseline_reg,
                    "optimization": opt,
                    "imbalance": baseline_imb,
                }
                experiments.append(exp)
        
        # Stage 4: Top 2 architectures with all imbalance handling
        top2_archs = ARCHITECTURES[:2]
        for arch in top2_archs:
            for imb in IMBALANCE_HANDLING:
                if imb == baseline_imb:
                    continue  # Skip, already tested
                exp = {
                    "name": (
                        f"{arch['name']}_{imb['loss']}"
                        f"{'_smote' if imb['use_smote'] else ''}"
                    ),
                    "architecture": arch,
                    "regularization": baseline_reg,
                    "optimization": baseline_opt,
                    "imbalance": imb,
                }
                experiments.append(exp)
    
    return experiments


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

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
    device = DEVICE
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Disable UCX to prevent HPC segmentation faults
    os.environ['UCX_ERROR_SIGNALS'] = ''
    os.environ['UCX_WARN_UNUSED_ENV_VARS'] = 'n'
    os.environ['OMPI_MCA_pml'] = 'ob1'
    os.environ['OMPI_MCA_btl'] = '^openib'
    
    # Set PyTorch threading for optimal CPU usage
    if DEVICE == "cpu":
        torch.set_num_threads(TORCH_THREADS)
        print(f"PyTorch threads set to: {TORCH_THREADS}")
    
    print("="*80)
    print("STANDALONE NEURAL NETWORK EXPERIMENT SWEEP")
    print("="*80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Device: {DEVICE}")
    print(f"DataLoader workers: {NUM_WORKERS}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print("="*80)
    print("✓ CPU OPTIMIZATION ACTIVE: PyTorch using all 16 cores")
    print("✓ DataLoader workers disabled (UCX/HPC compatibility)")
    print("✓ Expected speedup: 2.0x faster than default")
    print("✓ Estimated time: ~5-6 hours for 41 experiments")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = load_data(Path(DATASET_PATH))
    print(f"Loaded {len(df)} samples")
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)-1} features (excluding 'wave')")
    
    # Build experiment grid (exhaustive mode)
    all_experiments = build_experiment_grid("exhaustive")
    print(f"\nTotal experiments in grid: {len(all_experiments)}")
    
    # Skip first 10, reverse remaining 41
    experiments_to_run = all_experiments[10:][::-1]
    print(f"Running {len(experiments_to_run)} experiments (11-51 in reverse order)")
    print(f"Skipping first 10 experiments (already completed)")
    
    # Create output directory
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Run experiments
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENTS")
    print(f"{'='*80}")
    
    results = []
    for i, exp_config in enumerate(experiments_to_run, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(experiments_to_run)}: {exp_config['name']}")
        print(f"{'='*80}")
        
        try:
            result = run_one_experiment(exp_config, df, feature_cols, output_dir)
            results.append(result)
        except Exception as e:
            print(f"ERROR: Experiment {exp_config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ALL {len(results)} EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    
    # Print top 5 by ROC-AUC
    if results:
        valid_results = [r for r in results if r.get("status") != "failed"]
        if valid_results:
            sorted_results = sorted(
                valid_results,
                key=lambda x: x["aggregate_metrics"]["mean_roc_auc"],
                reverse=True
            )
            print(f"\nTop 5 Experiments by ROC-AUC:")
            print("-"*80)
            for i, r in enumerate(sorted_results[:5], 1):
                metrics = r["aggregate_metrics"]
                print(
                    f"{i}. {r['experiment_name']}: "
                    f"ROC-AUC={metrics['mean_roc_auc']:.4f}, "
                    f"F2={metrics['mean_f2_score']:.4f}"
                )
    
    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
