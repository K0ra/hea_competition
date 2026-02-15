"""PyTorch dataset and data loading for multi-task learning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, Dataset


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
    
    # Map task column names to suffixes
    from utils import DISEASE_SUFFIXES
    
    suffix_to_col = {v: k for k, v in DISEASE_SUFFIXES.items()}
    
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
        num_workers=0,
        pin_memory=True,
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
            num_workers=0,
            pin_memory=True,
        )
        val_loaders[suffix] = val_loader
    
    return train_loader, val_loaders
