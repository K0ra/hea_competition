"""Training infrastructure for neural multi-task models."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "multi_task_models"))

from utils import evaluate_multi_task


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
