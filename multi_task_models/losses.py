"""Loss functions for multi-task learning with class imbalance."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Balancing parameter (default: 0.25)
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
