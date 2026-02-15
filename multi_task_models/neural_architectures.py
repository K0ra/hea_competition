"""Neural network architectures for multi-task learning."""
from __future__ import annotations

import torch
import torch.nn as nn


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


class MultiTaskTabNet(nn.Module):
    """
    TabNet wrapper for multi-task learning.
    
    Uses sequential attention for feature selection.
    """
    
    def __init__(
        self,
        n_features: int,
        task_names: list[str],
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
    ):
        super().__init__()
        self.task_names = task_names
        
        # For simplicity, use separate TabNet models per task
        # (Full multi-task TabNet is complex to implement from scratch)
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            self.use_tabnet = True
            # Will be initialized in training script
        except ImportError:
            self.use_tabnet = False
            # Fallback to MLP
            self.fallback = MultiTaskMLP(
                n_features, [256, 128, 64], task_names, dropout=0.3
            )
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.use_tabnet:
            return self.fallback(x)
        # TabNet forward handled separately in training
        raise NotImplementedError(
            "TabNet forward handled in training script"
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    elif arch_type == "tabnet":
        n_d = architecture_config.get("n_d", 64)
        n_a = architecture_config.get("n_a", 64)
        n_steps = architecture_config.get("n_steps", 3)
        return MultiTaskTabNet(
            n_features, task_names, n_d, n_a, n_steps
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")
