"""Experiment configurations for neural network sweep."""
from __future__ import annotations

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


def get_experiment_count(mode: str = "exhaustive") -> int:
    """Get expected number of experiments for a mode."""
    if mode == "quick":
        return len(ARCHITECTURES)  # ~13
    elif mode == "comprehensive":
        # 13 archs × 3 reg × 2 opt × 2 imb
        return len(ARCHITECTURES) * 3 * 2 * 2  # ~156
    else:  # exhaustive
        # Stage 1: 13
        # Stage 2: 5 × 4 = 20
        # Stage 3: 3 × 4 = 12
        # Stage 4: 2 × 3 = 6
        # Total: ~51
        grid = build_experiment_grid(mode)
        return len(grid)
