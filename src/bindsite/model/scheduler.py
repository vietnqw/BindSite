"""Noam learning rate scheduler.

Implements the warmup + inverse square root decay schedule from
"Attention Is All You Need" (Vaswani et al., 2017), as used in
the original DeepProSite training.
"""

from __future__ import annotations

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


def create_noam_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup_steps: int,
    peak_lr: float = 4e-4,
) -> LambdaLR:
    """Create a Noam learning rate scheduler.

    The learning rate follows:
        lr = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    The factor is computed so that the peak learning rate equals `peak_lr`.

    Args:
        optimizer: The optimizer to schedule.
        d_model: Model hidden dimension (used in the Noam formula).
        warmup_steps: Number of warmup steps.
        peak_lr: Target peak learning rate (reached at warmup_steps).

    Returns:
        A PyTorch LambdaLR scheduler.
    """
    # Compute the factor that makes lr == peak_lr at step == warmup_steps.
    peak_noam = d_model ** (-0.5) * min(
        warmup_steps ** (-0.5), warmup_steps * warmup_steps ** (-1.5)
    )
    factor = peak_lr / peak_noam

    def lr_lambda(step: int) -> float:
        # Avoid division by zero at step 0.
        step = max(step, 1)
        return factor * d_model ** (-0.5) * min(
            step ** (-0.5), step * warmup_steps ** (-1.5)
        )

    return LambdaLR(optimizer, lr_lambda)


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    d_model: int,
    train_size: int,
    batch_size: int = 32,
    warmup_epochs: int = 5,
    peak_lr: float = 4e-4,
) -> tuple[Adam, LambdaLR]:
    """Create an Adam optimizer with Noam learning rate schedule.

    Args:
        model: The model whose parameters to optimize.
        d_model: Model hidden dimension.
        train_size: Number of proteins in the training set.
        batch_size: Training batch size.
        warmup_epochs: Number of warmup epochs.
        peak_lr: Target peak learning rate.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    optimizer = Adam(
        model.parameters(),
        lr=1.0,  # Will be overridden by the scheduler.
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    steps_per_epoch = max(1, train_size // batch_size)
    warmup_steps = warmup_epochs * steps_per_epoch

    scheduler = create_noam_scheduler(optimizer, d_model, warmup_steps, peak_lr)

    return optimizer, scheduler


def create_modern_optimizer_and_scheduler(
    model: torch.nn.Module,
    train_size: int,
    epochs: int,
    batch_size: int = 32,
    peak_lr: float = 5e-4,
    weight_decay: float = 1e-2,
) -> tuple[AdamW, OneCycleLR]:
    """Create an AdamW optimizer with OneCycleLR schedule.

    Args:
        model: The model whose parameters to optimize.
        train_size: Number of proteins in the training set (or num_samples_per_epoch).
        epochs: Total number of training epochs.
        batch_size: Training batch size.
        peak_lr: Maximum learning rate.
        weight_decay: Weight decay factor.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    optimizer = AdamW(
        model.parameters(),
        lr=peak_lr,
        weight_decay=weight_decay,
    )

    steps_per_epoch = max(1, train_size // batch_size)
    total_steps = epochs * steps_per_epoch

    scheduler = OneCycleLR(
        optimizer,
        max_lr=peak_lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    return optimizer, scheduler
