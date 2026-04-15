"""Training loop with K-fold cross-validation.

Orchestrates the training process following the original DeepProSite protocol:
  - 5-fold cross-validation with stratified splitting
  - Random oversampling per epoch (5× training set size)
  - BCEWithLogitsLoss with masking for variable-length sequences
  - Noam learning rate schedule
  - Early stopping based on validation AUPRC
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from tqdm import tqdm

from bindsite.config import ModelConfig, TrainingConfig
from bindsite.data.dataset import create_dataloader
from bindsite.model.graph_transformer import GraphTransformer
from bindsite.model.scheduler import create_optimizer_and_scheduler
from bindsite.training.metrics import MetricsResult, compute_metrics

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Trainer:
    """Manages training, validation, and checkpointing for a GraphTransformer.

    Args:
        model_config: Model architecture configuration.
        training_config: Training hyperparameter configuration.
        tensor_dir: Directory containing pre-computed protein tensors.
        output_dir: Directory for saving checkpoints and logs.
        device: Device to train on (auto-detected if None).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        tensor_dir: str | Path,
        output_dir: str | Path,
        device: str | None = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = training_config
        self.tensor_dir = Path(tensor_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        seed_everything(training_config.seed)
        logger.info("Training on device: %s", self.device)

    def _create_model(self) -> GraphTransformer:
        """Instantiate a fresh model from config."""
        model = GraphTransformer(
            node_features=self.model_config.node_features,
            edge_features=self.model_config.edge_features,
            hidden_dim=self.model_config.hidden_dim,
            num_encoder_layers=self.model_config.num_encoder_layers,
            num_attention_heads=self.model_config.num_attention_heads,
            k_neighbors=self.model_config.k_neighbors,
            augment_eps=self.model_config.augment_eps,
            dropout=self.model_config.dropout,
        )
        return model.to(self.device)

    def _train_one_epoch(
        self,
        model: GraphTransformer,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
    ) -> tuple[float, MetricsResult]:
        """Train for one epoch.

        Args:
            model: The model to train.
            dataloader: Training data loader.
            optimizer: Optimizer.
            scheduler: LR scheduler (stepped per batch).
            loss_fn: Loss function (BCEWithLogitsLoss).

        Returns:
            Tuple of (average_loss, training_metrics).
        """
        model.train()
        total_loss = 0.0
        total_residues = 0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch in tqdm(dataloader, desc="Training", leave=False):
            optimizer.zero_grad()

            coords = batch["coords"].to(self.device)
            node_feats = batch["node_features"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = model(coords, node_feats, mask)

            # Masked loss: only compute on real (non-padded) residues.
            loss = loss_fn(logits, labels) * mask.float()
            loss = loss.sum() / mask.sum()
            loss.backward()

            optimizer.step()
            scheduler.step()

            # Collect predictions for metrics.
            with torch.no_grad():
                probs = logits.sigmoid()
                preds_masked = torch.masked_select(probs, mask.bool())
                labels_masked = torch.masked_select(labels, mask.bool())

            all_preds.append(preds_masked.cpu().numpy())
            all_labels.append(labels_masked.cpu().numpy())

            n_residues = int(mask.sum().item())
            total_loss += n_residues * loss.item()
            total_residues += n_residues

        avg_loss = total_loss / total_residues
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = compute_metrics(preds, labels)

        return avg_loss, metrics

    @torch.no_grad()
    def _evaluate(
        self,
        model: GraphTransformer,
        dataloader: torch.utils.data.DataLoader,
    ) -> MetricsResult:
        """Evaluate model on a validation/test set.

        Args:
            model: Model to evaluate.
            dataloader: Evaluation data loader.

        Returns:
            Evaluation metrics.
        """
        model.eval()
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            coords = batch["coords"].to(self.device)
            node_feats = batch["node_features"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = model(coords, node_feats, mask)
            probs = logits.sigmoid()

            preds_masked = torch.masked_select(probs, mask.bool())
            labels_masked = torch.masked_select(labels, mask.bool())

            all_preds.append(preds_masked.cpu().numpy())
            all_labels.append(labels_masked.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        return compute_metrics(preds, labels)

    def train_kfold(self, train_df: pd.DataFrame) -> list[MetricsResult]:
        """Run K-fold cross-validation training.

        Args:
            train_df: Training DataFrame with 'ID', 'sequence', 'label' columns.

        Returns:
            List of best validation metrics for each fold.
        """
        cfg = self.train_config
        kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        all_fold_metrics: list[MetricsResult] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            logger.info("=" * 60)
            logger.info("Fold %d / %d", fold + 1, cfg.n_folds)
            logger.info("=" * 60)

            # Create data loaders.
            train_loader = create_dataloader(
                train_df.iloc[train_idx],
                self.tensor_dir,
                batch_size=cfg.batch_size,
                max_len=cfg.max_seq_len,
                num_workers=cfg.num_workers,
                num_samples=cfg.num_samples_per_epoch,
            )
            val_loader = create_dataloader(
                train_df.iloc[val_idx],
                self.tensor_dir,
                batch_size=cfg.batch_size,
                max_len=cfg.max_seq_len,
                num_workers=cfg.num_workers,
            )

            # Create model, optimizer, scheduler.
            model = self._create_model()
            train_size = cfg._train_sizes.get(cfg.task, 335)
            optimizer, scheduler = create_optimizer_and_scheduler(
                model,
                d_model=self.model_config.hidden_dim,
                train_size=train_size,
                batch_size=cfg.batch_size,
                warmup_epochs=cfg.warmup_epochs,
                peak_lr=cfg.peak_lr,
            )
            loss_fn = nn.BCEWithLogitsLoss(reduction="none")

            best_auprc = 0.0
            patience_counter = 0
            best_metrics: MetricsResult | None = None

            for epoch in range(cfg.epochs):
                # Train.
                train_loss, train_metrics = self._train_one_epoch(
                    model, train_loader, optimizer, scheduler, loss_fn
                )

                # Validate.
                val_metrics = self._evaluate(model, val_loader)

                # Check improvement.
                improved = val_metrics.auprc > best_auprc
                if improved:
                    best_auprc = val_metrics.auprc
                    best_metrics = val_metrics
                    patience_counter = 0

                    # Save checkpoint.
                    ckpt_path = self.output_dir / f"fold{fold}.pt"
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    patience_counter += 1

                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "[Epoch %2d] lr=%.6f loss=%.4f | "
                    "train AUC=%.4f AUPRC=%.4f | "
                    "val AUC=%.4f AUPRC=%.4f%s",
                    epoch + 1, lr, train_loss,
                    train_metrics.auc_roc, train_metrics.auprc,
                    val_metrics.auc_roc, val_metrics.auprc,
                    "" if improved else f" (patience {patience_counter}/{cfg.patience})",
                )

                if patience_counter >= cfg.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

            if best_metrics is not None:
                logger.info("Fold %d best: %s", fold + 1, best_metrics)
                all_fold_metrics.append(best_metrics)

        # Summary.
        if all_fold_metrics:
            mean_auprc = np.mean([m.auprc for m in all_fold_metrics])
            mean_auc = np.mean([m.auc_roc for m in all_fold_metrics])
            logger.info(
                "CV Summary: mean AUC-ROC=%.4f, mean AUPRC=%.4f",
                mean_auc, mean_auprc,
            )

        return all_fold_metrics
