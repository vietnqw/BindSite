"""Inference and evaluation pipeline.

Loads trained model checkpoints (from K-fold CV), performs ensemble
prediction by averaging across folds, and evaluates against ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bindsite.config import ModelConfig, TrainingConfig
from bindsite.data.dataset import create_dataloader
from bindsite.model.graph_transformer import GraphTransformer
from bindsite.training.metrics import MetricsResult, compute_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Inference pipeline using an ensemble of K-fold trained models.

    Args:
        model_config: Model architecture configuration.
        training_config: Training configuration (for batch size, max_len, etc.).
        checkpoint_dir: Directory containing fold checkpoint files (fold0.pt, ...).
        tensor_dir: Directory containing pre-computed protein tensors.
        device: Device for inference.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        checkpoint_dir: str | Path,
        tensor_dir: str | Path,
        device: str | None = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = training_config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tensor_dir = Path(tensor_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _load_models(self) -> list[GraphTransformer]:
        """Load all available fold checkpoints.

        Returns:
            List of loaded models in eval mode.
        """
        models: list[GraphTransformer] = []
        for ckpt_path in sorted(self.checkpoint_dir.glob("fold*.pt")):
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
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(self.device).eval()
            models.append(model)
            logger.info("Loaded checkpoint: %s", ckpt_path.name)

        logger.info("Loaded %d models for ensemble prediction.", len(models))
        return models

    @torch.no_grad()
    def predict(
        self,
        test_df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Run ensemble prediction on test data.

        Averages predictions from all fold models.

        Args:
            test_df: Test DataFrame with 'ID' and 'sequence' columns.

        Returns:
            Dict mapping protein_id -> predicted probabilities array.
        """
        models = self._load_models()
        if not models:
            raise RuntimeError(
                f"No checkpoint files found in {self.checkpoint_dir}"
            )

        dataloader = create_dataloader(
            test_df,
            self.tensor_dir,
            batch_size=self.train_config.batch_size,
            max_len=self.train_config.max_seq_len,
            num_workers=self.train_config.num_workers,
        )

        predictions: dict[str, np.ndarray] = {}

        for batch in tqdm(dataloader, desc="Predicting"):
            coords = batch["coords"].to(self.device)
            node_feats = batch["node_features"].to(self.device)
            mask = batch["mask"].to(self.device)
            pdb_ids = batch["pdb_ids"]

            # Ensemble: average sigmoid outputs from all models.
            all_outputs = torch.stack([
                model(coords, node_feats, mask).sigmoid()
                for model in models
            ], dim=0)
            avg_output = all_outputs.mean(dim=0)  # (B, L)

            # Extract per-protein predictions (remove padding).
            for i, pid in enumerate(pdb_ids):
                seq_len = int(mask[i].sum().item())
                predictions[pid] = avg_output[i, :seq_len].cpu().numpy()

        return predictions

    def evaluate(
        self,
        test_df: pd.DataFrame,
    ) -> MetricsResult:
        """Run prediction and evaluate against ground truth labels.

        Args:
            test_df: Test DataFrame with 'ID', 'sequence', and 'label' columns.

        Returns:
            Evaluation metrics.
        """
        import ast

        predictions = self.predict(test_df)

        all_preds: list[float] = []
        all_labels: list[int] = []

        for _, row in test_df.iterrows():
            pid = row["ID"]
            if pid not in predictions:
                logger.warning("Missing predictions for %s", pid)
                continue

            raw_label = row["label"]
            labels = ast.literal_eval(raw_label) if isinstance(raw_label, str) else list(raw_label)
            preds = predictions[pid]

            L = min(len(labels), len(preds))
            all_labels.extend(labels[:L])
            all_preds.extend(preds[:L].tolist())

        metrics = compute_metrics(
            np.array(all_preds), np.array(all_labels)
        )
        logger.info("Test results: %s", metrics)
        return metrics
