"""Configuration dataclasses for the BindSite model.

Centralizes all hyperparameters and settings into typed, immutable dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters for the Graph Transformer model.

    Attributes:
        node_features: Dimensionality of input node features (ProtT5 + DSSP).
        edge_features: Dimensionality of edge feature embeddings.
        hidden_dim: Hidden dimensionality throughout the transformer layers.
        num_encoder_layers: Number of stacked Graph Transformer layers.
        num_attention_heads: Number of attention heads in multi-head attention.
        k_neighbors: Number of nearest neighbors in the KNN graph.
        augment_eps: Scale of Gaussian noise added for data augmentation.
        dropout: Dropout probability applied in transformer layers.
    """

    node_features: int = 1038  # 1024 (ProtT5) + 14 (DSSP)
    edge_features: int = 16
    hidden_dim: int = 64
    num_encoder_layers: int = 4
    num_attention_heads: int = 4
    k_neighbors: int = 30
    augment_eps: float = 0.1
    dropout: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop and optimization hyperparameters.

    Attributes:
        task: Binding site prediction task — "PRO" (protein) or "PEP" (peptide).
        epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        batch_size: Number of proteins per training batch.
        n_folds: Number of cross-validation folds.
        seed: Random seed for reproducibility.
        max_seq_len: Maximum sequence length for padding.
        num_workers: DataLoader worker processes.
        peak_lr: Peak learning rate for the Noam scheduler.
        warmup_epochs: Number of warmup epochs for the Noam scheduler.
    """

    task: str = "PRO"
    epochs: int = 30
    patience: int = 8
    batch_size: int = 64
    n_folds: int = 5
    seed: int = 42
    max_seq_len: int = 1000
    num_workers: int = 4
    peak_lr: float = 4e-4
    warmup_epochs: int = 5

    # Dataset sizes used to compute number of samples per epoch.
    # Maps task name -> number of training proteins.
    _train_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "PRO": 335,
            "PEP": 1154,
        }
    )

    @property
    def num_samples_per_epoch(self) -> int:
        """Number of samples drawn per epoch (with replacement).

        Following the original paper: 5× the training set size per epoch.
        """
        return self._train_sizes.get(self.task, 335) * 5


@dataclass(frozen=True)
class PathConfig:
    """File system paths for data, features, and outputs.

    Attributes:
        data_dir: Root directory containing FASTA files and PDB structures.
        feature_dir: Directory for intermediate feature files.
        output_dir: Directory for model checkpoints and logs.
        pdb_dir: Directory containing PDB structure files.
        dssp_binary: Path to the DSSP executable.
        protrans_model: Path or HuggingFace ID for the ProtT5 model.
    """

    data_dir: Path = Path("./data")
    feature_dir: Path = Path("./features")
    output_dir: Path = Path("./output")
    pdb_dir: Path = Path("./data/pdb")
    dssp_binary: str = "mkdssp"
    protrans_model: str = "Rostlab/prot_t5_xl_uniref50"
