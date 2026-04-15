"""PyTorch Dataset and DataLoader factories for protein binding site data."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler


class ProteinDataset(Dataset):
    """Dataset for protein binding site prediction.

    Loads pre-computed padded tensors (coords, node_features, mask) from disk
    and optionally pairs them with per-residue binary labels from a CSV.

    Args:
        df: DataFrame with at least an 'ID' column and optionally 'sequence'
            and 'label' columns.
        tensor_dir: Directory containing pre-computed tensor files.
        max_len: Maximum sequence length (must match padding used in tensor creation).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tensor_dir: str | Path,
        max_len: int = 1000,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tensor_dir = Path(tensor_dir)
        self.max_len = max_len
        self.has_label = "label" in df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        pid = row["ID"]

        item: dict[str, torch.Tensor | str] = {
            "pdb_id": pid,
            "coords": torch.load(
                self.tensor_dir / f"{pid}_coords.pt", weights_only=True
            ),
            "node_features": torch.load(
                self.tensor_dir / f"{pid}_node_features.pt", weights_only=True
            ),
            "mask": torch.load(
                self.tensor_dir / f"{pid}_mask.pt", weights_only=True
            ),
        }

        if self.has_label:
            raw_label = row["label"]
            if isinstance(raw_label, str):
                label_list = ast.literal_eval(raw_label)
            else:
                label_list = list(raw_label)

            padded_label = np.zeros(self.max_len, dtype=np.float32)
            padded_label[: len(label_list)] = label_list
            item["label"] = torch.tensor(padded_label, dtype=torch.float32)

        return item


def collate_fn(
    batch: list[dict[str, torch.Tensor | str]],
) -> dict[str, torch.Tensor | list[str]]:
    """Collate a batch of protein samples into stacked tensors.

    Args:
        batch: List of sample dicts from ProteinDataset.__getitem__.

    Returns:
        Dict with stacked tensors and list of PDB IDs.
    """
    result: dict[str, torch.Tensor | list[str]] = {
        "pdb_ids": [item["pdb_id"] for item in batch],  # type: ignore[misc]
        "coords": torch.stack([item["coords"] for item in batch]),  # type: ignore[arg-type]
        "node_features": torch.stack([item["node_features"] for item in batch]),  # type: ignore[arg-type]
        "mask": torch.stack([item["mask"] for item in batch]),  # type: ignore[arg-type]
    }

    if "label" in batch[0]:
        result["label"] = torch.stack([item["label"] for item in batch])  # type: ignore[arg-type]

    return result


def create_dataloader(
    df: pd.DataFrame,
    tensor_dir: str | Path,
    batch_size: int = 32,
    max_len: int = 1000,
    num_workers: int = 4,
    shuffle: bool = False,
    num_samples: int | None = None,
) -> DataLoader:
    """Create a DataLoader for protein binding site data.

    Args:
        df: DataFrame with protein IDs and optional labels.
        tensor_dir: Directory containing pre-computed tensor files.
        batch_size: Batch size.
        max_len: Maximum sequence length for padding.
        num_workers: Number of data loading worker processes.
        shuffle: Whether to shuffle the data (ignored if num_samples is set).
        num_samples: If set, use random sampling with replacement (for training).

    Returns:
        Configured PyTorch DataLoader.
    """
    dataset = ProteinDataset(df, tensor_dir, max_len)

    sampler = None
    if num_samples is not None:
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        drop_last=(num_samples is not None),  # Drop last for training only.
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
