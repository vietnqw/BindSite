"""Feature merging, padding, and tensor creation.

Combines ProtT5 embeddings (1024-d) and DSSP features (14-d) into
final node feature tensors, along with Cα coordinate and mask tensors.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from bindsite.data.pdb import extract_ca_coords

logger = logging.getLogger(__name__)


def prepare_protein_tensors(
    protein_id: str,
    pdb_dir: str | Path,
    protrans_dir: str | Path,
    dssp_dir: str | Path,
    max_len: int = 1000,
    node_dim: int = 1038,
) -> dict[str, torch.Tensor]:
    """Prepare padded tensors for a single protein.

    Merges ProtT5 embeddings and DSSP features into a single node feature
    matrix, pads all tensors to a fixed length, and returns them as a dict.

    Args:
        protein_id: Protein identifier.
        pdb_dir: Directory containing PDB files (``{protein_id}.pdb``).
        protrans_dir: Directory containing normalized ProtT5 embeddings.
        dssp_dir: Directory containing DSSP feature files.
        max_len: Maximum sequence length for padding.
        node_dim: Expected node feature dimensionality (1024 + 14).

    Returns:
        Dict with keys:
            - "coords":        Cα coordinates, shape (max_len, 3)
            - "node_features": ProtT5 + DSSP features, shape (max_len, node_dim)
            - "mask":          Binary mask, shape (max_len,)

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    pdb_dir = Path(pdb_dir)
    protrans_dir = Path(protrans_dir)
    dssp_dir = Path(dssp_dir)

    # Load raw data.
    coords = extract_ca_coords(pdb_dir / f"{protein_id}.pdb")
    protrans = np.load(protrans_dir / f"{protein_id}.npy")
    dssp = np.load(dssp_dir / f"{protein_id}.npy")

    seq_len = coords.shape[0]

    # Merge node features.
    node_features = np.hstack([protrans[:seq_len], dssp[:seq_len]])
    assert node_features.shape[1] == node_dim, (
        f"Expected {node_dim} node features, got {node_features.shape[1]}"
    )

    # Pad to fixed length.
    padded_coords = np.zeros((max_len, 3), dtype=np.float32)
    padded_coords[:seq_len] = coords

    padded_features = np.zeros((max_len, node_dim), dtype=np.float32)
    padded_features[:seq_len] = node_features

    mask = np.zeros(max_len, dtype=np.int64)
    mask[:seq_len] = 1

    return {
        "coords": torch.tensor(padded_coords, dtype=torch.float32),
        "node_features": torch.tensor(padded_features, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.long),
    }


def prepare_and_save(
    protein_id: str,
    pdb_dir: str | Path,
    protrans_dir: str | Path,
    dssp_dir: str | Path,
    output_dir: str | Path,
    max_len: int = 1000,
    skip_existing: bool = True,
) -> bool:
    """Prepare and save tensors for a single protein.

    Args:
        protein_id: Protein identifier.
        pdb_dir: Directory containing PDB files.
        protrans_dir: Directory containing normalized ProtT5 embeddings.
        dssp_dir: Directory containing DSSP feature files.
        output_dir: Directory to save output tensors.
        max_len: Maximum sequence length for padding.
        skip_existing: If True, skip proteins whose tensors already exist.

    Returns:
        True if tensors were created, False if skipped or failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and (output_dir / f"{protein_id}_coords.pt").exists():
        return False

    try:
        tensors = prepare_protein_tensors(
            protein_id, pdb_dir, protrans_dir, dssp_dir, max_len
        )
        torch.save(tensors["coords"], output_dir / f"{protein_id}_coords.pt")
        torch.save(tensors["node_features"], output_dir / f"{protein_id}_node_features.pt")
        torch.save(tensors["mask"], output_dir / f"{protein_id}_mask.pt")
        return True
    except Exception as e:
        logger.warning("Failed to prepare features for %s: %s", protein_id, e)
        return False
