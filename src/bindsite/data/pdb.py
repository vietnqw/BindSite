"""PDB file parsing for extracting Cα atom coordinates.

Only reads ATOM records to extract alpha-carbon (CA) positions, which are
used as node positions in the protein graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_ca_coords(pdb_path: str | Path) -> np.ndarray:
    """Extract Cα (alpha-carbon) coordinates from a PDB file.

    Parses ATOM records and extracts one CA coordinate per residue. Handles
    multi-model PDB files by reading only the first model.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        Array of shape (L, 3) where L is the number of residues.

    Raises:
        FileNotFoundError: If the PDB file does not exist.
        ValueError: If no CA atoms are found.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    coords: list[np.ndarray] = []
    current_residue_idx = -9999
    current_ca: np.ndarray | None = None

    for line in pdb_path.read_text().splitlines():
        record_type = line[:6].strip()

        # Stop at TER or END to avoid reading multiple models.
        if record_type == "TER":
            # Flush the last residue before TER.
            if current_ca is not None:
                coords.append(current_ca)
                current_ca = None
            continue

        if record_type != "ATOM":
            continue

        residue_idx = int(line[22:26].strip())

        # New residue encountered — flush previous.
        if residue_idx != current_residue_idx:
            if current_ca is not None:
                coords.append(current_ca)
                current_ca = None
            current_residue_idx = residue_idx

        atom_name = line[12:16].strip()
        if atom_name == "CA":
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            current_ca = np.array([x, y, z], dtype=np.float32)

    # Flush trailing residue.
    if current_ca is not None:
        coords.append(current_ca)

    if not coords:
        raise ValueError(f"No CA atoms found in {pdb_path}")

    return np.stack(coords)
