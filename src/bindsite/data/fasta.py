"""FASTA file parsing utilities.

Supports both 2-line format (>ID / sequence) and 3-line format
(>ID / sequence / binary_label) used by DeepProSite datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FastaRecord:
    """A single parsed FASTA record.

    Attributes:
        id: Protein identifier (header line without '>').
        sequence: Amino acid sequence string.
        label: Optional per-residue binary labels (list of 0/1 ints).
    """

    id: str
    sequence: str
    label: list[int] | None = None


def parse_fasta(path: str | Path) -> list[FastaRecord]:
    """Parse a FASTA file into a list of records.

    Automatically detects whether the file uses 2-line or 3-line format
    by checking if the line after the sequence is a binary string (only
    contains '0' and '1' characters).

    Args:
        path: Path to the FASTA file.

    Returns:
        List of parsed FastaRecord objects.

    Raises:
        FileNotFoundError: If the FASTA file does not exist.
        ValueError: If a header line does not start with '>'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    records: list[FastaRecord] = []
    i = 0
    while i < len(lines):
        header = lines[i]
        if not header.startswith(">"):
            raise ValueError(f"Expected header starting with '>', got: {header!r}")

        protein_id = header[1:].strip()
        sequence = lines[i + 1].strip()

        # Check if the next line is a binary label string.
        label = None
        if (
            i + 2 < len(lines)
            and not lines[i + 2].startswith(">")
            and set(lines[i + 2]) <= {"0", "1"}
        ):
            label = [int(c) for c in lines[i + 2]]
            i += 3
        else:
            i += 2

        records.append(FastaRecord(id=protein_id, sequence=sequence, label=label))

    return records
