import numpy as np
from pathlib import Path
from dataclasses import dataclass
from ..core.logger import logger

@dataclass(frozen=True)
class BindingRecord:
    seq_id: str
    sequence: str
    labels: str

def parse_3line_fasta(path: Path) -> list[BindingRecord]:
    """Parses DeepProSite's 3-line FASTA format (ID, Sequence, Labels)."""
    if not path.exists():
        logger.error(f"FASTA file not found: {path}")
        return []
        
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3 != 0:
        logger.warning(f"Expected 3-line FASTA format in {path}, but got {len(lines)} lines.")

    records: list[BindingRecord] = []
    for idx in range(0, (len(lines) // 3) * 3, 3):
        header = lines[idx]
        sequence = lines[idx + 1]
        labels = lines[idx + 2]
        
        if not header.startswith(">"):
            logger.warning(f"Invalid FASTA header in {path}: {header}")
            continue
            
        records.append(BindingRecord(seq_id=header[1:], sequence=sequence, labels=labels))
    return records

def extract_ca_coordinates(pdb_path: Path) -> np.ndarray:
    """Extracts Alpha-Carbon coordinates from a PDB file."""
    coords = []
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except ValueError:
                        continue
    except Exception as e:
        logger.error(f"Error reading PDB {pdb_path}: {e}")
        
    return np.array(coords, dtype=np.float32)
