import os
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ProteinRecord:
    id: str
    sequence: str
    labels: Optional[str] = None

def parse_fasta_3line(file_path: str) -> List[ProteinRecord]:
    """
    Parses a FASTA-like file where every 3 lines represent:
    1. Header (>ID)
    2. Sequence
    3. Binary Labels (00111...)
    """
    records = []
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        
        # Process in chunks of 3
        for i in range(0, len(lines), 3):
            header = lines[i]
            if not header.startswith(">"):
                continue
                
            id_val = header[1:]
            seq_val = lines[i+1]
            label_val = lines[i+2] if i+2 < len(lines) else None
            
            records.append(ProteinRecord(id=id_val, sequence=seq_val, labels=label_val))
            
    return records

def get_all_fasta_files(raw_dir: str) -> List[str]:
    """Helper to find all .fa files in a directory."""
    return [
        os.path.join(raw_dir, f) 
        for f in os.listdir(raw_dir) 
        if f.endswith(".fa") or f.endswith(".fasta")
    ]
