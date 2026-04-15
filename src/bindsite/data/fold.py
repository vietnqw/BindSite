"""ESMFold protein structure prediction.

Leverages the Hugging Face `transformers` implementation of ESMFold
to predict 3D structures (.pdb files) from amino acid sequences.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from bindsite.data.fasta import parse_fasta

logger = logging.getLogger(__name__)


def run_esmfold(
    fasta_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    chunk_size: int = 64,
) -> None:
    """Predict PDB structures for proteins in a FASTA file using ESMFold.

    Args:
        fasta_path: Path to the input FASTA file.
        output_dir: Directory to save the predicted .pdb files.
        device: Device to run the model on (e.g., 'cuda:0', 'cpu').
        chunk_size: Processing chunk size for attention to save VRAM.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load sequences using the standard parser.
    records = parse_fasta(fasta_path)
    sequences = {r.id: r.sequence for r in records}
    logger.info("Loaded %d sequences from %s", len(sequences), fasta_path)

    # Filter out already predicted proteins.
    remaining = {
        pid: seq for pid, seq in sequences.items() 
        if not (out_dir / f"{pid}.pdb").exists()
    }
    
    if not remaining:
        logger.info("All PDBs already exist! Exiting.")
        return
        
    logger.info("%d sequences need prediction.", len(remaining))

    # Lazy import to avoid loading massive libraries if not needed.
    logger.info("Loading ESMFold model from Hugging Face...")
    from transformers import AutoTokenizer, EsmForProteinFolding
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    
    if chunk_size > 0:
        model.trunk.set_chunk_size(chunk_size)
    
    model = model.eval().to(device)
    logger.info("Model loaded on %s.", device)

    # Inference loop
    for pid, seq in tqdm(remaining.items(), desc="Folding"):
        pdb_path = out_dir / f"{pid}.pdb"
        try:
            with torch.no_grad():
                tokenized = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
                input_ids = tokenized["input_ids"].to(device)
                
                output = model(input_ids)
                
                # Convert the raw predicted structures to PDB format output
                pdb_lines = model.output_to_pdb(output)
                pdb_str = pdb_lines[0]
                
            pdb_path.write_text(pdb_str)
            
        except torch.cuda.OutOfMemoryError:
            logger.error("OOM error on %s (length %d). Skipping.", pid, len(seq))
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            logger.error("Failed folding %s: %s", pid, e)
            continue

    logger.info("Structure prediction complete.")
