"""Standalone script for ESMFold protein structure prediction.

Leverages the Hugging Face `transformers` implementation of ESMFold
to avoid the complex dependencies of the original `fair-esm` + `openfold` stack.
Predicts .pdb files from a FASTA file.
"""

import argparse
import ast
import logging
from pathlib import Path

import torch
from tqdm import tqdm


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("esmfold")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(handler)
    return logger


def parse_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into a mapping of ID -> sequence.
    Handles DeepProSite's 3-line format (ID / sequence / binary label).
    """
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    records = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        pid = lines[i][1:].strip()
        seq = lines[i + 1].strip()
        records[pid] = seq
        
        # Check if next line is a binary label and skip it if so
        if i + 2 < len(lines) and not lines[i + 2].startswith(">") and set(lines[i + 2]) <= {"0", "1"}:
            i += 3
        else:
            i += 2
    return records


def main():
    parser = argparse.ArgumentParser(description="Predict 3D structures using ESMFold.")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", required=True, help="Output directory for PDBs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size for attention to save VRAM")
    args = parser.parse_args()

    logger = setup_logger()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        logger.error(f"Input file not found: {fasta_path}")
        return

    sequences = parse_fasta(fasta_path)
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")

    # Determine which sequences need processing
    remaining = {pid: seq for pid, seq in sequences.items() if not (out_dir / f"{pid}.pdb").exists()}
    if not remaining:
        logger.info("All PDBs already exist! Exiting.")
        return
        
    logger.info(f"{len(remaining)} sequences need prediction.")

    # Lazy import to keep CLI fast help
    logger.info("Loading ESMFold model from Hugging Face...")
    from transformers import AutoTokenizer, EsmForProteinFolding
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    
    # Enable memory saving features
    if args.chunk_size > 0:
        model.trunk.set_chunk_size(args.chunk_size)
    
    model = model.eval().to(args.device)
    logger.info(f"Model loaded on {args.device}.")

    # Inference loop
    for pid, seq in tqdm(remaining.items(), desc="Folding"):
        pdb_path = out_dir / f"{pid}.pdb"
        try:
            with torch.no_grad():
                tokenized = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
                input_ids = tokenized["input_ids"].to(args.device)
                
                output = model(input_ids)
                
                # Convert the raw predicted structures to PDB format output
                pdb_lines = model.output_to_pdb(output)
                pdb_str = pdb_lines[0]
                
            pdb_path.write_text(pdb_str)
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM error on {pid} (length {len(seq)}). Skipping.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            logger.error(f"Failed folding {pid}: {e}")
            continue

    logger.info("Done.")

if __name__ == "__main__":
    main()
