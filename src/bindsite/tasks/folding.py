import logging
from pathlib import Path
from typing import List

import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

logger = logging.getLogger(__name__)

class ESMFolder:
    """Class to handle protein folding using ESMFold."""

    def __init__(self, model_name: str = "facebook/esmfold_v1", device: str = None, chunk_size: int = 128):
        """
        Initialize the ESMFold model and tokenizer.

        Args:
            model_name: The name or path of the pretrained model.
            device: The device to run the model on ('cuda' or 'cpu').
            chunk_size: Processing chunk size for attention to save VRAM.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForProteinFolding.from_pretrained(model_name, use_safetensors=True)
        
        if chunk_size > 0:
            self.model.trunk.set_chunk_size(chunk_size)
            
        self.model.to(self.device).eval()

    def _parse_fasta(self, fasta_path: Path):
        """
        Parses FASTA files, supporting both standard and 3-line (DeepProSite) formats.
        Yields: (protein_id, sequence)
        """
        with open(fasta_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(">"):
                protein_id = line[1:]
                # Check if next line is sequence
                if i + 1 < len(lines):
                    sequence = lines[i+1]
                    # If it's a 3-line format, the next line might be labels (0/1 string)
                    # We can detect this if it's purely digits.
                    # Standard fasta might have sequence split across multiple lines,
                    # but BindSite datasets use 3-line format.
                    i += 2
                    # Skip the label line if it looks like one
                    if i < len(lines) and not lines[i].startswith(">") and all(c in "01" for c in lines[i]):
                        i += 1
                    yield protein_id, sequence
                else:
                    i += 1
            else:
                i += 1

    def fold_fasta(self, fasta_paths: List[Path], output_dir: Path):
        """
        Fold sequences from FASTA files and save as PDB files.

        Args:
            fasta_paths: List of paths to .fa files.
            output_dir: Directory where the .pdb files will be saved.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        all_sequences = []
        for fasta_path in fasta_paths:
            if not fasta_path.exists():
                logger.warning(f"File {fasta_path} does not exist. Skipping.")
                continue
            for protein_id, sequence in self._parse_fasta(fasta_path):
                all_sequences.append((protein_id, sequence))

        if not all_sequences:
            logger.warning("No sequences found to fold.")
            return

        logger.info(f"Total sequences to fold: {len(all_sequences)}")

        for protein_id, sequence in tqdm(all_sequences, desc="Folding protein sequences"):
            # Sanitise protein_id for filename
            safe_id = protein_id.replace("/", "_").replace("|", "_")
            output_path = output_dir / f"{safe_id}.pdb"

            if output_path.exists():
                logger.debug(f"Skipping {safe_id}, already exists at {output_path}")
                continue

            try:
                # Clean sequence: ESMFold tokenizer only expects amino acids
                clean_sequence = "".join([c for c in sequence if c.isalpha()]).upper()
                seq_len = len(clean_sequence)
                
                if not clean_sequence:
                    logger.warning(f"Empty sequence for {protein_id}. Skipping.")
                    continue

                logger.debug(f"Folding {protein_id} (length: {seq_len})...")

                with torch.no_grad():
                    inputs = self.tokenizer(clean_sequence, return_tensors="pt", add_special_tokens=False)
                    input_ids = inputs["input_ids"].to(self.device)

                    outputs = self.model(input_ids)

                    # Fix pLDDT scaling: ESMFold outputs 0-1, PDB expects 0-100.
                    outputs.plddt *= 100

                    pdb_strings = self.model.output_to_pdb(outputs)

                    with open(output_path, "w") as f:
                        f.write(pdb_strings[0])

            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM error for {protein_id} (length: {len(sequence)}). Skipping.")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error folding {protein_id}: {e}")
