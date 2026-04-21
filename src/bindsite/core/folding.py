import os
import torch
from tqdm import tqdm
from Bio import SeqIO
from transformers import EsmTokenizer, EsmForProteinFolding
import logging

logger = logging.getLogger(__name__)

class ProteinFolder:
    def __init__(self, model_name="facebook/esmfold_v1"):
        logger.info(f"Loading ESMFold model: {model_name}")
        # Use EsmTokenizer explicitly as AutoTokenizer might pick up a generic one
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmForProteinFolding.from_pretrained(model_name)
        self.model.eval()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")

    def fold_sequence(self, sequence: str) -> str:
        """Predict structural PDB string for a given sequence."""
        # Use add_special_tokens=False to avoid embedding out-of-bounds issues 
        # with special tokens in the structure module
        with torch.no_grad():
            inputs = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            pdb_str = self.model.output_to_pdb(outputs)[0]
        return pdb_str

    def process_fasta(self, fasta_path: str, output_dir: str):
        """Process sequences in a FASTA file and save as PDBs."""
        from bindsite.core.data import parse_fasta_3line
        
        os.makedirs(output_dir, exist_ok=True)
        
        records = parse_fasta_3line(fasta_path)
            
        logger.info(f"Processing {len(records)} sequences from {fasta_path}")
        
        for record in tqdm(records, desc=f"Folding {os.path.basename(fasta_path)}"):
            pdb_filename = f"{record.id}.pdb"
            output_path = os.path.join(output_dir, pdb_filename)
            
            if os.path.exists(output_path):
                continue
                
            try:
                pdb_content = self.fold_sequence(record.sequence)
                with open(output_path, "w") as f:
                    f.write(pdb_content)
            except Exception as e:
                logger.error(f"Failed to fold {record.id}: {e}")
                # Ensure we clear cache even on failure
                if self.device == "cuda":
                    torch.cuda.empty_cache()
