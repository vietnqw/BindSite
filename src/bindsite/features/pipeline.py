import numpy as np
from pathlib import Path
from tqdm import tqdm
from .dssp import extract_dssp_features
from .prott5 import ProtT5Extractor
from ..core.logger import logger

from ..data.io import parse_3line_fasta

def process_dataset(
    fasta_path: Path,
    pdb_dir: Path,
    output_dir: Path,
    prott5_extractor: ProtT5Extractor,
    overwrite: bool = False,
):
    """Extracts features for all proteins in a FASTA file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    proteins = parse_3line_fasta(fasta_path)
    
    stats = {"processed": 0, "skipped_no_pdb": 0, "errors": 0}
    
    for record in tqdm(proteins, desc=f"Processing {fasta_path.name}"):
        seq_id, seq = record.seq_id, record.sequence
        out_path = output_dir / f"{seq_id}.npy"
        if out_path.exists() and not overwrite:
            stats["processed"] += 1
            continue
            
        pdb_path = pdb_dir / f"{seq_id}.pdb"
        if not pdb_path.exists():
            logger.debug(f"PDB not found for {seq_id}: {pdb_path}")
            stats["skipped_no_pdb"] += 1
            continue
            
        try:
            # 1. Extract DSSP features (14D)
            dssp_feats = extract_dssp_features(pdb_path, seq)
            
            # 2. Extract ProtT5 features (1024D)
            prott5_feats = prott5_extractor.extract(seq)
            
            # Check length alignment
            min_l = min(len(dssp_feats), len(prott5_feats))
            if len(dssp_feats) != len(prott5_feats):
                logger.warning(f"Feature length mismatch for {seq_id}: DSSP={len(dssp_feats)}, T5={len(prott5_feats)}")
                dssp_feats = dssp_feats[:min_l]
                prott5_feats = prott5_feats[:min_l]
            
            # 3. Concatenate (1038D) in reference order: ProtT5 + DSSP
            combined = np.concatenate([prott5_feats, dssp_feats], axis=1)
            
            # 4. Save
            np.save(out_path, combined)
            stats["processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {seq_id}: {e}")
            stats["errors"] += 1
            
    return stats

def compute_normalization_bounds(fasta_paths: list[Path], prott5_extractor: ProtT5Extractor):
    """Computes min and max values for ProtT5 embeddings across multiple FASTA files."""
    global_min = None
    global_max = None
    
    for fasta_path in fasta_paths:
        proteins = parse_3line_fasta(fasta_path)
        for record in tqdm(proteins, desc=f"Analyzing {fasta_path.name} for bounds"):
            seq = record.sequence
            # Extract raw embeddings (no normalization)
            prott5_extractor.set_normalization_bounds(None, None)
            emb = prott5_extractor.extract(seq)
            
            p_min = np.min(emb, axis=0)
            p_max = np.max(emb, axis=0)
            
            if global_min is None:
                global_min = p_min
                global_max = p_max
            else:
                global_min = np.minimum(global_min, p_min)
                global_max = np.maximum(global_max, p_max)
                
    return global_min, global_max

def run_extraction_pipeline(data_dir: Path, device: str = None, overwrite: bool = False):
    """Runs extraction for a specific dataset directory."""
    data_dir = Path(data_dir)
    prott5 = ProtT5Extractor(device=device)
    
    # Define task based on the provided data_dir
    task = {
        "name": data_dir.name, 
        "fasta_dir": data_dir / "fasta", 
        "pdb_dir": data_dir / "pdb", 
        "out_dir": data_dir / "features"
    }
    
    # 1. Collect all training FASTAs to compute global bounds
    train_fastas = list(task['fasta_dir'].glob("*Train*.fa"))
        
    if train_fastas:
        logger.info(f"Stage 1: Computing normalization bounds from training sets in {data_dir.name}...")
        min_val, max_val = compute_normalization_bounds(train_fastas, prott5)
        prott5.set_normalization_bounds(min_val, max_val)
        logger.info("Normalization bounds computed and set.")
        
        # Save bounds for future reference
        # We save bounds in data_dir/weights (e.g. data/PRO/weights)
        bounds_dir = data_dir / "weights"
        bounds_dir.mkdir(parents=True, exist_ok=True)
        np.save(bounds_dir / "Min_ProtTrans_repr.npy", min_val)
        np.save(bounds_dir / "Max_ProtTrans_repr.npy", max_val)
    
    # 2. Run normalized extraction for all datasets (both train and test)
    logger.info(f"Stage 2: Starting extraction for task directory: {data_dir}")
    fasta_files = list(task['fasta_dir'].glob("*.fa"))
    for fasta_file in fasta_files:
        stats = process_dataset(
            fasta_file,
            task['pdb_dir'],
            task['out_dir'],
            prott5,
            overwrite=overwrite,
        )
        logger.info(f"Stats for {fasta_file.name}: {stats}")
