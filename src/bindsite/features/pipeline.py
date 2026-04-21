import numpy as np
from pathlib import Path
from tqdm import tqdm
from .dssp import extract_dssp_features
from .esm2 import ESM2Extractor
from bindsite.utils import setup_logger
from bindsite.core.data import parse_fasta_3line

logger = setup_logger(__name__)


def process_dataset(
    fasta_path: Path,
    pdb_dir: Path,
    output_dir: Path,
    extractor: ESM2Extractor,
    dssp_bin: str,
    overwrite: bool = False,
):
    """Extracts features for all proteins in a FASTA file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    records = parse_fasta_3line(str(fasta_path))
    stats = {"processed": 0, "skipped_no_pdb": 0, "errors": 0}
    
    for record in tqdm(records, desc=f"Extracting features for {fasta_path.name}"):
        seq_id, seq = record.id, record.sequence
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
            dssp_feats = extract_dssp_features(pdb_path, seq, dssp_bin)
            
            # 2. Extract ESM-2 features (1280D, 480D, etc.)
            esm2_feats = extractor.extract(seq)
            
            # Check length alignment
            min_l = min(len(dssp_feats), len(esm2_feats))
            if len(dssp_feats) != len(esm2_feats):
                logger.warning(f"Length mismatch for {seq_id}: DSSP={len(dssp_feats)}, ESM2={len(esm2_feats)}")
                dssp_feats = dssp_feats[:min_l]
                esm2_feats = esm2_feats[:min_l]
            
            # 3. Concatenate
            combined = np.concatenate([esm2_feats, dssp_feats], axis=1)
            
            # 4. Save
            np.save(out_path, combined)
            stats["processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {seq_id}: {e}")
            stats["errors"] += 1
            
    return stats

def run_extraction_pipeline(fasta_dir: Path, pdb_dir: Path, output_dir: Path, model_name: str = "facebook/esm2_t33_650M_UR50D", dssp_bin: str = "bin/mkdssp", device: str = None, overwrite: bool = False):
    """Runs extraction pipeline using ESM-2 and DSSP with global normalization."""
    logger.info(f"Loading extraction pipeline with ESM-2 model: {model_name}")
    extractor = ESM2Extractor(model_name=model_name, device=device)
    
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    min_path = metadata_dir / "esm2_min.npy"
    max_path = metadata_dir / "esm2_max.npy"

    fasta_files = list(fasta_dir.glob("*.fa"))
    if not fasta_files:
        logger.warning(f"No FASTA files found in {fasta_dir}")
        return

    # Compute global min/max if they don't exist
    if not (min_path.exists() and max_path.exists()) or overwrite:
        logger.info("Normalization vectors not found or overwrite enabled. Computing global min/max from training data...")
        
        # Filter for train files only to avoid data leakage from test set
        train_fasta_files = [f for f in fasta_files if "train" in f.name.lower()]
        
        if not train_fasta_files:
            if overwrite:
                logger.error("Overwrite requested but no 'train' FASTA files found in raw directory. Cannot compute normalization vectors.")
            else:
                logger.warning("No 'train' FASTA files found to compute normalization vectors. Using default (unnormalized) or existing vectors if possible.")
        else:
            global_max = None
            global_min = None

            for fasta_file in train_fasta_files:
                records = parse_fasta_3line(str(fasta_file))
                for record in tqdm(records, desc=f"Scanning {fasta_file.name} for min/max stats"):
                    # We extract without normalization to get the bounds
                    extractor.set_normalization_bounds(None, None)
                    emb = extractor.extract(record.sequence)
                    
                    local_max = np.max(emb, axis=0)
                    local_min = np.min(emb, axis=0)

                    if global_max is None:
                        global_max = local_max
                        global_min = local_min
                    else:
                        global_max = np.maximum(global_max, local_max)
                        global_min = np.minimum(global_min, local_min)
            
            if global_max is not None:
                np.save(min_path, global_min)
                np.save(max_path, global_max)
                logger.info(f"Global normalization vectors computed from {len(train_fasta_files)} train files and saved to {metadata_dir}")
    
    # Load and set normalization bounds
    if min_path.exists() and max_path.exists():
        min_val = np.load(min_path)
        max_val = np.load(max_path)
        extractor.set_normalization_bounds(min_val, max_val)
        logger.info("Extractor normalization bounds set.")
        
    for fasta_file in fasta_files:
        stats = process_dataset(
            fasta_file,
            pdb_dir,
            output_dir,
            extractor,
            dssp_bin=dssp_bin,
            overwrite=overwrite,
        )
        logger.info(f"Extraction stats for {fasta_file.name}: {stats}")
