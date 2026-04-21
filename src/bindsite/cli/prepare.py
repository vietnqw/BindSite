import typer
import os
from pathlib import Path
from bindsite.utils import setup_logger

logger = setup_logger(__name__)

app = typer.Typer(help="Prepare assets for the pipeline (structures, features, datasets)")

@app.command("fold")
def prepare_fold(
    raw_dir: Path = Path("data/raw"),
    pdb_dir: Path = Path("data/pdb"),
    model_name: str = typer.Option("facebook/esmfold_v1", help="ESMFold model name"),
):
    """Predict 3D structures from sequences using ESMFold."""
    from bindsite.core.folding import ProteinFolder
    
    if not raw_dir.exists():
        logger.error(f"Raw directory {raw_dir} does not exist.")
        raise typer.Exit(code=1)
        
    folder = ProteinFolder(model_name)
    
    fasta_files = list(raw_dir.glob("*.fa"))
    if not fasta_files:
        logger.warning(f"No .fa files found in {raw_dir}")
        return

    for fasta_file in fasta_files:        
        logger.info(f"Processing {fasta_file.name} -> {pdb_dir}")
        folder.process_fasta(str(fasta_file), str(pdb_dir))

@app.command("extract")
def prepare_extract(
    raw_dir: Path = Path("data/raw"),
    pdb_dir: Path = Path("data/pdb"),
    feature_dir: Path = Path("data/features"),
    model_name: str = typer.Option("facebook/esm2_t33_650M_UR50D", help="ESM-2 model name"),
    dssp_bin: str = typer.Option("bin/mkdssp", help="Path to DSSP executable"),
    overwrite: bool = typer.Option(False, help="Overwrite existing features"),
):
    """Extract residue-level features (ESM-2 embeddings + DSSP)."""
    logger.info("Extracting features using ESM-2 and DSSP...")
    
    if not raw_dir.exists():
        logger.error(f"Raw directory {raw_dir} does not exist.")
        raise typer.Exit(code=1)
        
    if not pdb_dir.exists():
        logger.error(f"PDB directory {pdb_dir} does not exist. Please run fold first.")
        raise typer.Exit(code=1)
        
    from bindsite.features.pipeline import run_extraction_pipeline
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    run_extraction_pipeline(
        fasta_dir=raw_dir,
        pdb_dir=pdb_dir,
        output_dir=feature_dir,
        model_name=model_name,
        dssp_bin=dssp_bin,
        overwrite=overwrite
    )
    logger.info(f"Features extracted and saved to {feature_dir}")

@app.command("dataset")
def prepare_dataset(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/datasets"),
):
    """Parse raw FASTA files and create train/test splits in DeepProSite CSV format."""
    import csv
    
    if not raw_dir.exists():
        logger.error(f"Raw directory {raw_dir} does not exist.")
        raise typer.Exit(code=1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fasta_files = list(raw_dir.glob("*.fa"))
    if not fasta_files:
        logger.warning(f"No .fa files found in {raw_dir}")
        return

    from bindsite.core.data import parse_fasta_3line

    for fasta_file in fasta_files:
        output_file = output_dir / f"{fasta_file.stem}.csv"
        logger.info(f"Converting {fasta_file.name} -> {output_file.name}")
        
        records = parse_fasta_3line(str(fasta_file))
        dataset = []
        
        for record in records:
            if not record.labels:
                logger.warning(f"No labels found for {record.id} in {fasta_file}")
                continue
                
            if len(record.sequence) != len(record.labels):
                logger.warning(f"Sequence and label length mismatch for {record.id}: {len(record.sequence)} vs {len(record.labels)}")
            
            # Convert "00110" to "[0, 0, 1, 1, 0]"
            labels = [int(c) for c in record.labels]
            
            dataset.append({
                "ID": record.id,
                "sequence": record.sequence,
                "label": str(labels)
            })
            
        if dataset:
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["ID", "sequence", "label"])
                writer.writeheader()
                writer.writerows(dataset)
            logger.info(f"Successfully created {output_file}")
