import typer
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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
        typer.echo(f"Error: Raw directory {raw_dir} does not exist.")
        raise typer.Exit(code=1)
        
    folder = ProteinFolder(model_name)
    
    fasta_files = list(raw_dir.glob("*.fa"))
    if not fasta_files:
        typer.echo(f"No .fa files found in {raw_dir}")
        return

    for fasta_file in fasta_files:        
        typer.echo(f"Processing {fasta_file.name} -> {pdb_dir}")
        folder.process_fasta(str(fasta_file), str(pdb_dir))

@app.command("extract")
def prepare_extract():
    """Extract residue-level features (ESM-2 embeddings + DSSP)."""
    typer.echo("Extracting features...")

@app.command("dataset")
def prepare_dataset(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/datasets"),
):
    """Parse raw FASTA files and create train/test splits in DeepProSite CSV format."""
    import csv
    
    if not raw_dir.exists():
        typer.echo(f"Error: Raw directory {raw_dir} does not exist.")
        raise typer.Exit(code=1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fasta_files = list(raw_dir.glob("*.fa"))
    if not fasta_files:
        typer.echo(f"No .fa files found in {raw_dir}")
        return

    from bindsite.core.data import parse_fasta_3line

    for fasta_file in fasta_files:
        output_file = output_dir / f"{fasta_file.stem}.csv"
        typer.echo(f"Converting {fasta_file.name} -> {output_file.name}")
        
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
            typer.echo(f"Successfully created {output_file}")
