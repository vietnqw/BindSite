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
def prepare_dataset():
    """Parse raw FASTA files and create train/test splits."""
    typer.echo("Creating datasets...")
