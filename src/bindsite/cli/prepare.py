import typer

app = typer.Typer(help="Prepare assets for the pipeline (structures, features, datasets)")

@app.command("fold")
def prepare_fold():
    """Predict 3D structures from sequences using ESMFold."""
    typer.echo("Predicting 3D structures...")

@app.command("extract")
def prepare_extract():
    """Extract residue-level features (ESM-2 embeddings + DSSP)."""
    typer.echo("Extracting features...")

@app.command("dataset")
def prepare_dataset():
    """Parse raw FASTA files and create train/test splits."""
    typer.echo("Creating datasets...")
