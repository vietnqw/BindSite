import typer
from pathlib import Path

app = typer.Typer(help="Feature extraction commands.")

@app.command("extract")
def extract(
    data_dir: Path = typer.Option(..., "--data-dir", "-d", help="Path to task directory (e.g. data/PRO).")
):
    """Run DSSP and ProtT5 feature extraction for a specific dataset."""
    from ..features.pipeline import run_extraction_pipeline
    run_extraction_pipeline(data_dir=data_dir)
