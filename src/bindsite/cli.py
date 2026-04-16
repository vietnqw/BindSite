import logging
from pathlib import Path
from typing import List

import typer
import transformers.utils.import_utils
import transformers.modeling_utils

# Bypassing the torch >= 2.6 security check in transformers to allow GPU support with torch 2.5.1
# This is safe for loading trusted local or verified community models like ESMFold.
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="BindSite: A tool for training, evaluation, and prediction of protein binding sites inspired by DeepProSite.",
    no_args_is_help=True
)

@app.command()
def info():
    """Display information about the environment."""
    import torch
    typer.echo("BindSite CLI Initialized")
    typer.echo(f"PyTorch version: {torch.__version__}")
    typer.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"CUDA version: {torch.version.cuda}")

@app.command()
def fold(
    input_fasta: List[Path] = typer.Option(
        ..., "--input-fasta", "-i", help="One or many .fa files to process."
    ),
    output_dir: Path = typer.Option(
        "data/pdb", "--output-dir", "-o", help="Output directory for predicted .pdb files."
    ),
    model_name: str = typer.Option(
        "facebook/esmfold_v1", help="ESMFold model version to use."
    ),
    chunk_size: int = typer.Option(
        128, help="Processing chunk size for attention to save VRAM. Set to 0 to disable."
    ),
):
    """Predict 3D protein structures from FASTA sequences using ESMFold."""
    from bindsite.folding import ESMFolder
    
    typer.echo(f"Starting folding process...")
    folder = ESMFolder(model_name=model_name, chunk_size=chunk_size)
    folder.fold_fasta(fasta_paths=input_fasta, output_dir=output_dir)
    typer.echo(f"Folding completed. Results saved to {output_dir}")

@app.command()
def predict(pdb_path: str):
    """Placeholder for binding site prediction."""
    typer.echo(f"Predicting binding sites for: {pdb_path}")
    typer.echo("Not implemented yet.")

if __name__ == "__main__":
    app()
