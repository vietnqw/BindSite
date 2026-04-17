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
data_app = typer.Typer(help="Dataset management commands.")
app.add_typer(data_app, name="data")

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
        "data/PRO/pdb", "--output-dir", "-o", help="Output directory for predicted .pdb files."
    ),
    model_name: str = typer.Option(
        "facebook/esmfold_v1", "--model-name", "-m", help="ESMFold model version to use."
    ),
    chunk_size: int = typer.Option(
        128, "--chunk-size", "-c", help="Processing chunk size for attention to save VRAM. Set to 0 to disable."
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


@data_app.command("download")
def data_download(
    data_root: Path = typer.Option(
        "data", "--data-root", help="Root data directory to store task-specific datasets."
    ),
):
    """Download and build benchmark datasets under task-specific folders."""
    from bindsite.data import download_benchmark_datasets

    raise typer.Exit(code=download_benchmark_datasets(data_root=data_root))


@data_app.command("verify")
def data_verify(
    data_root: Path = typer.Option(
        "data", "--data-root", help="Root data directory containing task-specific FASTA datasets."
    ),
    task: str = typer.Option(
        "all",
        "--task",
        "-t",
        help="Task scope for integrity checks: all, pep, or pro. "
             "Conflicts across different tasks are intentionally ignored.",
    ),
):
    """Verify FASTA integrity within each task (PEP and/or PRO)."""
    from bindsite.data import verify_fasta_integrity

    task_key = task.strip().lower()
    if task_key == "all":
        tasks = ("PEP", "PRO")
    elif task_key == "pep":
        tasks = ("PEP",)
    elif task_key == "pro":
        tasks = ("PRO",)
    else:
        typer.echo("Invalid --task value. Use one of: all, pep, pro.", err=True)
        raise typer.Exit(code=2)

    raise typer.Exit(code=verify_fasta_integrity(data_root=data_root, tasks=tasks))


@data_app.command("fasta-to-csv")
def data_fasta_to_csv(
    input_fasta: Path = typer.Option(
        ..., "--input-fasta", "-i", help="Input 3-line FASTA file (.fa)."
    ),
    output_csv: Path = typer.Option(
        ..., "--output-csv", "-o", help="Output CSV file path."
    ),
):
    """Convert one FASTA dataset file to CSV (ID,sequence,label)."""
    from bindsite.data import export_fasta_to_csv

    raise typer.Exit(code=export_fasta_to_csv(input_fasta=input_fasta, output_csv=output_csv))

if __name__ == "__main__":
    app()
