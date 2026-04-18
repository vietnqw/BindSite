import typer
from pathlib import Path
from ..data.manager import download_benchmark_datasets, verify_fasta_integrity, export_fasta_to_csv, get_data_info

app = typer.Typer(help="Dataset management commands.")

@app.command("export-csv")
def export_csv(
    input_fasta: Path = typer.Option(..., "--input-fasta", "-i", help="Input 3-line FASTA file."),
    output_csv: Path = typer.Option(..., "--output-csv", "-o", help="Path to save the CSV file.")
):
    exit_code = export_fasta_to_csv(input_fasta, output_csv)
    raise typer.Exit(code=exit_code)

@app.command("download")
def download(
    data_root: Path = typer.Option("data", "--data-root", help="Root data directory.")
):
    """Download and build benchmark datasets."""
    exit_code = download_benchmark_datasets(data_root=data_root)
    raise typer.Exit(code=exit_code)

@app.command("verify")
def verify(
    data_root: Path = typer.Option("data", "--data-root", help="Root data directory."),
    task: str = typer.Option("all", "--task", "-t", help="Task scope: all, pep, or pro.")
):
    """Verify FASTA integrity within each task."""
    task_key = task.strip().lower()
    if task_key == "all":
        tasks = ("PEP", "PRO")
    elif task_key in ("pep", "pro"):
        tasks = (task_key.upper(),)
    else:
        typer.echo(f"Invalid task: {task}", err=True)
        raise typer.Exit(code=1)
        
    exit_code = verify_fasta_integrity(data_root=data_root, tasks=tasks)
    raise typer.Exit(code=exit_code)

@app.command("info")
def info(
    data_root: Path = typer.Option("data", "--data-root", help="Root data directory."),
    task: str = typer.Option("all", "--task", "-t", help="Task scope: all, pep, or pro.")
):
    """Check data-related information (PDB counts, item counts, etc.)."""
    task_key = task.strip().lower()
    if task_key == "all":
        tasks = ("PEP", "PRO")
    elif task_key in ("pep", "pro"):
        tasks = (task_key.upper(),)
    else:
        typer.echo(f"Invalid task: {task}", err=True)
        raise typer.Exit(code=1)
        
    exit_code = get_data_info(data_root=data_root, tasks=tasks)
    raise typer.Exit(code=exit_code)
