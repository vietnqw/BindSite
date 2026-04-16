import typer

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
def predict(pdb_path: str):
    """Placeholder for binding site prediction."""
    typer.echo(f"Predicting binding sites for: {pdb_path}")
    typer.echo("Not implemented yet.")

if __name__ == "__main__":
    app()
