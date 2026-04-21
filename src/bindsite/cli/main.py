import typer

app = typer.Typer(
    help="BindSite: Protein Binding Site Prediction Pipeline",
    no_args_is_help=True
)

@app.command()
def version():
    """Display the version of BindSite."""
    typer.echo("BindSite version 0.1.0")

@app.command()
def predict():
    """Predict binding sites for a given protein."""
    typer.echo("Prediction pipeline not yet implemented.")

if __name__ == "__main__":
    app()
