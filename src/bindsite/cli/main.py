import typer
from bindsite.cli import prepare, model

app = typer.Typer(
    help="BindSite: Protein Binding Site Prediction Pipeline",
    no_args_is_help=True
)

# Add modular sub-apps
app.add_typer(prepare.app, name="prepare")
app.add_typer(model.app, name="model")

@app.command()
def version():
    """Display the version of BindSite."""
    typer.echo("BindSite version 0.1.0")

if __name__ == "__main__":
    app()
