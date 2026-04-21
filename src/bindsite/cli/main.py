import typer
from bindsite.cli import prepare, model
from bindsite.utils import setup_logger

logger = setup_logger(__name__)

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
    logger.info("BindSite version 0.1.0", extra={"simple": True})


if __name__ == "__main__":
    app()
