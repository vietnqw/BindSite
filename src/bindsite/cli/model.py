import typer

app = typer.Typer(help="Train and run inference on the binding site prediction model")

@app.command("train")
def train():
    """Train the binding site prediction model."""
    typer.echo("Training pipeline not yet implemented.")

@app.command("predict")
def predict():
    """Predict binding sites for a given protein sequence or PDB."""
    typer.echo("Inference pipeline not yet implemented.")

@app.command("evaluate")
def evaluate():
    """Evaluate the model on a test dataset."""
    typer.echo("Evaluation pipeline not yet implemented.")
