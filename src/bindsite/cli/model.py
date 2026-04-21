import typer
from bindsite.utils import setup_logger

logger = setup_logger(__name__)

app = typer.Typer(help="Train and run inference on the binding site prediction model")

@app.command("train")
def train():
    """Train the binding site prediction model."""
    logger.info("Training pipeline not yet implemented.", extra={"simple": True})

@app.command("predict")
def predict():
    """Predict binding sites for a given protein sequence or PDB."""
    logger.info("Inference pipeline not yet implemented.", extra={"simple": True})

@app.command("evaluate")
def evaluate():
    """Evaluate the model on a test dataset."""
    logger.info("Evaluation pipeline not yet implemented.", extra={"simple": True})


