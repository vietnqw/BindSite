import typer
from pathlib import Path
from typing import List, Optional
from . import data, features
from ..core.config import (
    DEFAULT_HIDDEN_DIM, DEFAULT_NUM_LAYERS, DEFAULT_NUM_HEADS, 
    DEFAULT_DROPOUT, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_PATIENCE
)

app = typer.Typer(
    help="BindSite: A modular framework for protein binding site prediction.",
    no_args_is_help=True
)

# --- Grouped Sub-Apps ---
app.add_typer(data.app, name="data")
app.add_typer(features.app, name="features")

# --- Direct Action Commands (Verbs) ---

@app.command("fold")
def fold(
    data_dir: Path = typer.Option(..., "--data-dir", "-d", help="Path to task directory (e.g. data/PRO)."),
    model_name: str = typer.Option(
        "facebook/esmfold_v1", "--model-name", "-m", help="ESMFold model version to use."
    ),
    chunk_size: int = typer.Option(
        128, "--chunk-size", "-c", help="Processing chunk size for attention to save VRAM. Set to 0 to disable."
    ),
):
    """Predict 3D protein structures from FASTA sequences using ESMFold."""
    from ..tasks.folding import ESMFolder
    
    fasta_dir = data_dir / "fasta"
    output_dir = data_dir / "pdb"
    
    if not fasta_dir.exists():
        typer.echo(f"Error: fasta directory not found in {data_dir}")
        raise typer.Exit(1)
        
    fasta_files = list(fasta_dir.glob("*.fa"))
    if not fasta_files:
        typer.echo(f"No .fa files found in {fasta_dir}")
        raise typer.Exit()

    typer.echo(f"Found {len(fasta_files)} FASTA files. Starting folding process...")
    folder = ESMFolder(model_name=model_name, chunk_size=chunk_size)
    folder.fold_fasta(fasta_paths=fasta_files, output_dir=output_dir)
    typer.echo(f"Folding completed. Results saved to {output_dir}")


@app.command("train")
def train(
    task: str = typer.Option("PRO", "--task", help="Task name (PRO or PEP)."),
    fasta_file: Path = typer.Option(..., "--fasta", help="3-line FASTA training set."),
    data_root: Path = typer.Option("data", "--data-root", help="Root data directory."),
    epochs: int = typer.Option(DEFAULT_EPOCHS, "--epochs"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, "--batch-size"),
    output_dir: Path = typer.Option("output", "--output-dir"),
    fold_idx: int = typer.Option(0, "--fold", help="Specific fold (0-4) or -1 for all.")
):
    """Train DeepProSite model using 5-fold cross-validation."""
    from sklearn.model_selection import KFold
    from ..data.io import parse_3line_fasta
    from ..tasks.training import run_training_fold
    
    output_dir.mkdir(parents=True, exist_ok=True)
    proteins = parse_3line_fasta(fasta_file)
    
    config = {
        'node_features': 1038, 'hidden_dim': DEFAULT_HIDDEN_DIM,
        'num_encoder_layers': DEFAULT_NUM_LAYERS, 'dropout': DEFAULT_DROPOUT,
        'epochs': epochs, 'batch_size': batch_size,
        'warmup_epochs': 5, 'patience': DEFAULT_PATIENCE,
        'output_dir': output_dir
    }
    
    feature_dir = Path(data_root) / task / "features"
    pdb_dir = Path(data_root) / task / "pdb"
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(proteins))
    
    if fold_idx != -1:
        train_idx, val_idx = folds[fold_idx]
        train_data = [proteins[i] for i in train_idx]
        val_data = [proteins[i] for i in val_idx]
        run_training_fold(fold_idx, train_data, val_data, feature_dir, pdb_dir, config)
    else:
        for i, (train_idx, val_idx) in enumerate(folds):
            train_data = [proteins[i] for i in train_idx]
            val_data = [proteins[i] for i in val_idx]
            run_training_fold(i, train_data, val_data, feature_dir, pdb_dir, config)


@app.command("evaluate")
def evaluate(
    task: str = typer.Option("PRO", "--task"),
    fasta_file: Path = typer.Option(..., "--fasta"),
    data_root: Path = typer.Option("data", "--data-root"),
    model_dir: Path = typer.Option("output", "--model-dir")
):
    """Evaluate DeepProSite ensemble on a test set."""
    from ..data.io import parse_3line_fasta
    from ..tasks.evaluation import run_ensemble_evaluation
    
    proteins = parse_3line_fasta(fasta_file)
    model_paths = list(model_dir.glob("fold_*_best.pt"))
    
    config = {
        'node_features': 1038, 'hidden_dim': DEFAULT_HIDDEN_DIM,
        'num_encoder_layers': DEFAULT_NUM_LAYERS, 'dropout': DEFAULT_DROPOUT,
        'batch_size': DEFAULT_BATCH_SIZE
    }
    
    metrics = run_ensemble_evaluation(
        proteins, Path(data_root) / task / "features", Path(data_root) / task / "pdb", 
        model_paths, config
    )
    
    typer.echo("\n--- Results ---")
    for k, v in metrics.items():
        typer.echo(f"{k.upper()}: {v:.4f}")


@app.command("predict")
def predict(
    pdb_path: Path = typer.Option(..., "--pdb", "-p"),
    sequence: str = typer.Option(..., "--sequence", "-s"),
    model_dir: Path = typer.Option("output", "--model-dir"),
    weights_dir: Optional[Path] = typer.Option(None, "--weights-dir", help="Directory containing Min/Max_ProtTrans_repr.npy"),
    output: Optional[Path] = typer.Option(None, "--output", "-o")
):
    """Predict binding sites for a single protein."""
    import json
    import numpy as np
    from ..features.dssp import extract_dssp_features
    from ..features.prott5 import ProtT5Extractor
    from ..tasks.inference import run_single_prediction
    from ..data.io import extract_ca_coordinates
    
    model_paths = list(model_dir.glob("fold_*_best.pt"))
    
    # Feature extraction
    dssp_feats = extract_dssp_features(pdb_path, sequence)
    
    min_val, max_val = None, None
    if weights_dir and weights_dir.exists():
        min_path = weights_dir / "Min_ProtTrans_repr.npy"
        max_path = weights_dir / "Max_ProtTrans_repr.npy"
        if min_path.exists() and max_path.exists():
            min_val = np.load(min_path)
            max_val = np.load(max_path)
    
    if min_val is None:
        typer.echo("Warning: Normalization weights not found. Predictions might be suboptimal.")
    
    prott5 = ProtT5Extractor(min_val=min_val, max_val=max_val)
    prott5_feats = prott5.extract(sequence)
    
    min_l = min(len(dssp_feats), len(prott5_feats))
    features = np.concatenate([dssp_feats[:min_l], prott5_feats[:min_l]], axis=1)
    coords = extract_ca_coordinates(pdb_path)[:min_l]
    
    config = {
        'node_features': 1038, 'hidden_dim': DEFAULT_HIDDEN_DIM,
        'num_encoder_layers': DEFAULT_NUM_LAYERS, 'dropout': DEFAULT_DROPOUT
    }
    
    probs = run_single_prediction(coords, features, model_paths, config)
    
    results = [
        {"residue": i+1, "aa": sequence[i], "prob": float(p), "is_binding": bool(p > 0.5)}
        for i, p in enumerate(probs)
    ]
    
    for r in results[:10]:
        typer.echo(f"Res {r['residue']} ({r['aa']}): Prob={r['prob']:.4f}")
    
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)


@app.command()
def info():
    """Display environment information."""
    import torch
    typer.echo(f"PyTorch Version: {torch.__version__}")
    typer.echo(f"CUDA Available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    app()
