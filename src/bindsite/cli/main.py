import typer
import re
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


def _list_fold_weight_paths(model_dir: Path) -> List[Path]:
    """Return only fold weight files like fold_0.pt, fold_1.pt, ..."""
    pattern = re.compile(r"^fold_\d+\.pt$")
    return sorted([p for p in model_dir.glob("fold_*.pt") if pattern.match(p.name)])

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
    data_dir: Path = typer.Option("data/PRO", "--data-dir", help="Path to task directory."),
    train_data: List[Path] = typer.Option(..., "--train-data", help="Paths to one or more training CSV files."),
    output_dir: Path = typer.Option("output/PRO", "--output-dir", help="Where to save model weights."),
    epochs: int = typer.Option(DEFAULT_EPOCHS, "--epochs"),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, "--batch-size"),
    fold_idx: int = typer.Option(-1, "--fold", help="Specific fold (0-4) or -1 for all (default)."),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume training from saved per-fold resume checkpoints.",
    ),
):
    """Train DeepProSite model using 5-fold cross-validation."""
    from sklearn.model_selection import KFold
    from ..data.io import load_records_from_csv
    from ..tasks.training import run_training_fold
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    proteins = []
    for csv_file in train_data:
        typer.echo(f"Loading data from {csv_file}...")
        proteins.extend(load_records_from_csv(csv_file))
        
    if not proteins:
        typer.echo("Error: No protein records found in provided CSV(s).")
        raise typer.Exit(1)
        
    typer.echo(f"Total proteins loaded: {len(proteins)}")
    
    config = {
        'num_samples': len(proteins) * 5,
        'node_features': 1038, 'hidden_dim': DEFAULT_HIDDEN_DIM,
        'edge_features': 16, 'num_encoder_layers': DEFAULT_NUM_LAYERS,
        'num_heads': DEFAULT_NUM_HEADS, 'dropout': DEFAULT_DROPOUT,
        'k_neighbors': 30, 'augment_eps': 0.1,
        'epochs': epochs, 'batch_size': batch_size,
        'warmup_epochs': 5, 'patience': DEFAULT_PATIENCE,
        'output_dir': output_dir, 'task': data_dir.name.upper(),
        'resume': resume,
    }
    
    feature_dir = data_dir / "features"
    pdb_dir = data_dir / "pdb"
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(proteins))
    
    if fold_idx != -1:
        train_idx, val_idx = folds[fold_idx]
        train_records = [proteins[i] for i in train_idx]
        val_records = [proteins[i] for i in val_idx]
        run_training_fold(fold_idx, train_records, val_records, feature_dir, pdb_dir, config)
    else:
        for i, (train_idx, val_idx) in enumerate(folds):
            train_records = [proteins[j] for j in train_idx]
            val_records = [proteins[j] for j in val_idx]
            run_training_fold(i, train_records, val_records, feature_dir, pdb_dir, config)


@app.command("evaluate")
def evaluate(
    data_dir: Path = typer.Option("data/PRO", "--data-dir", help="Path to task directory (e.g. data/PRO)."),
    eval_data: Path = typer.Option(..., "--eval-data", help="Path to evaluation CSV file."),
    model_dir: Path = typer.Option("output/PRO", "--model-dir", help="Directory containing trained model folds."),
    threshold_mode: str = typer.Option(
        "all",
        "--threshold-mode",
        help=(
            "Decision threshold source for MCC/F1/Pre/Rec/Acc/Spe. "
            "'all' compares all supported approaches side by side. "
            "'fixed' uses 0.5 (clean). 'max-mcc' optimizes on the test set to "
            "match the paper's reporting (optimistic / leaks labels). "
            "'val-optimal' averages per-fold validation MCC-optimal thresholds "
            "saved during training (clean compromise)."
        ),
    ),
):
    """Evaluate DeepProSite ensemble on a test set."""
    from ..data.io import load_records_from_csv
    from ..tasks.evaluation import run_ensemble_evaluation

    if threshold_mode not in {"all", "fixed", "max-mcc", "val-optimal"}:
        typer.echo(
            f"Error: --threshold-mode must be one of all, fixed, max-mcc, val-optimal (got {threshold_mode!r})"
        )
        raise typer.Exit(2)

    proteins = load_records_from_csv(eval_data)
    if not proteins:
        typer.echo(f"Error: No records found in {eval_data}")
        raise typer.Exit(1)
        
    model_paths = _list_fold_weight_paths(model_dir)
    if not model_paths:
        typer.echo(f"Error: No fold weight files found in {model_dir} (expected fold_<n>.pt).")
        raise typer.Exit(1)
    
    config = {
        'node_features': 1038, 'hidden_dim': DEFAULT_HIDDEN_DIM,
        'edge_features': 16, 'num_encoder_layers': DEFAULT_NUM_LAYERS,
        'num_heads': DEFAULT_NUM_HEADS, 'dropout': DEFAULT_DROPOUT,
        'k_neighbors': 30, 'augment_eps': 0.1,
        'batch_size': DEFAULT_BATCH_SIZE
    }
    
    metrics = run_ensemble_evaluation(
        proteins, data_dir / "features", data_dir / "pdb",
        model_paths, config,
        threshold_mode=threshold_mode,
    )
    
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    
    # Mapping internal keys to descriptive display names
    metric_names = {
        'auc': 'Area Under ROC (AUC)',
        'auprc': 'Area Under PR (AUPRC)',
        'mcc': 'Matthews Correlation',
        'f1': 'F1 Score',
        'pre': 'Precision',
        'rec': 'Recall (Sensitivity)',
        'acc': 'Accuracy',
        'spe': 'Specificity',
        'threshold': 'Threshold',
    }

    if threshold_mode == "all":
        table = Table(title="[bold]DeepProSite Ensemble Performance[/bold]", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("fixed", justify="right", style="bold green")
        table.add_column("val-optimal", justify="right", style="bold yellow")
        table.add_column("max-mcc", justify="right", style="bold magenta")
        by_mode = metrics["by_mode"]
        for key, display_name in metric_names.items():
            table.add_row(
                display_name,
                f"{by_mode['fixed'][key]:.4f}",
                f"{by_mode['val-optimal'][key]:.4f}",
                f"{by_mode['max-mcc'][key]:.4f}",
            )
        subtitle = "all threshold modes"
    else:
        table = Table(title="[bold]DeepProSite Ensemble Performance[/bold]", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Score", justify="right", style="bold green")
        for key, display_name in metric_names.items():
            if key in metrics:
                table.add_row(display_name, f"{metrics[key]:.4f}")
        subtitle = f"threshold: {metrics.get('threshold_source', threshold_mode)}"

    console.print("\n")
    console.print(Panel(
        table,
        expand=False,
        border_style="blue",
        title="[bold white]Evaluation Complete[/bold white]",
        subtitle=subtitle,
    ))
    console.print("\n")


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
    
    model_paths = _list_fold_weight_paths(model_dir)
    if not model_paths:
        typer.echo(f"Error: No fold weight files found in {model_dir} (expected fold_<n>.pt).")
        raise typer.Exit(1)
    
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
    features = np.concatenate([prott5_feats[:min_l], dssp_feats[:min_l]], axis=1)
    coords = extract_ca_coordinates(pdb_path)[:min_l]
    
    config = {
        'node_features': 1038, 'hidden_dim': DEFAULT_HIDDEN_DIM,
        'edge_features': 16, 'num_encoder_layers': DEFAULT_NUM_LAYERS,
        'num_heads': DEFAULT_NUM_HEADS, 'dropout': DEFAULT_DROPOUT,
        'k_neighbors': 30, 'augment_eps': 0.1,
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
