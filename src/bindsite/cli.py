"""Command-line interface for BindSite.

Provides subcommands for the full pipeline:
  - fold:              Predict 3D structures from sequences using ESMFold
  - extract-features:  Extract ProtT5 + DSSP features from FASTA + PDB files
  - train:             Train the Graph Transformer model with K-fold CV
  - predict:           Run inference with trained model ensemble
  - evaluate:          Evaluate predictions against ground truth
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_cuda_environment() -> None:
    """Gracefully detect and warn if a system has a GPU but PyTorch can't use it."""
    import torch
    
    if torch.cuda.is_available():
        return

    # Check if there is an NVIDIA GPU present using nvidia-smi
    import subprocess
    logger = logging.getLogger(__name__)
    
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        # If nvidia-smi ran successfully but PyTorch CUDA isn't available, mismatched driver!
        logger.error("-" * 80)
        logger.error("CRITICAL GPU DRIVER WARNING !!!")
        logger.error(
            "Your system has an NVIDIA GPU, but PyTorch running inside 'uv' cannot detect it.\n"
            "This happens because PyTorch installed the newest CUDA binary, but your host\n"
            "NVIDIA Driver is older and incompatible."
        )
        logger.error(
            "To fix this gracefully without changing code, run the following command to\n"
            "install PyTorch compiled for a slightly older CUDA version (e.g. 12.1):\n"
            "\n"
            "    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            "\n"
        )
        logger.error("-" * 80)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # No GPU found (or nvidia-smi not installed). Safe to silently fall back to CPU
        pass

# --------------------------------------------------------------------------- #
#  Subcommand: fold
# --------------------------------------------------------------------------- #
def cmd_fold(args: argparse.Namespace) -> None:
    """Run ESMFold to predict PDB structures from FASTA sequences."""
    from bindsite.data.fold import run_esmfold
    run_esmfold(
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        device=args.device,
        chunk_size=args.chunk_size,
    )

# --------------------------------------------------------------------------- #
#  Subcommand: extract-features
# --------------------------------------------------------------------------- #
def cmd_extract_features(args: argparse.Namespace) -> None:
    """Extract features from FASTA + PDB files."""
    from bindsite.data.dssp import extract_dssp_features
    from bindsite.data.fasta import parse_fasta
    from bindsite.data.features import prepare_and_save
    from bindsite.data.protrans import (
        ProtTransExtractor,
        compute_normalization_stats,
        normalize_embeddings,
    )

    feature_dir = Path(args.feature_dir)
    raw_emb_dir = feature_dir / "raw_embeddings"
    norm_emb_dir = feature_dir / "protrans"
    dssp_dir = feature_dir / "dssp"
    tensor_dir = feature_dir / "tensors"

    for d in [raw_emb_dir, norm_emb_dir, dssp_dir, tensor_dir]:
        d.mkdir(parents=True, exist_ok=True)

    records = []
    fasta_paths = [args.fasta] if isinstance(args.fasta, str) else args.fasta
    for p in fasta_paths:
        records.extend(parse_fasta(p))
        
    sequences = {r.id: r.sequence for r in records}
    logging.info("Loaded %d sequences from %s", len(records), fasta_paths)

    # Step 1: ProtT5 embeddings.
    logging.info("Step 1/4: Extracting ProtT5 embeddings...")
    extractor = ProtTransExtractor(
        model_name_or_path=args.protrans_model,
        device=args.device,
    )
    extractor.extract(sequences, output_dir=raw_emb_dir, skip_existing=True)
    extractor.release()

    # Step 2: Normalize embeddings.
    logging.info("Step 2/4: Normalizing ProtT5 embeddings...")
    if args.norm_stats:
        stats = np.load(args.norm_stats).item()
        min_vals, max_vals = stats["min"], stats["max"]
    else:
        # Compute from current data (assumes this is the training set).
        min_vals, max_vals = compute_normalization_stats(
            raw_emb_dir, list(sequences.keys())
        )
        stats_path = feature_dir / "protrans_norm_stats.npz"
        np.savez(stats_path, min=min_vals, max=max_vals)
        logging.info("Saved normalization stats to %s", stats_path)

    normalize_embeddings(
        raw_emb_dir, norm_emb_dir, min_vals, max_vals,
        list(sequences.keys()), skip_existing=True,
    )

    # Step 3: DSSP features.
    logging.info("Step 3/4: Extracting DSSP features...")
    from tqdm import tqdm

    for pid, seq in tqdm(sequences.items(), desc="DSSP"):
        dssp_path = dssp_dir / f"{pid}.npy"
        if dssp_path.exists():
            continue
        pdb_path = Path(args.pdb_dir) / f"{pid}.pdb"
        if not pdb_path.exists():
            logging.warning("Missing PDB for %s, skipping DSSP.", pid)
            continue
        try:
            features = extract_dssp_features(pdb_path, seq, args.dssp_binary)
            np.save(dssp_path, features)
        except Exception as e:
            logging.warning("DSSP failed for %s: %s", pid, e)

    # Step 4: Merge features into tensors.
    logging.info("Step 4/4: Merging features into tensors...")
    from tqdm import tqdm

    success = 0
    for pid in tqdm(sequences, desc="Merging"):
        if prepare_and_save(
            pid, args.pdb_dir, norm_emb_dir, dssp_dir, tensor_dir,
            max_len=args.max_len, skip_existing=True,
        ):
            success += 1
    logging.info("Created tensors for %d / %d proteins.", success, len(sequences))


# --------------------------------------------------------------------------- #
#  Subcommand: generate-csv
# --------------------------------------------------------------------------- #
def cmd_generate_csv(args: argparse.Namespace) -> None:
    """Generate a CSV file from a FASTA file for training/testing."""
    from bindsite.data.fasta import parse_fasta

    fasta_paths = [args.fasta] if isinstance(args.fasta, str) else args.fasta
    records = []
    for p in fasta_paths:
        records.extend(parse_fasta(p))
    
    rows = []
    for r in records:
        row = {"ID": r.id, "sequence": r.sequence}
        if r.label is not None:
            row["label"] = str(r.label)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    logging.info("Saved %d records from %s to %s", len(df), fasta_paths, args.output)


# --------------------------------------------------------------------------- #
#  Subcommand: train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> None:
    """Train the model with K-fold cross-validation."""
    from bindsite.config import ModelConfig, TrainingConfig
    from bindsite.training.trainer import Trainer

    train_df = pd.read_csv(args.train_csv)
    logging.info("Training set: %d proteins", len(train_df))

    model_config = ModelConfig()
    training_config = TrainingConfig(
        task=args.task,
        seed=args.seed,
        epochs=args.epochs,
        max_seq_len=args.max_len,
        num_workers=args.num_workers,
    )

    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        tensor_dir=args.tensor_dir,
        output_dir=args.output_dir,
        device=args.device,
    )

    trainer.train_kfold(train_df)


# --------------------------------------------------------------------------- #
#  Subcommand: predict
# --------------------------------------------------------------------------- #
def cmd_predict(args: argparse.Namespace) -> None:
    """Run inference with trained model ensemble."""
    from bindsite.config import ModelConfig, TrainingConfig
    from bindsite.training.evaluator import Evaluator

    test_df = pd.read_csv(args.test_csv)
    logging.info("Test set: %d proteins", len(test_df))

    model_config = ModelConfig()
    training_config = TrainingConfig(
        task=args.task, max_seq_len=args.max_len, num_workers=args.num_workers,
    )

    evaluator = Evaluator(
        model_config=model_config,
        training_config=training_config,
        checkpoint_dir=args.checkpoint_dir,
        tensor_dir=args.tensor_dir,
        device=args.device,
    )

    predictions = evaluator.predict(test_df)

    # Save predictions as CSV.
    rows = []
    for _, row in test_df.iterrows():
        pid = row["ID"]
        if pid in predictions:
            rows.append({
                "ID": pid,
                "sequence": row["sequence"],
                "label": str(predictions[pid].tolist()),
            })
    pd.DataFrame(rows).to_csv(args.output, index=False)
    logging.info("Saved predictions to %s", args.output)


def smooth_probabilities(probs: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply a simple moving average to smooth residue probabilities.
    
    Helps remove isolated false positives and bridges small gaps in clusters.
    """
    if window_size <= 1:
        return probs
    
    pad = window_size // 2
    padded = np.pad(probs, pad, mode="edge")
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode="valid")
    # Ensure we return the same length as input
    return smoothed[:len(probs)]


# --------------------------------------------------------------------------- #
#  Subcommand: evaluate
# --------------------------------------------------------------------------- #
def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate predictions against ground truth."""
    from bindsite.training.metrics import compute_metrics

    pred_df = pd.read_csv(args.predictions)
    truth_df = pd.read_csv(args.ground_truth)

    pred_map = {
        row["ID"]: ast.literal_eval(row["label"])
        for _, row in pred_df.iterrows()
    }

    all_preds, all_labels = [], []
    for _, row in truth_df.iterrows():
        pid = row["ID"]
        if pid not in pred_map:
            continue
        labels = ast.literal_eval(row["label"]) if isinstance(row["label"], str) else list(row["label"])
        preds = pred_map[pid]
        L = min(len(labels), len(preds))
        all_labels.extend(labels[:L])
        
        # Apply spatial smoothing if requested
        p = np.array(preds[:L])
        if getattr(args, "smooth", 0) > 1:
            p = smooth_probabilities(p, args.smooth)
        all_preds.extend(p.tolist())

    metrics = compute_metrics(np.array(all_preds), np.array(all_labels), threshold=args.threshold)
    print(f"\nEvaluation Results:\n{metrics}")


# --------------------------------------------------------------------------- #
#  Main CLI parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="bindsite",
        description="BindSite: Protein binding site prediction using Graph Transformer",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- fold ---
    p_fold = subparsers.add_parser(
        "fold",
        help="Predict 3D structures from sequences using HF ESMFold",
    )
    p_fold.add_argument("--fasta", nargs="+", required=True, help="Input FASTA file(s)")
    p_fold.add_argument("--output-dir", required=True, help="Output directory for PDBs")
    p_fold.add_argument("--device", default=None, help="Device (e.g., cuda:0)")
    p_fold.add_argument("--chunk-size", type=int, default=64, help="Attention chunk size for VRAM saving")
    p_fold.set_defaults(func=cmd_fold)

    # --- extract-features ---
    p_extract = subparsers.add_parser(
        "extract-features",
        help="Extract ProtT5 + DSSP features from FASTA + PDB files",
    )
    p_extract.add_argument("--fasta", nargs="+", required=True, help="Input FASTA file(s)")
    p_extract.add_argument("--pdb-dir", required=True, help="Directory with PDB files")
    p_extract.add_argument("--feature-dir", default="./features", help="Output feature directory")
    p_extract.add_argument("--protrans-model", default="Rostlab/prot_t5_xl_uniref50")
    p_extract.add_argument("--dssp-binary", default=None, help="DSSP executable (defaults to auto-discovery)")
    p_extract.add_argument("--norm-stats", default=None, help="Path to normalization stats .npz")
    p_extract.add_argument("--device", default=None, help="Device (e.g., cuda:0)")
    p_extract.add_argument("--max-len", type=int, default=1000, help="Max sequence length")
    p_extract.set_defaults(func=cmd_extract_features)

    # --- generate-csv ---
    p_csv = subparsers.add_parser(
        "generate-csv",
        help="Generate train/test CSV from FASTA file",
    )
    p_csv.add_argument("--fasta", nargs="+", required=True, help="Input FASTA file(s)")
    p_csv.add_argument("--output", required=True, help="Output CSV file")
    p_csv.set_defaults(func=cmd_generate_csv)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train with K-fold CV")
    p_train.add_argument("--train-csv", required=True, help="Training CSV file")
    p_train.add_argument("--tensor-dir", required=True, help="Pre-computed tensor directory")
    p_train.add_argument("--output-dir", default="./output", help="Output directory")
    p_train.add_argument("--task", default="PRO", choices=["PRO", "PEP"])
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--max-len", type=int, default=1000)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--device", default=None)
    p_train.set_defaults(func=cmd_train)

    # --- predict ---
    p_predict = subparsers.add_parser("predict", help="Run inference")
    p_predict.add_argument("--test-csv", required=True, help="Test CSV file")
    p_predict.add_argument("--tensor-dir", required=True, help="Pre-computed tensor directory")
    p_predict.add_argument("--checkpoint-dir", required=True, help="Model checkpoint directory")
    p_predict.add_argument("--output", default="./predictions.csv", help="Output predictions CSV")
    p_predict.add_argument("--task", default="PRO", choices=["PRO", "PEP"])
    p_predict.add_argument("--max-len", type=int, default=1000)
    p_predict.add_argument("--num-workers", type=int, default=4)
    p_predict.add_argument("--device", default=None)
    p_predict.set_defaults(func=cmd_predict)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate predictions")
    p_eval.add_argument("--predictions", required=True, help="Predictions CSV")
    p_eval.add_argument("--ground-truth", required=True, help="Ground truth CSV")
    p_eval.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    p_eval.add_argument("--smooth", type=int, default=0, help="Smoothing window size (e.g. 3 or 5)")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))
    
    check_cuda_environment()
    
    args.func(args)


if __name__ == "__main__":
    main()
