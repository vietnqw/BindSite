# BindSite

BindSite is a modern, modular, and performant reimplementation of the [DeepProSite](https://academic.oup.com/bioinformatics/article/39/5/btad311/7160751) protein binding site prediction model.

It leverages structure-aware Graph Transformers, ESMFold-predicted structures, and ProtT5 language model embeddings to predict both protein-protein and protein-peptide binding sites directly from sequence.

## Features

- **Modern Architecture**: Fully rewritten using PyTorch 2.x and `uv` package manager.
- **Fast and Efficient**: Batched feature extraction, data loading with PyTorch DataLoaders, and highly optimized attention operations.
- **Device Agnostic**: Seamlessly runs on GPUs, MPS (Apple Silicon), or CPU.
- **Reproducible**: strict adherence to the original paper's cross-validation methodology, hyperparameter settings, and feature normalizations.

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) as its package manager.

```bash
# Clone the repository
git clone https://your-repo/bindsite.git
cd bindsite

# Install dependencies and the package in editable mode
uv pip install -e .
```

*Note: You also need the `mkdssp` binary installed on your system to extract secondary structure features. On most Linux distributions, you can install it via `apt install dssp` or `conda install -c salilab dssp`.*

## Pipeline Overview

The complete prediction pipeline follows three steps:

1. **Structure Prediction (ESMFold)**: Produce a 3D structural model (.pdb) from an amino acid sequence. *(Handled via an external tool)*
2. **Feature Extraction (`extract-features`)**: Extract 1024-d ProtT5 embeddings and 14-d DSSP features, and merge them into padded tensors.
3. **Training & Inference (`train` / `predict`)**: Run the Graph Transformer model to output per-residue binding probabilities.

## 1. Structure Prediction (ESMFold)

BindSite expects a directory of `.pdb` files corresponding to your protein sequences. Because ESMFold has heavyweight and complex dependencies (like `openfold`), it is **not** included as a direct project dependency.

You can generate PDBs using the official [ESMFold script](https://github.com/facebookresearch/esm) or any public API (like the ESM Metagenomic Atlas API).

A sample script is provided in `scripts/fold.py` if you have `fair-esm` installed in a standalone environment:
```bash
# In an environment with fair-esm[esmfold] installed:
python scripts/fold.py --fasta data/Train_335.fa --output-dir data/pdb --cpu-offload
```

## 2. Feature Extraction

Once you have your FASTA files and the corresponding PDB files, run the feature extraction pipeline. This will lazily load the ProtT5 model, run DSSP, and construct the PyTorch tensors.

```bash
# Example for a subset of data
bindsite extract-features \
    --fasta data/Train_335.fa \
    --pdb-dir data/pdb \
    --feature-dir features/
```

## 3. Training & Inference

The CLI provides subcommands for preparing data splits, training, and predicting.

### Generate CSV Manifests

Convert the raw FASTA files (which may contain inline binary labels in a 3-line format) into CSVs for the DataLoader:

```bash
bindsite generate-csv --fasta data/Train_335.fa --output data/PRO_train.csv
bindsite generate-csv --fasta data/Test_60.fa --output data/PRO_test60.csv
```

### Training

Start a 5-fold cross-validation training run:

```bash
bindsite train \
    --train-csv data/PRO_train.csv \
    --tensor-dir features/tensors \
    --output-dir output/checkpoints \
    --task PRO \
    --epochs 30
```

### Prediction & Evaluation

Predict on a test set using the ensemble of 5 trained models from the CV folds:

```bash
bindsite predict \
    --test-csv data/PRO_test60.csv \
    --tensor-dir features/tensors \
    --checkpoint-dir output/checkpoints \
    --output output/predictions_test60.csv

# Evaluate the predictions against ground truth
bindsite evaluate \
    --predictions output/predictions_test60.csv \
    --ground-truth data/PRO_test60.csv
```

## Development

To install development dependencies (e.g., pytest, ruff):

```bash
uv pip install -e ".[dev]"
```

Linting:
```bash
uvx ruff check .
```
