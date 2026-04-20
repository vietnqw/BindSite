# BindSite

BindSite is a modular framework for reproducing research in protein binding site prediction, currently focused on a clean, PyTorch-based implementation of the **DeepProSite** architecture.

## Installation

Requires [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
git clone git@github.com:vietnqw/BindSite.git
cd BindSite
uv sync
```

## Basic Workflow

The framework follows a standard pipeline from raw data to binding site prediction.

### 1. Data Preparation
Download and verify benchmark datasets:
```bash
uv run bindsite data download
uv run bindsite data verify
```

Export FASTA to CSV:
```bash
uv run bindsite data export-csv -i data/PRO/fasta/PRO_Train_335.fa -o data/PRO/csv/PRO_Train_335.csv
```

### 2. Protein Folding
Generate 3D structures (PDB) from FASTA using ESMFold:
```bash
uv run bindsite fold --data-dir data/PRO
```

### 3. Feature Extraction
Extract 1038D features (DSSP structural + ProtT5 sequence embeddings):
```bash
uv run bindsite features extract --data-dir data/PRO
```

### 4. Training
Train the 5-fold cross-validation ensemble model using a CSV manifest:
```bash
# Train all 5 folds
uv run bindsite train --data-dir data/PRO --train-data data/PRO/csv/PRO_Train_335.csv --output-dir output/PRO

# Train a single fold (0-4)
uv run bindsite train --data-dir data/PRO --train-data data/PRO/csv/PRO_Train_335.csv --output-dir output/PRO --fold 0

# Resume a fold from its saved checkpoint
uv run bindsite train --data-dir data/PRO --train-data data/PRO/csv/PRO_Train_335.csv --output-dir output/PRO --fold 1 --resume
```

Each fold writes `fold_{k}.pt` (best weights), `fold_{k}.resume.pt` (full training state), and `fold_{k}.threshold.json` (MCC-optimal threshold on that fold's validation split).

### 5. Evaluation & Prediction
Evaluate the ensemble on a test set or predict on a single protein:
```bash
# Evaluate ensemble (all threshold modes, default)
uv run bindsite evaluate --data-dir data/PRO --eval-data data/PRO/csv/PRO_Test_60.csv

# Predict on a single structural input
uv run bindsite predict --pdb protein.pdb --sequence "MAV..." --weights-dir data/PRO/weights
```

The `evaluate` command accepts `--threshold-mode` for the decision threshold used by MCC/F1/Pre/Rec/Acc/Spe (AUC and AUPRC are unaffected):

- `all` *(default)*: compares all modes side by side in one table.
- `fixed`: threshold = 0.5. Methodologically clean.
- `max-mcc`: threshold selected on the test predictions to maximize MCC. Matches the paper's reporting convention; leaks labels into threshold selection, so treat as optimistic.
- `val-optimal`: averages per-fold validation MCC-optimal thresholds saved during training. Clean compromise between the two.

```bash
uv run bindsite evaluate --data-dir data/PRO --eval-data data/PRO/csv/PRO_Test_60.csv --threshold-mode val-optimal
```

## Project Structure

- `src/bindsite/cli/`: Modular command-line interface.
- `src/bindsite/core/`: Centralized configuration and constants.
- `src/bindsite/data/`: FASTA/PDB parsing and dataset management.
- `src/bindsite/features/`: DSSP and ProtT5 extraction logic.
- `src/bindsite/models/`: DeepProSite Graph Transformer implementation.
- `src/bindsite/tasks/`: Training, evaluation, and inference workflows.

## Citation

If you use this work, please refer to the original DeepProSite paper:
> **DeepProSite**: A structure-aware protein binding site prediction framework.
