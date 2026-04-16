# BindSite

BindSite is a tool for the training, evaluation, and prediction of protein binding sites. Inspired by DeepProSite, it leverages state-of-the-art structural and sequence-based models to identify key binding regions in proteins.

## Installation

This project requires [uv](https://docs.astral.sh/uv/) for environment management.

1. **Clone the repository**:
   ```bash
   git clone git@github.com:vietnqw/BindSite.git
   cd BindSite
   ```

2. **Setup the environment**:
   ```bash
   uv sync
   ```
   This will create a virtual environment and install all necessary dependencies, including GPU-optimized PyTorch.

## Usage

BindSite comes with a built-in CLI. You can run it using `uv run`.

### Check Environment
Verify that your environment and CUDA are correctly configured:
```bash
uv run bindsite info
```

### 3D Structure Prediction (Folding)
Predict protein 3D structures from FASTA sequences using ESMFold:
```bash
uv run bindsite fold -i data/fasta/PRO_Test_60.fa
```
Options:
- `-i, --input-fasta`: Path to one or more FASTA files. Supports single-line and DeepProSite 3-line formats.
- `-o, --output-dir`: Output directory for `.pdb` files (default: `data/pdb`).
- `--model-name`: Specify the ESMFold model version (default: `facebook/esmfold_v1`).

### Predict Binding Sites
(In development) Run prediction on a PDB file:
```bash
uv run bindsite predict path/to/molecule.pdb
```
