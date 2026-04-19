import numpy as np
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
DSSP_BIN = BIN_DIR / "mkdssp"

# --- Biological Constants ---
# DSSP relative solvent accessibility max values per amino acid, used as
# ASA / max_ASA normalization (paper/reference implementation).
RASA_MAX = {
    "A": 115, "C": 135, "D": 150, "E": 190, "F": 210,
    "G": 75, "H": 195, "I": 175, "K": 200, "L": 170,
    "M": 185, "N": 160, "P": 145, "Q": 180, "R": 225,
    "S": 115, "T": 140, "V": 155, "W": 255, "Y": 230,
}

# Secondary-structure alphabet used by DSSP (8 states). Missing/unknown
# residues are represented by an extra 9th position in one-hot vectors.
SS_TYPES = ["H", "B", "E", "G", "I", "T", "S", "C"]

# --- Model Hyperparameters ---
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.2
DEFAULT_MAX_LEN = 1000

# --- Training Constants ---
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_PATIENCE = 8
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_TOP_LR = 0.0004

# --- Dataset URLs ---
PRO_FASTA_URLS = {
    "PRO_Train_335.fa": "https://raw.githubusercontent.com/WeiLab-Biology/DeepProSite/main/DeepProSite-main/datasets/Train_335.fa",
    "PRO_Test_60.fa": "https://raw.githubusercontent.com/WeiLab-Biology/DeepProSite/main/DeepProSite-main/datasets/Test_60.fa",
    "PRO_Test_315.fa": "https://raw.githubusercontent.com/WeiLab-Biology/DeepProSite/main/DeepProSite-main/datasets/Test_315.fa",
}

PEPBCL_TSV_URLS = {
    "Dataset1_train.tsv": "https://raw.githubusercontent.com/Ruheng-W/PepBCL/master/data/Dataset1_train.tsv",
    "Dataset1_test.tsv": "https://raw.githubusercontent.com/Ruheng-W/PepBCL/master/data/Dataset1_test.tsv",
    "Dataset2_train.tsv": "https://raw.githubusercontent.com/Ruheng-W/PepBCL/master/data/Dataset2_train.tsv",
    "Dataset2_test.tsv": "https://raw.githubusercontent.com/Ruheng-W/PepBCL/master/data/Dataset2_test.tsv",
}

SPRINT_STR_ZIP_URL = "https://raw.githubusercontent.com/GTaherzadeh/SPRINT-STR/master/Data.zip"

PEP_OUTPUT_MAPPING = {
    "Dataset1_train.tsv": "PEP_Train_1154.fa",
    "Dataset1_test.tsv": "PEP_Test_125.fa",
    "Dataset2_train.tsv": "PEP_Train_640.fa",
    "Dataset2_test.tsv": "PEP_Test_639.fa",
}
