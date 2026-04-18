import numpy as np
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
DSSP_BIN = BIN_DIR / "mkdssp"

# --- Biological Constants ---
# Normalized Mean/STD for Relative Accessible Surface Area (RASA)
# Based on the paper's 14D DSSP feature definition
RASA_STD = {
    'A': (0.32, 0.20), 'R': (0.54, 0.21), 'N': (0.47, 0.21), 'D': (0.47, 0.21),
    'C': (0.16, 0.17), 'Q': (0.50, 0.21), 'E': (0.52, 0.21), 'G': (0.36, 0.19),
    'H': (0.35, 0.21), 'I': (0.23, 0.18), 'L': (0.24, 0.18), 'K': (0.54, 0.22),
    'M': (0.29, 0.19), 'F': (0.20, 0.18), 'P': (0.44, 0.21), 'S': (0.42, 0.21),
    'T': (0.39, 0.20), 'W': (0.20, 0.18), 'Y': (0.26, 0.19), 'V': (0.26, 0.18)
}

# Secondary Structure types (H, B, E, G, I, T, S, P, -)
SS_TYPES = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'P', '-']

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
