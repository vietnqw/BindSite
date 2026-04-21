import subprocess
import numpy as np
from pathlib import Path
from Bio import pairwise2
from bindsite.utils import setup_logger

logger = setup_logger(__name__)

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

def extract_dssp_features(pdb_path: Path, fasta_seq: str, dssp_bin: str = "bin/mkdssp"):
    """
    Extracts 14D structural features from a PDB file, aligned to a FASTA sequence.
    Features: sin/cos of PHI/PSI (4D), normalized RASA (1D), and one-hot SS (9D).
    """
    try:
        # Run mkdssp and capture output
        cmd = [str(dssp_bin), "-i", str(pdb_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dssp_lines = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        logger.error(f"DSSP failed for {pdb_path}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"mkdssp binary not found. Please install dssp (e.g., conda install -c salilab dssp).")
        raise

    # 1. Parse DSSP output
    # Find start of data
    header_idx = 0
    for idx, line in enumerate(dssp_lines):
        if line.startswith("  #  RESIDUE"):
            header_idx = idx
            break
    
    data_lines = dssp_lines[header_idx + 1:]
    dssp_data = []
    
    for line in data_lines:
        if len(line) < 115:
            continue
        if line[13] in {"!", "*"}:
            continue

        aa = line[13]
        if aa.islower(): # DSSP represents some cysteines as a-z for bridges
            aa = 'C'
            
        ss = line[16] if line[16] != " " else "C"
        try:
            acc = float(line[34:38].strip())
            phi = float(line[103:109].strip())
            psi = float(line[109:115].strip())
            max_asa = RASA_MAX.get(aa, 1)
            rasa = min(100.0, round(acc / max_asa * 100.0)) / 100.0
            dssp_data.append({"aa": aa, "ss": ss, "rasa": rasa, "phi": phi, "psi": psi})
        except ValueError:
            continue

    # 2. Extract DSSP sequence and features
    dssp_seq = "".join([d['aa'] for d in dssp_data])
    
    # 3. Align DSSP sequence to FASTA sequence (to handle missing residues)
    alignments = pairwise2.align.globalxx(fasta_seq, dssp_seq)
    best_align = alignments[0]
    aligned_fasta, aligned_dssp = best_align[:2]
    
    # 4. Map features to FASTA sequence
    final_features = []
    dssp_ptr = 0
    
    for f_aa, d_aa in zip(aligned_fasta, aligned_dssp):
        if f_aa == "-":
            if d_aa != "-":
                dssp_ptr += 1
            continue

        if d_aa == "-":
            # [sin(phi), cos(phi), sin(psi), cos(psi), rasa, 9x ss-onehot]
            # For missing residues: neutral angles, zero rasa, unknown ss.
            feat = [0.0, 1.0, 0.0, 1.0, 0.0] + [0.0] * 8 + [1.0]
        else:
            data = dssp_data[dssp_ptr]
            phi_rad = np.radians(data['phi'])
            psi_rad = np.radians(data['psi'])

            ss_onehot = [0.0] * 9
            ss_idx = SS_TYPES.index(data["ss"]) if data["ss"] in SS_TYPES else 8
            ss_onehot[ss_idx] = 1.0

            feat = [
                np.sin(phi_rad), np.cos(phi_rad),
                np.sin(psi_rad), np.cos(psi_rad),
                data["rasa"],
            ] + ss_onehot

            dssp_ptr += 1

        final_features.append(feat)

    return np.array(final_features, dtype=np.float32)
