import subprocess
import numpy as np
from pathlib import Path
from Bio import pairwise2
from ..core.config import RASA_STD, SS_TYPES, DSSP_BIN
from ..core.logger import logger

def extract_dssp_features(pdb_path: Path, fasta_seq: str, mkdssp_bin: str = DSSP_BIN):
    """
    Extracts 14D structural features from a PDB file, aligned to a FASTA sequence.
    Features: sin/cos of PHI/PSI (4D), normalized RASA (1D), and one-hot SS (9D).
    """
    if not Path(mkdssp_bin).exists():
        # Fallback to system mkdssp if local binary not found
        mkdssp_bin = "mkdssp"

    try:
        # Run mkdssp and capture output
        cmd = [str(mkdssp_bin), "-i", str(pdb_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dssp_lines = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        logger.error(f"DSSP failed for {pdb_path}: {e}")
        raise

    # 1. Parse DSSP output
    # Find start of data
    header_idx = 0
    for idx, line in enumerate(dssp_lines):
        if line.startswith("  #  RESIDUE"):
            header_idx = idx
            break
    
    data_lines = dssp_lines[header_idx + 1:]
    dssp_data = [] # List of (res_idx, aa, ss, rasa, phi, psi)
    
    for line in data_lines:
        if line[13] == '!': continue # Missing residue
        
        aa = line[13]
        ss = line[16] if line[16] != ' ' else '-'
        try:
            rasa = float(line[35:38].strip())
            phi = float(line[103:109].strip())
            psi = float(line[109:115].strip())
            dssp_data.append({'aa': aa, 'ss': ss, 'rasa': rasa, 'phi': phi, 'psi': psi})
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
        if f_aa == '-': # Insertion in DSSP relative to FASTA (ignore)
            if d_aa != '-': dssp_ptr += 1
            continue
            
        if d_aa == '-': # Deletion in DSSP (missing structure)
            # Use neutral/zero features for missing residues
            feat = [0.0] * 14
        else:
            # Get data from dssp_data
            data = dssp_data[dssp_ptr]
            
            # Torsion angles (sin/cos)
            phi_rad = np.radians(data['phi'])
            psi_rad = np.radians(data['psi'])
            
            # Normalized ASA
            mean, std = RASA_STD.get(data['aa'], (0.0, 1.0))
            norm_rasa = (data['rasa'] - mean) / std
            
            # One-hot SS (9-state)
            ss_onehot = [0.0] * 9
            if data['ss'] in SS_TYPES:
                ss_onehot[SS_TYPES.index(data['ss'])] = 1.0
            else:
                ss_onehot[SS_TYPES.index('-')] = 1.0
                
            feat = [
                np.sin(phi_rad), np.cos(phi_rad),
                np.sin(psi_rad), np.cos(psi_rad),
                norm_rasa
            ] + ss_onehot
            
            dssp_ptr += 1
            
        final_features.append(feat)

    return np.array(final_features, dtype=np.float32)
