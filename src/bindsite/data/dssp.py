"""DSSP secondary structure feature extraction.

Extracts 14-dimensional structural features per residue from PDB files
using the DSSP algorithm (Kabsch & Sander, 1983):
  - 4 features: sin(φ), cos(φ), sin(ψ), cos(ψ)  (backbone torsion angles)
  - 1 feature:  relative solvent accessibility (RSA)
  - 9 features: one-hot secondary structure (H/B/E/G/I/T/S/C + unknown)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from Bio import pairwise2

logger = logging.getLogger(__name__)

# Standard amino acid types and their max ASA values for RSA normalization.
_AA_TYPES = "ACDEFGHIKLMNPQRSTVWY"
_SS_TYPES = "HBEGITSC"
_MAX_ASA = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
            185, 160, 145, 180, 225, 115, 140, 155, 255, 230]


def run_dssp(pdb_path: str | Path, dssp_binary: str = "mkdssp") -> str:
    """Run the DSSP program on a PDB file and return raw output.

    Args:
        pdb_path: Path to the input PDB file.
        dssp_binary: Path or name of the DSSP executable.

    Returns:
        Raw DSSP output as a string.

    Raises:
        FileNotFoundError: If PDB file or DSSP binary is not found.
        RuntimeError: If DSSP execution fails.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    try:
        result = subprocess.run(
            [dssp_binary, "-i", str(pdb_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except FileNotFoundError:
        raise FileNotFoundError(
            f"DSSP binary not found: '{dssp_binary}'. "
            "Install DSSP via: apt install dssp  or  conda install -c salilab dssp"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"DSSP failed on {pdb_path}: {e.stderr}")


def parse_dssp_output(dssp_text: str) -> tuple[str, list[np.ndarray]]:
    """Parse raw DSSP output into sequence and per-residue features.

    Args:
        dssp_text: Raw DSSP output string.

    Returns:
        Tuple of (sequence, list_of_feature_vectors) where each feature
        vector is 12-dimensional: [φ, ψ, RSA, SS_onehot(9)].
    """
    lines = dssp_text.splitlines()

    # Find the header separator line (starts with '#').
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            header_idx = i
            break

    sequence = ""
    features: list[np.ndarray] = []

    for line in lines[header_idx + 1:]:
        if len(line) < 115:
            continue

        aa = line[13]
        if aa in ("!", "*"):  # Chain break or missing residue.
            continue

        sequence += aa

        # Secondary structure (8 types + unknown).
        ss = line[16]
        if ss == " ":
            ss = "C"  # Coil
        ss_vec = np.zeros(9, dtype=np.float32)
        ss_idx = _SS_TYPES.find(ss)
        ss_vec[ss_idx if ss_idx >= 0 else 8] = 1.0  # 8 = unknown

        # Backbone torsion angles.
        phi = float(line[103:109].strip())
        psi = float(line[109:115].strip())

        # Relative solvent accessibility.
        acc = float(line[34:38].strip())
        aa_idx = _AA_TYPES.find(aa)
        if aa_idx >= 0:
            rsa = min(100.0, round(acc / _MAX_ASA[aa_idx] * 100)) / 100.0
        else:
            rsa = 0.0

        features.append(np.array([phi, psi, rsa, *ss_vec], dtype=np.float32))

    return sequence, features


def _align_dssp_to_reference(
    dssp_seq: str,
    dssp_features: list[np.ndarray],
    ref_seq: str,
) -> list[np.ndarray]:
    """Align DSSP features to the reference sequence using global alignment.

    Handles cases where the DSSP-parsed sequence differs from the original
    (e.g., due to missing residues in the PDB file).

    Args:
        dssp_seq: Sequence as parsed from DSSP output.
        dssp_features: Feature vectors corresponding to dssp_seq.
        ref_seq: Reference sequence from the FASTA file.

    Returns:
        Feature vectors aligned to ref_seq, with unknown features for gaps.
    """
    # Placeholder for missing residues: unknown SS, invalid angles.
    unknown_vec = np.zeros(12, dtype=np.float32)
    unknown_vec[0:2] = 360.0  # Invalid angle marker
    unknown_vec[11] = 1.0  # Unknown SS

    alignments = pairwise2.align.globalxx(ref_seq, dssp_seq)
    aligned_ref = alignments[0].seqA
    aligned_dssp = alignments[0].seqB

    # Map DSSP features to alignment positions.
    dssp_iter = iter(dssp_features)
    aligned_features: list[np.ndarray] = []
    for aa in aligned_dssp:
        if aa == "-":
            aligned_features.append(unknown_vec.copy())
        else:
            aligned_features.append(next(dssp_iter))

    # Extract features only at positions where the reference has residues.
    result = [
        aligned_features[i]
        for i in range(len(aligned_ref))
        if aligned_ref[i] != "-"
    ]
    return result


def _transform_angles(features: np.ndarray) -> np.ndarray:
    """Transform torsion angles from degrees to sin/cos representation.

    Input:  (L, 12) — [φ, ψ, RSA, SS(9)]
    Output: (L, 14) — [sin(φ), sin(ψ), cos(φ), cos(ψ), RSA, SS(9)]
    """
    angles = features[:, 0:2]
    rsa_ss = features[:, 2:]

    radians = np.deg2rad(angles)
    sin_cos = np.concatenate([np.sin(radians), np.cos(radians)], axis=1)

    return np.concatenate([sin_cos, rsa_ss], axis=1).astype(np.float32)


def extract_dssp_features(
    pdb_path: str | Path,
    ref_seq: str,
    dssp_binary: str = "mkdssp",
) -> np.ndarray:
    """Extract 14-dimensional DSSP features for a protein.

    End-to-end pipeline: run DSSP → parse → align → transform.

    Args:
        pdb_path: Path to the PDB file.
        ref_seq: Reference amino acid sequence for alignment.
        dssp_binary: Path or name of the DSSP executable.

    Returns:
        Array of shape (L, 14) with DSSP features per residue.
    """
    dssp_text = run_dssp(pdb_path, dssp_binary)
    dssp_seq, dssp_features = parse_dssp_output(dssp_text)

    if dssp_seq != ref_seq:
        logger.info(
            "DSSP sequence differs from reference for %s, aligning...",
            Path(pdb_path).stem,
        )
        dssp_features = _align_dssp_to_reference(dssp_seq, dssp_features, ref_seq)

    features = np.array(dssp_features, dtype=np.float32)
    return _transform_angles(features)
