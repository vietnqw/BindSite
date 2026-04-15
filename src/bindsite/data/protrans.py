"""ProtT5 language model embedding extraction and normalization.

Uses the ProtT5-XL-UniRef50 model to generate 1024-dimensional per-residue
embeddings, then applies min-max normalization based on training set statistics.
"""

from __future__ import annotations

import gc
import logging
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProtTransExtractor:
    """Extracts per-residue embeddings from the ProtT5-XL-UniRef50 model.

    This class manages model loading and batched inference. The model is
    loaded lazily on first use and can be explicitly freed.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device: Device string (e.g., "cuda:0" or "cpu").
        batch_size: Number of sequences per forward pass.
    """

    def __init__(
        self,
        model_name_or_path: str = "Rostlab/prot_t5_xl_uniref50",
        device: str | None = None,
        batch_size: int = 10,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazily load the ProtT5 model and tokenizer."""
        if self._model is not None:
            return

        from transformers import T5EncoderModel, T5Tokenizer

        logger.info("Loading ProtT5 model from %s ...", self.model_name_or_path)
        self._tokenizer = T5Tokenizer.from_pretrained(
            self.model_name_or_path, do_lower_case=False
        )
        self._model = T5EncoderModel.from_pretrained(self.model_name_or_path)
        self._model = self._model.to(self.device).eval()
        logger.info("ProtT5 model loaded on %s", self.device)

    def extract(
        self,
        sequences: dict[str, str],
        output_dir: str | Path | None = None,
        skip_existing: bool = True,
    ) -> dict[str, np.ndarray]:
        """Extract per-residue embeddings for a set of protein sequences.

        Args:
            sequences: Mapping of protein_id -> amino_acid_sequence.
            output_dir: If provided, save each embedding as a .npy file.
            skip_existing: If True, skip IDs whose .npy files already exist.

        Returns:
            Dict mapping protein_id -> embedding array of shape (L, 1024).
        """
        self._load_model()
        assert self._model is not None and self._tokenizer is not None

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Filter to sequences that need processing.
        ids_to_process = list(sequences.keys())
        if skip_existing and output_dir is not None:
            ids_to_process = [
                pid for pid in ids_to_process
                if not (output_dir / f"{pid}.npy").exists()
            ]

        if not ids_to_process:
            logger.info("All embeddings already exist, skipping extraction.")

        # Prepare space-separated sequences with rare AA substitution.
        prepared: list[tuple[str, str]] = [
            (pid, " ".join(re.sub(r"[UZOB]", "X", sequences[pid])))
            for pid in ids_to_process
        ]

        results: dict[str, np.ndarray] = {}

        for i in tqdm(
            range(0, len(prepared), self.batch_size),
            desc="Extracting ProtT5 embeddings",
        ):
            batch = prepared[i : i + self.batch_size]
            batch_ids = [pid for pid, _ in batch]
            batch_seqs = [seq for _, seq in batch]

            # Tokenize.
            encoded = self._tokenizer.batch_encode_plus(
                batch_seqs, add_special_tokens=True, padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Forward pass.
            with torch.no_grad():
                output = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            embeddings = output.last_hidden_state.cpu().numpy()

            # Extract per-sequence embeddings (remove padding and EOS token).
            for j, pid in enumerate(batch_ids):
                seq_len = int(attention_mask[j].sum()) - 1  # Exclude </s> token.
                embedding = embeddings[j, :seq_len].astype(np.float32)
                results[pid] = embedding

                if output_dir is not None:
                    np.save(output_dir / f"{pid}.npy", embedding)

        return results

    def release(self) -> None:
        """Free the model from memory."""
        del self._model, self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ProtT5 model released from memory.")


def compute_normalization_stats(
    embedding_dir: str | Path,
    protein_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute min-max normalization statistics from training set embeddings.

    Args:
        embedding_dir: Directory containing .npy embedding files.
        protein_ids: List of training protein IDs to compute stats from.

    Returns:
        Tuple of (min_values, max_values), each of shape (1024,).
    """
    embedding_dir = Path(embedding_dir)
    global_min: np.ndarray | None = None
    global_max: np.ndarray | None = None

    for pid in tqdm(protein_ids, desc="Computing normalization stats"):
        emb = np.load(embedding_dir / f"{pid}.npy")
        batch_min = emb.min(axis=0)
        batch_max = emb.max(axis=0)

        if global_min is None:
            global_min = batch_min
            global_max = batch_max
        else:
            global_min = np.minimum(global_min, batch_min)
            global_max = np.maximum(global_max, batch_max)

    assert global_min is not None and global_max is not None
    return global_min, global_max


def normalize_embeddings(
    embedding_dir: str | Path,
    output_dir: str | Path,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    protein_ids: list[str],
    skip_existing: bool = True,
) -> None:
    """Apply min-max normalization to ProtT5 embeddings.

    Normalizes each feature dimension to [0, 1] using training set statistics.

    Args:
        embedding_dir: Directory containing raw .npy embedding files.
        output_dir: Directory to save normalized .npy files.
        min_vals: Per-feature minimum values from training set, shape (1024,).
        max_vals: Per-feature maximum values from training set, shape (1024,).
        protein_ids: List of protein IDs to normalize.
        skip_existing: If True, skip IDs whose output files already exist.
    """
    embedding_dir = Path(embedding_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    denom = max_vals - min_vals
    # Avoid division by zero for constant features.
    denom[denom == 0] = 1.0

    for pid in tqdm(protein_ids, desc="Normalizing embeddings"):
        out_path = output_dir / f"{pid}.npy"
        if skip_existing and out_path.exists():
            continue

        raw = np.load(embedding_dir / f"{pid}.npy")
        normalized = ((raw - min_vals) / denom).astype(np.float32)
        np.save(out_path, normalized)
