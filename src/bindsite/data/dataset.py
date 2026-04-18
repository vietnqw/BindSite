import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from ..core.logger import logger
from ..core.config import DEFAULT_MAX_LEN
from .io import extract_ca_coordinates

class DeepProSiteDataset(Dataset):
    def __init__(self, data_list, feature_dir, pdb_dir, max_len=DEFAULT_MAX_LEN):
        """
        Args:
            data_list: List of BindingRecord objects.
            feature_dir: Path to directory containing .npy features (1038D).
            pdb_dir: Path to directory containing .pdb files.
            max_len: Maximum sequence length for padding.
        """
        self.data_list = data_list
        self.feature_dir = Path(feature_dir)
        self.pdb_dir = Path(pdb_dir)
        self.max_len = max_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        record = self.data_list[idx]
        seq_id = record.seq_id
        
        # 1. Load pre-computed features (1038D)
        feat_path = self.feature_dir / f"{seq_id}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
        features = np.load(feat_path) # [L, 1038]
        
        # 2. Load coordinates
        pdb_path = self.pdb_dir / f"{seq_id}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        coords = extract_ca_coordinates(pdb_path) # [L, 3]
        
        # 3. Convert labels string "010..." to tensor
        label_tensor = torch.tensor([int(c) for c in record.labels], dtype=torch.float32)
        
        # 4. Alignment and Length check
        L = len(features)
        if len(coords) < L:
            # Handle cases where PDB structure is shorter than sequence (e.g., missing parts)
            # This shouldn't happen with our realignment logic in extraction, but safety first
            logger.debug(f"PDB ({len(coords)}) shorter than feature sequence ({L}) for {seq_id}. Truncating.")
            L = len(coords)
            features = features[:L]
            label_tensor = label_tensor[:L]
        elif len(coords) > L:
            coords = coords[:L]
            
        # 5. Padding & Masking
        padded_feat = np.zeros((self.max_len, features.shape[1]), dtype=np.float32)
        padded_coords = np.zeros((self.max_len, 3), dtype=np.float32)
        padded_labels = np.zeros(self.max_len, dtype=np.float32)
        mask = np.zeros(self.max_len, dtype=np.float32)
        
        curr_l = min(L, self.max_len)
        padded_feat[:curr_l] = features[:curr_l]
        padded_coords[:curr_l] = coords[:curr_l]
        padded_labels[:curr_l] = label_tensor[:curr_l]
        mask[:curr_l] = 1.0
        
        return {
            'id': seq_id,
            'coords': torch.from_numpy(padded_coords),
            'features': torch.from_numpy(padded_feat),
            'labels': torch.from_numpy(padded_labels),
            'mask': torch.from_numpy(mask)
        }
