import torch
from transformers import EsmModel, EsmTokenizer
import numpy as np
import re
from bindsite.utils import setup_logger

logger = setup_logger(__name__)


class ESM2Extractor:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", device=None, min_val=None, max_val=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(self.device).eval()
        self.repr_layer = self.model.config.num_hidden_layers
        
        self.min_val = min_val
        self.max_val = max_val

    def set_normalization_bounds(self, min_val, max_val):
        """Sets the bounds for min-max normalization."""
        self.min_val = min_val
        self.max_val = max_val

    def extract(self, sequence: str):
        """Extracts and optionally normalizes features for a single sequence."""
        sequence = re.sub(r"[UZOB]", "X", sequence)
        
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # The last hidden state has shape [1, L+2, D]
            # We remove the first token (CLS) and last token (EOS) and the batch dimension
            embedding = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()
            
        if self.min_val is not None and self.max_val is not None:
            # Min-max normalization as per DeepProSite paper logic
            embedding = (embedding - self.min_val) / (self.max_val - self.min_val + 1e-9)
            embedding = np.clip(embedding, 0, 1)
            
        return embedding
