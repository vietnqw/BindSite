import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
import logging
from ..core.logger import logger

class ProtT5Extractor:
    def __init__(self, model_name="Rostlab/prot_t5_xl_uniref50", device=None, min_val=None, max_val=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Loading {model_name} on {self.device}...")
        # Note: use_fast=False is used to avoid tiktoken dependency in some environments
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False)
        self.model = T5EncoderModel.from_pretrained(model_name, use_safetensors=True).to(self.device).eval()
        
        self.min_val = min_val
        self.max_val = max_val

    def set_normalization_bounds(self, min_val, max_val):
        """Sets the bounds for min-max normalization."""
        self.min_val = min_val
        self.max_val = max_val

    def extract(self, sequence: str):
        """Extracts and normalizes embeddings for a single sequence."""
        # ProtT5 expects space-separated sequences
        seq_spaced = " ".join(list(sequence))
        
        inputs = self.tokenizer(seq_spaced, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # The last hidden state has shape [1, L+1, 1024]
            # We remove the last token (EOS) and the batch dimension
            embedding = outputs.last_hidden_state[0, :-1, :].cpu().numpy()
            
        if self.min_val is not None and self.max_val is not None:
            # Min-max normalization as per DeepProSite paper
            embedding = (embedding - self.min_val) / (self.max_val - self.min_val + 1e-9)
            embedding = np.clip(embedding, 0, 1)
            
        return embedding
