import torch
import numpy as np
from ..models.graph_transformer import DeepProSite

def run_single_prediction(coords, features, model_paths, config):
    """Predicts probabilities for a single protein structure/feature set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L = len(coords)
    max_len = config.get('max_len', 1000)
    
    padded_feat = torch.zeros((1, max_len, features.shape[1]), dtype=torch.float32)
    padded_coords = torch.zeros((1, max_len, 3), dtype=torch.float32)
    mask = torch.zeros((1, max_len), dtype=torch.float32)
    
    curr_l = min(L, max_len)
    padded_feat[0, :curr_l] = torch.from_numpy(features[:curr_l])
    padded_coords[0, :curr_l] = torch.from_numpy(coords[:curr_l])
    mask[0, :curr_l] = 1.0
    
    padded_feat = padded_feat.to(device)
    padded_coords = padded_coords.to(device)
    mask = mask.to(device)
    
    fold_preds = []
    with torch.no_grad():
        for path in model_paths:
            model = DeepProSite(
                node_features=config['node_features'],
                edge_features=config.get('edge_features', 16),
                hidden_dim=config['hidden_dim'],
                num_encoder_layers=config['num_encoder_layers'],
                num_heads=config.get('num_heads', 4),
                dropout=config['dropout'],
                k_neighbors=config.get('k_neighbors', 30),
                augment_eps=config.get('augment_eps', 0.1),
            ).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            
            logits = model(padded_coords, padded_feat, mask)
            fold_preds.append(torch.sigmoid(logits))
            
    avg_preds = torch.stack(fold_preds).mean(dim=0)[0] # [max_len]
    return avg_preds[:curr_l].cpu().numpy()
