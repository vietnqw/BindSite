import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from ..core.logger import logger
from ..models.transformer import DeepProSite
from ..data.dataset import DeepProSiteDataset

def compute_metrics(y_true, y_pred, threshold=0.5):
    """Computes a range of classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_bin = (y_pred > threshold).astype(int)
    
    auc_score = roc_auc_score(y_true, y_pred)
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_pred)
    auprc_score = auc(recall_pts, precision_pts)
    
    # Handle the case where confusion matrix might be smaller than 2x2
    try:
        cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
    
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_score = mcc_num / (mcc_den + 1e-9)
    
    return {
        'auc': auc_score, 'auprc': auprc_score, 'mcc': mcc_score,
        'acc': acc, 'pre': pre, 'rec': rec, 'spe': spe, 'f1': f1
    }

def run_ensemble_evaluation(test_data, feature_dir, pdb_dir, model_paths, config):
    """Evaluates an ensemble of models on a test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(DeepProSiteDataset(test_data, feature_dir, pdb_dir), batch_size=config['batch_size'], shuffle=False)
    
    models = []
    for path in model_paths:
        model = DeepProSite(
            node_features=config['node_features'],
            hidden_dim=config['hidden_dim'],
            num_encoder_layers=config['num_encoder_layers'],
            dropout=config['dropout']
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
        
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ensemble Evaluation"):
            X, V, mask, y = batch['coords'].to(device), batch['features'].to(device), batch['mask'].to(device), batch['labels'].to(device)
            
            # Average predictions from all models
            fold_preds = [torch.sigmoid(m(X, V, mask)) for m in models]
            avg_preds = torch.stack(fold_preds).mean(dim=0)
            
            y_true_all.extend(y[mask > 0.5].cpu().numpy())
            y_pred_all.extend(avg_preds[mask > 0.5].cpu().numpy())
            
    return compute_metrics(y_true_all, y_pred_all)
