import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ..core.logger import logger
from ..models.transformer import DeepProSite
from ..data.dataset import DeepProSiteDataset
from .evaluation import compute_metrics

class NoamOpt:
    """Learning rate schedule as defined in the paper."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model=64, warmup_steps=1000, top_lr=0.0004):
    warmup = float(warmup_steps)
    denom = d_model ** (-0.5) * min(warmup ** (-0.5), warmup * warmup ** (-1.5))
    factor = top_lr / denom
    adam = optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(d_model, factor, warmup_steps, adam)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    y_true_all, y_pred_all = [], []
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    for batch in tqdm(loader, desc="Training"):
        X, V, mask, y = batch['coords'].to(device), batch['features'].to(device), batch['mask'].to(device), batch['labels'].to(device)
        
        # Data augmentation: Add small Gaussian noise to features
        V = V + 0.1 * torch.randn_like(V) * mask.unsqueeze(-1)
        
        optimizer.zero_grad()
        logits = model(X, V, mask)
        loss = (criterion(logits, y) * mask).sum() / (mask.sum() + 1e-9)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        with torch.no_grad():
            preds = torch.sigmoid(logits)
            y_true_all.extend(y[mask > 0.5].cpu().numpy())
            y_pred_all.extend(preds[mask > 0.5].cpu().numpy())
            
    return total_loss / len(loader), compute_metrics(y_true_all, y_pred_all)

def validate(model, loader, device):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in loader:
            X, V, mask, y = batch['coords'].to(device), batch['features'].to(device), batch['mask'].to(device), batch['labels'].to(device)
            logits = model(X, V, mask)
            preds = torch.sigmoid(logits)
            y_true_all.extend(y[mask > 0.5].cpu().numpy())
            y_pred_all.extend(preds[mask > 0.5].cpu().numpy())
    return compute_metrics(y_true_all, y_pred_all)

def run_training_fold(fold_idx, train_data, val_data, feature_dir, pdb_dir, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(DeepProSiteDataset(train_data, feature_dir, pdb_dir), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(DeepProSiteDataset(val_data, feature_dir, pdb_dir), batch_size=config['batch_size'], shuffle=False)
    
    model = DeepProSite(
        node_features=config['node_features'],
        hidden_dim=config['hidden_dim'],
        num_encoder_layers=config['num_encoder_layers'],
        dropout=config['dropout']
    ).to(device)
    
    warmup_steps = config['warmup_epochs'] * len(train_loader)
    optimizer = get_std_opt(model.parameters(), d_model=config['hidden_dim'], warmup_steps=warmup_steps)
    
    best_auprc, patience_counter = 0, 0
    best_model_path = Path(config['output_dir']) / f"fold_{fold_idx}_best.pt"
    
    for epoch in range(config['epochs']):
        loss, _ = train_one_epoch(model, train_loader, optimizer, device)
        val_m = validate(model, val_loader, device)
        
        logger.info(f"Fold {fold_idx} Epoch {epoch}: Loss={loss:.4f}, Val AUPRC={val_m['auprc']:.4f}")
        
        if val_m['auprc'] > best_auprc:
            best_auprc = val_m['auprc']
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    return best_auprc
