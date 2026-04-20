import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ..core.logger import logger
from ..models.graph_transformer import DeepProSite
from ..data.dataset import DeepProSiteDataset
from .evaluation import compute_metrics, find_mcc_optimal_threshold

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

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "rate": self._rate,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self._step = state.get("step", 0)
        self._rate = state.get("rate", 0.0)

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
    metrics = compute_metrics(y_true_all, y_pred_all)
    return metrics, np.asarray(y_true_all), np.asarray(y_pred_all)

def run_training_fold(fold_idx, train_data, val_data, feature_dir, pdb_dir, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = DeepProSiteDataset(train_data, feature_dir, pdb_dir)
    # Match paper: 5x samples per epoch with replacement
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=config['num_samples'],
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
    
    val_loader = DataLoader(DeepProSiteDataset(val_data, feature_dir, pdb_dir), batch_size=config['batch_size'], shuffle=False)
    
    model = DeepProSite(
        node_features=config['node_features'],
        edge_features=config['edge_features'],
        hidden_dim=config['hidden_dim'],
        num_encoder_layers=config['num_encoder_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        k_neighbors=config['k_neighbors'],
        augment_eps=config['augment_eps'],
    ).to(device)

    # Standard warmup: 5 epochs (matches PRO and PEP papers)
    warmup_steps = 5 * len(train_loader)
    optimizer = get_std_opt(
        model.parameters(),
        d_model=config['hidden_dim'],
        warmup_steps=warmup_steps,
    )
    
    best_auprc, patience_counter = 0, 0
    start_epoch = 0
    best_model_path = Path(config['output_dir']) / f"fold_{fold_idx}.pt"
    resume_state_path = Path(config['output_dir']) / f"fold_{fold_idx}.resume.pt"
    threshold_path = Path(config['output_dir']) / f"fold_{fold_idx}.threshold.json"

    if config.get("resume", False):
        if resume_state_path.exists():
            checkpoint = torch.load(resume_state_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best_auprc = float(checkpoint.get("best_auprc", 0.0))
            patience_counter = int(checkpoint.get("patience_counter", 0))
            logger.info(
                "Resumed fold %s from epoch %s (best AUPRC=%.4f, patience=%s)",
                fold_idx,
                start_epoch,
                best_auprc,
                patience_counter,
            )
        else:
            logger.warning(
                "Resume requested for fold %s but no resume state found at %s. Starting fresh.",
                fold_idx,
                resume_state_path,
            )
    
    for epoch in range(start_epoch, config['epochs']):
        loss, _ = train_one_epoch(model, train_loader, optimizer, device)
        val_m, val_true, val_pred = validate(model, val_loader, device)
        
        logger.info(f"Fold {fold_idx} Epoch {epoch}: Loss={loss:.4f}, Val AUPRC={val_m['auprc']:.4f}")
        
        if val_m['auprc'] > best_auprc:
            best_auprc = val_m['auprc']
            torch.save(model.state_dict(), best_model_path)
            # Save the MCC-optimal threshold on this fold's validation split so
            # downstream evaluation can use a principled, non-test-leaking
            # threshold (see evaluate --threshold-mode val-optimal).
            val_threshold = find_mcc_optimal_threshold(val_true, val_pred)
            with open(threshold_path, "w") as f:
                json.dump(
                    {
                        "threshold": val_threshold,
                        "epoch": int(epoch),
                        "val_auprc": float(val_m['auprc']),
                        "val_auc": float(val_m['auc']),
                    },
                    f,
                    indent=2,
                )
            patience_counter = 0
        else:
            patience_counter += 1
        
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_auprc": best_auprc,
                "patience_counter": patience_counter,
            },
            resume_state_path,
        )

        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    return best_auprc
