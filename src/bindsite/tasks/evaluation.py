import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from ..core.logger import logger
from ..models.graph_transformer import DeepProSite
from ..data.dataset import DeepProSiteDataset


def find_mcc_optimal_threshold(y_true, y_pred, num_candidates: int = 100) -> float:
    """Return the threshold that maximizes MCC on the given predictions.
    Searches a linear grid between min and max of y_pred (clipped to [0.01, 0.99]).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lo = float(max(0.01, y_pred.min()))
    hi = float(min(0.99, y_pred.max()))
    if hi <= lo:
        return 0.5
    best_mcc = -1.0
    best_t = 0.5
    for t in np.linspace(lo, hi, num_candidates):
        y_bin = (y_pred > t).astype(int)
        cm = confusion_matrix(y_true, y_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / (mcc_den + 1e-9)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = float(t)
    return best_t


def compute_metrics(y_true, y_pred, threshold=0.5, optimize_threshold=False):
    """Computes a range of classification metrics.
    By default uses a fixed threshold (0.5). When ``optimize_threshold`` is
    True, a local MCC-maximizing threshold is used instead (useful for
    validation workflows only; never use on test predictions to report
    final numbers).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if optimize_threshold:
        threshold = find_mcc_optimal_threshold(y_true, y_pred)

    y_pred_bin = (y_pred > threshold).astype(int)
    auc_score = roc_auc_score(y_true, y_pred)
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_pred)
    auprc_score = auc(recall_pts, precision_pts)
    
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
    
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_score = mcc_num / (mcc_den + 1e-9)
    
    return {
        'auc': auc_score, 'auprc': auprc_score, 'mcc': mcc_score,
        'acc': acc, 'pre': pre, 'rec': rec, 'spe': spe, 'f1': f1,
        'threshold': threshold
    }

def _resolve_threshold(
    threshold_mode: str,
    model_paths,
    y_true,
    y_pred,
) -> tuple[float, str]:
    """Resolve a decision threshold based on the requested mode.

    Returns (threshold, source) where ``source`` is a short human-readable
    description of where the threshold came from.
    """
    mode = (threshold_mode or "fixed").lower()
    if mode == "fixed":
        return 0.5, "fixed=0.5"
    if mode == "max-mcc":
        t = find_mcc_optimal_threshold(y_true, y_pred)
        return t, f"max-mcc (optimized on test: {t:.4f})"
    if mode == "val-optimal":
        thresholds: list[float] = []
        missing: list[str] = []
        for mp in model_paths:
            tpath = Path(mp).with_suffix("").as_posix() + ".threshold.json"
            tpath = Path(tpath)
            if tpath.exists():
                with open(tpath) as f:
                    thresholds.append(float(json.load(f)["threshold"]))
            else:
                missing.append(str(tpath))
        if not thresholds:
            logger.warning(
                "No per-fold validation thresholds found (looked for %s). "
                "Falling back to fixed=0.5.",
                missing,
            )
            return 0.5, "val-optimal (missing; fell back to fixed=0.5)"
        t = float(np.mean(thresholds))
        return t, f"val-optimal (mean of {len(thresholds)} folds: {t:.4f})"
    raise ValueError(f"Unknown threshold_mode: {threshold_mode!r}")


def run_ensemble_evaluation(
    test_data,
    feature_dir,
    pdb_dir,
    model_paths,
    config,
    threshold_mode: str = "all",
):
    """Evaluates an ensemble of models on a test set.

    ``threshold_mode`` controls the decision threshold used for
    threshold-dependent metrics (MCC, F1, Precision, Recall, Accuracy,
    Specificity). AUC and AUPRC are unaffected.

    Modes:
        - ``all``: compute and return all thresholding approaches.
        - ``fixed``: threshold = 0.5 (methodologically clean).
        - ``max-mcc``: threshold is selected on the test predictions to
          maximize MCC. Reproduces the paper's reporting but leaks labels
          into threshold selection, so treat the result as optimistic.
        - ``val-optimal``: threshold is the mean of per-fold validation
          MCC-optimal thresholds saved during training. Good compromise:
          higher Recall/F1 than 0.5 without looking at test labels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(DeepProSiteDataset(test_data, feature_dir, pdb_dir), batch_size=config['batch_size'], shuffle=False)
    
    models = []
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
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models.append(model)
        
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ensemble Evaluation"):
            X, V, mask, y = batch['coords'].to(device), batch['features'].to(device), batch['mask'].to(device), batch['labels'].to(device)
            
            # Average predictions from all models (post-sigmoid)
            fold_preds = [torch.sigmoid(m(X, V, mask)) for m in models]
            avg_preds = torch.stack(fold_preds).mean(dim=0)
            
            y_true_all.extend(y[mask > 0.5].cpu().numpy())
            y_pred_all.extend(avg_preds[mask > 0.5].cpu().numpy())
            
    mode = (threshold_mode or "all").lower()
    if mode == "all":
        ordered_modes = ["fixed", "val-optimal", "max-mcc"]
        by_mode = {}
        for m in ordered_modes:
            threshold, source = _resolve_threshold(m, model_paths, y_true_all, y_pred_all)
            metrics = compute_metrics(y_true_all, y_pred_all, threshold=threshold)
            metrics["threshold_source"] = source
            by_mode[m] = metrics
        return {"by_mode": by_mode}

    threshold, source = _resolve_threshold(mode, model_paths, y_true_all, y_pred_all)
    logger.info("Evaluation using threshold source: %s", source)
    metrics = compute_metrics(y_true_all, y_pred_all, threshold=threshold)
    metrics["threshold_source"] = source
    return metrics
