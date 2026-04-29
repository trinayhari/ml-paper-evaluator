import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, brier_score_loss


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def classification_report_dict(y_true, y_prob, threshold: float = 0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'brier': float(brier_score_loss(y_true, y_prob)),
        'ece': expected_calibration_error(y_true, y_prob),
    }
    if len(set(y_true)) == 2:
        out['auroc'] = float(roc_auc_score(y_true, y_prob))
    return out
