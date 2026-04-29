
import numpy as np

def _as_1d(arr, name='array'):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    return arr

def _check_len(*arrays):
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"Arrays have inconsistent lengths: {sorted(lengths)}")

def _per_class_counts(y_true, y_pred, labels):
    """Return (TP, FP, FN) arrays — one entry per class — fully vectorised."""
    k = labels.size
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    def _map(arr):
        idx = np.full(arr.shape, -1, dtype=int)
        for lbl, i in label_to_idx.items():
            idx[arr == lbl] = i
        return idx

    ti = _map(y_true);  pi = _map(y_pred)
    valid = (ti >= 0) & (pi >= 0)
    ti, pi = ti[valid], pi[valid]

    tp = np.bincount(ti[ti == pi], minlength=k).astype(float)
    fp = np.bincount(pi, minlength=k).astype(float) - tp
    fn = np.bincount(ti, minlength=k).astype(float) - tp
    return tp, fp, fn

#confusion matrix

def confusion_matrix(y_true, y_pred, labels=None):
    y_true = _as_1d(np.asarray(y_true), 'y_true')
    y_pred = _as_1d(np.asarray(y_pred), 'y_pred')
    _check_len(y_true, y_pred)
    if labels is None:
        labels = np.union1d(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    k = labels.size
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    def _map(arr):
        idx = np.full(arr.shape, -1, dtype=int)
        for lbl, i in label_to_idx.items():
            idx[arr == lbl] = i
        return idx

    ti = _map(y_true);  pi = _map(y_pred)
    valid = (ti >= 0) & (pi >= 0)
    cm = np.bincount(ti[valid] * k + pi[valid], minlength=k*k).reshape(k, k)
    return cm, labels

#accuracy
def accuracy(y_true, y_pred):
    y_true = _as_1d(np.asarray(y_true), 'y_true')
    y_pred = _as_1d(np.asarray(y_pred), 'y_pred')
    _check_len(y_true, y_pred)
    if y_true.size == 0:
        raise ValueError("y_true is empty.")
    return float(np.mean(y_true == y_pred))
# precision / recall / f1
def _clf_metric(metric, y_true, y_pred, average, pos_label, zero_division):
    _OK = {'binary', 'macro', 'micro', 'none'}
    if average not in _OK:
        raise ValueError(f"average={average!r} must be one of {_OK}")

    y_true = _as_1d(np.asarray(y_true), 'y_true')
    y_pred = _as_1d(np.asarray(y_pred), 'y_pred')
    _check_len(y_true, y_pred)

    labels      = np.union1d(y_true, y_pred)
    tp, fp, fn  = _per_class_counts(y_true, y_pred, labels)

    if metric == 'precision':
        denom      = tp + fp
        per_class  = np.where(denom == 0, zero_division, tp / denom)
        micro_num  = tp.sum();  micro_den = (tp + fp).sum()
    elif metric == 'recall':
        denom      = tp + fn
        per_class  = np.where(denom == 0, zero_division, tp / denom)
        micro_num  = tp.sum();  micro_den = (tp + fn).sum()
    else:  # f1
        denom      = 2*tp + fp + fn
        per_class  = np.where(denom == 0, zero_division, 2*tp / denom)
        micro_num  = 2*tp.sum();  micro_den = (2*tp + fp + fn).sum()

    if average == 'none'  : return per_class
    if average == 'macro' : return float(np.mean(per_class))
    if average == 'micro' :
        return float(micro_num / micro_den) if micro_den > 0 else zero_division
    if pos_label not in labels:
        return zero_division
    idx = np.where(labels == pos_label)[0][0]
    return float(per_class[idx])


def precision(y_true, y_pred, average='binary', pos_label=1, zero_division=0.0):
    return _clf_metric('precision', y_true, y_pred, average, pos_label, zero_division)


def recall(y_true, y_pred, average='binary', pos_label=1, zero_division=0.0):
    return _clf_metric('recall', y_true, y_pred, average, pos_label, zero_division)


def f1(y_true, y_pred, average='binary', pos_label=1, zero_division=0.0):
    return _clf_metric('f1', y_true, y_pred, average, pos_label, zero_division)

# regression metrics

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Inputs are empty.")
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean(np.abs(y_true - y_pred)))
def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

# ROC curve + AUC 
def roc_curve(y_true, y_score, pos_label=1):
    y_true  = _as_1d(np.asarray(y_true),          'y_true')
    y_score = _as_1d(np.asarray(y_score, float),  'y_score')
    _check_len(y_true, y_score)

    pos   = (y_true == pos_label).astype(float)
    n_pos = pos.sum();  n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least one positive and one negative sample.")

    order      = np.argsort(-y_score, kind='stable')
    sorted_pos = pos[order]
    thresholds = y_score[order]

    tp_cum = np.cumsum(sorted_pos)
    fp_cum = np.cumsum(1.0 - sorted_pos)

    tpr = np.concatenate(([0.], tp_cum / n_pos))
    fpr = np.concatenate(([0.], fp_cum / n_neg))
    if fpr.size > 2:
        keep = np.ones(fpr.size, dtype=bool)
        keep[1:-1] = (np.diff(fpr[:-1]) != 0) | (np.diff(tpr[:-1]) != 0)
        fpr        = fpr[keep]
        tpr        = tpr[keep]
        thresholds = thresholds[keep[1:]]

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape.")
    if fpr.size < 2:
        raise ValueError("Need at least 2 points to compute AUC.")
    order = np.argsort(fpr, kind='stable')
    return float(np.trapz(tpr[order], fpr[order]))

print("metrics.py loaded ✓")
