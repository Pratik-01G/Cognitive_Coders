
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