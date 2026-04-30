import numpy as np
def rank(data, method='average'):
    """
    Rank array elements with tie-handling.

    Parameters
    ----------
    data   : 1-D or 2-D np.ndarray
    method : 'average' | 'dense' | 'ordinal'

    Returns
    -------
    ranks : np.ndarray, same shape as data, dtype float64 (1-based)

    Examples
    --------
    >>> rank(np.array([3, 1, 4, 1, 5, 9, 2, 6]), method='average')
    array([4. , 1.5, 5. , 1.5, 6. , 8. , 3. , 7. ])
    """
    _METHODS = {'average', 'dense', 'ordinal'}
    if method not in _METHODS:
        raise ValueError(f"method={method!r} must be one of {_METHODS}")

    data = np.asarray(data, dtype=float)
    original_shape = data.shape
    flat = data.ravel()
    n    = flat.size

    if n == 0:
        return np.empty(original_shape, dtype=float)

    sorter = np.argsort(flat, kind='stable') 

    if method == 'ordinal':
        ranks_flat = np.empty(n, dtype=float)
        ranks_flat[sorter] = np.arange(1, n + 1, dtype=float)

    elif method == 'dense':
        sorted_vals = flat[sorter]
        obs = np.empty(n, dtype=bool)
        obs[0]  = True
        obs[1:] = sorted_vals[1:] != sorted_vals[:-1]   
        dense_sorted = np.cumsum(obs).astype(float)     
        ranks_flat = np.empty(n, dtype=float)
        ranks_flat[sorter] = dense_sorted

    else:
        sorted_vals = flat[sorter]
        obs = np.empty(n, dtype=bool)
        obs[0]  = True
        obs[1:] = sorted_vals[1:] != sorted_vals[:-1]

        dense_sorted = np.cumsum(obs)                    
        group_start  = np.concatenate(([0], np.where(obs[1:])[0] + 1))
        group_count  = np.diff(np.concatenate((group_start, [n])))
        group_avg    = group_start + (group_count - 1) / 2.0 + 1.0
        avg_sorted   = group_avg[dense_sorted - 1]
        ranks_flat   = np.empty(n, dtype=float)
        ranks_flat[sorter] = avg_sorted

    return ranks_flat.reshape(original_shape)


def percentile(data, q, interpolation='linear', axis=None, ignore_nan=False):
     """
    Compute the q-th percentile(s) of data.

    Parameters
    ----------
    data          : np.ndarray
    q             : float or array-like, values in [0, 100]
    interpolation : 'linear' | 'lower' | 'higher' | 'midpoint' | 'nearest'
    axis          : int or None  (None → operate on flattened array)
    ignore_nan    : bool  (if True, NaNs are dropped before computing)

    Returns
    -------
    float or np.ndarray

    Examples
    --------
    >>> percentile(np.arange(1, 11), 50)
    5.5
    >>> percentile(np.arange(1, 11), [25, 50, 75])
    array([3.25, 5.5 , 7.75])
    """
    _INTERP = {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
    if interpolation not in _INTERP:
        raise ValueError(f"interpolation={interpolation!r} must be one of {_INTERP}")

    data      = np.asarray(data, dtype=float)
    q         = np.asarray(q,    dtype=float)
    scalar_q  = q.ndim == 0
    q         = np.atleast_1d(q)

    if np.any((q < 0) | (q > 100)):
        raise ValueError("All values in q must be in [0, 100].")

    def _pct_1d(arr, qs):
        if ignore_nan:
            arr = arr[~np.isnan(arr)]
        n = arr.size
        if n == 0:
            return np.full(qs.shape, np.nan)

        s   = np.sort(arr, kind='stable')
        vi  = qs / 100.0 * (n - 1)          
        lo  = np.clip(np.floor(vi).astype(int), 0, n - 1)
        hi  = np.clip(np.ceil(vi).astype(int),  0, n - 1)
        fr  = vi - lo                         

        if   interpolation == 'linear'  : return s[lo] + fr * (s[hi] - s[lo])
        elif interpolation == 'lower'   : return s[lo].astype(float)
        elif interpolation == 'higher'  : return s[hi].astype(float)
        elif interpolation == 'midpoint': return (s[lo] + s[hi]) / 2.0
        else:
            return np.where(fr >= 0.5, s[hi], s[lo]).astype(float)

    if axis is None:
        result = _pct_1d(data.ravel(), q)
    else:
        result = np.apply_along_axis(_pct_1d, axis, data, q)

    if scalar_q:
        result = result.squeeze()
        if result.ndim == 0:
            return float(result)
    return result

print("rank.py loaded ✓")
