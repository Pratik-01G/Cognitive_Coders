import numpy as np

# --------------------------------------------------------------------------
# Statistical Functions
# --------------------------------------------------------------------------
def mean(x: np.ndarray, axis: int = None, ignore_nan: bool = False) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanmean(x, axis=axis)
  return np.mean(x, axis=axis)

def median(x: np.ndarray, axis: int = None, ignore_nan: bool = False) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanmedian(x, axis=axis)
  return np.median(x, axis=axis)

def variance(x: np.ndarray,axis: int = None,ddof: int = 0,ignore_nan: bool = False,) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanvar(x, axis=axis, ddof=ddof)
  return np.var(x, axis=axis, ddof=ddof)

def std(x: np.ndarray,axis: int = None,ddof: int = 0,ignore_nan: bool = False,) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanstd(x, axis=axis, ddof=ddof)
  return np.std(x, axis=axis, ddof=ddof)

def mode(x: np.ndarray) -> tuple:
  x = np.asarray(x).ravel()
  if x.size == 0:
    raise ValueError("Cannot compute mode of an empty array.")
  values, counts = np.unique(x, return_counts=True)
  idx = np.argmax(counts)
  return values[idx], counts[idx]

def skewness(x: np.ndarray, axis: int = None, bias: bool = True) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  mu = np.mean(x, axis=axis, keepdims=True)
  sigma = np.std(x, axis=axis, keepdims=True, ddof=0)
  if np.any(sigma == 0):
    raise ValueError("Standard deviation is 0 — skewness is undefined.")
  m3 = np.mean(((x - mu) / sigma) ** 3, axis=axis)
  if not bias:
    n = x.shape[axis] if axis is not None else x.size
    m3 = m3 * np.sqrt(n * (n - 1)) / (n - 2)
  return m3

def kurtosis(x: np.ndarray, axis: int = None, excess: bool = True) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  mu = np.mean(x, axis=axis, keepdims=True)
  sigma = np.std(x, axis=axis, keepdims=True, ddof=0)
  if np.any(sigma == 0):
    raise ValueError("Standard deviation is 0 — kurtosis is undefined.")
  m4 = np.mean(((x - mu) / sigma) ** 4, axis=axis)
  return m4 - 3.0 if excess else m4
