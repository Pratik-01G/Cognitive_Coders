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

# --------------------------------------------------------------------------
# Quantiles and Percentile
# --------------------------------------------------------------------------
def quantile(x: np.ndarray,q: float | list,axis: int = None,ignore_nan: bool = False) -> np.ndarray:
  q_arr = np.asarray(q)
  if np.any(q_arr < 0) or np.any(q_arr > 1):
    raise ValueError(f"All quantile values must be in [0, 1]; got {q}.")
  x = np.asarray(x, dtype=float)
  if ignore_nan:
    return np.nanquantile(x, q, axis=axis, method="linear")
  return np.quantile(x, q, axis=axis, method="linear")

def percentile(x: np.ndarray,p: float | list,axis: int = None,ignore_nan: bool = False) -> np.ndarray:
  p_arr = np.asarray(p)
  if np.any(p_arr < 0) or np.any(p_arr > 100):
    raise ValueError(f"All percentile values must be in [0, 100]; got {p}.")
  return quantile(x, p_arr / 100.0, axis=axis, ignore_nan=ignore_nan)

def iqr(x: np.ndarray, axis: int = None, ignore_nan: bool = False) -> np.ndarray:
  q1 = quantile(x, 0.25, axis=axis, ignore_nan=ignore_nan)
  q3 = quantile(x, 0.75, axis=axis, ignore_nan=ignore_nan)
  return q3 - q1

# ----------------------------------------------------
# Histogram
# ----------------------------------------------------
def histogram(x: np.ndarray,bins: int | np.ndarray = 10,range: tuple = None,density: bool = False) -> tuple:
  x = np.asarray(x, dtype=float).ravel()
  if x.size == 0:
    raise ValueError("Cannot compute histogram of an empty array.")
  if isinstance(bins, int):
    if bins < 1:
      raise ValueError(f"Bins must be >= 1, got {bins}.")
    lo = x.min() if range is None else range[0]
    hi = x.max() if range is None else range[1]
    if lo == hi:
      lo, hi = lo - 0.5, hi + 0.5
    bin_edges = np.linspace(lo, hi, bins + 1)
  else:
    bin_edges = np.asarray(bins, dtype=float)
    if bin_edges.ndim != 1 or len(bin_edges) < 2:
      raise ValueError("Bins array must be 1-D with at least 2 elements.")
  indices = np.searchsorted(bin_edges, x, side="right") - 1
  indices = np.clip(indices, 0, len(bin_edges) - 2)
  counts = np.bincount(indices, minlength=len(bin_edges) - 1).astype(float)
  if density:
    widths = np.diff(bin_edges)
    counts = counts/(counts.sum()*widths)
  return counts, bin_edges

# ----------------------------------------------------
# Covariance and Correlation
# ----------------------------------------------------
def covariance_matrix(X: np.ndarray, ddof: int = 1) -> np.ndarray:
  X = np.asarray(X, dtype=float)
  if X.ndim == 1:
    X = X[:, None]
  n, d = X.shape
  if n <= ddof:
    raise ValueError(f"Need at least {ddof + 1} observations for ddof={ddof}.")
  X_centred = X - X.mean(axis=0)
  return (X_centred.T @ X_centred) / (n - ddof)

def correlation_matrix(X: np.ndarray) -> np.ndarray:
  cov = covariance_matrix(X, ddof=1)
  d = np.sqrt(np.diag(cov))
  denom = np.outer(d, d)
  with np.errstate(invalid="ignore"):
    corr = np.where(denom == 0, np.nan, cov / denom)
  return corr
