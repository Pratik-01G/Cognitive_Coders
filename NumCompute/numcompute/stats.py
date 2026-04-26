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

# ----------------------------------------------------
# Welford's Online/Streaming algorithm
# ----------------------------------------------------
class WelfordStatistics:
  def __init__(self, shape: tuple = None):
    self._n = 0
    self._mean = None
    self._M2 = None
    self._shape = shape

  def _init_arrays(self, x: np.ndarray):
    # On 1st update, initialize the internal array
    self._mean = np.zeros_like(x, dtype=float)
    self._M2 = np.zeros_like(x, dtype=float)

  def update(self, x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if self._mean is None:
      self._init_arrays(x)
    self._n += 1
    delta = x - self._mean
    self._mean += delta / self._n
    delta2 = x - self._mean
    self._M2 += delta * delta2
    return self

  def update_batch(self, X: np.ndarray):
    X = np.asarray(X, dtype=float)
    for row in X:
      self.update(row)
    return self

  def mean(self)->np.ndarray:
    if self._n==0:
      return None
    return self._mean.copy()

  def variance(self,ddof:int=0)->np.ndarray:
    if self._n==0:
      return None
    if self._n<=ddof:
      raise ValueError(f"Require atleast {ddof+1}observations for ddof={ddof}")
    return self._M2/(self._n-ddof)

  def std(self,ddof:int=0)->np.ndarray:
    v=self.variance(ddof=ddof)
    return None if v is None else np.sqrt(v)

  @property
  def n(self)->int: #Number of observations so far
    return self._n

  def reset(self): #Reset all accumulaor to initial state
    self._n=0
    self._mean=None
    self._M2=None
    return self

  def merge(self,other:"WelfordStatistics")->"WelfordStatistics":
    if other._n==0:
      return self
    if self._n==0:
      self._n=other._n
      self._mean=other.mean().copy()
      self._M2=other._M2.copy()
      return self
    if self._mean.shape!=other._mean.shape:
      raise ValueError(f"Shape Mismatched")
    num_a,num_b=self._n,other._n
    num_combined=num_a+num_b
    delta=other._mean-self._mean
    new_mean=self._mean+delta*(num_b/num_combined)
    new_M2=self._M2+other._M2+(delta**2)*(num_a*num_b/num_combined)
    self._n=num_combined
    self._mean=new_mean
    self._M2=new_M2
    return self

  def __repr__(self)->str:
    return f"Welfordstatistics(n={self._n},mean={self._mean},std={self.std()})"

# ----------------------------------------------------
# Summary Utility
# ----------------------------------------------------
def describe(x:np.ndarray, ddof:int=1)->dict:
  x=np.asarray(x,dtype=float).ravel()
  if x.size==0:
    raise ValueError(f"Empty array cannot be described")
  nan_mask=~np.isnan(x)
  x_clean=x[nan_mask]
  if x_clean.size==0:
    raise ValueError(f"All Values are NaN")
  q1,med,q3=np.quantile(x_clean,[0.25,0.50,0.75],method="linear")
  try:
    skew=float(skewness(x_clean))
  except ValueError:
    skew=float("nan")
  try:
    kurt=float(kurtosis(x_clean))
  except ValueError:
    kurt=float("nan")

  return{
      "n": int(x_clean.size),
      "nan_count": int(x.size-x_clean.size),
      "mean": float(np.mean(x_clean)),
      "std": float(np.std(x_clean,ddof=ddof)),
      "variance": float(np.var(x_clean,ddof=ddof)),
      "min": float(np.min(x_clean)),
      "max": float(np.max(x_clean)),
      "q25": float(q1),
      "median": float(med),
      "q75": float(q3),
      "iqr": float(q3-q1),
      "skew": float(skew),
      "kurtosis": float(kurt)
  }
