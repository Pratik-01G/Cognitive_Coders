# -------------------------------------------------------------------------------------------------
# Distance functions
# -------------------------------------------------------------------------------------------------
def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
  X = np.atleast_2d(np.asarray(X, dtype=float))
  Y = np.atleast_2d(np.asarray(Y, dtype=float))
  if X.shape[1] != Y.shape[1]:
    raise ValueError(f"Feature dimension mismatch: X has {X.shape[1]} features")
  XX = np.sum(X ** 2, axis=1, keepdims=True)   # (m, 1)
  YY = np.sum(Y ** 2, axis=1, keepdims=True)   # (n, 1)
  D_sq = XX + YY.T - 2.0 * (X @ Y.T)           # (m, n)
  return np.sqrt(np.maximum(D_sq, 0.0))
def manhattan_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
  X = np.atleast_2d(np.asarray(X, dtype=float))
  Y = np.atleast_2d(np.asarray(Y, dtype=float))
  if X.shape[1] != Y.shape[1]:
    raise ValueError(f"Feature dimension mismatch.")
  return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
  X = np.atleast_2d(np.asarray(X, dtype=float))
  Y = np.atleast_2d(np.asarray(Y, dtype=float))
  if X.shape[1] != Y.shape[1]:
    raise ValueError(f"Feature dimension mismatch.")
  X_norm = np.linalg.norm(X, axis=1, keepdims=True)   # (m, 1)
  Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)   # (n, 1)
  # Avoid division by zero for zero vectors
  X_safe = np.where(X_norm == 0, 1.0, X_norm)
  Y_safe = np.where(Y_norm == 0, 1.0, Y_norm)
  X_unit = X / X_safe
  Y_unit = Y / Y_safe
  S = X_unit @ Y_unit.T
  # Zero vectors: set their rows/cols to 0
  zero_rows = (X_norm == 0).ravel()
  zero_cols = (Y_norm == 0).ravel()
  S[zero_rows, :] = 0.0
  S[:, zero_cols] = 0.0
  return np.clip(S, -1.0, 1.0)
def minkowski_distance(X: np.ndarray, Y: np.ndarray, p: float = 2.0) -> np.ndarray:
  X = np.atleast_2d(np.asarray(X, dtype=float))
  Y = np.atleast_2d(np.asarray(Y, dtype=float))
  if X.shape[1] != Y.shape[1]:
    raise ValueError(f"Feature dimension mismatch.")
  if p < 1:
    raise ValueError(f"p must be >= 1, got {p}.")
  diff = np.abs(X[:, None, :] - Y[None, :, :])  # (m, n, d)
  if np.isinf(p):
    return np.max(diff, axis=2)
  return np.sum(diff ** p, axis=2) ** (1.0 / p)
