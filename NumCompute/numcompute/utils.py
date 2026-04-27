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
# -------------------------------------------------------------------------------------------------
# Activation Functions
# -------------------------------------------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
  return np.maximum(0.0, np.asarray(x, dtype=float))
def relu_derivative(x: np.ndarray) -> np.ndarray:
  return (np.asarray(x, dtype=float) > 0).astype(float)
def sigmoid(x: np.ndarray) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  pos_mask = x >= 0
  result = np.empty_like(x)
  # For positive values: 1 / (1 + exp(-x))
  result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
  # For negative values: exp(x) / (1 + exp(x))
  exp_x = np.exp(x[~pos_mask])
  result[~pos_mask] = exp_x / (1.0 + exp_x)
  return result
def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
  s = sigmoid(x)
  return s * (1.0 - s)
def tanh_activation(x: np.ndarray) -> np.ndarray:
  return np.tanh(np.asarray(x, dtype=float))
def tanh_derivative(x: np.ndarray) -> np.ndarray:
  return 1.0 - np.tanh(np.asarray(x, dtype=float)) ** 2
def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
  if alpha < 0:
    raise ValueError(f"alpha must be non-negative, got {alpha}.")
  x = np.asarray(x, dtype=float)
  return np.where(x >= 0, x, alpha * x)
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  x_shifted = x - np.max(x, axis=axis, keepdims=True)
  exp_x = np.exp(x_shifted)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
