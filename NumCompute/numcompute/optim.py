"""
optim.py — Numerical gradient and Jacobian via finite differences.
Uses forward or central differences to approximate derivatives.

"""

import numpy as np
from numpy.typing import NDArray

def _validate_x(x: NDArray) -> NDArray:
    """Validate and convert x to a finite 1-D float array.
    Parameters
    ----------
    x : array-like
        Input point.
    Returns
    -------
    ndarray, shape (d,)
        Validated 1-D float64 array.
    Raises
    ------
    TypeError
        If x cannot be converted to numeric values.
    ValueError
        If x is empty, non-1D, or contains NaN/Inf.
    Notes
    -----
    Time complexity: O(d).
    Space complexity: O(d).
    """
    try:
         x = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"x must be numeric, got {type(x).__name__}.") from exc

    if x.ndim == 0:
        x = x.reshape(1)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}.")
    if x.size == 0:
        raise ValueError("x must not be empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or Inf")
    return x

  
def _validate_h(h: float) -> float:
    """Validate and convert x to a finite 1-D float array.
    Parameters
    ----------
    x : array-like
        Input point.
    Returns
    -------
    ndarray, shape (d,)
        Validated 1-D float64 array.
    Raises
    ------
    TypeError
        If x cannot be converted to numeric values.
    ValueError
        If x is empty, non-1D, or contains NaN/Inf.
    Notes
    -----
    Time complexity: O(d).
    Space complexity: O(d).
    """
    if not isinstance(h, (int, float, np.integer, np.floating)):
        raise TypeError(f"h must be a number, got {type(h).__name__}.")
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"h must be positive and finite, got {h}.")
    return h


def _validate_method(method: str) -> str:
    """Validate finite difference method name.
    Parameters
    ----------
    method : str
        Either "central" or "forward".
    Returns
    -------
    str
        Normalised method string.
    Raises
    ------
    TypeError
        If method is not a string.
    ValueError
        If method is not "central" or "forward".
    Notes
    -----
    Time complexity: O(1).
    Space complexity: O(1).
    """
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    method = method.strip().lower()
    if method not in ("central", "forward"):
        raise ValueError(f"method must be 'central' or 'forward', got '{method}'.")
    return method

def _validate_scalar_output(y, name="f"):
    """Validate scalar output from a scalar-valued function.
    Parameters
    ----------
    y : object
        Function output to validate.
    name : str
        Function name used in error messages.
    Returns
    -------
    float
        Scalar output as a float.
    Raises
    ------
    ValueError
        If output is not scalar or is NaN/Inf.
    Notes
    -----
    Time complexity: O(1).
    Space complexity: O(1).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 0:
        raise ValueError(f"{name} must return a scalar value.")
    if not np.isfinite(y):
        raise ValueError(f"{name} returned NaN or Inf.")
    return float(y)

# Core API 
def grad(
    f,
    x: NDArray,
    h: float = 1e-5,
    method: str = "central",
) -> NDArray:
    """Estimate gradient of a scalar function using finite differences.
    Parameters
    ----------
    f : callable
        Scalar function with signature f(x) -> float.
    x : array-like, shape (d,)
        Point where the gradient is evaluated.
    h : float
        Positive finite difference step size.
    method : {"central", "forward"}
        Finite difference method. Central is usually more accurate;
        forward uses fewer function evaluations.
    Returns
    -------
    ndarray, shape (d,)
        Estimated gradient vector.
    Raises
    ------
    TypeError
        If f is not callable or x/h have invalid types.
    ValueError
        If x, h, method, or f output is invalid.
    Notes
    -----
    Central difference uses 2d function evaluations.
    Forward difference uses d + 1 function evaluations.
    Time complexity: O(d * cost(f)).
    Space complexity: O(d^2) because perturbation points are built as a matrix.
    """
    if not callable(f):
        raise TypeError("f must be callable.")  

    x = _validate_x(x)
    h = _validate_h(h)
    method = _validate_method(method)

    d = x.shape[0]

    perturbations = np.eye(d, dtype=np.float64) * h

    x_plus = x + perturbations   # (d, d) — each row is x shifted in one dim
    x_minus = x - perturbations

    if method == "central":
        f_plus = np.array([_validate_scalar_output(f(x_plus[i])) for i in range(d)], dtype=np.float64)
        f_minus = np.array([_validate_scalar_output(f(x_minus[i])) for i in range(d)], dtype=np.float64)
        return (f_plus - f_minus) / (2.0 * h)
    else:
        f0 = _validate_scalar_output(f(x))
        f_plus = np.array([_validate_scalar_output(f(x_plus[i])) for i in range(d)], dtype=np.float64)
        return (f_plus - f0) / h


def jacobian(
    F,
    x: NDArray,
    h: float = 1e-5,
    method: str = "central",
) -> NDArray:
    """Estimate Jacobian of a vector function using finite differences.
    Parameters
    ----------
    F : callable
        Vector function with signature F(x) -> array-like of shape (m,).
        Scalar outputs are treated as shape (1,).
    x : array-like, shape (d,)
        Point where the Jacobian is evaluated.
    h : float
        Positive finite difference step size.
    method : {"central", "forward"}
        Finite difference method.
    Returns
    -------
    ndarray, shape (m, d)
        Estimated Jacobian matrix where rows correspond to output
        dimensions and columns correspond to input dimensions.
    Raises
    ------
    TypeError
        If F is not callable or x/h have invalid types.
    ValueError
        If x, h, method, or F outputs are invalid or inconsistent.
    Notes
    -----
    Central difference uses 2d function evaluations.
    Forward difference uses d + 1 function evaluations.
    Time complexity: O(d * cost(F)).
    Space complexity: O(d*m + d^2), due to perturbation and output arrays.
    """
    if not callable(F):
        raise TypeError("F must be callable.")

    x = _validate_x(x)
    h = _validate_h(h)
    method = _validate_method(method)

    d = x.shape[0]
    perturbations = np.eye(d, dtype=np.float64) * h

    x_plus = x + perturbations
    x_minus = x - perturbations

    def _stack_and_check_outputs(points, label):
        results = []

        for j, out in enumerate(points):
            val = np.asarray(out, dtype=np.float64)
            if val.ndim == 0:
                val = val.reshape(1)
            if val.ndim != 1:
                raise ValueError(
                    f"F must return a scalar or 1-D array, got shape {val.shape} "
                    f"at perturbation dim {j} ({label})."
                )
            if not np.all(np.isfinite(val)):
                raise ValueError(f"F returned NaN or Inf at perturbation dim {j} ({label}).")
            results.append(val)

        m = results[0].shape[0]
        for j, r in enumerate(results):
            if r.shape != (m,):
                raise ValueError(
                    f"F output shape inconsistent, expected ({m},), "
                    f"got {r.shape} at perturbation dim {j} ({label})."
                )
        return np.stack(results, axis=0), m  # (d, m)

    if method == "central":
        F_plus, m = _stack_and_check_outputs(
            [F(x_plus[j]) for j in range(d)], "x+h"
        )
        F_minus, _ = _stack_and_check_outputs(
            [F(x_minus[j]) for j in range(d)], "x-h"
        )
        jac = (F_plus - F_minus) / (2.0 * h)  # (d, m)
        return jac.T  # (m, d)
    else:
        F0, _ = _stack_and_check_outputs([F(x)], "x_base")
        F0 = F0[0]
        F_plus, _ = _stack_and_check_outputs(
            [F(x_plus[j]) for j in range(d)], "x+h"
        )
        jac = (F_plus - F0) / h 
        return jac.T
