import numpy as np
from typing import Tuple


# =========================
# Sorting Utilities


def argsort(X: np.ndarray, axis: int = -1, stable: bool = True) -> np.ndarray:
    """
    Return indices that would sort an array.

    Parameters
    ----------
    X : np.ndarray
    axis : int
    stable : bool

    Returns
    -------
    np.ndarray

    Notes
    -----
    Uses NumPy's stable sort if requested.

    Complexity
    ----------
    O(N log N)
    """
    kind = "stable" if stable else "quicksort"
    return np.argsort(X, axis=axis, kind=kind)


def sort(X: np.ndarray, axis: int = -1, stable: bool = True) -> np.ndarray:
    """
    Sort array along axis.

    Returns
    -------
    np.ndarray
    """
    idx = argsort(X, axis=axis, stable=stable)
    return np.take_along_axis(X, idx, axis=axis)


# =========================
# Top-K / Argpartition
# =========================

def topk(X: np.ndarray, k: int, axis: int = -1, largest: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top-k elements and their indices.

    Parameters
    ----------
    X : np.ndarray
    k : int
    axis : int
    largest : bool

    Returns
    -------
    values : np.ndarray
    indices : np.ndarray

    Notes
    -----
    Uses argpartition (O(N)) then sorts top-k (O(k log k)).

    Complexity
    ----------
    O(N + k log k)
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    if k > X.shape[axis]:
        raise ValueError("k cannot exceed dimension size")

    if largest:
        idx = np.argpartition(X, -k, axis=axis)
        top_idx = np.take(idx, indices=range(X.shape[axis] - k, X.shape[axis]), axis=axis)
    else:
        idx = np.argpartition(X, k, axis=axis)
        top_idx = np.take(idx, indices=range(k), axis=axis)

    values = np.take_along_axis(X, top_idx, axis=axis)

    # sort within top-k
    order = np.argsort(values, axis=axis)
    if largest:
        order = np.flip(order, axis=axis)

    top_idx = np.take_along_axis(top_idx, order, axis=axis)
    values = np.take_along_axis(values, order, axis=axis)

    return values, top_idx


# =========================
# Quickselect (k-th smallest)
# =========================

def quickselect(X: np.ndarray, k: int) -> float:
    """
    Return k-th smallest element (0-based index).

    Parameters
    ----------
    X : np.ndarray (1D)
    k : int

    Returns
    -------
    float

    Notes
    -----
    In-place partition-based selection.

    Complexity
    ----------
    Average: O(N)
    Worst: O(N^2)
    """
    if X.ndim != 1:
        raise ValueError("Quickselect requires 1D array")

    if k < 0 or k >= len(X):
        raise ValueError("k out of bounds")

    arr = X.copy()

    def partition(left, right, pivot_index):
        pivot = arr[pivot_index]
        arr[pivot_index], arr[right] = arr[right], arr[pivot_index]

        store = left
        for i in range(left, right):
            if arr[i] < pivot:
                arr[store], arr[i] = arr[i], arr[store]
                store += 1

        arr[right], arr[store] = arr[store], arr[right]
        return store

    left, right = 0, len(arr) - 1

    while True:
        pivot_index = (left + right) // 2
        pivot_index = partition(left, right, pivot_index)

        if k == pivot_index:
            return arr[k]
        elif k < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


# =========================
# Binary Search
# =========================

def binary_search(X: np.ndarray, target, left: bool = True) -> int:
    """
    Binary search on sorted array.

    Parameters
    ----------
    X : np.ndarray (sorted 1D)
    target : scalar
    left : bool
        If True → first occurrence (lower bound)
        If False → last occurrence (upper bound)

    Returns
    -------
    int
        Index of target or insertion point.

    Notes
    -----
    Equivalent to np.searchsorted.

    Complexity
    ----------
    O(log N)
    """
    if X.ndim != 1:
        raise ValueError("binary_search requires 1D sorted array")

    side = "left" if left else "right"
    return np.searchsorted(X, target, side=side)


# =========================
# Rank / Percentile Helper
# =========================

def percentile(X: np.ndarray, q: float) -> float:
    """
    Compute percentile using quickselect.

    Parameters
    ----------
    X : np.ndarray
    q : float in [0, 100]

    Returns
    -------
    float

    Notes
    -----
    Avoids full sort.

    Complexity
    ----------
    O(N)
    """
    if not (0 <= q <= 100):
        raise ValueError("q must be in [0, 100]")

    X = X.ravel()
    k = int((q / 100.0) * (len(X) - 1))

    return quickselect(X, k)