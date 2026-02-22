from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from aeon.datasets import load_classification


def _to_list_of_paths_TxC(X: Any) -> list[np.ndarray]:
    """
    Convert aeon classification data into a list of (T, C) arrays.
    aeon typically returns:
      - equal length: ndarray (N, C, T)
      - unequal length: list of arrays (C, T_i) (sometimes)
    """
    if isinstance(X, list):
        paths = []
        for xi in X:
            xi = np.asarray(xi)
            if xi.ndim != 2:
                raise ValueError(f"Expected 2D per case, got shape {xi.shape}")
            # Assume (C, T) if channels smaller than timepoints; else already (T, C)
            path_TxC = xi.T if xi.shape[0] <= xi.shape[1] else xi
            paths.append(path_TxC.astype(np.float64, copy=False))
        return paths

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N,C,T), got shape {X.shape}")

    # (N,C,T) -> list of (T,C)
    return [X[i].T.astype(np.float64, copy=False) for i in range(X.shape[0])]


def load_dataset(name: str) -> Tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")

    Xtr = _to_list_of_paths_TxC(X_train)
    Xte = _to_list_of_paths_TxC(X_test)

    ytr = np.asarray(y_train)
    yte = np.asarray(y_test)
    return Xtr, ytr, Xte, yte