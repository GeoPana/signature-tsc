from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score

# aeon MiniRocketClassifier (scikit-learn style API)
from aeon.classification.convolution_based import MiniRocketClassifier  # :contentReference[oaicite:1]{index=1}


def _paths_TxC_to_aeon_X(paths: list[np.ndarray]) -> Any:
    """
    Convert list of (T, C) arrays into aeon expected input.
    - If all T are equal: return ndarray of shape (N, C, T)
    - If variable length: return list of arrays of shape (C, T_i)
    """
    if not paths:
        raise ValueError("No paths provided.")

    Ts = [p.shape[0] for p in paths]
    Cs = [p.shape[1] for p in paths]
    if len(set(Cs)) != 1:
        raise ValueError(f"All paths must have same number of channels. Got: {set(Cs)}")

    C = Cs[0]
    equal_len = len(set(Ts)) == 1

    if equal_len:
        T = Ts[0]
        X = np.empty((len(paths), C, T), dtype=np.float64)
        for i, p in enumerate(paths):
            if p.ndim != 2:
                raise ValueError(f"Expected (T,C), got shape {p.shape}")
            X[i] = p.T  # (C, T)
        return X

    # Variable length: list of (C, T_i)
    X_list = []
    for p in paths:
        if p.ndim != 2:
            raise ValueError(f"Expected (T,C), got shape {p.shape}")
        X_list.append(p.T.astype(np.float64, copy=False))
    return X_list


@dataclass(frozen=True)
class MiniRocketResult:
    accuracy: float
    details: Dict[str, Any]


def train_eval_minirocket(
    Xtr_paths: list[np.ndarray],
    ytr: np.ndarray,
    Xte_paths: list[np.ndarray],
    yte: np.ndarray,
    params: Dict[str, Any],
) -> MiniRocketResult:
    """
    Train and evaluate MiniRocketClassifier on raw time series.
    Params (optional):
      - n_kernels (int, default 10000)
      - max_dilations_per_kernel (int, default 32)
      - n_jobs (int, default -1)
      - random_state (int or None)
    """
    Xtr = _paths_TxC_to_aeon_X(Xtr_paths)
    Xte = _paths_TxC_to_aeon_X(Xte_paths)

    clf = MiniRocketClassifier(
        n_kernels=int(params.get("n_kernels", 10_000)),
        max_dilations_per_kernel=int(params.get("max_dilations_per_kernel", 32)),
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=params.get("random_state", None),
    )

    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = float(accuracy_score(yte, pred))
    return MiniRocketResult(
        accuracy=acc,
        details={
            "accuracy": acc,
            "n_kernels": int(params.get("n_kernels", 10_000)),
            "max_dilations_per_kernel": int(params.get("max_dilations_per_kernel", 32)),
            "n_jobs": int(params.get("n_jobs", -1)),
            "random_state": params.get("random_state", None),
        },
    )