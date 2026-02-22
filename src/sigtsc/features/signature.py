from __future__ import annotations

import numpy as np
import iisignature


def znormalize(path_TxC: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-normalization of a (T, C) path."""
    mu = path_TxC.mean(axis=0, keepdims=True)
    sd = path_TxC.std(axis=0, keepdims=True)
    return (path_TxC - mu) / (sd + eps)


def add_time_channel(path_TxC: np.ndarray) -> np.ndarray:
    """Append normalized time in [0,1] as an extra channel."""
    T = path_TxC.shape[0]
    t = np.linspace(0.0, 1.0, T, dtype=path_TxC.dtype).reshape(T, 1)
    return np.concatenate([path_TxC, t], axis=1)


def logsig_features_global(
    paths: list[np.ndarray],
    level: int = 3,
    with_time: bool = False,
) -> np.ndarray:
    """
    Compute global log-signature features for a list of paths.
    Each path: (T, C). Returns array (N, F).
    """
    if not paths:
        raise ValueError("No paths provided.")

    # Prepare based on dimension of first path (after optional time channel)
    d0 = paths[0].shape[1] + (1 if with_time else 0)
    s = iisignature.prepare(d0, level)

    feats = []
    for path in paths:
        if path.ndim != 2:
            raise ValueError(f"Expected (T,C), got shape {path.shape}")
        p = znormalize(path.astype(np.float64, copy=False))
        if with_time:
            p = add_time_channel(p)
        # iisignature expects (T, d)
        feats.append(iisignature.logsig(p, s))

    return np.vstack(feats)