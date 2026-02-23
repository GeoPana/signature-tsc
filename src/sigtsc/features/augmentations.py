from __future__ import annotations

import numpy as np


def lead_lag(path: np.ndarray) -> np.ndarray:
    """
    Lead–lag transform.

    Input:
      path: (T, C)

    Output:
      ll: (2T - 1, 2C)

    Construction:
      For each original point x[t], we add:
        - a "lag" copy (previous) and a "lead" copy (current) in 2C dims
      Interleaving produces a new path that encodes increments/order.

    This variant produces length (2T-1):
      y[2t]   = [x[t], x[t]]           (lag = x[t], lead = x[t])
      y[2t+1] = [x[t], x[t+1]]         (lag = x[t], lead = x[t+1])   for t=0..T-2
    """
    path = np.asarray(path)
    if path.ndim != 2:
        raise ValueError(f"lead_lag expects (T,C), got shape {path.shape}")

    T, C = path.shape
    if T < 2:
        # Nothing meaningful to do; return doubled channels with same single point
        return np.concatenate([path, path], axis=1)

    out = np.empty((2 * T - 1, 2 * C), dtype=path.dtype)

    # even indices
    out[0::2, :C] = path
    out[0::2, C:] = path

    # odd indices (except last which doesn't exist)
    out[1::2, :C] = path[:-1]
    out[1::2, C:] = path[1:]

    return out