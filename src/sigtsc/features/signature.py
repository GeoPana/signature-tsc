from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import iisignature

from sigtsc.features.augmentations import lead_lag


def znormalize(path_TxC: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-normalization of a (T, C) path."""
    mu = path_TxC.mean(axis=0, keepdims=True)
    sd = path_TxC.std(axis=0, keepdims=True)
    return (path_TxC - mu) / (sd + eps)

def _preprocess_path(path: np.ndarray, with_time: bool, lead_lag_flag: bool) -> np.ndarray:
    # path is (T, C)
    x = path.astype(np.float64, copy=False)

    if with_time:
        T = x.shape[0]
        t = np.linspace(0.0, 1.0, T, dtype=x.dtype).reshape(T, 1)
        x = np.concatenate([t, x], axis=1)

    if lead_lag_flag:
        x = lead_lag(x)

    return x

def _validate_pool(pool: Sequence[str]) -> List[str]:
    allowed = {"mean", "max", "std"}
    pool = [p.strip().lower() for p in pool]
    unknown = [p for p in pool if p not in allowed]
    if unknown:
        raise ValueError(f"Unknown pool ops: {unknown}. Allowed: {sorted(allowed)}")
    if len(pool) == 0:
        raise ValueError("pool must contain at least one op (e.g. ['mean']).")
    return pool


def _windows(path: np.ndarray, window: int, step: int) -> Iterable[np.ndarray]:
    """
    Generate sliding windows of shape (window, C) from (T, C).
    Always yields at least one window (falls back to full path if window > T).
    """
    T = path.shape[0]
    if window >= T:
        yield path
        return
    if window < 2:
        yield path
        return
    step = max(1, step)
    for start in range(0, T - window + 1, step):
        yield path[start : start + window]


def _pool_windows(W: np.ndarray, pool: List[str]) -> np.ndarray:
    """
    Pool window-level features W of shape (n_windows, F) to a single vector.
    Concatenates requested pooling ops in given order.
    """
    feats = []
    if "mean" in pool:
        feats.append(W.mean(axis=0))
    if "max" in pool:
        feats.append(W.max(axis=0))
    if "std" in pool:
        feats.append(W.std(axis=0))
    return np.concatenate(feats, axis=0)


@dataclass(frozen=True)
class LogSigWindowConfig:
    """
    Multiscale windowing configuration.
    - window_fracs: fractions of T to use as window sizes, e.g. [0.125, 0.25, 1.0]
    - step_frac: stride as fraction of window size, e.g. 0.5 means 50% overlap
    - min_window: minimum window length in samples
    """
    window_fracs: Sequence[float]
    step_frac: float = 0.5
    min_window: int = 8


def logsig_features(
    paths: list[np.ndarray],
    level: int = 3,
    with_time: bool = False,
    lead_lag: bool = False,
    windowing: LogSigWindowConfig | None = None,
    pool: Sequence[str] = ("mean", "max"),
) -> np.ndarray:
    """
    Compute log-signature features for each path.
    - If windowing is None or window_fracs is empty => global logsig only.
    - Else => multiscale windowed logsig, pooled per scale and concatenated.

    Inputs:
      paths: list of (T, C) arrays (T may vary across samples)
      level: logsig truncation level
      with_time: append time channel (warp-sensitive)
      windowing: multiscale window config
      pool: pooling operations across windows per scale (mean/max/std)

    Output:
      X: (N, F) feature matrix
    """
    if not paths:
        raise ValueError("No paths provided.")

    pool_ops = _validate_pool(pool)

    # Determine final dimension AFTER with_time and lead_lag
    base_d = paths[0].shape[1] + (1 if with_time else 0)
    final_d = base_d * (2 if lead_lag else 1)

    s = iisignature.prepare(int(final_d), int(level))

    def logsig_one(seg: np.ndarray) -> np.ndarray:
        # iisignature expects (T, d) float array
        return iisignature.logsig(seg, s)

    use_windowing = windowing is not None and windowing.window_fracs is not None and len(windowing.window_fracs) > 0

    feats_all = []
    for path in paths:
        if path.ndim != 2:
            raise ValueError(f"Expected (T,C), got shape {path.shape}")

        # Start from float
        p = path.astype(np.float64, copy=False)

        # Normalize ORIGINAL channels (before time and lead-lag)
        p = znormalize(p)

        # Preprocess the normalized path 
        p = _preprocess_path(p, with_time=with_time, lead_lag_flag=lead_lag)

        # Sanity check (helps catch dimension mismatches early)
        if p.shape[1] != final_d:
            raise RuntimeError(
                f"Preprocess produced dim={p.shape[1]} but expected dim={final_d}. "
                f"(with_time={with_time}, lead_lag={lead_lag})"
            )

        if not use_windowing:
            feats_all.append(logsig_one(p))
            continue

        T = p.shape[0]
        per_scale_feats = []

        for frac in windowing.window_fracs:
            if frac <= 0:
                raise ValueError(f"window_frac must be > 0, got {frac}")

            w = int(round(frac * T))
            w = max(int(windowing.min_window), w)
            w = min(w, T)  # cap at T

            step = int(round(windowing.step_frac * w))
            step = max(1, step)

            # Compute logsig for each window
            win_feats = [logsig_one(seg) for seg in _windows(p, w, step)]
            W = np.vstack(win_feats)  # (n_windows, F)

            # Pool within this scale
            pooled = _pool_windows(W, pool_ops)
            per_scale_feats.append(pooled)

        feats_all.append(np.concatenate(per_scale_feats, axis=0))

    return np.vstack(feats_all)