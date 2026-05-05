from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import iisignature

from sigtsc.features.augmentations import lead_lag


Window = tuple[int, int]


def znormalize(path_TxC: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-normalization of a (T, C) path."""
    mu = path_TxC.mean(axis=0, keepdims=True)
    sd = path_TxC.std(axis=0, keepdims=True)
    return (path_TxC - mu) / (sd + eps)


def _preprocess_path(
    path: np.ndarray,
    with_time: bool,
    basepoint: bool,
    lead_lag_flag: bool,
) -> np.ndarray:
    # path is (T, C)
    x = path.astype(np.float64, copy=False)

    if with_time:
        T = x.shape[0]
        t = np.linspace(0.0, 1.0, T, dtype=x.dtype).reshape(T, 1)
        x = np.concatenate([t, x], axis=1)

    if basepoint:
        zero = np.zeros((1, x.shape[1]), dtype=x.dtype)
        x = np.vstack([zero, x])

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


def _deduplicate_windows(windows: Sequence[Window]) -> list[Window]:
    """Remove duplicate windows while preserving deterministic order."""
    out: list[Window] = []
    seen: set[Window] = set()
    for start, end in windows:
        start = int(start)
        end = int(end)
        if start < 0 or end <= start:
            continue
        window = (start, end)
        if window not in seen:
            seen.add(window)
            out.append(window)
    return out


def _sliding_windows(n: int, window: int, step: int) -> list[Window]:
    """Generate legacy sliding windows using exclusive end indices."""
    if window >= n or window < 2:
        return [(0, n)]
    step = max(1, step)
    return [(start, start + window) for start in range(0, n - window + 1, step)]


def _expanding_windows(
    n: int,
    num_windows: int,
    min_window: int,
    initial_frac: float | None = None,
) -> list[Window]:
    """
    Generate expanding windows `(0, end)` with the full path included last.

    If `n < min_window`, returns only `(0, n)`.
    """
    if num_windows < 1:
        raise ValueError(f"expanding num_windows must be >= 1, got {num_windows}")
    if n < min_window:
        return [(0, n)]

    start_frac = (1.0 / num_windows) if initial_frac is None else float(initial_frac)
    if not (0.0 < start_frac <= 1.0):
        raise ValueError(f"expanding initial_frac must be in (0, 1], got {initial_frac}")

    ends = [int(round(frac * n)) for frac in np.linspace(start_frac, 1.0, num_windows)]
    windows = [(0, min(n, end)) for end in ends if end >= min_window]
    windows.append((0, n))
    return _deduplicate_windows(windows)


def _dyadic_windows(n: int, depth: int, min_window: int) -> list[Window]:
    """
    Generate hierarchical dyadic windows, ordered coarse-to-fine and left-to-right.

    If `n < min_window`, returns only `(0, n)`.
    """
    if depth < 1:
        raise ValueError(f"dyadic depth must be >= 1, got {depth}")
    if n < min_window:
        return [(0, n)]

    windows: list[Window] = []
    for level in range(depth):
        n_segments = 2 ** level
        bounds = np.rint(np.linspace(0, n, n_segments + 1)).astype(int)
        for start, end in zip(bounds[:-1], bounds[1:]):
            if int(end) - int(start) >= min_window:
                windows.append((int(start), int(end)))
    return _deduplicate_windows(windows)


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
    Windowing configuration for signature/log-signature features.

    Backward-compatible parsing in run_experiment still supports older
    `features.window_fracs` configs, but the canonical config shape is the
    top-level `windowing` block.

    Supported `type` values:
    - global: one full-window feature
    - sliding: multiscale sliding windows from `window_fracs`
    - expanding: windows `(0, end)` with increasing end indices
    - dyadic: hierarchical dyadic windows, coarse-to-fine

    - window_fracs: fractions of T to use as window sizes, e.g. [0.125, 0.25, 1.0]
    - step_frac: stride as fraction of window size, e.g. 0.5 means 50% overlap
    - min_window: minimum window length in samples
    """
    window_fracs: Sequence[float] | None = None
    step_frac: float = 0.5
    min_window: int = 8
    type: str = "sliding"
    aggregation: str = "pool"
    num_windows: int = 4
    depth: int = 1
    initial_frac: float | None = None


def _normalize_window_type(window_type: str) -> str:
    t = str(window_type).strip().lower()
    aliases = {
        "": "global",
        "none": "global",
        "multiscale": "sliding",
        "sliding_multiscale": "sliding",
        "hierarchical_dyadic": "dyadic",
    }
    return aliases.get(t, t)


def _normalize_aggregation(aggregation: str) -> str:
    agg = str(aggregation).strip().lower()
    if agg not in {"concat", "pool"}:
        raise ValueError(f"Unknown window aggregation '{aggregation}'. Supported: concat, pool")
    return agg


def _normalize_transform_type(transform_type: str) -> str:
    t = str(transform_type).strip().lower()
    aliases = {
        "logsignature": "logsig",
        "log_signature": "logsig",
        "log-signature": "logsig",
        "sig": "signature",
    }
    t = aliases.get(t, t)
    if t not in {"logsig", "signature"}:
        raise ValueError(f"Unknown feature type '{transform_type}'. Supported: logsig, signature")
    return t


def make_windows(n: int, config: LogSigWindowConfig | None = None) -> list[Window]:
    """
    Build deterministic `(start, end)` windows for a path length `n`.

    End indices are exclusive. Returned windows are deduplicated and never
    empty. Sliding behavior matches the original multiscale implementation:
    windows are emitted scale-by-scale in `window_fracs` order.
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if config is None:
        return [(0, n)]

    window_type = _normalize_window_type(config.type)
    min_window = max(1, int(config.min_window))

    if window_type == "global":
        return [(0, n)]

    if window_type == "sliding":
        if config.window_fracs is None or len(config.window_fracs) == 0:
            return [(0, n)]

        windows: list[Window] = []
        for frac in config.window_fracs:
            if frac <= 0:
                raise ValueError(f"window_frac must be > 0, got {frac}")

            w = int(round(float(frac) * n))
            w = max(min_window, w)
            w = min(w, n)

            step = int(round(float(config.step_frac) * w))
            step = max(1, step)

            windows.extend(_sliding_windows(n, w, step))
        return _deduplicate_windows(windows)

    if window_type == "expanding":
        return _expanding_windows(
            n=n,
            num_windows=int(config.num_windows),
            min_window=min_window,
            initial_frac=config.initial_frac,
        )

    if window_type == "dyadic":
        return _dyadic_windows(n=n, depth=int(config.depth), min_window=min_window)

    raise ValueError(
        f"Unknown windowing type '{config.type}'. "
        "Supported: global, sliding, expanding, dyadic"
    )


def _concat_window_features(W: np.ndarray) -> np.ndarray:
    return W.reshape(-1)


def _aggregate_window_features(W: np.ndarray, aggregation: str, pool_ops: List[str]) -> np.ndarray:
    aggregation = _normalize_aggregation(aggregation)
    if aggregation == "concat":
        return _concat_window_features(W)
    return _pool_windows(W, pool_ops)


def path_signature_features(
    paths: list[np.ndarray],
    level: int = 3,
    transform_type: str = "logsig",
    with_time: bool = False,
    basepoint: bool = False,
    lead_lag: bool = False,
    windowing: LogSigWindowConfig | None = None,
    pool: Sequence[str] = ("mean", "max"),
) -> np.ndarray:
    """
    Compute signature or log-signature features for each path.

    - Global/no windowing computes one transform over the full path.
    - Sliding windowing keeps per-scale pooling by default.
    - Expanding/dyadic windowing supports concatenation or pooling.

    Inputs:
      paths: list of (T, C) arrays (T may vary across samples)
      level: transform truncation level
      transform_type: "logsig" or "signature"
      with_time: append time channel (warp-sensitive)
      basepoint: prepend a zero vector after time augmentation and before lead-lag
      windowing: window config
      pool: pooling operations across windows (mean/max/std)

    Output:
      X: (N, F) feature matrix
    """
    if not paths:
        raise ValueError("No paths provided.")

    transform_type = _normalize_transform_type(transform_type)
    pool_ops = _validate_pool(pool)

    # Determine final dimension AFTER with_time, basepoint, and lead_lag.
    # Basepoint changes path length, not channel dimension.
    base_d = paths[0].shape[1] + (1 if with_time else 0)
    final_d = base_d * (2 if lead_lag else 1)

    prep = iisignature.prepare(int(final_d), int(level)) if transform_type == "logsig" else None

    def transform_one(seg: np.ndarray) -> np.ndarray:
        # iisignature expects (T, d) float array
        if transform_type == "logsig":
            return iisignature.logsig(seg, prep)
        return iisignature.sig(seg, int(level))

    window_type = "global" if windowing is None else _normalize_window_type(windowing.type)
    aggregation = "pool" if windowing is None else _normalize_aggregation(windowing.aggregation)
    sliding_has_scales = (
        windowing is not None
        and windowing.window_fracs is not None
        and len(windowing.window_fracs) > 0
    )
    use_windowing = (
        windowing is not None
        and window_type != "global"
        and (window_type != "sliding" or sliding_has_scales)
    )
    use_legacy_sliding_pool = use_windowing and window_type == "sliding" and aggregation == "pool"

    feats_all: list[np.ndarray] = []
    expected_dim: int | None = None
    for path in paths:
        if path.ndim != 2:
            raise ValueError(f"Expected (T,C), got shape {path.shape}")

        # Start from float
        p = path.astype(np.float64, copy=False)

        # Normalize ORIGINAL channels (before time and lead-lag)
        p = znormalize(p)

        # Preprocess the normalized path
        p = _preprocess_path(
            p,
            with_time=with_time,
            basepoint=basepoint,
            lead_lag_flag=lead_lag,
        )

        # Sanity check (helps catch dimension mismatches early)
        if p.shape[1] != final_d:
            raise RuntimeError(
                f"Preprocess produced dim={p.shape[1]} but expected dim={final_d}. "
                f"(with_time={with_time}, basepoint={basepoint}, lead_lag={lead_lag})"
            )

        if not use_windowing:
            feat = transform_one(p)
            feats_all.append(feat)
            expected_dim = len(feat) if expected_dim is None else expected_dim
            continue

        if use_legacy_sliding_pool:
            T = p.shape[0]
            per_scale_feats: list[np.ndarray] = []

            for frac in windowing.window_fracs or []:
                if frac <= 0:
                    raise ValueError(f"window_frac must be > 0, got {frac}")

                w = int(round(float(frac) * T))
                w = max(int(windowing.min_window), w)
                w = min(w, T)  # cap at T

                step = int(round(float(windowing.step_frac) * w))
                step = max(1, step)

                win_feats = [
                    transform_one(p[start:end])
                    for start, end in _sliding_windows(T, w, step)
                ]
                W = np.vstack(win_feats)  # (n_windows, F)
                per_scale_feats.append(_pool_windows(W, pool_ops))

            feat = np.concatenate(per_scale_feats, axis=0)
        else:
            windows = make_windows(p.shape[0], windowing)
            win_feats = [transform_one(p[start:end]) for start, end in windows]
            W = np.vstack(win_feats)
            feat = _aggregate_window_features(W, aggregation, pool_ops)

        if expected_dim is None:
            expected_dim = int(feat.shape[0])
        elif int(feat.shape[0]) != expected_dim:
            raise ValueError(
                "Windowed feature dimension changed across samples. "
                f"Expected {expected_dim}, got {feat.shape[0]}. "
                "For concat aggregation, ensure all paths produce the same number "
                "of windows or use aggregation: pool."
            )

        feats_all.append(feat)

    return np.vstack(feats_all)


def logsig_features(
    paths: list[np.ndarray],
    level: int = 3,
    with_time: bool = False,
    basepoint: bool = False,
    lead_lag: bool = False,
    windowing: LogSigWindowConfig | None = None,
    pool: Sequence[str] = ("mean", "max"),
) -> np.ndarray:
    """Backward-compatible wrapper for log-signature features."""
    return path_signature_features(
        paths=paths,
        level=level,
        transform_type="logsig",
        with_time=with_time,
        basepoint=basepoint,
        lead_lag=lead_lag,
        windowing=windowing,
        pool=pool,
    )
