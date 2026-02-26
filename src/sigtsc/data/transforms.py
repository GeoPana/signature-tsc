from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class TransformSpec:
    warp: float | None = None        # e.g. 0.20
    shift: float | None = None       # fraction of length, e.g. 0.20
    noise: float | None = None       # noise strength, e.g. 0.05


def parse_dataset_spec(name: str) -> Tuple[str, TransformSpec]:
    """
    Parse "BaseDataset@warp=0.2,shift=0.1,noise=0.05" -> ("BaseDataset", TransformSpec(...))

    If no '@' is present, returns (name, TransformSpec()).
    """
    if "@" not in name:
        return name, TransformSpec()

    base, tail = name.split("@", 1)
    parts = [p.strip() for p in tail.split(",") if p.strip()]

    kv: Dict[str, float] = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad dataset transform token '{p}'. Use key=value, e.g. warp=0.2")
        k, v = p.split("=", 1)
        k = k.strip().lower()
        v = float(v.strip())
        kv[k] = v

    return base.strip(), TransformSpec(
        warp=kv.get("warp"),
        shift=kv.get("shift"),
        noise=kv.get("noise"),
    )


def _interp_resample(path: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """
    Resample path (T,C) at fractional indices t_new in [0, T-1] using linear interpolation.
    Keeps output length len(t_new).
    """
    T, C = path.shape
    t_old = np.arange(T, dtype=np.float64)
    out = np.empty((len(t_new), C), dtype=np.float64)
    for c in range(C):
        out[:, c] = np.interp(t_new, t_old, path[:, c])
    return out


def time_warp(path: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """
    Random monotone time warp, returns same length (T,C).
    strength in [0, 1): 0 = no warp, larger = more speed variation.
    """
    x = np.asarray(path, dtype=np.float64)
    T, _ = x.shape
    if T < 3 or strength <= 0:
        return x

    # Random positive increments around 1.0 -> cumulative -> normalized to [0, T-1]
    lo = max(1e-6, 1.0 - strength)
    hi = 1.0 + strength
    inc = rng.uniform(lo, hi, size=T - 1)
    cum = np.concatenate([[0.0], np.cumsum(inc)])
    cum = cum / cum[-1] * (T - 1)

    return _interp_resample(x, cum)


def phase_shift(path: np.ndarray, max_frac: float, rng: np.random.Generator) -> np.ndarray:
    """
    Circular shift by a random integer k in [-K, K], where K = round(max_frac * T).
    """
    x = np.asarray(path, dtype=np.float64)
    T, _ = x.shape
    if T < 2 or max_frac <= 0:
        return x

    K = int(round(max_frac * T))
    if K <= 0:
        return x

    k = int(rng.integers(-K, K + 1))
    return np.roll(x, shift=k, axis=0)


def add_noise(path: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add Gaussian noise with per-channel sigma = strength * std(channel).
    strength ~ 0.01..0.2 typical.
    """
    x = np.asarray(path, dtype=np.float64)
    if strength <= 0:
        return x

    sd = x.std(axis=0, keepdims=True)
    noise = rng.normal(0.0, 1.0, size=x.shape) * (strength * (sd + 1e-8))
    return x + noise


def apply_transforms(paths: List[np.ndarray], spec: TransformSpec, seed: int) -> List[np.ndarray]:
    """
    Apply the specified transforms to a list of paths, deterministically given seed.
    We use a single RNG and draw per-sample randomness from it.
    """
    if spec.warp is None and spec.shift is None and spec.noise is None:
        return paths

    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for p in paths:
        x = np.asarray(p, dtype=np.float64)
        if spec.warp is not None:
            x = time_warp(x, spec.warp, rng)
        if spec.shift is not None:
            x = phase_shift(x, spec.shift, rng)
        if spec.noise is not None:
            x = add_noise(x, spec.noise, rng)
        out.append(x)
    return out