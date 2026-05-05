from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import sqrt
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CoordinateProjectionConfig:
    """Configuration for deterministic coordinate projection streams."""

    enabled: bool = False
    mode: str = "none"


@dataclass(frozen=True)
class RandomProjectionConfig:
    """Configuration for deterministic random projection streams."""

    enabled: bool = False
    output_dim: int = 1
    num_projections: int = 1
    seed: int = 42


@dataclass(frozen=True)
class AugmentationConfig:
    """Multi-stream augmentation configuration."""

    coordinate_projection: CoordinateProjectionConfig = field(
        default_factory=CoordinateProjectionConfig
    )
    random_projection: RandomProjectionConfig = field(default_factory=RandomProjectionConfig)


def lead_lag(path: np.ndarray) -> np.ndarray:
    """
    Lead-lag transform.

    Input:
      path: (T, C)

    Output:
      ll: (2T - 1, 2C)

    Construction:
      For each original point x[t], we add:
        - a lag copy (previous) and a lead copy (current) in 2C dims
      Interleaving produces a new path that encodes increments/order.

    This variant produces length (2T - 1):
      y[2t]   = [x[t], x[t]]
      y[2t+1] = [x[t], x[t+1]] for t=0..T-2
    """
    path = np.asarray(path)
    if path.ndim != 2:
        raise ValueError(f"lead_lag expects (T,C), got shape {path.shape}")

    T, C = path.shape
    if T < 2:
        return np.concatenate([path, path], axis=1)

    out = np.empty((2 * T - 1, 2 * C), dtype=path.dtype)
    out[0::2, :C] = path
    out[0::2, C:] = path
    out[1::2, :C] = path[:-1]
    out[1::2, C:] = path[1:]
    return out


def add_time_channel(path: np.ndarray) -> np.ndarray:
    """Prepend a normalized time channel to a (T, C) path."""
    path = _as_path(path, "add_time_channel")
    T = path.shape[0]
    t = np.linspace(0.0, 1.0, T, dtype=path.dtype).reshape(T, 1)
    return np.concatenate([t, path], axis=1)


def add_basepoint(path: np.ndarray) -> np.ndarray:
    """Prepend a zero vector to a (T, C) path."""
    path = _as_path(path, "add_basepoint")
    zero = np.zeros((1, path.shape[1]), dtype=path.dtype)
    return np.vstack([zero, path])


def invisibility_reset(path: np.ndarray) -> np.ndarray:
    """
    Add a visibility coordinate and reset the path to the origin.

    For x_1, ..., x_n in R^d, returns:
      (1, x_1), ..., (1, x_n), (0, x_n), (0, 0)

    The output has shape (n + 2, d + 1).
    """
    path = _as_path(path, "invisibility_reset")
    T, C = path.shape
    out = np.zeros((T + 2, C + 1), dtype=path.dtype)
    out[:T, 0] = 1.0
    out[:T, 1:] = path
    out[T, 1:] = path[-1]
    return out


def coordinate_combinations(input_dim: int, mode: str) -> list[tuple[int, ...]]:
    """Return deterministic value-channel index combinations for a projection mode."""
    d = int(input_dim)
    normalized = _normalize_projection_mode(mode)
    if normalized == "none":
        return [tuple(range(d))]

    sizes = {"singletons": 1, "pairs": 2, "triplets": 3}
    size = sizes[normalized]
    if d < size:
        raise ValueError(
            f"coordinate_projection mode '{normalized}' requires at least {size} "
            f"value channels, got {d}"
        )
    return [tuple(c) for c in combinations(range(d), size)]


def coordinate_project_streams(path: np.ndarray, mode: str) -> list[np.ndarray]:
    """Project a value path into singleton, pair, or triplet coordinate streams."""
    path = _as_path(path, "coordinate_project_streams")
    combos = coordinate_combinations(path.shape[1], mode)
    return [path[:, combo].astype(path.dtype, copy=False) for combo in combos]


def random_projection_matrices(
    input_dim: int,
    output_dim: int,
    num_projections: int,
    seed: int,
) -> list[np.ndarray]:
    """
    Build deterministic Gaussian random projection matrices.

    Entries use N(0, 1 / sqrt(input_dim)) scaling, which keeps projected
    coordinate magnitudes stable as input dimension changes.
    """
    d = int(input_dim)
    e = int(output_dim)
    p = int(num_projections)
    if d < 1:
        raise ValueError(f"random_projection requires input_dim >= 1, got {d}")
    if e < 1:
        raise ValueError(f"random_projection output_dim must be >= 1, got {e}")
    if p < 1:
        raise ValueError(f"random_projection num_projections must be >= 1, got {p}")

    rng = np.random.default_rng(int(seed))
    scale = 1.0 / sqrt(d)
    return [rng.normal(0.0, scale, size=(d, e)) for _ in range(p)]


def random_project_streams(
    path: np.ndarray,
    output_dim: int,
    num_projections: int,
    seed: int,
) -> list[np.ndarray]:
    """Project a value path through deterministic random projection heads."""
    path = _as_path(path, "random_project_streams")
    matrices = random_projection_matrices(
        input_dim=path.shape[1],
        output_dim=output_dim,
        num_projections=num_projections,
        seed=seed,
    )
    return [
        np.sum(path[:, :, None] * A[None, :, :], axis=1).astype(path.dtype, copy=False)
        for A in matrices
    ]


def apply_augmentations(
    path: np.ndarray,
    *,
    with_time: bool = False,
    basepoint: bool = False,
    invisibility_reset_enabled: bool = False,
    lead_lag_enabled: bool = False,
    config: AugmentationConfig | None = None,
) -> list[np.ndarray]:
    """
    Apply the configured augmentation pipeline to a normalized value path.

    Ordering is:
      value projection -> time channel -> basepoint/invisibility-reset -> lead-lag.

    The input and all returned streams use (T, C) orientation.
    """
    path = _as_path(path, "apply_augmentations").astype(np.float64, copy=False)
    cfg = config or AugmentationConfig()
    _validate_augmentation_config(
        input_dim=path.shape[1],
        basepoint=basepoint,
        invisibility_reset_enabled=invisibility_reset_enabled,
        config=cfg,
    )

    if cfg.coordinate_projection.enabled:
        streams = coordinate_project_streams(path, cfg.coordinate_projection.mode)
    elif cfg.random_projection.enabled:
        streams = random_project_streams(
            path,
            output_dim=cfg.random_projection.output_dim,
            num_projections=cfg.random_projection.num_projections,
            seed=cfg.random_projection.seed,
        )
    else:
        streams = [path]

    out: list[np.ndarray] = []
    for stream in streams:
        x = stream
        if with_time:
            x = add_time_channel(x)
        if basepoint:
            x = add_basepoint(x)
        if invisibility_reset_enabled:
            x = invisibility_reset(x)
        if lead_lag_enabled:
            x = lead_lag(x)
        out.append(x)
    return out


def augmentation_metadata(streams: Sequence[np.ndarray]) -> dict[str, object]:
    """Return simple shape metadata for augmented streams."""
    channels = [int(s.shape[1]) for s in streams]
    lengths = [int(s.shape[0]) for s in streams]
    return {
        "num_augmented_streams": len(streams),
        "channels_per_augmented_stream": channels,
        "lengths_per_augmented_stream": lengths,
    }


def _as_path(path: np.ndarray, caller: str) -> np.ndarray:
    path = np.asarray(path)
    if path.ndim != 2:
        raise ValueError(f"{caller} expects (T,C), got shape {path.shape}")
    if path.shape[0] < 1:
        raise ValueError(f"{caller} expects at least one time point")
    if path.shape[1] < 1:
        raise ValueError(f"{caller} expects at least one channel")
    return path


def _normalize_projection_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "off": "none",
        "false": "none",
        "singleton": "singletons",
        "single": "singletons",
        "pair": "pairs",
        "triplet": "triplets",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"none", "singletons", "pairs", "triplets"}:
        raise ValueError(
            f"Unknown coordinate_projection mode '{mode}'. "
            "Supported: none, singletons, pairs, triplets"
        )
    return normalized


def _validate_augmentation_config(
    *,
    input_dim: int,
    basepoint: bool,
    invisibility_reset_enabled: bool,
    config: AugmentationConfig,
) -> None:
    coord_enabled = bool(config.coordinate_projection.enabled)
    random_enabled = bool(config.random_projection.enabled)
    if coord_enabled and random_enabled:
        raise ValueError(
            "coordinate_projection and random_projection cannot both be enabled"
        )
    if basepoint and invisibility_reset_enabled:
        raise ValueError("basepoint and invisibility_reset cannot both be enabled")

    if coord_enabled:
        mode = _normalize_projection_mode(config.coordinate_projection.mode)
        if mode == "none":
            raise ValueError(
                "coordinate_projection.enabled=true requires mode "
                "singletons, pairs, or triplets"
            )
        coordinate_combinations(input_dim, mode)

    if random_enabled:
        random_projection_matrices(
            input_dim=input_dim,
            output_dim=config.random_projection.output_dim,
            num_projections=config.random_projection.num_projections,
            seed=config.random_projection.seed,
        )
