from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from sigtsc.features.augmentations import (
    AugmentationConfig,
    CoordinateProjectionConfig,
    RandomProjectionConfig,
    apply_augmentations,
    coordinate_combinations,
    invisibility_reset,
)
from sigtsc.features.signature import LogSigWindowConfig, path_signature_features


def test_invisibility_reset_shape_and_values() -> None:
    X = np.arange(15, dtype=float).reshape(5, 3)

    out = invisibility_reset(X)

    assert out.shape == (7, 4)
    assert np.allclose(out[:5, 0], 1.0)
    assert np.allclose(out[5:, 0], 0.0)
    assert np.allclose(out[:5, 1:], X)
    assert np.allclose(out[5, 1:], X[-1])
    assert np.allclose(out[-1], 0.0)


def test_coordinate_projection_singletons_with_time() -> None:
    X = np.arange(20, dtype=float).reshape(5, 4)
    cfg = AugmentationConfig(
        coordinate_projection=CoordinateProjectionConfig(enabled=True, mode="singletons")
    )

    streams = apply_augmentations(X, with_time=True, config=cfg)

    assert len(streams) == 4
    t = np.linspace(0.0, 1.0, X.shape[0])
    for i, stream in enumerate(streams):
        assert stream.shape == (5, 2)
        assert np.allclose(stream[:, 0], t)
        assert np.allclose(stream[:, 1], X[:, i])


def test_coordinate_projection_pairs_order() -> None:
    X = np.arange(20, dtype=float).reshape(5, 4)
    cfg = AugmentationConfig(
        coordinate_projection=CoordinateProjectionConfig(enabled=True, mode="pairs")
    )

    streams = apply_augmentations(X, config=cfg)

    expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert coordinate_combinations(4, "pairs") == expected
    assert len(streams) == 6
    for stream, combo in zip(streams, expected):
        assert stream.shape == (5, 2)
        assert np.allclose(stream, X[:, combo])


def test_coordinate_projection_triplets_order() -> None:
    X = np.arange(20, dtype=float).reshape(5, 4)
    cfg = AugmentationConfig(
        coordinate_projection=CoordinateProjectionConfig(enabled=True, mode="triplets")
    )

    streams = apply_augmentations(X, config=cfg)

    expected = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    assert coordinate_combinations(4, "triplets") == expected
    assert len(streams) == 4
    for stream, combo in zip(streams, expected):
        assert stream.shape == (5, 3)
        assert np.allclose(stream, X[:, combo])


def test_random_projection_reproducibility() -> None:
    X = np.arange(20, dtype=float).reshape(5, 4)
    cfg = AugmentationConfig(
        random_projection=RandomProjectionConfig(
            enabled=True,
            output_dim=3,
            num_projections=2,
            seed=123,
        )
    )
    other_seed = AugmentationConfig(
        random_projection=RandomProjectionConfig(
            enabled=True,
            output_dim=3,
            num_projections=2,
            seed=456,
        )
    )

    streams_1 = apply_augmentations(X, with_time=True, config=cfg)
    streams_2 = apply_augmentations(X, with_time=True, config=cfg)
    streams_3 = apply_augmentations(X, with_time=True, config=other_seed)

    assert len(streams_1) == 2
    assert all(s.shape == (5, 4) for s in streams_1)
    for a, b in zip(streams_1, streams_2):
        assert np.allclose(a, b)
    assert any(not np.allclose(a, b) for a, b in zip(streams_1, streams_3))


def test_augmentation_validation_errors() -> None:
    X = np.ones((5, 1))

    with pytest.raises(ValueError, match="requires at least 2"):
        apply_augmentations(
            X,
            config=AugmentationConfig(
                coordinate_projection=CoordinateProjectionConfig(enabled=True, mode="pairs")
            ),
        )

    with pytest.raises(ValueError, match="requires at least 3"):
        apply_augmentations(
            np.ones((5, 2)),
            config=AugmentationConfig(
                coordinate_projection=CoordinateProjectionConfig(enabled=True, mode="triplets")
            ),
        )

    with pytest.raises(ValueError, match="output_dim must be >= 1"):
        apply_augmentations(
            X,
            config=AugmentationConfig(
                random_projection=RandomProjectionConfig(enabled=True, output_dim=0)
            ),
        )

    with pytest.raises(ValueError, match="num_projections must be >= 1"):
        apply_augmentations(
            X,
            config=AugmentationConfig(
                random_projection=RandomProjectionConfig(
                    enabled=True,
                    output_dim=2,
                    num_projections=0,
                )
            ),
        )

    with pytest.raises(ValueError, match="cannot both be enabled"):
        apply_augmentations(
            np.ones((5, 3)),
            config=AugmentationConfig(
                coordinate_projection=CoordinateProjectionConfig(enabled=True, mode="pairs"),
                random_projection=RandomProjectionConfig(enabled=True, output_dim=2),
            ),
        )

    with pytest.raises(ValueError, match="basepoint and invisibility_reset"):
        apply_augmentations(X, basepoint=True, invisibility_reset_enabled=True)


def test_augmented_signature_features_are_deterministic() -> None:
    rng = np.random.default_rng(123)
    paths = [rng.normal(size=(24, 4)), rng.normal(size=(24, 4))]
    dyadic = LogSigWindowConfig(type="dyadic", depth=2, min_window=1, aggregation="concat")

    invis = path_signature_features(
        paths,
        level=2,
        transform_type="logsig",
        with_time=True,
        invisibility_reset=True,
        windowing=LogSigWindowConfig(type="global", aggregation="concat"),
    )

    coord_cfg = AugmentationConfig(
        coordinate_projection=CoordinateProjectionConfig(enabled=True, mode="pairs")
    )
    coord_1 = path_signature_features(
        paths,
        level=2,
        transform_type="logsig",
        with_time=True,
        augmentation=coord_cfg,
        windowing=dyadic,
    )
    coord_2 = path_signature_features(
        paths,
        level=2,
        transform_type="logsig",
        with_time=True,
        augmentation=coord_cfg,
        windowing=dyadic,
    )

    random_cfg = AugmentationConfig(
        random_projection=RandomProjectionConfig(
            enabled=True,
            output_dim=3,
            num_projections=2,
            seed=42,
        )
    )
    random_sig = path_signature_features(
        paths,
        level=2,
        transform_type="signature",
        with_time=True,
        augmentation=random_cfg,
        windowing=LogSigWindowConfig(type="global", aggregation="concat"),
    )

    assert invis.ndim == 2
    assert coord_1.ndim == 2
    assert random_sig.ndim == 2
    assert np.allclose(coord_1, coord_2)
    assert coord_1.shape[1] > invis.shape[1]
    assert random_sig.shape[1] > 0


def test_run_experiment_records_augmentation_metadata(monkeypatch) -> None:
    from sigtsc.experiments import run_experiment

    t = np.linspace(0.0, 1.0, 18)
    Xtr = [
        np.column_stack([t, t**2, np.sin(t)]),
        np.column_stack([t + 1, t**2 + 1, np.cos(t)]),
    ]
    Xte = [np.column_stack([t + 0.5, t**2 + 0.5, np.sin(t + 0.5)])]
    ytr = np.array([0, 1])
    yte = np.array([0])

    def fake_load_dataset(name: str, seed: int = 42):
        return Xtr, ytr, Xte, yte

    def fake_train_eval_logreg(Xtr_feat, ytr_arr, Xte_feat, yte_arr, params):
        assert Xtr_feat.shape[1] == Xte_feat.shape[1]
        return 1.0, {"accuracy": 1.0}

    monkeypatch.setattr(run_experiment, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(run_experiment, "_train_eval_logreg", fake_train_eval_logreg)

    results_dir = Path("results/test_augmentations") / uuid.uuid4().hex
    try:
        cfg = {
            "seed": 42,
            "results_dir": str(results_dir),
            "dataset": {"name": "FakeDataset"},
            "features": {
                "type": "logsig",
                "level": 2,
                "with_time": True,
                "basepoint": False,
                "invisibility_reset": False,
            },
            "augmentation": {
                "coordinate_projection": {"enabled": True, "mode": "pairs"},
            },
            "windowing": {"type": "global", "aggregation": "concat"},
            "model": {"type": "logreg", "params": {}},
        }

        out, _ = run_experiment.run_one_experiment_dict(cfg)
        features = out["features"]

        assert features["coordinate_projection_mode"] == "pairs"
        assert features["num_augmented_streams"] == 3
        assert features["channels_per_augmented_stream"] == [3, 3, 3]
        assert features["total_windows"] == 3
        assert features["feature_dim"] == features["dim"]
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)
