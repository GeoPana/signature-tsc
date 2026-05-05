from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np

from sigtsc.features.signature import LogSigWindowConfig, make_windows, path_signature_features


def test_global_windows() -> None:
    cfg = LogSigWindowConfig(type="global")
    assert make_windows(100, cfg) == [(0, 100)]


def test_sliding_windows() -> None:
    cfg = LogSigWindowConfig(
        type="sliding",
        window_fracs=[0.25],
        step_frac=1.0,
        min_window=1,
        aggregation="pool",
    )

    assert make_windows(100, cfg) == [(0, 25), (25, 50), (50, 75), (75, 100)]


def test_expanding_windows() -> None:
    cfg = LogSigWindowConfig(type="expanding", num_windows=4, min_window=1)
    windows = make_windows(100, cfg)

    assert len(windows) == 4
    assert windows[-1] == (0, 100)
    assert all(start == 0 for start, _ in windows)
    assert [end for _, end in windows] == sorted(end for _, end in windows)


def test_dyadic_windows() -> None:
    cfg = LogSigWindowConfig(type="dyadic", depth=3, min_window=1)

    assert make_windows(100, cfg) == [
        (0, 100),
        (0, 50),
        (50, 100),
        (0, 25),
        (25, 50),
        (50, 75),
        (75, 100),
    ]


def test_dyadic_min_window_skips_short_windows() -> None:
    cfg = LogSigWindowConfig(type="dyadic", depth=5, min_window=20)
    windows = make_windows(64, cfg)

    assert windows == [(0, 64), (0, 32), (32, 64)]
    assert all(end > start for start, end in windows)
    assert all(end - start >= 20 for start, end in windows)


def test_short_sequences_return_full_window() -> None:
    expanding = LogSigWindowConfig(type="expanding", num_windows=4, min_window=8)
    dyadic = LogSigWindowConfig(type="dyadic", depth=3, min_window=8)

    assert make_windows(5, expanding) == [(0, 5)]
    assert make_windows(5, dyadic) == [(0, 5)]


def test_dyadic_deduplicates_small_high_depth_windows() -> None:
    cfg = LogSigWindowConfig(type="dyadic", depth=6, min_window=1)
    windows = make_windows(5, cfg)

    assert len(windows) == len(set(windows))
    assert all(end > start for start, end in windows)


def test_windowed_features_are_deterministic_and_expand_dimension() -> None:
    rng = np.random.default_rng(123)
    paths = [rng.normal(size=(32, 2)), rng.normal(size=(32, 2))]

    global_X = path_signature_features(
        paths,
        level=2,
        transform_type="logsig",
        with_time=True,
        basepoint=True,
    )

    expanding_cfg = LogSigWindowConfig(
        type="expanding",
        num_windows=4,
        min_window=1,
        aggregation="concat",
    )
    expanding_X1 = path_signature_features(
        paths,
        level=2,
        transform_type="logsig",
        with_time=True,
        basepoint=True,
        windowing=expanding_cfg,
    )
    expanding_X2 = path_signature_features(
        paths,
        level=2,
        transform_type="logsig",
        with_time=True,
        basepoint=True,
        windowing=expanding_cfg,
    )

    dyadic_cfg = LogSigWindowConfig(
        type="dyadic",
        depth=3,
        min_window=1,
        aggregation="concat",
    )
    dyadic_X = path_signature_features(
        paths,
        level=2,
        transform_type="logsig",
        with_time=True,
        basepoint=True,
        windowing=dyadic_cfg,
    )
    signature_X = path_signature_features(
        paths,
        level=2,
        transform_type="signature",
        with_time=True,
        basepoint=True,
        windowing=dyadic_cfg,
    )

    assert global_X.ndim == 2
    assert expanding_X1.ndim == 2
    assert dyadic_X.ndim == 2
    assert signature_X.ndim == 2
    assert np.allclose(expanding_X1, expanding_X2)
    assert dyadic_X.shape[1] > global_X.shape[1]
    assert signature_X.shape[1] > global_X.shape[1]


def test_run_experiment_records_window_metadata(monkeypatch) -> None:
    from sigtsc.experiments import run_experiment

    t = np.linspace(0.0, 1.0, 24)
    class_zero = [np.column_stack([np.sin(2 * np.pi * t), t]) for _ in range(4)]
    class_one = [np.column_stack([np.cos(2 * np.pi * t), t**2]) for _ in range(4)]
    Xtr = class_zero[:3] + class_one[:3]
    Xte = class_zero[3:] + class_one[3:]
    ytr = np.array([0, 0, 0, 1, 1, 1])
    yte = np.array([0, 1])

    def fake_load_dataset(name: str, seed: int = 42):
        return Xtr, ytr, Xte, yte

    def fake_train_eval_logreg(Xtr_feat, ytr_arr, Xte_feat, yte_arr, params):
        assert Xtr_feat.ndim == 2
        assert Xte_feat.ndim == 2
        assert Xtr_feat.shape[1] == Xte_feat.shape[1]
        return 1.0, {"accuracy": 1.0}

    monkeypatch.setattr(run_experiment, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(run_experiment, "_train_eval_logreg", fake_train_eval_logreg)

    results_dir = Path("results/test_windowing") / uuid.uuid4().hex
    try:
        cfg = {
            "seed": 42,
            "results_dir": str(results_dir),
            "dataset": {"name": "FakeDataset"},
            "features": {
                "type": "logsig",
                "level": 2,
                "with_time": True,
                "basepoint": True,
            },
            "windowing": {
                "type": "expanding",
                "num_windows": 4,
                "min_window": 1,
                "aggregation": "concat",
            },
            "model": {"type": "logreg", "params": {"C": 1.0, "max_iter": 1000}},
        }

        out, run_dir = run_experiment.run_one_experiment_dict(cfg)
        features = out["features"]

        assert run_dir.exists()
        assert features["type"] == "logsig"
        assert features["basepoint"] is True
        assert features["window_type"] == "expanding"
        assert features["window_aggregation"] == "concat"
        assert features["num_windows"] == 4
        assert features["expanding_num_windows"] == 4
        assert features["feature_dim"] == features["dim"]
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)


def test_run_experiment_accepts_homogeneous_global_and_sliding_configs(monkeypatch) -> None:
    from sigtsc.experiments import run_experiment

    t = np.linspace(0.0, 1.0, 16)
    Xtr = [np.column_stack([t, t**2]), np.column_stack([t + 1, t**2 + 1])]
    Xte = [np.column_stack([t + 0.5, t**2 + 0.5])]
    ytr = np.array([0, 1])
    yte = np.array([0])

    def fake_load_dataset(name: str, seed: int = 42):
        return Xtr, ytr, Xte, yte

    monkeypatch.setattr(run_experiment, "load_dataset", fake_load_dataset)

    base_cfg = {
        "seed": 42,
        "dataset": {"name": "FakeDataset"},
        "features": {
            "type": "logsig",
            "level": 2,
            "with_time": False,
            "basepoint": False,
        },
    }

    global_window = run_experiment._build_window_config(
        base_cfg["features"],
        {**base_cfg, "windowing": {"type": "global", "aggregation": "concat"}},
    )
    sliding_window = run_experiment._build_window_config(
        base_cfg["features"],
        {
            **base_cfg,
            "windowing": {
                "type": "sliding",
                "window_fracs": [0.5],
                "step_frac": 1.0,
                "min_window": 1,
                "aggregation": "pool",
                "pool": ["mean", "max"],
            },
        },
    )

    global_meta = run_experiment._window_metadata(
        Xtr,
        windowing=global_window,
        basepoint=False,
        lead_lag=False,
    )
    sliding_meta = run_experiment._window_metadata(
        Xtr,
        windowing=sliding_window,
        basepoint=False,
        lead_lag=False,
    )

    assert global_meta["window_type"] == "global"
    assert global_meta["window_aggregation"] == "concat"
    assert global_meta["num_windows"] == 1
    assert global_meta["min_window"] is None
    assert sliding_meta["window_type"] == "sliding"
    assert sliding_meta["window_aggregation"] == "pool"
    assert sliding_meta["num_windows"] == 2
