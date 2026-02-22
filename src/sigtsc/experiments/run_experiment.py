from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sigtsc.models.baselines import train_eval_minirocket
from sigtsc.data.loaders import load_dataset
from sigtsc.features.signature import LogSigWindowConfig, logsig_features
from sigtsc.utils.io import load_yaml, save_json, save_yaml
from sigtsc.utils.seed import set_seed


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    metrics_path: Path
    config_snapshot_path: Path


def _make_run_dir(results_dir: str) -> RunPaths:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(results_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.json",
        config_snapshot_path=run_dir / "config_snapshot.yaml",
    )


def _train_eval_logreg(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, params: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 5000)),
            n_jobs=-1,
        ),
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    return acc, {"accuracy": float(acc)}


def run_from_config(config_path: str) -> None:
    cfg = load_yaml(config_path)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    results_dir = str(cfg.get("results_dir", "results/runs"))
    paths = _make_run_dir(results_dir)

    # Save config snapshot for reproducibility
    save_yaml(paths.config_snapshot_path, cfg)

    dataset_name = cfg["dataset"]["name"]

    feature_level = int(cfg["features"].get("level", 3))
    with_time = bool(cfg["features"].get("with_time", False))

    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "logreg")
    model_params = model_cfg.get("params", {})

    # Load data
    Xtr_paths, ytr, Xte_paths, yte = load_dataset(dataset_name)

    # Features (global or multiscale windowed logsig)
    win_cfg = None
    pool_ops = cfg["features"].get("pool", ["mean", "max"])

    window_fracs = cfg["features"].get("window_fracs", [])
    if window_fracs:
        win_cfg = LogSigWindowConfig(
            window_fracs=window_fracs,
            step_frac=float(cfg["features"].get("step_frac", 0.5)),
            min_window=int(cfg["features"].get("min_window", 8)),
        )

    Xtr = logsig_features(
        Xtr_paths,
        level=feature_level,
        with_time=with_time,
        windowing=win_cfg,
        pool=pool_ops,
    )
    Xte = logsig_features(
        Xte_paths,
        level=feature_level,
        with_time=with_time,
        windowing=win_cfg,
        pool=pool_ops,
    )

    # Train/eval
    if model_type == "logreg":
        acc, metrics = _train_eval_logreg(Xtr, ytr, Xte, yte, model_params)

    elif model_type == "minirocket":
        # MiniROCKET works on raw series, not on signature features:
        res = train_eval_minirocket(Xtr_paths, ytr, Xte_paths, yte, model_params)
        acc = res.accuracy
        metrics = {"accuracy": acc}

    else:
        raise ValueError(f"Unknown model.type: {model_type}. Supported: 'logreg', 'minirocket'")

    if model_type == "minirocket":
        features_out = {"type": "raw", "dim": None}
    else:
        features_out = {
            "type": "logsig",
            "level": feature_level,
            "with_time": with_time,
            "window_fracs": cfg["features"].get("window_fracs", []),
            "step_frac": cfg["features"].get("step_frac", None),
            "min_window": cfg["features"].get("min_window", None),
            "pool": cfg["features"].get("pool", None),
            "dim": int(Xtr.shape[1]),
        }

    out = {
        "dataset": dataset_name,
        "seed": seed,
        "features": features_out,
        "model": {
            "type": model_type,
            "params": model_params,
        },
        "metrics": metrics,
    }

    save_json(paths.metrics_path, out)

    print(f"[sigtsc] dataset={dataset_name} acc={acc:.4f}")
    print(f"[sigtsc] saved: {paths.metrics_path}")