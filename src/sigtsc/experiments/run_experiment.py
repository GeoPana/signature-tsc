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

from sigtsc.data.loaders import load_dataset
from sigtsc.features.signature import logsig_features_global
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

    # Features
    Xtr = logsig_features_global(Xtr_paths, level=feature_level, with_time=with_time)
    Xte = logsig_features_global(Xte_paths, level=feature_level, with_time=with_time)

    # Train/eval
    if model_type != "logreg":
        raise ValueError(f"Minimal slice only supports model.type='logreg'. Got: {model_type}")

    acc, metrics = _train_eval_logreg(Xtr, ytr, Xte, yte, model_params)

    out = {
        "dataset": dataset_name,
        "seed": seed,
        "features": {
            "type": "logsig",
            "level": feature_level,
            "with_time": with_time,
            "dim": int(Xtr.shape[1]),
        },
        "model": {
            "type": model_type,
            "params": model_params,
        },
        "metrics": metrics,
    }

    save_json(paths.metrics_path, out)

    print(f"[sigtsc] dataset={dataset_name} acc={acc:.4f}")
    print(f"[sigtsc] saved: {paths.metrics_path}")