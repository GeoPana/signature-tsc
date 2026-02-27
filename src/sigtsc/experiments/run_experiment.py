from __future__ import annotations

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
from sigtsc.features.signature import LogSigWindowConfig, logsig_features
from sigtsc.models.baselines import train_eval_minirocket
from sigtsc.utils.git import get_git_commit
from sigtsc.utils.io import load_yaml, save_json, save_yaml
from sigtsc.utils.seed import set_seed


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    metrics_path: Path
    config_snapshot_path: Path


def _make_run_dir(results_dir: str) -> RunPaths:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = Path(results_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.json",
        config_snapshot_path=run_dir / "config_snapshot.yaml",
    )


def _train_eval_logreg(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    params: Dict[str, Any],
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


def run_one_experiment_dict(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Path]:
    """
    Run a single experiment given a config dict.
    Returns: (result_dict, run_dir_path)
    """
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    results_dir = str(cfg.get("results_dir", "results/runs"))
    paths = _make_run_dir(results_dir)

    # Save config snapshot
    save_yaml(paths.config_snapshot_path, cfg)

    # ------------------------------------------------------------------
    # Load data
    #
    # NOTE: load_dataset now supports transform specs in dataset name:
    #   "NATOPS@warp=0.20" etc.
    # ------------------------------------------------------------------
    dataset_name = cfg["dataset"]["name"]
    Xtr_paths, ytr, Xte_paths, yte = load_dataset(dataset_name, seed=seed)

    # ------------------------------------------------------------------
    # Model config
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "logreg")
    model_params = model_cfg.get("params", {})

    # ------------------------------------------------------------------
    # If model is MiniROCKET, skip signature features
    # ------------------------------------------------------------------
    if model_type == "minirocket":
        res = train_eval_minirocket(Xtr_paths, ytr, Xte_paths, yte, model_params)
        acc = float(res.accuracy)
        metrics = {"accuracy": acc}
        features_out = {"type": "raw", "dim": None}

    else:
        # ------------------------------------------------------------------
        # Signature features config
        # ------------------------------------------------------------------
        feats = cfg.get("features", {})
        feature_level = int(feats.get("level", 3))
        with_time = bool(feats.get("with_time", False))
        lead_lag = bool(feats.get("lead_lag", False))
        pool_ops = feats.get("pool", ["mean", "max"])

        # IMPORTANT:
        # window_fracs == None => GLOBAL (no windowing)
        # window_fracs == list => multiscale windowing
        window_fracs = feats.get("window_fracs", None)
        if window_fracs is None:
            windowing = None
        else:
            windowing = LogSigWindowConfig(
                window_fracs=window_fracs,
                step_frac=float(feats.get("step_frac", 0.5)),
                min_window=int(feats.get("min_window", 8)),
            )

        # Compute signature features
        Xtr = logsig_features(
            Xtr_paths,
            level=feature_level,
            with_time=with_time,
            lead_lag=lead_lag,
            windowing=windowing,
            pool=pool_ops,
        )
        Xte = logsig_features(
            Xte_paths,
            level=feature_level,
            with_time=with_time,
            lead_lag=lead_lag,
            windowing=windowing,
            pool=pool_ops,
        )

        acc, metrics = _train_eval_logreg(Xtr, ytr, Xte, yte, model_params)

        # Record features used
        features_out = {
            "type": "logsig",
            "level": feature_level,
            "with_time": with_time,
            "lead_lag": lead_lag,
            "window_fracs": window_fracs,  # can be None for global
            "step_frac": feats.get("step_frac", None),
            "min_window": feats.get("min_window", None),
            "pool": feats.get("pool", None),
            "dim": int(Xtr.shape[1]),
        }

    git_commit = get_git_commit()

    out = {
        "dataset": dataset_name,
        "seed": seed,
        "git_commit": git_commit,
        "features": features_out,
        "model": {"type": model_type, "params": model_params},
        "metrics": metrics,
        # suite runner should set cfg["variant"] when looping
        "variant": cfg.get("variant", None),
    }

    save_json(paths.metrics_path, out)
    return out, paths.run_dir


def run_from_config(config_path: str) -> None:
    cfg = load_yaml(config_path)
    out, run_dir = run_one_experiment_dict(cfg)
    print(f"[sigtsc] dataset={out['dataset']} acc={out['metrics']['accuracy']:.4f}")
    print(f"[sigtsc] saved: {run_dir / 'metrics.json'}")