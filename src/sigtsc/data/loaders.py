from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile
import tempfile
import shutil
import os

import numpy as np
from aeon.datasets import load_classification

from sigtsc.data.transforms import parse_dataset_spec, apply_transforms

def _to_list_of_paths_TxC(X: Any) -> list[np.ndarray]:
    """
    Convert aeon classification data into a list of (T, C) arrays.
    aeon typically returns:
      - equal length: ndarray (N, C, T)
      - unequal length: list of arrays (C, T_i) (sometimes)
    """
    if isinstance(X, list):
        paths = []
        for xi in X:
            xi = np.asarray(xi)
            if xi.ndim != 2:
                raise ValueError(f"Expected 2D per case, got shape {xi.shape}")
            # Assume (C, T) -> (T, C)
            path_TxC = xi.T
            paths.append(path_TxC.astype(np.float64, copy=False))
        return paths

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N,C,T), got shape {X.shape}")

    # (N,C,T) -> list of (T,C)
    return [X[i].T.astype(np.float64, copy=False) for i in range(X.shape[0])]


def _download_and_extract_with_headers(dataset: str, cache_dir: str) -> str:
    """
    Download https://timeseriesclassification.com/aeon-toolkit/{dataset}.zip
    using a browser-like User-Agent, then extract into cache_dir/dataset/.

    Returns the dataset directory path (cache_dir/dataset).
    """
    url = f"https://timeseriesclassification.com/aeon-toolkit/{dataset}.zip"

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = cache_root / dataset

    # If already extracted, don't re-download
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        return str(dataset_dir)

    dl_dir = Path(tempfile.mkdtemp())
    zip_path = dl_dir / f"{dataset}.zip"

    req = Request(
        url,
        headers={
            # Many sites block default Python user agents; emulate a normal browser.
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        },
    )

    try:
        with urlopen(req, timeout=60) as resp, open(zip_path, "wb") as out:
            out.write(resp.read())

        dataset_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dataset_dir)

        return str(dataset_dir)
    finally:
        shutil.rmtree(dl_dir, ignore_errors=True)


def load_dataset(
    name: str,
    cache_dir: str = "data/aeon_cache",
    seed: int = 42,
) -> Tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Load dataset via aeon.

    Supports transform specs in dataset string, e.g.:
      - "NATOPS@warp=0.20"
      - "CharacterTrajectories@shift=0.20"
      - "X@warp=0.20,shift=0.10,noise=0.05"

    If remote download fails with HTTP 401, fall back to manual download
    (browser-like headers) and then load from local cache.

    Transforms are applied deterministically using `seed` (separate train/test seeds).
    """
    base_name, tf_spec = parse_dataset_spec(name)

    try:
        X_train, y_train = load_classification(base_name, split="train")
        X_test, y_test = load_classification(base_name, split="test")
    except HTTPError as e:
        if e.code != 401:
            raise
        # Fall back: download ourselves into cache_dir and re-load from there
        _download_and_extract_with_headers(base_name, cache_dir)
        X_train, y_train = load_classification(base_name, split="train", extract_path=cache_dir)
        X_test, y_test = load_classification(base_name, split="test", extract_path=cache_dir)
    except URLError:
        # Network/DNS issues: try local cache if present
        X_train, y_train = load_classification(base_name, split="train", extract_path=cache_dir)
        X_test, y_test = load_classification(base_name, split="test", extract_path=cache_dir)

    Xtr = _to_list_of_paths_TxC(X_train)
    Xte = _to_list_of_paths_TxC(X_test)

    # Apply transforms after conversion to list of (T,C) arrays
    if tf_spec.warp is not None or tf_spec.shift is not None or tf_spec.noise is not None:
        Xtr = apply_transforms(Xtr, tf_spec, seed + 101)
        Xte = apply_transforms(Xte, tf_spec, seed + 202)

    ytr = np.asarray(y_train)
    yte = np.asarray(y_test)
    return Xtr, ytr, Xte, yte