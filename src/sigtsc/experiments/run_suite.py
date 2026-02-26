from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sigtsc.experiments.run_experiment import run_one_experiment_dict
from sigtsc.utils.io import load_yaml, save_json, save_yaml


@dataclass(frozen=True)
class SuitePaths:
    suite_dir: Path
    summary_csv: Path
    suite_config_snapshot: Path
    results_jsonl: Path


def _make_suite_dir(results_dir: str, suite_name: str) -> SuitePaths:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suite_dir = Path(results_dir) / f"{suite_name}_{ts}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    return SuitePaths(
        suite_dir=suite_dir,
        summary_csv=suite_dir / "summary.csv",
        suite_config_snapshot=suite_dir / "suite_config_snapshot.yaml",
        results_jsonl=suite_dir / "results.jsonl",
    )


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stable column order (common keys first)
    preferred = [
        "suite",
        "dataset",
        "variant",
        "model_type",
        "features_type",
        "level",
        "with_time",
        "window_fracs",
        "pool",
        "dim",
        "accuracy",
        "run_dir",
    ]
    # Include any extra keys at end
    all_keys = set().union(*(r.keys() for r in rows))
    cols = preferred + [k for k in sorted(all_keys) if k not in preferred]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_suite_from_config(config_path: str) -> None:
    cfg = load_yaml(config_path)

    if "suite" not in cfg:
        raise ValueError("Suite config must contain a top-level 'suite' key.")

    seed = int(cfg.get("seed", 42))
    results_dir = str(cfg.get("results_dir", "results/suites"))

    suite_name = str(cfg["suite"].get("name", "suite"))
    datasets = list(cfg["suite"]["datasets"])
    variants = list(cfg["suite"]["variants"])

    paths = _make_suite_dir(results_dir, suite_name)

    # Snapshot suite config for reproducibility
    save_yaml(paths.suite_config_snapshot, cfg)

    summary_rows: List[Dict[str, Any]] = []
    jsonl_lines: List[Dict[str, Any]] = []

    print(f"[sigtsc] Running suite '{suite_name}'")
    print(f"[sigtsc] Datasets: {len(datasets)} | Variants: {len(variants)}")
    print(f"[sigtsc] Output: {paths.suite_dir}")

    for ds in datasets:
        for v in variants:
            variant_name = str(v.get("name", "variant"))
            # Build a single-experiment config dict compatible with the normal runner
            exp_cfg: Dict[str, Any] = {
                "seed": seed,
                "variant": variant_name,
                "results_dir": str(paths.suite_dir / "runs"),
                "dataset": {"name": ds},
                # features may be missing for minirocket; runner handles it
                "features": v.get("features", cfg.get("features", {})),
                "model": v.get("model", cfg.get("model", {})),
            }

            print(f"[sigtsc] -> dataset={ds} variant={variant_name}")

            try:
                out, run_dir = run_one_experiment_dict(exp_cfg)
                # out is the same dict you write to metrics.json in single runs
                acc = float(out["metrics"]["accuracy"])
                features = out.get("features", {})
                model = out.get("model", {})

                row = {
                    "suite": suite_name,
                    "dataset": ds,
                    "variant": variant_name,
                    "model_type": model.get("type"),
                    "features_type": features.get("type"),
                    "level": features.get("level"),
                    "with_time": features.get("with_time"),
                    "window_fracs": features.get("window_fracs"),
                    "pool": features.get("pool"),
                    "dim": features.get("dim"),
                    "accuracy": acc,
                    "run_dir": str(run_dir),
                }
                summary_rows.append(row)

                jsonl_rec = {"suite": suite_name, "dataset": ds, "variant": variant_name, **out, "run_dir": str(run_dir)}
                jsonl_lines.append(jsonl_rec)

            except Exception as e:
                # Do not crash the whole suite; record failure
                row = {
                    "suite": suite_name,
                    "dataset": ds,
                    "variant": variant_name,
                    "model_type": exp_cfg.get("model", {}).get("type"),
                    "features_type": exp_cfg.get("features", {}).get("type", "logsig"),
                    "accuracy": None,
                    "run_dir": None,
                    "error": repr(e),
                }
                summary_rows.append(row)
                print(f"[sigtsc] !! failed dataset={ds} variant={variant_name}: {e}")

    # Save outputs
    _write_summary_csv(paths.summary_csv, summary_rows)
    save_json(paths.suite_dir / "summary.json", {"rows": summary_rows})

    # Save JSONL (one record per run) for easy parsing later
    with open(paths.results_jsonl, "w", encoding="utf-8") as f:
        for rec in jsonl_lines:
            f.write(json.dumps(rec) + "\n")

    print(f"[sigtsc] Suite complete.")
    print(f"[sigtsc] Summary CSV: {paths.summary_csv}") 