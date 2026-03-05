from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import os

import concurrent.futures as cf

from sigtsc.experiments.run_experiment import run_one_experiment_dict
from sigtsc.experiments.aggregate_results import aggregate_results
from sigtsc.utils.io import load_yaml, save_json, save_yaml
from sigtsc.experiments.plot_results import generate_plots, print_plot_summary


@dataclass(frozen=True)
class SuitePaths:
    suite_dir: Path
    summary_csv: Path
    suite_config_snapshot: Path
    results_jsonl: Path
    agg_dir: Path
    agg_summary_csv: Path
    agg_report_csv: Path
    agg_robustness_csv: Path
    agg_winners_csv: Path
    status_running: Path
    status_done: Path
    status_failed: Path


def _make_suite_dir(results_dir: str, suite_name: str) -> SuitePaths:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suite_dir = Path(results_dir) / f"{suite_name}_{ts}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    agg_dir = suite_dir / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)

    return SuitePaths(
        suite_dir=suite_dir,
        summary_csv=suite_dir / "summary.csv",
        suite_config_snapshot=suite_dir / "suite_config_snapshot.yaml",
        results_jsonl=suite_dir / "results.jsonl",
        agg_dir=agg_dir,
        agg_summary_csv=agg_dir / "summary.csv",
        agg_report_csv=agg_dir / "report.csv",
        agg_robustness_csv=agg_dir / "robustness.csv",
        agg_winners_csv=agg_dir / "robustness_winners.csv",
        status_running=suite_dir / "STATUS_RUNNING",
        status_done=suite_dir / "STATUS_DONE",
        status_failed=suite_dir / "STATUS_FAILED",
    )


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
    all_keys = set().union(*(r.keys() for r in rows))
    cols = preferred + [k for k in sorted(all_keys) if k not in preferred]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _touch(path: Path, text: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("" if text is None else text, encoding="utf-8")


def _aggregate_suite(paths: SuitePaths) -> None:
    runs_root = str(paths.suite_dir / "runs")
    aggregate_results(
        results_root=runs_root,
        out_summary_csv=str(paths.agg_summary_csv),
        out_report_csv=str(paths.agg_report_csv),
        out_robustness_csv=str(paths.agg_robustness_csv),
        out_robustness_winners_csv=str(paths.agg_winners_csv),
    )


def _plot_suite(paths: SuitePaths, cfg: Dict[str, Any]) -> None:
    # support both top-level plotting and suite.plotting
    plotting = cfg.get("plotting")
    if plotting is None:
        plotting = cfg.get("suite", {}).get("plotting", {})
    if not bool(plotting.get("enabled", False)):
        return

    datasets = plotting.get("datasets", None)
    if isinstance(datasets, str):
        datasets = [datasets]

    out_dir_cfg = plotting.get("out_dir", None)
    out_dir = Path(out_dir_cfg) if out_dir_cfg else (paths.suite_dir / "plots")

    plot_paths = generate_plots(
        summary_csv=str(paths.agg_summary_csv),
        report_csv=str(paths.agg_report_csv),
        robustness_csv=str(paths.agg_robustness_csv),
        out_dir=str(out_dir),
        datasets=datasets,
    )
    print_plot_summary(plot_paths)


def _safe_name(s: str) -> str:
    return (
        str(s)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("@", "_at_")
        .replace(",", "_")
        .replace("=", "_")
        .replace(" ", "")
    )


def run_suite_from_config(config_path: str, workers: int = 1) -> None:
    cfg = load_yaml(config_path)
    if "suite" not in cfg:
        raise ValueError("Suite config must contain a top-level 'suite' key.")
    
    # Normalize worker count and leave one CPU core free
    avail = os.cpu_count() or 1
    cap = max(1, avail - 1)   #leave one CPU core free
    requested = max(1, int(workers))
    if requested > cap:
        print(f"[sigtsc] workers={requested} exceeds cap={cap} (avail={avail}, leaving one core free); clamping.")
    workers = min(requested, cap)

    seed = int(cfg.get("seed", 42))
    results_dir = str(cfg.get("results_dir", "results/suites"))

    suite_name = str(cfg["suite"].get("name", "suite"))
    datasets = list(cfg["suite"]["datasets"])
    variants = list(cfg["suite"]["variants"])

    paths = _make_suite_dir(results_dir, suite_name)

    _touch(paths.status_running, text=f"started_at={datetime.now().isoformat()}\n")
    save_yaml(paths.suite_config_snapshot, cfg)

    summary_rows: List[Dict[str, Any]] = []
    jsonl_lines: List[Dict[str, Any]] = []
    failed_any = False

    print(f"[sigtsc] Running suite '{suite_name}'")
    print(f"[sigtsc] Datasets: {len(datasets)} | Variants: {len(variants)}")
    print(f"[sigtsc] Output: {paths.suite_dir}")

    try:
        # -------------------- build all jobs first --------------------
        jobs: List[Dict[str, Any]] = []
        for ds in datasets:
            for v in variants:
                variant_name = str(v.get("name", "variant"))

                # Prevent collisions: separate dirs per dataset+variant
                runs_root = paths.suite_dir / "runs" / _safe_name(ds) / _safe_name(variant_name)

                exp_cfg: Dict[str, Any] = {
                    "seed": seed,
                    "variant": variant_name,
                    "results_dir": str(runs_root),
                    "dataset": {"name": ds},
                    "features": v.get("features", cfg.get("features", {})),
                    "model": v.get("model", cfg.get("model", {})),
                }

                # Avoid core over-subscription when suite-level parallelism is used
                if workers > 1 and exp_cfg.get("model", {}).get("type") == "minirocket":
                    exp_cfg.setdefault("model", {}).setdefault("params", {})
                    exp_cfg["model"]["params"].setdefault("n_jobs", 1)

                jobs.append({"dataset": ds, "variant": variant_name, "exp_cfg": exp_cfg})

                print(f"[sigtsc] -> dataset={ds} variant={variant_name}")

        print(f"[sigtsc] Execution mode: {'parallel' if workers > 1 else 'serial'} (workers={workers})")

        if workers <= 1:
            for j in jobs:
                ds = j["dataset"]
                variant_name = j["variant"]
                exp_cfg = j["exp_cfg"]
                print(f"[sigtsc] -> dataset={ds} variant={variant_name}")
                try:
                    out, run_dir = run_one_experiment_dict(exp_cfg)
                    acc = float(out["metrics"]["accuracy"])
                    features = out.get("features", {})
                    model = out.get("model", {})
                    summary_rows.append(
                        {
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
                    )
                    jsonl_lines.append({"suite": suite_name, "dataset": ds, "variant": variant_name, **out, "run_dir": str(run_dir)})
                except Exception as e:
                    failed_any = True
                    summary_rows.append(
                        {
                            "suite": suite_name,
                            "dataset": ds,
                            "variant": variant_name,
                            "model_type": exp_cfg.get("model", {}).get("type"),
                            "features_type": exp_cfg.get("features", {}).get("type", "logsig"),
                            "accuracy": None,
                            "run_dir": None,
                            "error": repr(e),
                        }
                    )
                    print(f"[sigtsc] !! failed dataset={ds} variant={variant_name}: {e}")
        else:
            with cf.ProcessPoolExecutor(max_workers=workers) as ex:
                fut_to_job = {ex.submit(run_one_experiment_dict, j["exp_cfg"]): j for j in jobs}
                for fut in cf.as_completed(fut_to_job):
                    j = fut_to_job[fut]
                    ds = j["dataset"]
                    variant_name = j["variant"]
                    exp_cfg = j["exp_cfg"]

                    print(f"[sigtsc] -> dataset={ds} variant={variant_name}")

                    try:
                        out, run_dir = fut.result()
                        acc = float(out["metrics"]["accuracy"])
                        features = out.get("features", {})
                        model = out.get("model", {})

                        summary_rows.append(
                            {
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
                        )

                        jsonl_lines.append(
                            {"suite": suite_name, 
                             "dataset": ds, 
                             "variant": variant_name, 
                             **out, "run_dir": str(run_dir)
                             }
                        )
                     
                    except Exception as e:
                        failed_any = True
                        summary_rows.append(
                            {
                                "suite": suite_name,
                                "dataset": ds,
                                "variant": variant_name,
                                "model_type": exp_cfg.get("model", {}).get("type"),
                                "features_type": exp_cfg.get("features", {}).get("type", "logsig"),
                                "accuracy": None,
                                "run_dir": None,
                                "error": repr(e),
                            }
                        )
                        print(f"[sigtsc] !! failed dataset={ds} variant={variant_name}: {e}")

    finally:
        _write_summary_csv(paths.summary_csv, summary_rows)
        save_json(paths.suite_dir / "summary.json", {"rows": summary_rows})

        with open(paths.results_jsonl, "w", encoding="utf-8") as f:
            for rec in jsonl_lines:
                f.write(json.dumps(rec) + "\n")

        try:
            _aggregate_suite(paths)
            print(f"[sigtsc] Aggregated suite into: {paths.agg_dir}")
            print(f"[sigtsc] - {paths.agg_report_csv}")
            print(f"[sigtsc] - {paths.agg_robustness_csv}")
            print(f"[sigtsc] - {paths.agg_winners_csv}")
            try:
                _plot_suite(paths, cfg)
            except Exception as e:
                # Keep suite successful even if plotting fails
                print(f"[sigtsc] Plot generation failed: {e}")
        except Exception as e:
            failed_any = True
            print(f"[sigtsc] Aggregation failed: {e}")

        if failed_any:
            _touch(paths.status_failed, text=f"finished_at={datetime.now().isoformat()}\n")
        else:
            _touch(paths.status_done, text=f"finished_at={datetime.now().isoformat()}\n")

        try:
            if paths.status_running.exists():
                paths.status_running.unlink()
        except Exception:
            pass

    print(f"[sigtsc] Suite complete.")
    print(f"[sigtsc] Summary CSV: {paths.summary_csv}")