from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from collections import defaultdict


def _iter_metrics_files(root: Path) -> Iterable[Path]:
    yield from root.glob("**/metrics.json")


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, None) for c in columns})


def aggregate_results(
    results_root: str = "results",
    out_summary_csv: str = "results/summary.csv",
    out_report_csv: str = "results/report.csv",
    signature_method_name: str = "logreg",
    baseline_method_name: str = "minirocket",
) -> Tuple[str, str]:
    """
    Scans results_root/**/metrics.json and writes:
      1) out_summary_csv: one row per run
      2) out_report_csv: mean accuracy per method, average rank, and per-dataset gaps
    Returns: (summary_csv_path, report_csv_path)
    """
    root = Path(results_root)
    summary_rows: List[Dict[str, Any]] = []

    # ------------- Build summary rows -------------
    for p in _iter_metrics_files(root):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        timestamp = p.parent.name
        dataset = rec.get("dataset")
        git_commit = rec.get("git_commit")

        model_type = _safe_get(rec, "model", "type")
        accuracy = _to_float(_safe_get(rec, "metrics", "accuracy"))

        features_type = _safe_get(rec, "features", "type")
        level = _safe_get(rec, "features", "level")
        with_time = _safe_get(rec, "features", "with_time")
        dim = _safe_get(rec, "features", "dim")
        window_fracs = _safe_get(rec, "features", "window_fracs")
        pool = _safe_get(rec, "features", "pool")

        summary_rows.append(
            {
                "dataset": dataset,
                "method": model_type,
                "features_type": features_type,
                "level": level,
                "with_time": with_time,
                "window_fracs": window_fracs,
                "pool": pool,
                "dim": dim,
                "accuracy": accuracy,
                "timestamp": timestamp,
                "git_commit": git_commit,
                "metrics_path": str(p),
            }
        )

    summary_cols = [
        "dataset",
        "method",
        "features_type",
        "level",
        "with_time",
        "window_fracs",
        "pool",
        "dim",
        "accuracy",
        "timestamp",
        "git_commit",
        "metrics_path",
    ]
    _write_csv(Path(out_summary_csv), summary_rows, summary_cols)

    # ------------- Compute report stats -------------
    # We compute:
    # - mean accuracy per method
    # - average rank per method (within each dataset across methods)
    # - per-dataset gaps: signature_method_name vs baseline_method_name

    # Group by dataset -> method -> best accuracy (in case multiple runs exist)
    best_by_dataset_method: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in summary_rows:
        ds = r.get("dataset")
        m = r.get("method")
        acc = r.get("accuracy")
        if ds is None or m is None or acc is None:
            continue
        prev = best_by_dataset_method[ds].get(m)
        if prev is None or acc > prev:
            best_by_dataset_method[ds][m] = acc

    datasets = sorted(best_by_dataset_method.keys())
    methods = sorted({m for ds in datasets for m in best_by_dataset_method[ds].keys()})

    # Mean accuracy per method across datasets where it exists
    accs_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    for ds in datasets:
        for m in methods:
            if m in best_by_dataset_method[ds]:
                accs_by_method[m].append(best_by_dataset_method[ds][m])

    mean_acc: Dict[str, float | None] = {}
    for m, accs in accs_by_method.items():
        mean_acc[m] = (sum(accs) / len(accs)) if accs else None

    # Average rank per method (1 = best) per dataset.
    # Ties: average rank among tied positions.
    ranks_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    for ds in datasets:
        items = list(best_by_dataset_method[ds].items())  # (method, acc)
        # sort desc by accuracy
        items.sort(key=lambda x: x[1], reverse=True)

        # assign ranks with tie handling
        i = 0
        while i < len(items):
            j = i
            while j < len(items) and items[j][1] == items[i][1]:
                j += 1
            # items i..j-1 are tied; ranks are i+1..j
            avg_rank = ( (i + 1) + j ) / 2.0
            for k in range(i, j):
                ranks_by_method[items[k][0]].append(avg_rank)
            i = j

    avg_rank: Dict[str, float | None] = {}
    for m, rs in ranks_by_method.items():
        avg_rank[m] = (sum(rs) / len(rs)) if rs else None

    # Per-dataset gaps: signature - baseline
    gaps_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        sig = best_by_dataset_method[ds].get(signature_method_name)
        base = best_by_dataset_method[ds].get(baseline_method_name)
        gap = (sig - base) if (sig is not None and base is not None) else None
        gaps_rows.append(
            {
                "dataset": ds,
                f"{signature_method_name}_acc": sig,
                f"{baseline_method_name}_acc": base,
                "gap_sig_minus_base": gap,
            }
        )

    # Compose report CSV as a single table with sections (type column)
    report_rows: List[Dict[str, Any]] = []

    # Section 1: method summary
    for m in methods:
        report_rows.append(
            {
                "section": "method_summary",
                "method": m,
                "mean_accuracy": mean_acc[m],
                "avg_rank": avg_rank[m],
            }
        )

    # Section 2: dataset gaps
    for r in gaps_rows:
        report_rows.append(
            {
                "section": "dataset_gap",
                "dataset": r["dataset"],
                "signature_method": signature_method_name,
                "baseline_method": baseline_method_name,
                "signature_acc": r[f"{signature_method_name}_acc"],
                "baseline_acc": r[f"{baseline_method_name}_acc"],
                "gap_sig_minus_base": r["gap_sig_minus_base"],
            }
        )

    report_cols = [
        "section",
        "method",
        "mean_accuracy",
        "avg_rank",
        "dataset",
        "signature_method",
        "baseline_method",
        "signature_acc",
        "baseline_acc",
        "gap_sig_minus_base",
    ]
    _write_csv(Path(out_report_csv), report_rows, report_cols)

    return out_summary_csv, out_report_csv