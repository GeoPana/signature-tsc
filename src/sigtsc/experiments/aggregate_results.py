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


def _split_dataset_name(dataset: str) -> Tuple[str, str | None]:
    """
    "NATOPS@warp=0.2,shift=0.1" -> ("NATOPS", "warp=0.2,shift=0.1")
    "NATOPS" -> ("NATOPS", None)
    """
    if dataset is None:
        return "", None
    if "@" not in dataset:
        return dataset, None
    base, tail = dataset.split("@", 1)
    base = base.strip()
    tail = tail.strip()
    return base, (tail if tail else None)


def aggregate_results(
    results_root: str = "results",
    out_summary_csv: str = "results/summary.csv",
    out_report_csv: str = "results/report.csv",
    out_robustness_csv: str = "results/robustness.csv",
    out_robustness_winners_csv: str = "results/robustness_winners.csv",
    signature_method_name: str = "logreg",
    baseline_method_name: str = "minirocket",
) -> Tuple[str, str, str, str]:
    """
    Writes:
      1) summary.csv: one row per run, includes model_type and variant labels
      2) report.csv: METHOD-LEVEL summary using model_type grouping (logreg vs minirocket)
      3) robustness.csv: VARIANT-LEVEL robustness using variant label (fallback to model_type)
      4) robustness_winners.csv: winners (min drop) per base_dataset+transform (variant-level)

    This keeps your existing report behavior intact, while making robustness
    comparisons meaningful across signature variants.
    """
    root = Path(results_root)
    summary_rows: List[Dict[str, Any]] = []

    # -------------------- Build summary rows --------------------
    for p in _iter_metrics_files(root):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        timestamp = p.parent.name
        dataset = rec.get("dataset")
        git_commit = rec.get("git_commit")

        variant = rec.get("variant", None)
        model_type = _safe_get(rec, "model", "type")

        # labels
        method_model = model_type
        method_variant = variant if variant else model_type

        accuracy = _to_float(_safe_get(rec, "metrics", "accuracy"))

        features_type = _safe_get(rec, "features", "type")
        level = _safe_get(rec, "features", "level")
        with_time = _safe_get(rec, "features", "with_time")
        lead_lag = _safe_get(rec, "features", "lead_lag")
        dim = _safe_get(rec, "features", "dim")
        window_fracs = _safe_get(rec, "features", "window_fracs")
        pool = _safe_get(rec, "features", "pool")

        summary_rows.append(
            {
                "dataset": dataset,
                "method_model": method_model,
                "method_variant": method_variant,
                "model_type": model_type,
                "variant": variant,
                "features_type": features_type,
                "level": level,
                "with_time": with_time,
                "lead_lag": lead_lag,
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
        "method_model",
        "method_variant",
        "model_type",
        "variant",
        "features_type",
        "level",
        "with_time",
        "lead_lag",
        "window_fracs",
        "pool",
        "dim",
        "accuracy",
        "timestamp",
        "git_commit",
        "metrics_path",
    ]
    _write_csv(Path(out_summary_csv), summary_rows, summary_cols)

    # ============================================================
    # REPORT (method-level): group by method_model (logreg vs minirocket)
    # ============================================================
    best_by_dataset_model: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in summary_rows:
        ds = r.get("dataset")
        m = r.get("method_model")
        acc = r.get("accuracy")
        if ds is None or m is None or acc is None:
            continue
        prev = best_by_dataset_model[ds].get(m)
        if prev is None or acc > prev:
            best_by_dataset_model[ds][m] = acc

    datasets_model = sorted(best_by_dataset_model.keys())
    methods_model = sorted({m for ds in datasets_model for m in best_by_dataset_model[ds].keys()})

    # Mean accuracy per method_model
    accs_by_method: Dict[str, List[float]] = {m: [] for m in methods_model}
    for ds in datasets_model:
        for m in methods_model:
            if m in best_by_dataset_model[ds]:
                accs_by_method[m].append(best_by_dataset_model[ds][m])

    mean_acc: Dict[str, float | None] = {}
    for m, accs in accs_by_method.items():
        mean_acc[m] = (sum(accs) / len(accs)) if accs else None

    # Average rank per method_model
    ranks_by_method: Dict[str, List[float]] = {m: [] for m in methods_model}
    for ds in datasets_model:
        items = list(best_by_dataset_model[ds].items())
        items.sort(key=lambda x: x[1], reverse=True)

        i = 0
        while i < len(items):
            j = i
            while j < len(items) and items[j][1] == items[i][1]:
                j += 1
            avg_rank = ((i + 1) + j) / 2.0
            for k in range(i, j):
                ranks_by_method[items[k][0]].append(avg_rank)
            i = j

    avg_rank: Dict[str, float | None] = {}
    for m, rs in ranks_by_method.items():
        avg_rank[m] = (sum(rs) / len(rs)) if rs else None

    # Per-dataset gap: signature_method_name - baseline_method_name (model-level)
    gaps_rows: List[Dict[str, Any]] = []
    for ds in datasets_model:
        sig = best_by_dataset_model[ds].get(signature_method_name)
        base = best_by_dataset_model[ds].get(baseline_method_name)
        gap = (sig - base) if (sig is not None and base is not None) else None
        gaps_rows.append(
            {
                "dataset": ds,
                "signature_method": signature_method_name,
                "baseline_method": baseline_method_name,
                "signature_acc": sig,
                "baseline_acc": base,
                "gap_sig_minus_base": gap,
            }
        )

    report_rows: List[Dict[str, Any]] = []
    for m in methods_model:
        report_rows.append(
            {
                "section": "method_summary",
                "method": m,
                "mean_accuracy": mean_acc[m],
                "avg_rank": avg_rank[m],
            }
        )
    for r in gaps_rows:
        report_rows.append({"section": "dataset_gap", **r})

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

    # ============================================================
    # ROBUSTNESS (variant-level): group by method_variant (sig_* vs minirocket)
    # ============================================================
    best_by_dataset_variant: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in summary_rows:
        ds = r.get("dataset")
        m = r.get("method_variant")
        acc = r.get("accuracy")
        if ds is None or m is None or acc is None:
            continue
        prev = best_by_dataset_variant[ds].get(m)
        if prev is None or acc > prev:
            best_by_dataset_variant[ds][m] = acc

    datasets_variant = sorted(best_by_dataset_variant.keys())
    methods_variant = sorted({m for ds in datasets_variant for m in best_by_dataset_variant[ds].keys()})

    # Index datasets by (base, transform_tag) using dataset names from variant view
    by_base_tag: Dict[Tuple[str, str | None], str] = {}
    for ds in datasets_variant:
        base, tag = _split_dataset_name(ds)
        by_base_tag[(base, tag)] = ds

    robustness_rows: List[Dict[str, Any]] = []
    winners_rows: List[Dict[str, Any]] = []

    base_datasets = sorted({base for (base, tag) in by_base_tag.keys() if tag is not None})

    for base in base_datasets:
        clean_ds = by_base_tag.get((base, None))
        if clean_ds is None:
            continue

        tags = sorted({tag for (b, tag) in by_base_tag.keys() if b == base and tag is not None})

        for tag in tags:
            transformed_ds = by_base_tag.get((base, tag))
            if transformed_ds is None:
                continue

            drops: List[Tuple[str, float]] = []
            for m in methods_variant:
                clean_acc_m = best_by_dataset_variant.get(clean_ds, {}).get(m)
                trans_acc_m = best_by_dataset_variant.get(transformed_ds, {}).get(m)
                if clean_acc_m is None or trans_acc_m is None:
                    continue
                drop = clean_acc_m - trans_acc_m
                robustness_rows.append(
                    {
                        "base_dataset": base,
                        "transform": tag,
                        "method": m,
                        "clean_dataset": clean_ds,
                        "transformed_dataset": transformed_ds,
                        "clean_acc": clean_acc_m,
                        "transformed_acc": trans_acc_m,
                        "drop_clean_minus_transformed": drop,
                    }
                )
                drops.append((m, drop))

            if not drops:
                continue

            drops.sort(key=lambda x: x[1])
            best_drop = drops[0][1]
            winners = [m for (m, d) in drops if d == best_drop]

            winners_rows.append(
                {
                    "base_dataset": base,
                    "transform": tag,
                    "winner_method": "|".join(winners),
                    "winner_drop_clean_minus_transformed": best_drop,
                    "n_methods_compared": len(drops),
                }
            )

    robustness_cols = [
        "base_dataset",
        "transform",
        "method",
        "clean_dataset",
        "transformed_dataset",
        "clean_acc",
        "transformed_acc",
        "drop_clean_minus_transformed",
    ]
    _write_csv(Path(out_robustness_csv), robustness_rows, robustness_cols)

    winners_cols = [
        "base_dataset",
        "transform",
        "winner_method",
        "winner_drop_clean_minus_transformed",
        "n_methods_compared",
    ]
    _write_csv(Path(out_robustness_winners_csv), winners_rows, winners_cols)

    return out_summary_csv, out_report_csv, out_robustness_csv, out_robustness_winners_csv