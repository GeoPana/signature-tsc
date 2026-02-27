from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# =========================
# IO helpers
# =========================

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
            row_out: Dict[str, Any] = {}
            for c in columns:
                val = r.get(c, None)
                if isinstance(val, dict):
                    val = json.dumps(val, ensure_ascii=False, sort_keys=True)
                row_out[c] = val
            w.writerow(row_out)


# =========================
# Transform parsing + canonicalization
# =========================

def _parse_transform_tag(tag: str | None) -> Tuple[str | None, float | None, Dict[str, float]]:
    """
    Parse dataset transform tag(s).

    Examples:
      "warp=0.2" -> ("warp", 0.2, {"warp": 0.2})
      "shift=0.05" -> ("shift", 0.05, {"shift": 0.05})
      "warp=0.2,shift=0.1" -> ("shift+warp", None, {"warp": 0.2, "shift": 0.1})
      None -> (None, None, {})

    Returns:
      transform_type: "warp", "shift", "shift+warp", etc (sorted keys)
      severity: float if exactly one transform present, else None
      params: dict of parsed transform->value
    """
    if tag is None:
        return None, None, {}

    tag = str(tag).strip()
    if not tag:
        return None, None, {}

    parts = [p.strip() for p in tag.split(",") if p.strip()]
    params: Dict[str, float] = {}

    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        try:
            params[k] = float(v)
        except Exception:
            continue

    if not params:
        return "unknown", None, {}

    keys = sorted(params.keys())
    transform_type = "+".join(keys)
    severity = params[keys[0]] if len(keys) == 1 else None
    return transform_type, severity, params


def _canonical_tag(tag: str | None) -> str | None:
    """
    Canonicalize transform tag strings so:
      "shift=0.10" -> "shift=0.1"
      "warp=0.20,shift=0.05" -> "shift=0.05,warp=0.2" (sorted keys, normalized floats)
    """
    if tag is None:
        return None
    tag = str(tag).strip()
    if not tag:
        return None

    _, _, params = _parse_transform_tag(tag)
    if not params:
        return tag

    items: List[str] = []
    for k in sorted(params.keys()):
        v = params[k]
        v_str = format(v, "g")  # 0.10->0.1, 1.0->1
        items.append(f"{k}={v_str}")
    return ",".join(items)


def _split_dataset_name(dataset: str | None) -> Tuple[str, str | None]:
    """
    "NATOPS@warp=0.2,shift=0.1" -> ("NATOPS", "shift=0.1,warp=0.2")  (canonical tag)
    "NATOPS" -> ("NATOPS", None)
    """
    if dataset is None:
        return "", None
    dataset = str(dataset).strip()
    if not dataset:
        return "", None
    if "@" not in dataset:
        return dataset, None
    base, tail = dataset.split("@", 1)
    base = base.strip()
    tail = tail.strip()
    tag = tail if tail else None
    tag = _canonical_tag(tag)
    return base, tag


def _canonical_dataset_name(dataset: str | None) -> str | None:
    if dataset is None:
        return None
    base, tag = _split_dataset_name(dataset)
    if not base:
        return None
    if tag is None:
        return base
    return f"{base}@{tag}"


# =========================
# Aggregation
# =========================

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

    Canonicalizes dataset transform tags to prevent "shift=0.10" vs "shift=0.1" duals.
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
        dataset_raw = rec.get("dataset")
        dataset = _canonical_dataset_name(dataset_raw)
        if dataset is None:
            continue

        git_commit = rec.get("git_commit")

        variant = rec.get("variant", None)
        model_type = _safe_get(rec, "model", "type")

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
                "dataset_raw": dataset_raw,
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
        "dataset_raw",
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
    # REPORT (method-level): group by method_model
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

    accs_by_method: Dict[str, List[float]] = {m: [] for m in methods_model}
    for ds in datasets_model:
        for m in methods_model:
            if m in best_by_dataset_model[ds]:
                accs_by_method[m].append(best_by_dataset_model[ds][m])

    mean_acc: Dict[str, float | None] = {}
    for m, accs in accs_by_method.items():
        mean_acc[m] = (sum(accs) / len(accs)) if accs else None

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
    # ROBUSTNESS (variant-level): group by method_variant
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
    expected_methods = set(methods_variant)

    by_base_tag: Dict[Tuple[str, str | None], List[str]] = defaultdict(list)
    for ds in datasets_variant:
        base, tag = _split_dataset_name(ds)
        by_base_tag[(base, tag)].append(ds)

    def _pick_one(names: List[str]) -> str:
        return sorted(names)[0]

    robustness_rows: List[Dict[str, Any]] = []
    winners_rows: List[Dict[str, Any]] = []

    base_datasets = sorted({base for (base, tag) in by_base_tag.keys() if tag is not None})

    for base in base_datasets:
        clean_names = by_base_tag.get((base, None), [])
        if not clean_names:
            continue
        clean_ds = _pick_one(clean_names)

        tags = sorted({tag for (b, tag) in by_base_tag.keys() if b == base and tag is not None})

        for tag in tags:
            trans_names = by_base_tag.get((base, tag), [])
            if not trans_names:
                continue
            transformed_ds = _pick_one(trans_names)

            drops: List[Tuple[str, float]] = []
            transform_type, severity, params = _parse_transform_tag(tag)
            methods_present: set[str] = set()

            for m in methods_variant:
                clean_acc_m = best_by_dataset_variant.get(clean_ds, {}).get(m)
                trans_acc_m = best_by_dataset_variant.get(transformed_ds, {}).get(m)
                if clean_acc_m is None or trans_acc_m is None:
                    continue

                drop = clean_acc_m - trans_acc_m
                drop_clipped = max(0.0, drop)

                robustness_rows.append(
                    {
                        "base_dataset": base,
                        "transform": tag,
                        "transform_type": transform_type,
                        "severity": severity,
                        "method": m,
                        "clean_dataset": clean_ds,
                        "transformed_dataset": transformed_ds,
                        "clean_acc": clean_acc_m,
                        "transformed_acc": trans_acc_m,
                        "drop_clean_minus_transformed": drop,
                        "drop_clipped": drop_clipped,
                        "transform_params": params,
                    }
                )
                drops.append((m, drop_clipped))
                methods_present.add(m)

            if not drops:
                continue

            if methods_present != expected_methods:
                missing = sorted(expected_methods - methods_present)
                extra = sorted(methods_present - expected_methods)
                winners_rows.append(
                    {
                        "base_dataset": base,
                        "transform": tag,
                        "transform_type": transform_type,
                        "severity": severity,
                        "winner_method": "",
                        "winner_drop_clipped": None,
                        "n_methods_compared": len(drops),
                        "transform_params": params,
                        "missing_methods": "|".join(missing),
                        "extra_methods": "|".join(extra),
                    }
                )
                continue

            drops.sort(key=lambda x: x[1])
            best_drop = drops[0][1]
            winners = [m for (m, d) in drops if d == best_drop]

            winners_rows.append(
                {
                    "base_dataset": base,
                    "transform": tag,
                    "transform_type": transform_type,
                    "severity": severity,
                    "winner_method": "|".join(winners),
                    "winner_drop_clipped": best_drop,
                    "n_methods_compared": len(drops),
                    "transform_params": params,
                    "missing_methods": "",
                    "extra_methods": "",
                }
            )

    robustness_cols = [
        "base_dataset",
        "transform",
        "transform_type",
        "severity",
        "method",
        "clean_dataset",
        "transformed_dataset",
        "clean_acc",
        "transformed_acc",
        "drop_clean_minus_transformed",
        "drop_clipped",
        "transform_params",
    ]
    _write_csv(Path(out_robustness_csv), robustness_rows, robustness_cols)

    winners_cols = [
        "base_dataset",
        "transform",
        "transform_type",
        "severity",
        "winner_method",
        "winner_drop_clipped",
        "n_methods_compared",
        "transform_params",
        "missing_methods",
        "extra_methods",
    ]
    _write_csv(Path(out_robustness_winners_csv), winners_rows, winners_cols)

    return out_summary_csv, out_report_csv, out_robustness_csv, out_robustness_winners_csv