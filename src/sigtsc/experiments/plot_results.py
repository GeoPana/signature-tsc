from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)


def _base_dataset_name(dataset: str) -> str:
    return dataset.split("@", 1)[0] if "@" in dataset else dataset


def _parse_list_like(x: Any) -> list[Any]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
    except Exception:
        return []
    return v if isinstance(v, list) else []


def _coerce_bool(x: Any) -> Any:
    if pd.isna(x):
        return x
    s = str(x).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return x


def _matches_dataset(name: str, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    name_s = str(name).strip()
    base = _base_dataset_name(name_s)
    fset = {str(f).strip() for f in filters if str(f).strip()}
    return name_s in fset or base in fset


def _filter_frames(
    summary_df: pd.DataFrame,
    report_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    datasets: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = list(datasets or [])
    if not ds:
        return summary_df, report_df, robust_df

    s = summary_df.copy()
    r = report_df.copy()
    rb = robust_df.copy()

    if "dataset" in s.columns:
        s = s[s["dataset"].map(lambda x: _matches_dataset(x, ds))]
    if "dataset" in r.columns:
        r = r[r["dataset"].map(lambda x: _matches_dataset(x, ds))]
    if "base_dataset" in rb.columns:
        rb = rb[rb["base_dataset"].map(lambda x: _matches_dataset(x, ds))]

    return s, r, rb


def _transform_type_from_dataset_name(dataset: str) -> str:
    """
    Map dataset name to transform bucket:
      - no '@'            -> clean
      - '@warp=...'       -> warp
      - '@shift=...'      -> shift
      - '@warp=...,shift' -> shift+warp
      - unknown tags      -> other
    """
    s = str(dataset).strip()
    if "@" not in s:
        return "clean"

    _, tail = s.split("@", 1)
    keys: list[str] = []
    for tok in tail.split(","):
        tok = tok.strip()
        if "=" not in tok:
            continue
        k = tok.split("=", 1)[0].strip().lower()
        if k:
            keys.append(k)

    if not keys:
        return "other"
    keys = sorted(set(keys))
    if keys == ["shift"]:
        return "shift"
    if keys == ["warp"]:
        return "warp"
    return "+".join(keys)


def _wrap_vals(vals: Sequence[Any], width: int = 18) -> list[str]:
    out = []
    for v in vals:
        s = str(v)
        out.append("\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s)
    return out


def _finalize(ax: plt.Axes, rotate_x: int = 0, legend_out: bool = False) -> None:
    if rotate_x:
        for t in ax.get_xticklabels():
            t.set_rotation(rotate_x)
            t.set_ha("right")
    if legend_out and ax.get_legend() is not None:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            frameon=False,
            title=ax.get_legend().get_title().get_text() if ax.get_legend().get_title() else None,
        )
    ax.figure.tight_layout()


def _save_close(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Core transformations
# -----------------------------------------------------------------------------
def _best_per_dataset_method(summary_df: pd.DataFrame) -> pd.DataFrame:
    req = {"dataset", "method_model", "accuracy"}
    if not req.issubset(summary_df.columns):
        missing = sorted(req - set(summary_df.columns))
        raise ValueError(f"summary.csv missing required columns: {missing}")

    df = summary_df.dropna(subset=["dataset", "method_model", "accuracy"]).copy()
    if df.empty:
        return df

    idx = df.groupby(["dataset", "method_model"], as_index=False)["accuracy"].idxmax()
    out = df.loc[idx["accuracy"].values].copy()
    return out.sort_values(["dataset", "method_model"])


# -----------------------------------------------------------------------------
# Plot builders
# -----------------------------------------------------------------------------
def _plot_best_method_heatmap(best_df: pd.DataFrame, out_path: Path) -> None:
    if best_df.empty:
        return

    pivot = best_df.pivot(index="dataset", columns="method_model", values="accuracy")
    if pivot.empty:
        return

    n_rows, n_cols = pivot.shape
    fig_w = max(8.0, 1.2 * n_cols + 2.0)
    fig_h = max(5.0, 0.42 * n_rows + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    annot = n_rows <= 18 and n_cols <= 8  # prevent text clutter on large matrices
    sns.heatmap(
        pivot,
        annot=annot,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "accuracy"},
        ax=ax,
    )

    ax.set_title("Best Accuracy Per Dataset/Method")
    ax.set_xlabel("method")
    ax.set_ylabel("dataset")
    ax.set_xticklabels(_wrap_vals(pivot.columns, width=14))
    ax.set_yticklabels(_wrap_vals(pivot.index, width=26), rotation=0)
    _finalize(ax, rotate_x=30)
    _save_close(fig, out_path)

def _plot_method_mean_bar(best_df: pd.DataFrame, out_path: Path) -> None:
    if best_df.empty:
        return

    means = (
        best_df.groupby("method_model", as_index=False)["accuracy"]
        .mean()
        .sort_values("accuracy", ascending=False)
    )
    if means.empty:
        return

    fig_w = max(7.5, 1.0 * len(means) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.6))
    sns.barplot(data=means, x="method_model", y="accuracy", ax=ax)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean accuracy")
    ax.set_xlabel("method")
    ax.set_title("Method Mean Accuracy (Best Per Dataset)")
    
    xlabels = _wrap_vals(means["method_model"].tolist(), width=14)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)


    for i, v in enumerate(means["accuracy"].tolist()):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    _finalize(ax, rotate_x=25)
    _save_close(fig, out_path)


def _plot_report_gaps(report_df: pd.DataFrame, out_path: Path) -> None:
    req = {"section", "dataset", "gap_sig_minus_base"}
    if not req.issubset(report_df.columns):
        return

    df = report_df[report_df["section"] == "dataset_gap"].copy()
    df = df.dropna(subset=["dataset", "gap_sig_minus_base"])
    if df.empty:
        return

    idx = df.groupby("dataset", as_index=False)["gap_sig_minus_base"].idxmax()
    df = df.loc[idx["gap_sig_minus_base"].values].sort_values("gap_sig_minus_base")

    fig_h = max(4.5, 0.32 * len(df) + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in df["gap_sig_minus_base"]]
    ax.barh(_wrap_vals(df["dataset"].tolist(), width=28), df["gap_sig_minus_base"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("gap (signature - baseline)")
    ax.set_title("Dataset Gap: Signature vs Baseline")
    _finalize(ax)
    _save_close(fig, out_path)


def _plot_robustness_curves(robust_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    req = {"base_dataset", "transform_type", "severity", "method", "drop_clipped"}
    if not req.issubset(robust_df.columns):
        return []

    df = robust_df.dropna(
        subset=["base_dataset", "transform_type", "severity", "method", "drop_clipped"]
    ).copy()
    if df.empty:
        return []

    grouped = (
        df.groupby(["base_dataset", "transform_type", "severity", "method"], as_index=False)["drop_clipped"]
        .mean()
        .sort_values(["base_dataset", "transform_type", "method", "severity"])
    )

    out_paths: list[Path] = []
    for (base_dataset, transform_type), g in grouped.groupby(["base_dataset", "transform_type"]):
        n_methods = g["method"].nunique()
        fig_w = max(8.0, 1.2 * n_methods + 4.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.8))
        sns.lineplot(
            data=g,
            x="severity",
            y="drop_clipped",
            hue="method",
            marker="o",
            linewidth=2,
            ax=ax,
        )

        ax.set_xlabel("severity")
        ax.set_ylabel("accuracy drop (clipped)")
        ax.set_title(f"Robustness: {base_dataset} | {transform_type}")
        _finalize(ax, legend_out=(n_methods > 4))

        safe_base = str(base_dataset).replace("/", "_").replace("\\", "_")
        safe_tf = str(transform_type).replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"robustness_{safe_base}_{safe_tf}.png"
        _save_close(fig, out_path)
        out_paths.append(out_path)

    return out_paths


def _plot_param_sweeps(summary_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    req = {"dataset", "accuracy", "level", "with_time", "lead_lag", "window_fracs", "method_variant"}
    if not req.issubset(summary_df.columns):
        return []

    df = summary_df.dropna(subset=["dataset", "accuracy"]).copy()
    if df.empty:
        return []

    df["with_time"] = df["with_time"].map(_coerce_bool)
    df["lead_lag"] = df["lead_lag"].map(_coerce_bool)
    df["window_fracs_parsed"] = df["window_fracs"].map(_parse_list_like)
    df["n_scales"] = df["window_fracs_parsed"].map(len)
    df["base_dataset"] = df["dataset"].astype(str).map(_base_dataset_name)

    out_paths: list[Path] = []

    # 1) level sweep
    lvl = df.dropna(subset=["level"])
    lvl_agg = (
        lvl.groupby(["base_dataset", "level"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["base_dataset", "level"])
    )
    for base, g in lvl_agg.groupby("base_dataset"):
        if g["level"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        sns.lineplot(data=g, x="level", y="accuracy", marker="o", linewidth=2, ax=ax)
        ax.set_title(f"Parameter Sweep (level): {base}")
        ax.set_ylabel("mean accuracy")
        _finalize(ax)
        out_path = out_dir / f"param_level_{base}.png"
        _save_close(fig, out_path)
        out_paths.append(out_path)

    # 2) with_time / lead_lag
    for col in ["with_time", "lead_lag"]:
        gg = df[df[col].isin([True, False])]
        agg = (
            gg.groupby(["base_dataset", col], as_index=False)["accuracy"]
            .mean()
            .sort_values(["base_dataset", col])
        )
        for base, g in agg.groupby("base_dataset"):
            if g[col].nunique() < 2:
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            sns.barplot(data=g, x=col, y="accuracy", ax=ax)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Parameter Sweep ({col}): {base}")
            ax.set_ylabel("mean accuracy")
            _finalize(ax)
            out_path = out_dir / f"param_{col}_{base}.png"
            _save_close(fig, out_path)
            out_paths.append(out_path)

    # 3) n_scales
    wf = df[df["n_scales"] > 0]
    agg = (
        wf.groupby(["base_dataset", "n_scales"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["base_dataset", "n_scales"])
    )
    for base, g in agg.groupby("base_dataset"):
        if g["n_scales"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        sns.lineplot(data=g, x="n_scales", y="accuracy", marker="o", linewidth=2, ax=ax)
        ax.set_title(f"Parameter Sweep (n_scales): {base}")
        ax.set_ylabel("mean accuracy")
        _finalize(ax)
        out_path = out_dir / f"param_n_scales_{base}.png"
        _save_close(fig, out_path)
        out_paths.append(out_path)

    # 4) window-fracs combo comparison
    #    - overall mean (across all runs)
    #    - split means by transform type (clean/shift/warp/...)
    def _combo_label(xs: list[Any]) -> str:
        if not xs:
            return "global(None)"
        vals = []
        for x in xs:
            try:
                vals.append(f"{float(x):g}")
            except Exception:
                vals.append(str(x))
        return "[" + ", ".join(vals) + "]"

    df_combo = df.copy()
    df_combo["window_combo"] = df_combo["window_fracs_parsed"].map(_combo_label)
    df_combo["condition"] = df_combo["dataset"].astype(str).map(_transform_type_from_dataset_name)

    # Overall means
    overall = (
        df_combo.groupby(["base_dataset", "window_combo"], as_index=False)["accuracy"]
        .mean()
        .assign(condition="overall")
    )

    # Condition-specific means
    by_condition = (
        df_combo.groupby(["base_dataset", "window_combo", "condition"], as_index=False)["accuracy"]
        .mean()
    )

    combo_agg = pd.concat([overall, by_condition], ignore_index=True)

    for base, g in combo_agg.groupby("base_dataset"):
        # choose top combos by overall mean for readability
        top_combos = (
            g[g["condition"] == "overall"]
            .sort_values("accuracy", ascending=False)
            .head(8)["window_combo"]
            .tolist()
        )
        if len(top_combos) < 2:
            continue

        g = g[g["window_combo"].isin(top_combos)].copy()

        # ordered categories for cleaner legend / y-axis
        cond_order = ["overall", "clean", "shift", "warp", "shift+warp", "other"]
        present_conds = [c for c in cond_order if c in set(g["condition"])]
        combo_order = (
            g[g["condition"] == "overall"]
            .sort_values("accuracy", ascending=False)["window_combo"]
            .tolist()
        )

        fig_h = max(5.2, 0.6 * len(combo_order) + 1.8)
        fig, ax = plt.subplots(figsize=(11.0, fig_h))
        sns.barplot(
            data=g,
            x="accuracy",
            y="window_combo",
            hue="condition",
            order=combo_order,
            hue_order=present_conds,
            orient="h",
            ax=ax,
        )

        # annotate each bar with its value
        for p in ax.patches:
            w = p.get_width()
            if pd.isna(w) or w <= 1e-6:   # skip empty bars
                continue
            y = p.get_y() + p.get_height() / 2
            ax.text(w + 0.002, y, f"{w:.3f}", va="center", ha="left", fontsize=8)


        ax.set_title(f"Window-Fracs Combo Comparison (Overall + Transform Split): {base}")
        ax.set_xlabel("mean accuracy")
        ax.set_ylabel("window_fracs setting")
        xmax = g["accuracy"].max() if not g.empty else 1.0
        ax.set_xlim(0.0, min(1.05, xmax + 0.08))

        _finalize(ax, legend_out=True)

        out_path = out_dir / f"param_window_combo_{base}.png"
        _save_close(fig, out_path)
        out_paths.append(out_path)



    return out_paths


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def generate_plots(
    summary_csv: str = "results/summary.csv",
    report_csv: str = "results/report.csv",
    robustness_csv: str = "results/robustness.csv",
    out_dir: str = "results/plots",
    datasets: Sequence[str] | None = None,
) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_df = _read_csv(summary_csv)
    report_df = _read_csv(report_csv)
    robust_df = _read_csv(robustness_csv)

    summary_df, report_df, robust_df = _filter_frames(summary_df, report_df, robust_df, datasets)

    created: list[Path] = []

    best_df = _best_per_dataset_method(summary_df)

    p_heat = out / "best_accuracy_heatmap.png"
    _plot_best_method_heatmap(best_df, p_heat)
    if p_heat.exists():
        created.append(p_heat)

    p_bar = out / "method_mean_accuracy_bar.png"
    _plot_method_mean_bar(best_df, p_bar)
    if p_bar.exists():
        created.append(p_bar)

    p_gap = out / "dataset_gap_signature_vs_baseline.png"
    _plot_report_gaps(report_df, p_gap)
    if p_gap.exists():
        created.append(p_gap)

    created.extend(_plot_robustness_curves(robust_df, out))
    created.extend(_plot_param_sweeps(summary_df, out))

    dedup: list[str] = []
    seen: set[str] = set()
    for p in created:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            dedup.append(sp)
    return dedup


def print_plot_summary(paths: Iterable[str]) -> None:
    paths = list(paths)
    print(f"[sigtsc] plots generated: {len(paths)}")
    for p in paths:
        print(f"[sigtsc] - {p}")
