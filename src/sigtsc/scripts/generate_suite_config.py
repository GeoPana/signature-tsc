from itertools import product
from pathlib import Path
from datetime import datetime

from sigtsc.utils.io import save_yaml

# ------------------------------------------------------------
# Edit only this section for different config generations
# ------------------------------------------------------------
MODE = "global"  # "global" or "per_dataset"
TAG = "dataset_sweeps"   # change manually if needed

SEED = 42
RESULTS_DIR = "results/suites"

BASE_DATASETS = [
    "BasicMotions",
    "ArticularyWordRecognition",
    "CharacterTrajectories",
    "NATOPS",
    "Epilepsy",
]

# Transform dataset generation
INCLUDE_CLEAN = True
WARP_LEVELS = [0.10, 0.20, 0.40]      # set [] to disable
SHIFT_LEVELS = [0.05, 0.10, 0.20]     # set [] to disable
COMBINED_WARP_SHIFT = [(0.20, 0.10)]  # list of (warp, shift), or []

# Feature sweep
LEVELS = [3]
WITH_TIME_OPTIONS = [False, True]
LEAD_LAG_OPTIONS = [False, True]

# name, value
WINDOW_OPTIONS = [
    ("global", None),
    ("w125_250_1000", [0.125, 0.25, 1.0]),
    ("w050_100_200", [0.05, 0.10, 0.20]),
]

COMMON_FEATURES = {
    "step_frac": 0.5,
    "min_window": 12,
    "pool": ["mean", "max"],
}

# Model sweeps
# Each model has:
# - type
# - base_params (always included)
# - params_grid (cartesian-expanded into variants)
MODEL_SPECS = [
    {
        "name": "logreg",
        "type": "logreg",
        "base_params": {},
        "params_grid": {
            "C": [0.3, 1.0, 3.0],
            "max_iter": [10000],
            "solver": ["saga"],
            "n_jobs": [1],
        },
    },
    {
        "name": "linearsvc",
        "type": "linearsvc",
        "base_params": {},
        "params_grid": {
            "C": [0.3, 1.0, 3.0],
            "max_iter": [10000],
            "dual": ["auto"],
        },
    },
    {
        "name": "mlp",
        "type": "mlp",
        "base_params": {},
        "params_grid": {
            "hidden_layer_sizes": [[256, 128], [128, 64]],
            "alpha": [1e-4, 1e-3],
            "max_iter": [400],
            "random_state": [42],
        },
    },
    {
        "name": "minirocket",
        "type": "minirocket",
        "base_params": {},
        "params_grid": {
            "n_kernels": [10000],
            "max_dilations_per_kernel": [32],
            "n_jobs": [1],         # good when suite workers > 1
            "random_state": [42],
        },
    },
]

PLOTTING = {
    "enabled": True,
    # "out_dir": "results/custom_plots",  # optional
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def grid_to_param_dicts(grid: dict):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


def val_for_name(v):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return "-".join(str(x) for x in v)
    return str(v)


def short_param_tag(params: dict, keep_keys: list[str]):
    parts = []
    for k in keep_keys:
        if k in params:
            parts.append(f"{k}{val_for_name(params[k])}")
    return "_".join(parts) if parts else "default"


def feature_variant_name(level, with_time, lead_lag, w_name):
    return f"L{level}_{'time' if with_time else 'notime'}_{'ll' if lead_lag else 'noll'}_{w_name}"


def dataset_family(base_dataset: str):
    out = []
    if INCLUDE_CLEAN:
        out.append(base_dataset)
    for w in WARP_LEVELS:
        out.append(f"{base_dataset}@warp={w:.2f}")
    for s in SHIFT_LEVELS:
        out.append(f"{base_dataset}@shift={s:.2f}")
    for w, s in COMBINED_WARP_SHIFT:
        out.append(f"{base_dataset}@warp={w:.2f},shift={s:.2f}")
    return out


def build_all_datasets():
    ds = []
    for b in BASE_DATASETS:
        ds.extend(dataset_family(b))
    return ds


def build_variants():
    variants = []
    for model in MODEL_SPECS:
        m_name = model["name"]
        m_type = model["type"]
        base_params = dict(model.get("base_params", {}))
        params_grid = model.get("params_grid", {})

        for p in grid_to_param_dicts(params_grid):
            model_params = {**base_params, **p}

            if m_type == "minirocket":
                ptag = short_param_tag(model_params, ["n_kernels", "max_dilations_per_kernel"])
                variants.append(
                    {"name": f"{m_name}_{ptag}", "model": {"type": m_type, "params": model_params}}
                )
                continue

            ptag = short_param_tag(
                model_params, ["C", "solver", "alpha", "hidden_layer_sizes", "dual"]
            )

            for level, with_time, lead_lag, (w_name, w_val) in product(
                LEVELS, WITH_TIME_OPTIONS, LEAD_LAG_OPTIONS, WINDOW_OPTIONS
            ):
                feats = {
                    "level": level,
                    "with_time": with_time,
                    "lead_lag": lead_lag,
                    "window_fracs": w_val,
                    **COMMON_FEATURES,
                }
                fname = feature_variant_name(level, with_time, lead_lag, w_name)
                variants.append(
                    {
                        "name": f"{m_name}_{ptag}_{fname}",
                        "features": feats,
                        "model": {"type": m_type, "params": model_params},
                    }
                )
    return variants


def base_dataset_names(datasets: list[str]) -> list[str]:
    return list(dict.fromkeys(d.split("@", 1)[0] for d in datasets))


def build_cfg(suite_name: str, datasets: list[str], variants: list[dict]):
    return {
        "seed": SEED,
        "results_dir": RESULTS_DIR,
        "suite": {
            "name": suite_name,
            "datasets": datasets,
            "variants": variants,
            "plotting": {
                "enabled": PLOTTING["enabled"],
                "datasets": base_dataset_names(datasets),
            },
        },
    }

# ============================================================
# Main
# ============================================================
def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    variants = build_variants()

    if MODE == "global":
        datasets = build_all_datasets()
        suite_name = f"sig_{TAG}_{ts}"
        out = Path(f"configs/{suite_name}.yaml")
        cfg = build_cfg(suite_name, datasets, variants)
        save_yaml(out, cfg)
        print(f"[sigtsc] wrote {out}")
        print(f"[sigtsc] variants={len(variants)} datasets={len(datasets)} total_runs={len(variants)*len(datasets)}")
        return

    if MODE == "per_dataset":
        out_dir = Path("configs/per_dataset")
        written = 0
        for base in BASE_DATASETS:
            datasets = dataset_family(base)
            suite_name = f"sig_{TAG}_{base}_{ts}"
            out = out_dir / f"{suite_name}.yaml"
            cfg = build_cfg(suite_name, datasets, variants)
            save_yaml(out, cfg)
            written += 1
            print(f"[sigtsc] wrote {out}")
            print(f"[sigtsc]  variants={len(variants)} datasets={len(datasets)} total_runs={len(variants)*len(datasets)}")
        print(f"[sigtsc] wrote {written} per-dataset configs in {out_dir}")
        return

    raise ValueError(f"Unknown MODE={MODE!r}. Use 'global' or 'per_dataset'.")


if __name__ == "__main__":
    main()
