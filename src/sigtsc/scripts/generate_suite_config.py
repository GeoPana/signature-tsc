from itertools import product
from pathlib import Path
import yaml

from datetime import datetime


# ------------------------------------------------------------
# Edit only this section for different config generations
# ------------------------------------------------------------
SEED = 42
RESULTS_DIR = "results/suites"

# Date and time
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TAG = "test2"  # change manually if needed

SUITE_NAME = f"sig_models_grid_{TAG}_{ts}"
OUT_PATH = f"configs/{SUITE_NAME}.yaml"

BASE_DATASETS = [
    "BasicMotions",
    "ArticularyWordRecognition",
    "CharacterTrajectories",
    "NATOPS",
    "Epilepsy",
]

INCLUDE_CLEAN = True
WARP_LEVELS = [0.10, 0.20, 0.40]      # set [] to disable
SHIFT_LEVELS = [0.05, 0.10, 0.20]     # set [] to disable

DATASETS = []
if INCLUDE_CLEAN:
    DATASETS.extend(BASE_DATASETS)

for ds in BASE_DATASETS:
    for w in WARP_LEVELS:
        DATASETS.append(f"{ds}@warp={w:.2f}")
    for s in SHIFT_LEVELS:
        DATASETS.append(f"{ds}@shift={s:.2f}")

COMBINED_WARP_SHIFT = [(0.20, 0.10)]  # list of (warp, shift), or []
for ds in BASE_DATASETS:
    for w, s in COMBINED_WARP_SHIFT:
        DATASETS.append(f"{ds}@warp={w:.2f},shift={s:.2f}")

# name, value
WINDOW_OPTIONS = [
    ("global", None),
    ("w125_250_1000", [0.125, 0.25, 1.0]),
    ("w050_100_200", [0.05, 0.10, 0.20]),
]

LEVELS = [3]
WITH_TIME_OPTIONS = [False, True]
LEAD_LAG_OPTIONS = [False, True]

MODELS = [
    {
        "name": "minirocket",
        "cfg": {
            "type": "minirocket",
            "params": {
                "n_kernels": 10000,
                "max_dilations_per_kernel": 32,
                "n_jobs": -1,
                "random_state": 42,
            },
        },
    },
    {
        "name": "logreg",
        "cfg": {
            "type": "logreg",
            "params": {"C": 1.0, "max_iter": 5000},
        },
    },
    {
        "name": "linearsvc",
        "cfg": {
            "type": "linearsvc",
            "params": {"C": 1.0, "max_iter": 10000},
        },
    },
    {
        "name": "mlp",
        "cfg": {
            "type": "mlp",
            "params": {
                "hidden_layer_sizes": [256, 128],
                "alpha": 1e-4,
                "max_iter": 400,
                "random_state": 42,
            },
        },
    },
]

PLOTTING = {
    "enabled": True,
    "datasets": DATASETS,
    # "out_dir": "results/custom_plots",  # optional
}

# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------
variants = []

for model in MODELS:
    model_name = model["name"]
    model_cfg = model["cfg"]

    if model_cfg["type"] == "minirocket":
        variants.append({"name": model_name, "model": model_cfg})
        continue

    for level, with_time, lead_lag, (w_name, w_val) in product(
        LEVELS,
        WITH_TIME_OPTIONS,
        LEAD_LAG_OPTIONS,
        WINDOW_OPTIONS,
    ):
        variant_name = (
            f"{model_name}_"
            f"L{level}_"
            f"{'time' if with_time else 'notime'}_"
            f"{'ll' if lead_lag else 'noll'}_"
            f"{w_name}"
        )

        features = {
            "level": level,
            "with_time": with_time,
            "lead_lag": lead_lag,
            "window_fracs": w_val,
            "step_frac": 0.5,
            "min_window": 12,
            "pool": ["mean", "max"],
        }

        variants.append(
            {
                "name": variant_name,
                "features": features,
                "model": model_cfg,
            }
        )

cfg = {
    "seed": SEED,
    "results_dir": RESULTS_DIR,
    "suite": {
        "name": SUITE_NAME,
        "datasets": DATASETS,
        "variants": variants,
        "plotting": PLOTTING,
    },
}

out = Path(OUT_PATH)
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[sigtsc] wrote {out}")
print(f"[sigtsc] variants: {len(variants)}")
print(f"[sigtsc] datasets: {len(DATASETS)}")
print(f"[sigtsc] estimated runs: {len(variants) * len(DATASETS)}")
