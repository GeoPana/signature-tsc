# sig-tsc

Time Series Classification using Signature and Log-Signature Features  
(iisignature-based research framework)

---

## Overview

`sig-tsc` is a reproducible research framework for extracting signature and
log-signature features from multivariate time series and evaluating them on
standard Time Series Classification (TSC) benchmarks.

The project focuses on:

- Multivariate time series
- Phase-shift robustness
- Time-warp invariance vs sensitivity
- Controlled benchmarking against UCR/UEA datasets
- Clean, reproducible experiment pipelines

The implementation uses:

- `iisignature` for fast log-signature computation
- `aeon` for dataset loading (official UCR/UEA splits)
- `scikit-learn` for classification and evaluation

---

## Motivation

The signature transform provides an algebraic and theoretically grounded
feature map for paths. This project investigates its practical effectiveness
as a feature extractor for time series classification.

Key research questions:

- Can global log-signatures compete with modern TSC methods?
- Does multiscale windowing improve phase-shift robustness?
- When is including time as a channel beneficial?
- How does truncation level affect performance vs dimensionality?

---

## Project Structure

```
sig-tsc/
├─ src/sigtsc/
│  ├─ data/           # Dataset loading and preprocessing
│  ├─ features/       # Signature / log-signature extraction
│  ├─ models/         # Classifiers and evaluation
│  ├─ experiments/    # Experiment runner
│  └─ utils/          # Logging, reproducibility utilities
├─ configs/           # YAML experiment configurations
├─ results/           # Saved experiment outputs (not versioned)
├─ tests/             # Unit tests
├─ environment.yml    # Conda environment specification
├─ pyproject.toml     # Package definition
└─ README.md
```

---

## Installation

### 1) Create the Conda environment

```bash
conda env create -f environment.yml
conda activate sig-tsc
```

### 2) Install the package in editable mode

```bash
pip install -e .
```

---

## Running an Experiment

To run the default experiment:

```bash
sigtsc run --config configs/default.yaml
```

This will:

1. Load the specified dataset using official train/test splits
2. Extract log-signature features
3. Train the selected classifier
4. Evaluate performance
5. Save metrics and a configuration snapshot to:

```
results/runs/<timestamp>/
```

---

## Configuration System

Experiments are controlled via YAML configuration files in `configs/`.

Example:

```yaml
dataset:
  name: BasicMotions

features:
  type: logsig
  level: 3
  with_time: false
  window_fracs: []
  pool: ["mean", "max"]

model:
  type: logreg
  params:
    C: 1.0
    max_iter: 5000
```

This design ensures:

- Clean separation between experiment settings and implementation
- Reproducible experiment runs
- Easy hyperparameter exploration

---

### Generating Suite Configs

The helper script `src/sigtsc/scripts/generate_suite_config.py` generates
timestamped YAML suite configs for the current dataset, transform, feature,
and model sweeps.

Before running it, edit the small settings block at the top of the script:

```python
MODE = "global"       # "global" or "per_dataset"
TAG = "dataset_sweeps"
```

Run the generator with:

```bash
python src/sigtsc/scripts/generate_suite_config.py
```

`MODE = "global"` writes one config:

```text
configs/sig_dataset_sweeps_<timestamp>.yaml
```

The current global sweep contains:

- 5 base datasets
- 40 suite datasets after clean, warp, shift, and combined warp/shift variants
- 121 model/feature variants
- 4840 total runs

`MODE = "per_dataset"` writes one config per base dataset:

```text
configs/per_dataset/sig_dataset_sweeps_<dataset>_<timestamp>.yaml
```

Each per-dataset config contains:

- 1 base dataset family
- 8 suite datasets: clean, 3 warp levels, 3 shift levels, and 1 combined warp/shift setting
- 121 model/feature variants
- 968 total runs

The generated suite datasets include transformed names such as:

```text
BasicMotions
BasicMotions@warp=0.10
BasicMotions@shift=0.05
BasicMotions@warp=0.20,shift=0.10
```

The generated variants cover:

- Signature models: `logreg`, `linearsvc`, `mlp`
- Baseline model: `minirocket`
- Signature feature options: `level`, `with_time`, `lead_lag`, and `window_fracs`
- Model parameter grids, including `C`, `alpha`, hidden layer sizes, and MiniROCKET kernel settings

Generated configs enable suite-level plotting by default and use deduplicated
base dataset names for plot filters, so transformed warp/shift variants are
still included in plots without repeated dataset entries.

Run a generated suite with:

```bash
sigtsc run --config configs/sig_dataset_sweeps_<timestamp>.yaml
```

For per-dataset configs:

```bash
sigtsc run --config configs/per_dataset/sig_dataset_sweeps_BasicMotions_<timestamp>.yaml
```

Suite runs can also use parallel workers:

```bash
sigtsc run --config <suite_config>.yaml --workers 4
```

---

## Datasets

Datasets are loaded via `aeon` and use official UCR/UEA splits.

Example datasets:

- BasicMotions
- ArticularyWordRecognition
- CharacterTrajectories
- NATOPS
- Epilepsy

All comparisons should use the provided train/test splits unless explicitly
performing resampling experiments.

---

## Feature Extraction Notes

Currently supported (or intended as core features):

- Global log-signature features
- Optional time-channel augmentation (`with_time`)
- Optional lead-lag augmentation (`lead_lag`)
- Optional multiscale windowing (`window_fracs`)
- Pooling across windows (mean / max)
- Clean and transformed dataset variants for warp/shift robustness sweeps

Feature dimensionality depends on:

- Number of channels `d` (including optional time channel and lead-lag augmentation)
- Truncation level `m`

Log-signature dimension can be checked via:

```python
import iisignature
iisignature.logsiglength(d, m)
```

---

## Reproducibility

Each experiment stores:

- Configuration snapshot
- Evaluation metrics
- Timestamped run directory

To reproduce a result:

1. Checkout the corresponding git commit
2. Activate the Conda environment
3. Run with the saved config file

This project is structured to support publishable, reproducible research.

---

## Development

Run tests:

```bash
pytest
```

(Optional) If you use pre-commit hooks:

```bash
pre-commit run --all-files
```

---

## Roadmap

Possible extensions:

- Statistical significance testing across datasets
- Additional baseline comparisons
- Broader hyperparameter sweep presets
- CI coverage for generated suite configs and plotting outputs

---

## Core Dependencies

- aeon
- iisignature
- scikit-learn
- numpy
- scipy

---

## Plotting From Aggregated Results and Auto-Plotting via Config

This project now supports:

1. Manual plot generation from aggregated CSV files.
2. Automatic plot generation at the end of a run/suite when enabled in config.

### What gets plotted

The plotting pipeline reads aggregated CSV files (`summary.csv`, `report.csv`, `robustness.csv`) and generates PNGs for:

- Best accuracy heatmap by dataset and method
- Mean method accuracy bar plot
- Signature vs baseline dataset gap plot
- Robustness curves (accuracy drop vs transform severity)
- Parameter sensitivity plots, including:
  - log-signature level
  - `with_time`
  - `lead_lag`
  - number of window scales
  - window fraction sensitivity

### Manual plotting CLI

Generate plots from existing aggregate files:

```bash
sigtsc plot \
  --summary-csv results/summary.csv \
  --report-csv results/report.csv \
  --robustness-csv results/robustness.csv \
  --out-dir results/plots
```

Optional dataset filtering (repeatable):

```bash
sigtsc plot --dataset NATOPS
sigtsc plot --dataset NATOPS --dataset CharacterTrajectories
```

Filter matching behavior:
- Exact dataset name is supported (e.g. `NATOPS@warp=0.2`)
- Base dataset name is also supported (e.g. `NATOPS` matches transformed variants like `NATOPS@warp=...`, `NATOPS@shift=...`)

### Auto-plotting from config

You can enable plotting directly in run or suite config files by adding a `plotting` block:

```yaml
plotting:
  enabled: true
  # optional dataset filter(s)
  # datasets: [NATOPS, CharacterTrajectories]
  # optional custom output directory
  # out_dir: results/custom_plots
```

#### Behavior for single runs (`sigtsc run --config <single_config>.yaml`)

When `plotting.enabled: true`:

1. The single run is executed and saved in its timestamped run directory.
2. Aggregation is performed for that run directory.
3. Plots are generated and saved to:
   - Default: `<run_dir>/plots/`
   - Or `plotting.out_dir` if provided.

#### Behavior for suites (`sigtsc run --config <suite_config>.yaml`)

When `plotting.enabled: true`:

1. All suite experiments are executed.
2. Suite-level aggregation is written to `<suite_dir>/agg/`.
3. Plots are generated from suite aggregate CSVs and saved to:
   - Default: `<suite_dir>/plots/`
   - Or `plotting.out_dir` if provided.

### Recommended usage pattern

For reproducible workflow:

1. Run suite:
```bash
sigtsc run --config configs/suite_stress_grid.yaml
```

2. Let config-driven auto-plotting produce suite-local plots under that suite folder.

3. Optionally rerun manual plotting with specific filters:
```bash
sigtsc plot --summary-csv <suite_dir>/agg/summary.csv --report-csv <suite_dir>/agg/report.csv --robustness-csv <suite_dir>/agg/robustness.csv --dataset NATOPS --out-dir <suite_dir>/plots_natops
```

### Dependencies

Plotting requires:

- `matplotlib`
- `seaborn`
- `pandas`

Ensure these are present in both:
- `pyproject.toml` dependencies
- `environment.yaml` dependencies (for conda-based environment creation)


## License

MIT in `LICENSE`.

---

## Author

Georgios Panagiotopoulos
2026
