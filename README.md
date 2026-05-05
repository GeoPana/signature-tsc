# sig-tsc

Time series classification with signature and log-signature features.

`sig-tsc` is a small research framework for benchmarking signature-based
feature extraction on multivariate time series classification tasks. It uses
`iisignature` for signature/log-signature transforms, `aeon` for UCR/UEA
dataset loading, and scikit-learn style classifiers for evaluation.

## Overview

The project supports experiments over:

- Multivariate UCR/UEA time series datasets
- Signature and log-signature feature extraction
- Optional time-channel, basepoint, invisibility-reset, and lead-lag path augmentations
- Coordinate and random projection augmentation streams
- Global, sliding, expanding, and hierarchical dyadic windows
- Window aggregation by concatenation or pooling
- Logistic regression, linear SVM, MLP, and MiniROCKET baselines
- Controlled dataset transforms for warp, shift, and noise robustness checks
- Suite execution, aggregation, and plotting

## Project Structure

```text
sig-tsc/
|- src/sigtsc/
|  |- data/           # Dataset loading and deterministic transforms
|  |- features/       # Signature/log-signature extraction and windowing
|  |- models/         # Classifiers and baselines
|  |- experiments/    # Single-run, suite, aggregation, and plotting code
|  |- scripts/        # Config generation helpers
|  `- utils/          # IO, seed, and git helpers
|- configs/           # YAML experiment and suite configurations
|- data/              # Local dataset cache
|- results/           # Experiment outputs
|- tests/             # Unit tests
|- environment.yaml   # Conda environment specification
|- pyproject.toml     # Package metadata and CLI entry point
`- README.md
```

## Installation

Create the Conda environment and install the package in editable mode:

```bash
conda env create -f environment.yaml
conda activate sig-tsc
pip install -e .
```

The editable install reads `pyproject.toml` and installs the Python package
dependencies, including `iisignature`.

## CLI

Run the default config:

```bash
sigtsc run --config configs/default.yaml
```

Run a suite config. The CLI detects suite configs by the presence of a top-level
`suite:` block:

```bash
sigtsc run --config configs/suite_stress_grid.yaml
```

Run a suite with parallel workers:

```bash
sigtsc run --config configs/suite_stress_grid.yaml --workers 4
```

Aggregate existing `metrics.json` files:

```bash
sigtsc aggregate \
  --results-root results \
  --out-summary results/summary.csv \
  --out-report results/report.csv \
  --out-robustness results/robustness.csv \
  --out-winners results/robustness_winners.csv
```

Generate plots from aggregate CSV files:

```bash
sigtsc plot \
  --summary-csv results/summary.csv \
  --report-csv results/report.csv \
  --robustness-csv results/robustness.csv \
  --out-dir results/plots
```

## Configuration

Experiments are controlled by YAML files in `configs/`.

Minimal signature/log-signature config:

```yaml
seed: 42
results_dir: results/runs

dataset:
  name: BasicMotions

features:
  type: logsig        # logsig or signature
  level: 3
  rescaling: none     # none, pre, or post
  with_time: true
  basepoint: false
  invisibility_reset: false
  lead_lag: false

windowing:
  type: global
  aggregation: concat

model:
  type: logreg
  params:
    C: 1.0
    max_iter: 5000
```

The canonical windowing format is always a top-level `windowing:` block.
Older configs that keep sliding settings under `features.window_fracs` are still
accepted for backward compatibility, but newly generated configs use the
homogeneous format.

`features.rescaling` controls explicit signature-term rescaling. Supported
values are `none`, `pre`, and `post`. This is separate from the existing
per-path z-normalization and from the classifier-level `StandardScaler`.

MiniROCKET uses raw time series and ignores signature feature settings:

```yaml
model:
  type: minirocket
  params:
    n_kernels: 10000
    max_dilations_per_kernel: 32
    n_jobs: -1
    random_state: 42
```

## Augmentations

Signature and log-signature models use a composable augmentation pipeline.
Inputs and outputs use `(time, channels)` orientation.

The current order is:

```text
z-normalize raw value channels
-> coordinate/random projection, if enabled
-> time channel, if with_time=true
-> basepoint or invisibility-reset
-> lead-lag, if lead_lag=true
```

`basepoint` and `invisibility_reset` are mutually exclusive. Coordinate
projection and random projection are also mutually exclusive.

Invisibility-reset:

```yaml
features:
  type: logsig
  level: 3
  with_time: true
  basepoint: false
  invisibility_reset: true
  lead_lag: false
```

Coordinate projections create multiple deterministic streams:

```yaml
augmentation:
  coordinate_projection:
    enabled: true
    mode: pairs        # singletons, pairs, or triplets
```

Random projections create deterministic projected streams from a fixed seed:

```yaml
augmentation:
  random_projection:
    enabled: true
    output_dim: 6
    num_projections: 5
    seed: 42
```

## Windowing

All signature/log-signature window modes use the same `windowing:` block.

Global:

```yaml
windowing:
  type: global
  aggregation: concat
```

Sliding:

```yaml
windowing:
  type: sliding
  window_fracs: [0.125, 0.25, 1.0]
  step_frac: 0.5
  min_window: 8
  aggregation: pool
  pool: ["mean", "max"]
```

Expanding:

```yaml
windowing:
  type: expanding
  num_windows: 4
  min_window: 8
  aggregation: concat
```

Hierarchical dyadic:

```yaml
windowing:
  type: dyadic
  depth: 3
  min_window: 8
  aggregation: concat
```

Supported aggregation modes:

- `concat`: compute one transform per window and concatenate window features in
  deterministic order.
- `pool`: pool window features with operations such as `mean`, `max`, and `std`.
  This preserves the existing pooled sliding-window behavior.

For concat aggregation, all samples in a run must produce the same number of
windows. If variable-length samples make that impossible, use `aggregation: pool`
or switch to a fixed-window-count mode.

## Example Configs

Useful checked-in configs:

- `configs/default.yaml`: log-signature features on `NATOPS@warp=0.20` with
  pooled sliding windows and logistic regression.
- `configs/logsig_expanding.yaml`: log-signature features with expanding
  windows.
- `configs/logsig_dyadic.yaml`: log-signature features with dyadic windows.
- `configs/signature_dyadic.yaml`: full signature features with dyadic windows.
- `configs/logsig_invisibility_reset_dyadic.yaml`: log-signature features with
  invisibility-reset and dyadic windows.
- `configs/logsig_coordinate_pairs_global.yaml`: log-signature features over
  coordinate-pair streams.
- `configs/logsig_coordinate_singletons_expanding.yaml`: log-signature features
  over singleton coordinate streams with expanding windows.
- `configs/signature_random_projection_dyadic.yaml`: full signature features
  over random projection streams with dyadic windows.
- `configs/logsig_dyadic_rescaling_none.yaml`: explicit no-rescaling
  log-signature example.
- `configs/logsig_dyadic_rescaling_pre.yaml`: pre-signature rescaling example.
- `configs/signature_dyadic_rescaling_post.yaml`: post-signature rescaling
  example for full signatures.
- `configs/minirocket.yaml`: MiniROCKET raw-series baseline.

Run any example with:

```bash
sigtsc run --config configs/logsig_dyadic.yaml
```

## Suite Config Generation

The helper script `src/sigtsc/scripts/generate_suite_config.py` generates
timestamped suite YAML files.

Edit the settings block at the top of the script:

```python
MODE = "global"       # "global" or "per_dataset"
TAG = "dataset_sweeps"
```

Run:

```bash
python src/sigtsc/scripts/generate_suite_config.py
```

With the current script settings, `MODE = "global"` writes one suite config:

```text
configs/sig_dataset_sweeps_<timestamp>.yaml
```

Current global sweep size:

- 5 base datasets
- 40 suite datasets after clean, warp, shift, and combined warp/shift variants
- 401 model/feature/window variants
- 16040 total runs

`MODE = "per_dataset"` writes one config per base dataset:

```text
configs/per_dataset/sig_dataset_sweeps_<dataset>_<timestamp>.yaml
```

Current per-dataset sweep size:

- 1 base dataset family
- 8 suite datasets: clean, 3 warp levels, 3 shift levels, and 1 combined
  warp/shift setting
- 401 model/feature/window variants
- 3208 total runs

Generated dataset names include transform tags such as:

```text
BasicMotions
BasicMotions@warp=0.10
BasicMotions@shift=0.05
BasicMotions@warp=0.20,shift=0.10
```

Generated variants currently cover:

- Models: `logreg`, `linearsvc`, `mlp`, and `minirocket`
- Log-signature level 3 feature variants
- `with_time`, `basepoint`, and `lead_lag` on/off combinations
- An augmentation hook with no extra projection/reset variants enabled by default
- Homogeneous `windowing` blocks for `global`, two `sliding` presets,
  `expanding`, and `dyadic`
- Model parameter grids for regularization, solver settings, MLP shape, and
  MiniROCKET kernel settings

Generated suite configs enable plotting by default. Plot dataset filters use
deduplicated base names, so transformed variants remain included without noisy
repeated entries.

## Datasets and Transforms

Datasets are loaded with `aeon` using official train/test splits. The loader
converts cases to `(time, channels)` paths before feature extraction.

Common datasets used in the configs:

- BasicMotions
- ArticularyWordRecognition
- CharacterTrajectories
- NATOPS
- Epilepsy

Dataset transform tags are parsed directly from the dataset name:

```text
NATOPS@warp=0.20
NATOPS@shift=0.10
NATOPS@warp=0.20,shift=0.10
NATOPS@noise=0.05
```

Transforms are applied deterministically from the configured seed.

## Outputs

Single runs write to a timestamped directory under `results_dir`, usually:

```text
results/runs/<timestamp>/
```

Each run stores:

- `config.yaml`: configuration snapshot
- `metrics.json`: metrics, model metadata, feature metadata, and git commit

Feature metadata includes fields such as:

- `type`
- `level`
- `rescaling`
- `with_time`
- `basepoint`
- `invisibility_reset`
- `lead_lag`
- `coordinate_projection_mode`
- `random_projection_output_dim`
- `random_projection_num_projections`
- `random_projection_seed`
- `num_augmented_streams`
- `channels_per_augmented_stream`
- `window_type`
- `window_aggregation`
- `num_windows`
- `total_windows`
- `dyadic_depth`
- `expanding_num_windows`
- `min_window`
- `window_fracs`
- `pool`
- `dim` / `feature_dim`

Suite runs write per-run outputs plus suite-level summaries and aggregate CSVs.

## Aggregation and Plotting

Aggregation writes:

- `summary.csv`: one row per run
- `report.csv`: method-level summaries and signature-vs-baseline gaps
- `robustness.csv`: robustness rows by transformed dataset and method variant
- `robustness_winners.csv`: lowest-drop method variants per transform condition

Config-driven plotting is supported for both single runs and suites:

```yaml
plotting:
  enabled: true
  # datasets: [NATOPS, CharacterTrajectories]
  # out_dir: results/custom_plots
```

For single runs, plots are saved under the run directory unless `out_dir` is
provided. For suites, aggregation is written under `<suite_dir>/agg/` and plots
are saved under `<suite_dir>/plots/` unless `out_dir` is provided.

The plotting pipeline can generate:

- Best accuracy heatmap by dataset and method
- Mean method accuracy bar plot
- Signature-vs-baseline dataset gap plot
- Robustness curves by transform severity
- Parameter sensitivity plots for level, time channel, lead-lag, and sliding
  window fraction settings

Dataset filters match exact names and base names. For example, `--dataset NATOPS`
also matches `NATOPS@warp=...` and `NATOPS@shift=...`.

## Feature Dimension Notes

Feature dimensionality depends on:

- Transform type: `logsig` or `signature`
- Number of channels after optional time channel and lead-lag augmentation
- Truncation level
- Windowing mode and number of emitted windows
- Aggregation mode and pooling operations

Basepoint changes the path length, not the channel dimension. It prepends a zero
vector after optional time augmentation and before optional lead-lag
augmentation.

Log-signature and signature dimensions can be checked with:

```python
import iisignature

iisignature.logsiglength(d, level)
iisignature.siglength(d, level)
```

## Development

Run tests:

```bash
python -m pytest
```

Compile-check the edited Python files if needed:

```bash
python -m py_compile src/sigtsc/features/signature.py src/sigtsc/experiments/run_experiment.py
```

## Core Dependencies

The code uses these main packages. Package dependencies are managed through
`pyproject.toml`, with `environment.yaml` providing the local Conda development
environment.

- aeon
- iisignature
- joblib
- matplotlib
- numpy
- pandas
- pyyaml
- rich
- scikit-learn
- scipy
- seaborn
- tqdm

`environment.yaml` provides the Conda environment used for local development.

## License

MIT in `LICENSE`.

## Author

Georgios Panagiotopoulos, 2026
