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
sigtsc --config configs/default.yaml
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

## Datasets

Datasets are loaded via `aeon` and use official UCR/UEA splits.

Example datasets:

- BasicMotions
- ArticularyWordRecognition
- CharacterTrajectories
- GunPoint
- ItalyPowerDemand

All comparisons should use the provided train/test splits unless explicitly
performing resampling experiments.

---

## Feature Extraction Notes

Currently supported (or intended as core features):

- Global log-signature features
- Optional time-channel augmentation (`with_time`)
- Optional multiscale windowing (`window_fracs`)
- Pooling across windows (mean / max)

Feature dimensionality depends on:

- Number of channels `d` (including optional time channel)
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

Planned extensions:

- Lead–lag augmentation
- Baseline comparisons (ROCKET / MiniROCKET)
- Statistical significance testing across datasets
- Hyperparameter sweeps / grid search
- Benchmark suite runner and results aggregation

---

## Core Dependencies

- aeon
- iisignature
- scikit-learn
- numpy
- scipy

---

## License

MIT in `LICENSE`.

---

## Author

Georgios Panagiotopoulos
2026