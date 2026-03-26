# EEG Motor Imagery Classification Pipeline

A modular Python pipeline for benchmarking classical and deep learning models on EEG motor imagery datasets. Supports single-dataset evaluation, multi-dataset merging, and four cross-validation strategies including Leave-One-Subject-Out (LOSO).

Built on [Braindecode](https://braindecode.org/), [MNE-Python](https://mne.tools/), [MOABB](https://moabb.neurotechx.com/), and [PyTorch Lightning](https://lightning.ai/).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Models](#models)
- [Evaluation Strategies](#evaluation-strategies)
- [Data Pipeline](#data-pipeline)
- [Output](#output)

---

## Overview

The pipeline loads publicly available EEG datasets via MOABB, applies a standardised preprocessing chain, and evaluates a configurable set of models under one or more cross-validation strategies. Multiple datasets can be merged into a single experiment to study cross-dataset generalisation.

Key features:

- Automatic dataset download and caching via MOABB/pooch
- Flexible per-experiment overrides for task, subjects, and evaluation strategy
- Multi-dataset merging with automatic subject ID offsetting and shape alignment
- Four evaluation strategies
- Persistent results CSV — each run appends rows, enabling longitudinal comparison
- Automatic learning rate finder via PyTorch Lightning
- GPU support with automatic fallback to CPU

---

## Project Structure

```
repo-root/
├── config.py           # Config class and Experiment dataclass
├── datasets.py         # DATASET_EVENT_MAPPINGS registry
├── main.py             # Entry point
├── requirements.txt
├── README.md
├── .gitignore
├── data/               # Dataset cache — created on first run, gitignored
├── outputs/            # Results and plots — created on first run, gitignored
├── data/
│   ├── loading.py      # Dataset fetch, preprocessing, epoching, merging
│   └── splitting.py    # Cross-validation strategies
├── models/
│   ├── lightning.py    # LightningModule wrapper (no project imports)
│   ├── factory.py      # Model and DataLoader construction
│   └── training.py     # Training loop, fold iteration
└── results/
    ├── persistence.py  # CSV append and load
    └── plotting.py     # Confusion matrices and model comparison plots
```

---

## Installation

**Requirements:** Python 3.11, pip

```bash
git clone <repo-url>
cd Braindecode

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

**GPU support (optional):** Install the CUDA-enabled PyTorch build matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

**Running the pipeline:**

```bash
python main.py
```



**Dataset cache:** Datasets are downloaded automatically on first use and cached in the `data/` folder inside the repository root. Subsequent runs read from cache without re-downloading. The `data/` folder is gitignored and will not be committed.

---



## Configuration

All settings are centralised in `config.py`.

### Experiments

Defined as a list of `Experiment` objects. Each experiment specifies which datasets to load, the task type, subject limit, and evaluation strategy. Any parameter left as `None` falls back to the global default in `Config`.

```python
EXPERIMENTS = [
    # Single dataset
    Experiment("AlexMI_LOSO", ["AlexMI"],
               task_type="movement_vs_rest",
               eval_strategy="loso"),

    # Merged dataset — subjects are automatically offset to avoid ID conflicts
    Experiment("Combined_LR", ["AlexMI", "Weibo2014"],
               task_type="movement_vs_rest",
               max_subjects=8),
]
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Display name used in plots and CSV |
| `datasets` | `list[str]` | One or more dataset keys from `DATASET_EVENT_MAPPINGS` |
| `task_type` | `str \| None` | Overrides `Config.TASK_TYPE` |
| `max_subjects` | `int \| None` | Loads subjects 1..N; `None` loads all |
| `eval_strategy` | `str \| None` | Overrides `Config.EVAL_STRATEGY` |

### Models

```python
MODELS = ['CSP+LDA', 'EEGNet', 'ShallowConvNet', 'Deep4Net']
```

### Global Defaults

| Parameter | Default | Description |
|---|---|---|
| `TASK_TYPE` | `'movement_vs_rest'` | Default task for all experiments |
| `MAX_SUBJECTS` | `None` | Load all subjects unless overridden |
| `CHANNELS` | `['C3', 'Cz', 'C4']` | Motor cortex channels; falls back to all EEG if not found |
| `EVAL_STRATEGY` | `'loso'` | Default evaluation strategy |

### Preprocessing

| Parameter | Default | Description |
|---|---|---|
| `LOW_CUT_HZ` | `8` | Bandpass filter lower edge (mu/beta band) |
| `HIGH_CUT_HZ` | `30` | Bandpass filter upper edge |
| `RESAMPLE_FREQ` | `100` | Target sampling frequency in Hz |
| `TMIN` | `0.0` | Epoch start relative to event onset (seconds) |
| `TMAX` | `4.0` | Epoch end relative to event onset (seconds) |

### Training

| Parameter | Default | Description |
|---|---|---|
| `BATCH_SIZE` | `64` | Mini-batch size for deep models |
| `MAX_EPOCHS` | `100` | Maximum training epochs (early stopping may terminate sooner) |
| `LEARNING_RATE` | `0.001` | Fixed LR; ignored when `USE_LR_FINDER=True` |
| `WEIGHT_DECAY` | `0.01` | L2 regularisation coefficient |
| `USE_LR_FINDER` | `True` | Auto-detect LR for the second model in `MODELS` |
| `USE_LR_FINDER_FOR_ALL` | `False` | Run LR finder for every model |
| `LR_FINDER_NUM_STEPS` | `50` | Steps in the LR sweep |

### Evaluation

| Parameter | Default | Description |
|---|---|---|
| `EVAL_STRATEGY` | `'loso'` | See [Evaluation Strategies](#evaluation-strategies) |
| `TEST_SIZE` | `0.2` | Test fraction for `pooled_split` |
| `KFOLD_N_SPLITS` | `5` | Number of folds for `stratified_kfold` |
| `SUBJECT_TEST_RATIO` | `0.2` | Fraction of subjects held out for `subject_split` |

---

## Datasets

Datasets are registered in `datasets.py` with their event mappings per task. The pipeline supports any MOABB-compatible dataset — add an entry to `DATASET_EVENT_MAPPINGS` to include a new one.

| Dataset | Subjects | Classes | Notes |
|---|---|---|---|
| `AlexMI` | 8 | right hand, feet, rest | 3-class MI |
| `Weibo2014` | 10 | left/right hand, feet, rest | 4-class MI |
| `Ofner2017` | 15 (14 usable) | 6 upper limb movements + rest | Subject 5 has corrupted file |
| `BNCI2014_001` | 9 | left/right hand, feet, tongue | BCI Competition IV 2a |
| `PhysionetMI` | 109 | left/right hand, feet, rest | Large-scale dataset |
| `Beetl2021_A` | 4 | left/right hand | BCI competition dataset |
| `GrosseWentrup2009` | 10 | left/right hand | MEG-based |

### Task types

`movement_vs_rest` — all movement classes collapsed to label 1, rest to label 0 (binary).

`left_vs_right` — left hand = 0, right hand = 1 (binary, only datasets with both classes).

Event mappings for each dataset/task combination are defined in `datasets.py`.

---

## Models

| Model | Type | Notes |
|---|---|---|
| `CSP+LDA` | Classical | Common Spatial Patterns + Linear Discriminant Analysis; no GPU |
| `EEGNet` | Deep (CNN) | Compact depthwise separable CNN; ~1.5K parameters |
| `ShallowConvNet` | Deep (CNN) | Shallow architecture optimised for oscillatory features |
| `Deep4Net` | Deep (CNN) | Deeper residual-style architecture for temporal features |

Deep models are implemented via Braindecode and trained with PyTorch Lightning. All share the same training loop with early stopping (patience 20), learning rate reduction on plateau, and best-checkpoint saving based on validation accuracy.

---

## Evaluation Strategies

| Strategy | Description | Suitable for |
|---|---|---|
| `within_subject_split` | Random 80/20 trial split; same subjects in train and test 
| `stratified_kfold` | K-fold on pooled trials, class-balanced
| `subject_split` | Hold out a fixed fraction of subjects entirely
| `loso` | Leave-One-Subject-Out; every subject is test set once


---

## Data Pipeline

```
MOABBDataset (Braindecode)
    └── BaseConcatDataset[BaseDataset[mne.io.Raw]]
            │
            ▼ preprocess()
        pick_types(eeg=True)          drop non-EEG channels
        ×1e6                          V → µV
        bandpass filter 8–30 Hz       mu / beta band
        resample to 100 Hz            reduce from 512 Hz
            │
            ▼ per session
        mne.events_from_annotations() → events [N × 3]
        mne.Epochs()                  → picks C3, Cz, C4
        epochs.get_data()             → float32 [trials × 3 × 401]
            │
            ▼ concatenate all sessions
        X  [N × C × T]   float32
        y  [N]           int32
        subjects [N]     int
            │
            ▼ (merged experiments only)
        offset subject IDs per dataset
        truncate to minimum C and T across datasets
        concatenate along trial axis
            │
            ▼
        TensorDataset → DataLoader → LightningModule
```

Array dimensions: **N** = total trials, **C** = channels (3 after selection), **T** = timepoints (100 Hz × 4 s = 400, +1 for endpoint = 401).

---

## Output

All outputs are written to `outputs/` (configured via `Config.OUTPUT_DIR`).

```
outputs/
├── results_all_runs.csv      # Persistent results; one row per model per experiment per run
├── checkpoints/              # Best model weights per fold
└── plots/
    ├── model_comparison_<run_id>.png
    └── confusion_matrix_<experiment>_<run_id>.png
```

The results CSV accumulates across runs. Each row contains: `run_id`, `experiment`, `eval_strategy`, `model`, `accuracy`, `std`, `n_trials`, `n_subjects`, `channels`, `dataset_names`, and `dataset_detail`.

Summary statistics printed at the end of each run show mean/min/max accuracy grouped by model, experiment, and evaluation strategy, plus the all-time best result across all stored runs.
