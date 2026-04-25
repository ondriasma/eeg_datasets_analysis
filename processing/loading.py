"""
data/loading.py
---------------
Fetching, preprocessing, epoching, and merging EEG datasets.

"""

from __future__ import annotations

import gc
from typing import Optional

import mne
import numpy as np
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess

from config import Config
from datasets import DATASET_EVENT_MAPPINGS


# Helper functions

def _resolve_picks(raw: mne.io.BaseRaw) -> list | str:
    """
    Picks channels according to Config.CHANNELS.
    Falls back to 'eeg' (all EEG) when none are found. TODO maybe better to stop and inform user and research how often this happens
    """
    if not Config.CHANNELS:
        return 'eeg'

    available = [ch for ch in Config.CHANNELS if ch in raw.ch_names]
    if available:
        return available

    # Case-insensitive check
    ch_lower = {ch.lower(): ch for ch in raw.ch_names}
    available = [ch_lower[c.lower()] for c in Config.CHANNELS if c.lower() in ch_lower]
    if available:
        print(f"Channel name case mismatch — using {available}")
        return available

    print(
        f"Warning - None of {Config.CHANNELS} found "
        f"(first 8: {raw.ch_names[:8]}). Falling back to all EEG channels."
    )
    return 'eeg'


def _load_single_raw(dataset_name: str, max_subjects: Optional[int], task_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Load, preprocess, and epoch one dataset.

    Returns
    X        : float32 [trials x channels x time]
    y        : int32   [trials]
    subjects : int     [trials]
    """
    print(f"Loading dataset: {dataset_name}")
    print(f"{'-' * 70}")

    dataset_info = DATASET_EVENT_MAPPINGS.get(dataset_name)
    if not dataset_info:
        raise ValueError(f"'{dataset_name}' not in DATASET_EVENT_MAPPINGS")

    event_mapping = dataset_info.get(task_type)
    if event_mapping is None:
        available = [k for k, v in dataset_info.items() if isinstance(v, dict) and v is not None]
        raise ValueError(
            f"Task '{task_type}' not available for {dataset_name}. "
            f"Available tasks: {available}"
        )

    print(f"Description : {dataset_info['description']}")
    print(f"Task : {task_type} -> {event_mapping}")# event_mapping example: {'rest': 0, 'right_hand': 1, 'feet': 1}
    if Config.CHANNELS:
        print(f"Channels : {Config.CHANNELS} (falls back to all EEG if not specified)")

    # Loading the dataset from MOABB
    subject_ids = list(range(1, max_subjects + 1)) if max_subjects else None
    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subject_ids)
    print(f"Sessions loaded: {len(dataset.datasets)}")



    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: np.multiply(data, 1e6)),
        Preprocessor('filter', l_freq=Config.LOW_CUT_HZ, h_freq=Config.HIGH_CUT_HZ),
        Preprocessor('resample', sfreq=Config.RESAMPLE_FREQ),
    ]
    preprocess(dataset, preprocessors)
    print(
        f"Preprocessing: {Config.LOW_CUT_HZ}-{Config.HIGH_CUT_HZ} Hz, "
        f"resampled to {Config.RESAMPLE_FREQ} Hz"
    )


    all_data, all_labels, all_subjects = [], [], []
    # Epoching - iterating over every recording session
    
    for idx, ds in enumerate(dataset.datasets):
        print(f"Recording {idx + 1}/{len(dataset.datasets)}...", end=' ')

        # Braindecode → MNE structures
        raw = ds.raw
        events, event_id = mne.events_from_annotations(raw)
        subject_id = ds.description.get('subject', idx)

        valid_event_ids = {# filters only the events relevant to our task
            name: event_id[name]
            for name in event_mapping if name in event_id
        }
        if not valid_event_ids:
            print("skip (no matching events)")
            gc.collect()
            continue

        valid_codes = list(valid_event_ids.values())
        valid_events = events[np.isin(events[:, 2], valid_codes)]
        if len(valid_events) == 0:
            print("skip (no events after filtering)")
            gc.collect()
            continue

        picks  = _resolve_picks(raw)

        #making epochs from raw data
        epochs = mne.Epochs(
            raw, valid_events, event_id=valid_event_ids,
            tmin=Config.TMIN, tmax=Config.TMAX,
            baseline=None, preload=True, reject=None,
            on_missing='ignore', verbose=False,
            picks=picks,
        )

        # MNE → NumPy  (cast to float32 immediately to save RAM usage (todo, change if done on more powerful computer))
        data = epochs.get_data()
        
        # Z-score each trial independently across time
        mean = data.mean(axis=-1, keepdims=True)
        std  = data.std(axis=-1, keepdims=True) + 1e-6
        data = (data - mean) / std
        codes = epochs.events[:, 2] # maps the dataset events into the two classes we want
        labels = np.zeros(len(codes), dtype=np.int32)
        for event_name, target_label in event_mapping.items():
            if event_name in valid_event_ids:
                labels[codes == valid_event_ids[event_name]] = target_label

        all_data.append(data)
        all_labels.append(labels)
        all_subjects.extend([subject_id] * len(data))
        print(f"ok ({len(data)} trials, {data.shape[1]} ch)")

        del epochs, raw
        gc.collect()

    # Concatenate all sessions into single arrays
    # Result: X[N × C × T], y[N], subjects[N], N = total trials, C = nuber of channels, T = number of samples per trial
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subjects = np.array(all_subjects)

    del all_data, all_labels #todo - ram saver
    gc.collect()

    print(f"Shape : {X.shape}  dtype: {X.dtype}")
    print(f"RAM : {X.nbytes / 1024 ** 2:.1f} MB")
    print(f"Classes : {np.bincount(y)}")
    print(f"Subjects : {len(np.unique(subjects))}  ids={np.unique(subjects).tolist()}")

    return X, y, subjects


def load_data(spec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None, None]:
    """
    Load and return (X, y, subjects) for any Experiment.

    For a single dataset, delegates straight to _load_single_raw.
    For a merged one it loads each dataset separately then concatenates,
    truncating channels and timepoints to the common minimum.
    Subject IDs are offset per dataset to avoid conflicts.
    """
    task_type = spec.resolve_task()
    max_subjects = spec.resolve_max_subjects()

    if not spec.is_merged: # using only one dataset
        X, y, subj = _load_single_raw(spec.datasets[0], max_subjects, task_type)
        if X is None: return None, None, None, None
        # Single dataset: all trials belong to ID 0
        dataset_ids = np.zeros(len(X), dtype=int)
        return X, y, subj, dataset_ids


    print(f"Merging {len(spec.datasets)} datasets -> '{spec.name}'")
    print(f"{'-' * 70}")

    X_parts, y_parts, subj_parts = [], [], []
    n_channels_list, n_timepoints_list = [], []
    subj_offset = 0

    dataset_id_parts = []

    for ds_idx, dataset_name in enumerate(spec.datasets):
        X, y, subj = _load_single_raw(dataset_name, max_subjects, task_type)
        if X is None:
            print(f"Warning - skipping {dataset_name} - load failed")
            continue
        X_parts.append(X)
        y_parts.append(y)
        subj_parts.append(subj + subj_offset)
        subj_offset += int(subj.max()) + 1
        dataset_id_parts.append(np.full(len(X), ds_idx, dtype=int))
        n_channels_list.append(X.shape[1])
        n_timepoints_list.append(X.shape[2])
        print(f"+ {dataset_name}: {len(X)} trials")
        gc.collect()


    min_ch = min(n_channels_list)
    min_t  = min(n_timepoints_list)
    if len(set(n_channels_list)) > 1:
        print(f"Alert - Channel mismatch {n_channels_list} -> truncating to {min_ch}")
    if len(set(n_timepoints_list)) > 1:
        print(f"Alert - Timepoint mismatch {n_timepoints_list} -> truncating to {min_t}")

    X_parts = [X[:, :min_ch, :min_t] for X in X_parts]
    X_merged = np.concatenate([X[:, :min_ch, :min_t] for X in X_parts], axis=0)
    y_merged = np.concatenate(y_parts, axis=0)
    subj_merged = np.concatenate(subj_parts, axis=0)
    dataset_ids_merged = np.concatenate(dataset_id_parts, axis=0)

    del X_parts, y_parts, subj_parts
    gc.collect()

    print(f"Merged shape : {X_merged.shape}  dtype: {X_merged.dtype}")
    print(f"RAM          : {X_merged.nbytes / 1024 ** 2:.1f} MB")
    print(f"Classes      : {np.bincount(y_merged)}")
    print(f"Subjects     : {len(np.unique(subj_merged))}")

    return X_merged, y_merged, subj_merged, dataset_ids_merged
