"""
data/splitting.py
-----------------
Evaluation strategy functions.
Each strategy returns a list of (train_idx, test_idx) NumPy index arrays.
train_model() iterates over these pairs and aggregates fold results.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import Config


# Implementations of all possible strategies

def _splits_within_subject(
    y: np.ndarray,
    subjects: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Single train/test split (splits randomly across individual trials, ignoring which subject they came from)."""
    idx = np.arange(len(y))
    train, test = train_test_split(
        idx,
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.RANDOM_SEED,
    )
    return [(train, test)]


def _splits_kfold(
    y: np.ndarray,
    subjects: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-Fold cross-validation on trials."""
    skf = StratifiedKFold(
        n_splits=Config.KFOLD_N_SPLITS,
        shuffle=True,
        random_state=Config.RANDOM_SEED,
    )
    idx = np.arange(len(y))
    return list(skf.split(idx, y))


def _splits_loso(
    y: np.ndarray,
    subjects: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Subject-Out.
    Each subject is in the test set exactly once
    """
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError(f"LOSO requires at least 2 subjects, got {len(unique_subjects)}.")
        
    

    splits = []
    for held_out in unique_subjects:
        test_idx  = np.where(subjects == held_out)[0]
        train_idx = np.where(subjects != held_out)[0]
        splits.append((train_idx, test_idx))
    return splits


def _splits_subject_split(
    y: np.ndarray,
    subjects: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Hold out a random fraction of subjects entirely.
    Remaining subjects form the training set.
    """
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError(f"Subject split requires at least 2 subjects, got {len(unique_subjects)}.")

    rng = np.random.default_rng(Config.RANDOM_SEED)
    n_test = max(1, int(np.ceil(len(unique_subjects) * Config.SUBJECT_TEST_RATIO)))
    test_subjects  = rng.choice(unique_subjects, size=n_test, replace=False)
    train_subjects = np.setdiff1d(unique_subjects, test_subjects)

    print(
        f"Subject split: train subjects={train_subjects.tolist()}, "
        f"test subjects={test_subjects.tolist()}"
    )

    test_idx  = np.where(np.isin(subjects, test_subjects))[0]
    train_idx = np.where(np.isin(subjects, train_subjects))[0]
    return [(train_idx, test_idx)]

def _dataset_split(
    y: np.ndarray, 
    subjects: np.ndarray, 
    dataset_ids: np.ndarray = None  # Add these to match signature
) -> list[tuple[np.ndarray, np.ndarray]]:
    if dataset_ids is None:
        raise ValueError("dataset_ids required for 'dataset_split' strategy.")
        
    unique_ids = np.unique(dataset_ids)
    if len(unique_ids) < 2:
        # If you only have 1 dataset, we can't do a cross-dataset split
        raise ValueError(f"dataset_split needs at least 2 datasets, found {len(unique_ids)}")

    return [
        (np.where(dataset_ids == unique_ids[0])[0], np.where(dataset_ids == unique_ids[1])[0]),
        (np.where(dataset_ids == unique_ids[1])[0], np.where(dataset_ids == unique_ids[0])[0]),
    ]



# functions mapping
SPLIT_FUNCTIONS = {
    'within_subject_split': _splits_within_subject,
    'stratified_kfold': _splits_kfold,
    'loso': _splits_loso,
    'subject_split': _splits_subject_split,
    'dataset_split': _dataset_split,
}


def get_splits(eval_strategy: str, y: np.ndarray, subjects: np.ndarray, dataset_ids: np.ndarray = None) -> list[tuple[np.ndarray, np.ndarray]]:
    
    fn = SPLIT_FUNCTIONS.get(eval_strategy)
    if fn is None:
        raise ValueError(
            f"Unknown eval_strategy '{eval_strategy}'. "
            f"Choose from: {list(SPLIT_FUNCTIONS)}"
        )
    if eval_strategy == 'dataset_split':
        splits = fn(y, subjects, dataset_ids)
    else:
        splits = fn(y, subjects)
    print(f"  Evaluation: {eval_strategy}  ({len(splits)} fold(s))")
    return splits
