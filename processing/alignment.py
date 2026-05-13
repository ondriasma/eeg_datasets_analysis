"""
processing/alignment.py
-----------------------
Euclidean Alignment for cross-subject EEG normalisation.
It removes between-subject differences in signal
scale and spatial correlation without requiring a separate calibration
session from the test subject.
"""

from __future__ import annotations

import numpy as np


def _mean_covariance(X: np.ndarray) -> np.ndarray:
    """
    Compute the mean covariance matrix across trials.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)

    Returns
    -------
    R : ndarray, shape (n_channels, n_channels)
        Mean covariance matrix, regularised to guarantee positive-definiteness.
    """
    
    n_times = X.shape[2]# (n_trials, n_channels, n_times) -> (n_channels, n_channels)
    covs = np.einsum('ict,idt->icd', X, X) / (n_times - 1)
    R = covs.mean(axis=0)                                       

    # Tikhonov regularisation
    R += np.eye(R.shape[0]) * 1e-6
    return R


def _invsqrtm(M: np.ndarray) -> np.ndarray:
    """
    Compute M^{-1/2} via eigendecomposition.

    Parameters
    ----------
    M : ndarray, shape (C, C), symmetric positive-definite

    Returns
    -------
    ndarray, shape (C, C)
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)           # clamp negatives from float errors
    inv_sqrt_diag = np.diag(1.0 / np.sqrt(eigvals))
    return eigvecs @ inv_sqrt_diag @ eigvecs.T



# Strategies where EA is meaningful (cross-subject boundary exists)
_CROSS_SUBJECT_STRATEGIES = {'loso', 'subject_split', 'dataset_split'}


def should_align(eval_strategy: str) -> bool:
    """
    Returns True when the eval strategy has a cross-subject boundary
    and EA is therefore beneficial.
    """
    return eval_strategy in _CROSS_SUBJECT_STRATEGIES


def apply_euclidean_alignment(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    subjects_train: np.ndarray,
    subjects_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit per-subject alignment matrices on training subjects and apply them
    to both train and test data.

    For test subjects that are unseen (the cross-subject case), we compute
    their alignment matrix from their own test trials.  This is valid
    because EA uses only the unlabelled signal covariance — no label
    information leaks.

    Parameters
    ----------
    X_train         : ndarray (N_train, C, T)
    X_test          : ndarray (N_test,  C, T)
    subjects_train  : ndarray (N_train,)  subject ID per trial
    subjects_test   : ndarray (N_test,)   subject ID per trial

    Returns
    -------
    X_train_aligned : ndarray (N_train, C, T)
    X_test_aligned  : ndarray (N_test,  C, T)
    """
    X_train_aligned = X_train.copy()
    X_test_aligned  = X_test.copy()
    subj_matrices = {}
    #aligning training subjects
    for subj in np.unique(subjects_train):
        mask = subjects_train == subj
        R_inv = _invsqrtm(_mean_covariance(X_train[mask]))
        subj_matrices[subj] = R_inv
        X_train_aligned[mask] = np.einsum('cd,idt->ict', R_inv, X_train[mask])

    #aligning testing subjects
    # For subjects also present in training, reuse their training matrix.
    # For truly unseen subjects (cross-subject case), compute from test trials.
    train_subj_set = set(np.unique(subjects_train).tolist())

    for subj in np.unique(subjects_test):
        test_mask = subjects_test == subj
        if subj in train_subj_set:
            #subject present - reuse training alignment
            mask = subjects_train == subj
            R_inv = subj_matrices[subj]
        else:
            #subject unseen - compute from their test trials (no label leakage)
            R_inv = _invsqrtm(_mean_covariance(X_test[mask]))
        X_test_aligned[mask] = np.einsum('cd,idt->ict', R_inv, X_test[test_mask])
    return X_train_aligned, X_test_aligned