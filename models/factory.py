"""
models/factory.py
Pure construction functions: no training logic, no side effects beyond

"""

from __future__ import annotations

import numpy as np
import torch
from braindecode.models import Deep4Net, EEGNetv4, ShallowFBCSPNet
from torch.utils.data import DataLoader, TensorDataset

from config import Config


def create_model(
    model_name: str,
    n_channels: int,
    n_classes: int,
    input_window_samples: int,
) -> torch.nn.Module:
    """Braindecode model wrappers."""
    if model_name == 'EEGNet':
        return EEGNetv4(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
            final_conv_length='auto',
            drop_prob=0.5,
        )
    if model_name == 'ShallowConvNet':
        return ShallowFBCSPNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
            final_conv_length='auto',
        )
    if model_name == 'Deep4Net':
        return Deep4Net(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
            final_conv_length='auto',
        )
    raise ValueError(f"Unknown model: '{model_name}'. Choose from: EEGNet, ShallowConvNet, Deep4Net")


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    """Wraps NumPy arrays in a TensorDataset and returns a DataLoader."""
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY and torch.cuda.is_available(),
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=Config.NUM_WORKERS > 0,
    )
