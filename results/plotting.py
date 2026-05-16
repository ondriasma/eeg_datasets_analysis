"""
results/plotting.py
All visualisation functions.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config import Config

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

_MODEL_COLORS = {
    'CSP+LDA':        '#7fc7af',
    'EEGNet':         '#88b7d5',
    'ShallowConvNet': '#c9c9c9',
    'Deep4Net':       '#fce762',
}


def plot_confusion_matrices(predictions_dict: dict, spec, run_id: str) -> None:
    """
    One confusion matrix panel per model, saved as a single PNG.

    Handles two formats in predictions_dict:
    - Standard (all strategies except dataset_split):
        predictions_dict[model] = (y_true, y_pred)
        -> one matrix per model showing aggregated results across all folds

    - dataset_split:
        predictions_dict[model] = [(y_true_A, y_pred_A), (y_true_B, y_pred_B)]
        -> two matrices per model: one per transfer direction
        'Train A -> Test B' and 'Train B -> Test A'
    """
    task_type   = spec.resolve_task()
    n_models    = len(predictions_dict)
    is_dataset_split = any(isinstance(v, list) for v in predictions_dict.values())

    if n_models == 0:
        return

    class_names = (
        ['Left Hand', 'Right Hand'] if task_type == 'left_vs_right' else ['Rest', 'Movement']
    )

    # dataset_split needs 2 columns per model, others need 1
    n_cols = n_models * 2 if is_dataset_split else n_models
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    col = 0
    for model_name, preds in predictions_dict.items():

        if isinstance(preds, list):
            # dataset_split: one subplot per transfer direction
            dataset_names = spec.datasets
            for fold_idx, (y_true, y_pred) in enumerate(preds):
                ax   = axes[col]
                cm   = confusion_matrix(y_true, y_pred)
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=class_names
                )
                disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
                acc = float(np.mean(y_true == y_pred))

                if len(dataset_names) >= 2:
                    train_ds  = dataset_names[fold_idx % len(dataset_names)]
                    test_ds   = dataset_names[(fold_idx + 1) % len(dataset_names)]
                    direction = f"Train: {train_ds}\nTest: {test_ds}"
                else:
                    direction = f"Direction {fold_idx + 1}"

                ax.set_title(
                    f'{model_name}\n{direction}\nAcc: {acc:.3f}',
                    fontsize=11, fontweight='bold',
                )
                col += 1

        else:
            # Standard: single aggregated matrix across all folds
            y_true, y_pred = preds
            ax   = axes[col]
            cm   = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=class_names
            )
            disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
            acc = float(np.mean(y_true == y_pred))
            ax.set_title(
                f'{model_name}\nAccuracy: {acc:.3f}',
                fontsize=12, fontweight='bold',
            )
            col += 1

    plt.suptitle(
        f'Confusion Matrices - {spec.name} ({task_type})  [{run_id}]',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    path = (
        Config.OUTPUT_DIR / 'plots' /
        f'confusion_matrix_{spec.name}_{task_type}_{run_id}.png'
    )
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Confusion matrices saved: {path}")
    plt.close()