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

'''
def plot_model_comparison(results_df: pd.DataFrame, run_id: str) -> None:
    """Grouped bar chart of accuracy for every model experiment in this run."""
    run_df = results_df[results_df['run_id'] == run_id]
    if run_df.empty:
        return

    experiments = run_df['experiment'].unique()
    models = run_df['model'].unique()

    x = np.arange(len(experiments))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(10, 3 * len(experiments)), 6))

    for i, model in enumerate(models):
        mdata  = run_df[run_df['model'] == model]
        accs = [
            mdata.loc[mdata['experiment'] == e, 'accuracy'].values[0]
            if e in mdata['experiment'].values else 0
            for e in experiments
        ]
        stds = [
            mdata.loc[mdata['experiment'] == e, 'std'].values[0]
            if e in mdata['experiment'].values else 0
            for e in experiments
        ]
        offset = width * (i - len(models) / 2 + 0.5)
        ax.bar(
            x + offset, accs, width,
            label=model,
            color=_MODEL_COLORS.get(model, '#cccccc'),
            yerr=stds, capsize=3,
            error_kw={'linewidth': 1.5},
        )

    ax.set_xlabel('Experiment', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy',   fontsize=14, fontweight='bold')
    ax.set_title(f'Model Comparison  [{run_id}]', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance level')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = Config.OUTPUT_DIR / 'plots' / f'model_comparison_{run_id}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

'''
def plot_confusion_matrices(predictions_dict: dict, spec, run_id: str) -> None:

    """One confusion matrix panel per model, saved as a single PNG."""
    task_type = spec.resolve_task()
    n_models  = len(predictions_dict)
    if n_models == 0:
        return

    class_names = (
        ['Left Hand', 'Right Hand'] if task_type == 'left_vs_right' else ['Rest', 'Movement'])

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, (y_true, y_pred)) in zip(axes, predictions_dict.items()):
        cm   = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        acc = np.mean(y_true == y_pred)
        ax.set_title(f'{model_name}\nAccuracy: {acc:.3f}', fontsize=12, fontweight='bold')

    plt.suptitle(
        f'Confusion Matrices — {spec.name} ({task_type})  [{run_id}]',
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
