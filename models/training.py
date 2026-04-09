"""
models/training.py
------------------
Per-fold training and evaluation for both CSP+LDA and deep models.
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.tuner import Tuner
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from mne.decoding import CSP

from config import Config
from data.splitting import get_splits
from models.factory import create_model, make_loader
from models.lightning import LightningModel


def _run_one_fold_csp(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train and evaluate CSP+LDA on a single fold.
    Builds a scikit learn pipeline.
    """
    clf = Pipeline([
        ('CSP', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ('LDA', LDA()),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_test, y_pred


def _run_one_fold_deep(
    model_name:    str,
    X_train:       np.ndarray,
    X_test:        np.ndarray,
    y_train:       np.ndarray,
    y_test:        np.ndarray,
    spec_name:     str,
    fold_idx:      int,
    use_lr_finder: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train and evaluate a deep model on a single fold.
    Uses Pytorch LightningModule.
    """

    sub_train_idx, sub_val_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=0.15,
        stratify=y_train,
        random_state=Config.RANDOM_SEED,
    )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)
    val_loader   = make_loader(X_train[sub_val_idx], y_train[sub_val_idx], shuffle=False)

    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    input_window_samples = X_train.shape[2]

    base_model = create_model(model_name, n_channels, n_classes, input_window_samples)
    lightning_model = LightningModel(
        base_model, n_classes,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    ckpt_name = f'{spec_name}_{model_name}_fold{fold_idx:02d}_{{epoch:02d}}_{{val_acc:.3f}}'
    checkpoint = ModelCheckpoint(
        dirpath=Config.OUTPUT_DIR / 'checkpoints',
        filename=ckpt_name,
        monitor='val_acc', mode='max', save_top_k=1, verbose=False,
    )
    early_stop = EarlyStopping(
        monitor='val_acc', patience=20, mode='max', verbose=False,
    )

    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        callbacks=[checkpoint, early_stop],
        enable_progress_bar=True,
        enable_model_summary=(fold_idx == 0),
        accelerator='auto', devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        logger=False,
    )

    if use_lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            min_lr=1e-5, max_lr=1e-1,
            num_training=Config.LR_FINDER_NUM_STEPS,
        )
        suggested_lr = lr_finder.suggestion()
        lightning_model.learning_rate = suggested_lr
        print(f"Suggested LR: {suggested_lr:.2e}")
        fig = lr_finder.plot(suggest=True)
        fig.savefig(
            Config.OUTPUT_DIR / 'plots' /
            f'lr_finder_{spec_name}_{model_name}_fold{fold_idx}.png',
            dpi=150, bbox_inches='tight',
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    trainer.fit(lightning_model, train_loader, val_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model = LightningModel.load_from_checkpoint(
        checkpoint.best_model_path,
        model=base_model,
        n_classes=n_classes,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    ).to(device)
    best_model.eval()

    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = best_model(xb.to(best_model.device)).argmax(dim=1).cpu().numpy()
            all_y_true.extend(yb.numpy())
            all_y_pred.extend(preds)

    return np.array(all_y_true), np.array(all_y_pred)



def train_model(
    model_name:       str,
    X:                np.ndarray,
    y:                np.ndarray,
    subjects:         np.ndarray,
    spec,
    predictions_dict: dict,
    use_lr_finder:    bool = False,
    dataset_ids:        np.array = None
) -> tuple[float, float]:
    """
    Train and evaluate model_name using the strategy defined in spec.

    Iterates over all folds returned by get_splits(), calling the
    appropriate fold function (_run_one_fold_csp or _run_one_fold_deep).

    Parameters
    ----------
    predictions_dict : dict
        Mutated in-place: stores the last fold's (y_true, y_pred) for
        confusion matrix generation in the caller.

    Returns mean and standard accuracy aggregated across all folds. 
    """
    eval_strategy = spec.resolve_eval_strategy()

    print(f"TRAINING: {model_name}  |  strategy: {eval_strategy}  |  {spec.name}")
    #print(f"Train class dist: {np.bincount(y_train)}, Test class dist: {np.bincount(y_test)}")
    print(f"{'-' * 70}")

    splits = get_splits(eval_strategy, y, subjects, dataset_ids=dataset_ids)
    fold_accs = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if len(splits) > 1:
            print(
                f" --- Fold {fold_idx + 1}/{len(splits)} "
                f"(train={len(train_idx)}, test={len(test_idx)}) ---"
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_name == 'CSP+LDA':
            y_true, y_pred = _run_one_fold_csp(X_train, X_test, y_train, y_test)
        else:
            run_lr = use_lr_finder and (fold_idx == 0)
            y_true, y_pred = _run_one_fold_deep(
                model_name, X_train, X_test, y_train, y_test,
                spec.name, fold_idx, run_lr,
            )

        fold_acc = float(np.mean(y_true == y_pred))
        fold_accs.append(fold_acc)
        print(f"  Fold {fold_idx + 1} accuracy: {fold_acc:.4f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    print(f"\n  Final: {mean_acc:.4f} ± {std_acc:.4f}  ({len(fold_accs)} fold(s))")

    # Store the last fold's predictions for confusion matrix in the caller
    predictions_dict[model_name] = (y_true, y_pred)

    return mean_acc, std_acc
