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
from sklearn.metrics import cohen_kappa_score, f1_score

from processing.alignment import apply_euclidean_alignment, should_align


from mne.decoding import CSP

from config import Config
from processing.splitting import get_splits
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
    #use_lr_finder: bool,
    current_lr: float,
    subjects_train: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train and evaluate a deep model on a single fold.
    Uses Pytorch LightningModule.
    """
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    if subjects_train is not None:
        # Nested LOSO: hold out one training subject for validation.
        # Rotate through available subjects so each fold gets a different one.
        unique_train_subj = np.unique(subjects_train)
        val_subj          = unique_train_subj[fold_idx % len(unique_train_subj)]

        val_mask   = subjects_train == val_subj
        pure_mask  = ~val_mask

        print(f"  Validation subject: {val_subj}  "
            f"({val_mask.sum()} trials)  |  "
            f"Pure train: {pure_mask.sum()} trials")

        train_loader = make_loader(X_train[pure_mask], y_train[pure_mask], shuffle=True)
        val_loader   = make_loader(X_train[val_mask],  y_train[val_mask],  shuffle=False)
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=0.15,
            stratify=y_train,
            random_state=Config.RANDOM_SEED,
        )
        train_loader = make_loader(X_train[train_idx], y_train[train_idx], shuffle=True)
        val_loader   = make_loader(X_train[val_idx], y_train[val_idx], shuffle=False)

    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    input_window_samples = X_train.shape[2]

    base_model = create_model(model_name, n_channels, n_classes, input_window_samples)
    lightning_model = LightningModel(
        base_model, n_classes,
        learning_rate=current_lr,
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

    

    trainer.fit(lightning_model, train_loader, val_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model = LightningModel.load_from_checkpoint(
        checkpoint.best_model_path,
        model=base_model,
        n_classes=n_classes,
        learning_rate=current_lr,
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

def _find_lr(
    model_name: str,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
) -> float:
    """
    Run the LR finder on a small probe model using a single random split of
    the provided training data. The probe model is discarded afterwards.

    Returns the suggested LR, or Config.LEARNING_RATE if the finder fails.
    """
    tr_idx, val_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=0.15,
        stratify=y_train,
        random_state=Config.RANDOM_SEED,
    )
    train_loader = make_loader(X_train[tr_idx],  y_train[tr_idx],  shuffle=True)
    val_loader   = make_loader(X_train[val_idx], y_train[val_idx], shuffle=False)

    n_channels           = X_train.shape[1]
    n_classes            = len(np.unique(y_train))
    input_window_samples = X_train.shape[2]

    probe_model = LightningModel(
        create_model(model_name, n_channels, n_classes, input_window_samples),
        n_classes,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    probe_trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator='auto', devices=1,
        logger=False,
    )

    lr_finder    = Tuner(probe_trainer).lr_find(
        probe_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        min_lr=1e-5, max_lr=1e-1,
        num_training=Config.LR_FINDER_NUM_STEPS,
    )
    suggested_lr = lr_finder.suggestion()

    import matplotlib.pyplot as plt
    plt.close(lr_finder.plot(suggest=True))

    del probe_model, probe_trainer

    if suggested_lr is not None:
        print(f"  LR finder suggested : {suggested_lr:.2e}")
        return suggested_lr

    print(f"  LR finder did not converge, using default: {Config.LEARNING_RATE:.2e}")
    return Config.LEARNING_RATE


def train_model(
    model_name:       str,
    X:                np.ndarray,
    y:                np.ndarray,
    subjects:         np.ndarray,
    spec,
    predictions_dict: dict,
    use_lr_finder:    bool = False,
    dataset_ids:        np.array = None
) -> tuple[float, float, float, float, float, float]:
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
    use_ea = Config.USE_EUCLIDEAN_ALIGNMENT and should_align(eval_strategy)
    if use_ea:
        print(f"  Euclidean Alignment: ON")


    print(f"TRAINING: {model_name}  |  strategy: {eval_strategy}  |  {spec.name}")
    #print(f"Train class dist: {np.bincount(y_train)}, Test class dist: {np.bincount(y_test)}")
    print(f"{'-' * 70}")

    splits = get_splits(eval_strategy, y, subjects, dataset_ids=dataset_ids)

    current_lr = Config.LEARNING_RATE
    if use_lr_finder and model_name != 'CSP+LDA':
        print(f"  Running LR finder on fold-0 training split...")
        train_idx_0 = splits[0][0]
        current_lr  = _find_lr(model_name, X[train_idx_0], y[train_idx_0])



    fold_accs = []
    fold_kappa_values = []
    fold_f1_values = []
    is_dataset_split = eval_strategy == 'dataset_split'
    agg_y_true = []
    agg_y_pred = []
    per_fold_preds = [] #only for dataset split

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if len(splits) > 1:
            print(
                f" --- Fold {fold_idx + 1}/{len(splits)} "
                f"(train={len(train_idx)}, test={len(test_idx)}) ---"
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_ea:
            subjects_train = subjects[train_idx]
            subjects_test  = subjects[test_idx]
            X_train, X_test = apply_euclidean_alignment(
                X_train, X_test,
                subjects_train, subjects_test,
            )


        if model_name == 'CSP+LDA':
            y_true, y_pred = _run_one_fold_csp(X_train, X_test, y_train, y_test)
        else:
            #run_lr = use_lr_finder and (fold_idx == 0) and (model_name in Config.LR_FINDER_USE)
            subj_train_for_fold = subjects[train_idx] if eval_strategy == 'loso' else None
            y_true, y_pred = _run_one_fold_deep(
                model_name, X_train, X_test, y_train, y_test,
                spec.name, fold_idx,
                current_lr = current_lr,
                subjects_train = subj_train_for_fold,
            )

        fold_acc = float(np.mean(y_true == y_pred))
        fold_kappa = float(cohen_kappa_score(y_true, y_pred))
        fold_f1 = float(f1_score(y_true, y_pred, average='macro'))
        fold_accs.append(fold_acc)
        fold_kappa_values.append(fold_kappa)
        fold_f1_values.append(fold_f1)
        print(f"  Fold {fold_idx + 1} accuracy: {fold_acc:.4f}, kappa: {fold_kappa:.4f}, f1: {fold_f1:.4f}")

        if is_dataset_split:
            per_fold_preds.append((y_true.copy(), y_pred.copy()))
        else:
            agg_y_true.extend(y_true)
            agg_y_pred.extend(y_pred)

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    mean_kappa = float(np.mean(fold_kappa_values))
    std_kappa  = float(np.std(fold_kappa_values))
    mean_f1 = float(np.mean(fold_f1_values))
    std_f1 = float(np.std(fold_f1_values))
    print(f"\n  Final accuracy : {mean_acc:.4f} ± {std_acc:.4f}  ({len(fold_accs)} fold(s))")
    print(f"  Final Kappa : {mean_kappa:.4f} ± {std_kappa:.4f}")
    print(f"  Final F1 score : {mean_f1:.4f} ± {std_f1:.4f}")

    if is_dataset_split:

        predictions_dict[model_name] = per_fold_preds

    else:
        predictions_dict[model_name] = (np.array(agg_y_true), np.array(agg_y_pred))

    return mean_acc, std_acc, mean_kappa, std_kappa, mean_f1, std_f1
