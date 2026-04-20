"""
config.py
Central configuration class Config and one experiment config class Experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import mne

@dataclass
class Experiment:
    """
    Descriptor for one experiment

    Parameters
    ----------
    name : str
        Display name used in plots, CSV rows, and checkpoint filenames.
    datasets : list[str]
        One entry = single dataset.  Multiple entries = auto-merged.
    task_type : str | None
        Overrides Config.TASK_TYPE for this experiment only.
    max_subjects : int | None
        Overrides Config.MAX_SUBJECTS for this experiment only.
    eval_strategy : str | None
        Overrides Config.EVAL_STRATEGY for this experiment only.
        One of: 'within_subject_split', 'stratified_kfold', 'loso', 'subject_split'.

    Examples
    --------
    Experiment("BNCI2014_004", ["BNCI2014_004"])
    Experiment("Combined_LR",  ["BNCI2014_004", "GrosseWentrup2009"])
    Experiment("PhysionetMI_LOSO", ["PhysionetMI"],task_type="movement_vs_rest", max_subjects=10, eval_strategy="loso")
    """
    name:          str
    datasets:      list
    task_type:     Optional[str] = None
    max_subjects:  Optional[int] = None
    eval_strategy: Optional[str] = None

    def resolve_task(self) -> str: return self.task_type or Config.TASK_TYPE
    def resolve_max_subjects(self) -> Optional[int]: return self.max_subjects if self.max_subjects is not None else Config.MAX_SUBJECTS
    def resolve_eval_strategy(self) -> str: return self.eval_strategy or Config.EVAL_STRATEGY

    @property
    def is_merged(self) -> bool:
        return len(self.datasets) > 1


class Config:
    """
    Central configuration.  Per-experiment overrides live in DataSpec.
    """

    EXPERIMENTS = [
        Experiment("Combined_LeftRight", ["Weibo2014"], task_type="movement_vs_rest", max_subjects=2),
        
        #Experiment("Combined_LeftRight2", ["Weibo2014", "Beetl2021_A"], task_type="movement_vs_rest", max_subjects=None),

        #Experiment("Combined_LeftRight3", ["Beetl2021_A", "PhysionetMI"], task_type="movement_vs_rest", max_subjects=None),

        #Experiment("Combined_LeftRight", ["PhysionetMI", "Weibo2014"], task_type="movement_vs_rest", max_subjects=None),
        
    ]

    MODELS = [
        'CSP+LDA',
        'EEGNet',
        'ShallowConvNet',
        #'Deep4Net',
        #'BENDR',
        #'CBraMod',
    ]

    LR_FINDER_USE = {'EEGNet', 'ShallowConvNet', 'Deep4Net'}


    TASK_TYPE = 'movement_vs_rest' #or left vs right
    MAX_SUBJECTS = None


    CHANNELS = ['C3', 'Cz', 'C4']

    # Evaluation strategy - implemented possibilities
    # 'within_subject_split'     — random trial split
    # 'stratified_kfold' — K-fold on trials
    # 'loso'             — Leave-One-Subject-Out
    # 'subject_split'    — holds out a fraction of subjects entirely
    EVAL_STRATEGY = 'within_subject_split'
    TEST_SIZE = 0.2 # ratio for within subject split
    KFOLD_N_SPLITS = 5
    SUBJECT_TEST_RATIO = 0.2 # ratio for subject_split

    # Preprocessing config
    LOW_CUT_HZ    = 8
    HIGH_CUT_HZ   = 30
    RESAMPLE_FREQ = 100
    TMIN = 0.0
    TMAX = 3.99

    # Training config
    BATCH_SIZE    = 64
    MAX_EPOCHS    = 100
    LEARNING_RATE = 0.001 # not used if USE_LR_FINDER = True
    WEIGHT_DECAY  = 0.01
    

    USE_LR_FINDER = True
    USE_LR_FINDER_FOR_ALL = False #if LR finding runs for every model or just one
    LR_FINDER_NUM_STEPS = 50

    NUM_WORKERS = 0 #number of background processes DataLoader uses to prefetch batches
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2 #how many batches each worker preloads ahead of time

    # Paths — all relative to the repo root (the folder containing this file)
    _ROOT       = Path(__file__).parent
    OUTPUT_DIR  = _ROOT / 'outputs'
    DATA_DIR    = _ROOT / 'data'
    RANDOM_SEED = 42

    @classmethod
    def setup(cls) -> None:
        """Create output directories, configure MNE data cache, print GPU info."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        mne.set_config('MNE_DATA', 'data')

        config = mne.get_config()
        for key in list(config.keys()):
            if key.startswith('MNE_DATASETS_') and key.endswith('_PATH'):
                mne.set_config(key, None)

        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (cls.OUTPUT_DIR / 'checkpoints').mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / 'plots').mkdir(exist_ok=True)
        pl.seed_everything(cls.RANDOM_SEED)

        print("GPU info")
        print(f"{'-' * 70}")
        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version    : {torch.version.cuda}")
            print(f"GPU             : {torch.cuda.get_device_name(0)}")
            print(f"GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('medium')
            print("GPU optimisation enabled")
        else:
            print("Running on CPU")
        print(f"{'-' * 70}\n")
