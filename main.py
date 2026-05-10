"""
main.py
"""

import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from config import Config
from datasets import DATASET_EVENT_MAPPINGS, print_dataset_summary
from processing.loading import load_data
from models.training import train_model
from results.persistence import append_results
from results.plotting import plot_confusion_matrices, plot_model_comparison


def main() -> None:
    Config.setup()

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = Config.OUTPUT_DIR / 'results_all_runs.csv'

    print("Experiment info:") # visit configure.py to configure experiments and datasets.py for datasets info
    print("-" * 70)
    print(f"Run ID           : {run_id}")
    print(f"Eval strategy    : {Config.EVAL_STRATEGY}")
    print(f"Global task type : {Config.TASK_TYPE}")
    print(f"Models           : {Config.MODELS}")
    print(f"Channels         : {Config.CHANNELS or 'all EEG channels loaded'}")
    print(f"Results CSV      : {csv_path}  (rows appended)")
    print(f"\nExperiments ({len(Config.EXPERIMENTS)} total):")
    for spec in Config.EXPERIMENTS:
        tag      = f"merged({len(spec.datasets)})" if spec.is_merged else "single"
        task_str = spec.task_type     or f"[{Config.TASK_TYPE}]"
        eval_str = spec.eval_strategy or f"[{Config.EVAL_STRATEGY}]"
        print(f"  {spec.name:<30} {tag:<18} task={task_str}  eval={eval_str}")

    #print_dataset_summary()

    new_rows = []

    for spec in Config.EXPERIMENTS:
        print(f"Experiment: {spec.name}")
        print(f"{'-' * 70}")

        X, y, subjects, dataset_ids = load_data(spec)

        predictions_dict = {}

        for model_idx, model_name in enumerate(Config.MODELS):
            try:
                use_lr = (
                    Config.USE_LR_FINDER and
                    model_name in Config.LR_FINDER_USE
                )
                mean_acc, std_acc, mean_kappa, std_kappa, mean_f1, std_f1 = train_model(
                    model_name, X, y, subjects, spec,
                    predictions_dict, use_lr,
                    dataset_ids=dataset_ids
                )

                max_subj = spec.resolve_max_subjects()
                dataset_detail_parts = []
                for ds_name in spec.datasets:
                    info  = DATASET_EVENT_MAPPINGS.get(ds_name, {})
                    total = info.get('n_subjects', '?')
                    used  = min(max_subj, total) if max_subj else total
                    dataset_detail_parts.append(f"{ds_name}(n={used})")
                dataset_detail = '+'.join(dataset_detail_parts)

                new_rows.append({
                    'experiment':     spec.name,
                    'dataset_names':  '+'.join(spec.datasets),
                    'dataset_detail': dataset_detail,
                    'eval_strategy':  spec.resolve_eval_strategy(),
                    'is_merged':      spec.is_merged,
                    'task':           spec.resolve_task(),
                    'n_datasets':     len(spec.datasets),
                    'channels':       str(Config.CHANNELS),
                    'model':          model_name,
                    'accuracy':       round(mean_acc, 6),
                    'std':            round(std_acc, 6),
                    'mean kappa':     round(mean_kappa, 6),
                    'std kappa':      round(std_kappa, 6),
                    'mean f1':        round(mean_f1, 6),
                    'std f1':         round(std_f1, 6),
                    'n_trials':       len(X),
                    'n_subjects':     int(len(np.unique(subjects))),
                    'kfold_splits':   Config.KFOLD_N_SPLITS,
                })

            except Exception as e:
                import traceback
                print(f"\n  x Error training {model_name}: {str(e)[:150]}")
                traceback.print_exc()

        if predictions_dict:
            plot_confusion_matrices(predictions_dict, spec, run_id)

        del X, y, subjects
        gc.collect()


    print("Saving results")
    print("-" * 70)

    all_results = append_results(new_rows, csv_path, run_id)

    this_run = all_results[all_results['run_id'] == run_id]
    if not this_run.empty:
        print("This run:")
        print(this_run[[
            'experiment', 'eval_strategy', 'model',
            'accuracy', 'std', 'n_trials', 'n_subjects',
        ]].to_string(index=False))

    plot_model_comparison(all_results, run_id)

    print("SUMMARY  (this run only)")
    print("-" * 70)

    if not this_run.empty:
        print("\nAccuracy by model:")
        print(
            this_run.groupby('model')['accuracy']
            .agg(['mean', 'min', 'max'])
            .sort_values('mean', ascending=False)
            .round(4)
        )

        if this_run['experiment'].nunique() > 1:
            print("\nAccuracy by experiment:")
            print(
                this_run.groupby('experiment')['accuracy']
                .agg(['mean', 'min', 'max'])
                .sort_values('mean', ascending=False)
                .round(4)
            )

        if this_run['eval_strategy'].nunique() > 1:
            print("\nAccuracy by evaluation strategy:")
            print(
                this_run.groupby('eval_strategy')['accuracy']
                .agg(['mean', 'min', 'max'])
                .sort_values('mean', ascending=False)
                .round(4)
            )

        best = this_run.loc[this_run['accuracy'].idxmax()]
        print(
            f"\nBest this run: {best['model']} on {best['experiment']} "
            f"({best['eval_strategy']})  ->  {best['accuracy']:.4f}"
        )

    print("COMPLETE")
    print("-" * 70)
    print(f"Run ID      : {run_id}")
    print(f"Results CSV : {csv_path}")
    print(f"Plots       : {Config.OUTPUT_DIR / 'plots'}")


if __name__ == '__main__':
    main()
