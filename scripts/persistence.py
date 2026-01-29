import os
import argparse
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from torch import ones_like
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
import pandas as pd

import torch.multiprocessing as mp
from multiprocessing import cpu_count
import psutil

from MAST_benchmark.tasks import get_task_config, get_task_metadata, get_signals_metadata
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_benchmark.data import initialize_MAST_dataset, initialize_model_dataset
from MAST_benchmark.evaluator import WindowMetricsWriter, compute_task_metrics, compute_all_metrics


class PersistanceTransform:

    def __init__(self, signals):
        self.signals = signals

    def __call__(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        sample = {}

        sample['shot_id'] = segment['shot_id']
        sample['window_id'] = segment['window_index']

        def get_signals_data(label):
            subset = {}
            for signal_name in self.signals:
                if signal_name in segment[label]:
                    subset[signal_name] = segment[label][signal_name]['values']
            return subset

        sample['x'] = get_signals_data("input")
        sample['y'] = get_signals_data("output")

        return sample


def MAST_collate_fn(batch, verbose=True):
    # Flatten the batch of lists into a single list
    flattened_batch = []

    for shot in batch:
        for sample in shot:
            def contains_nans(data):
                return any(np.isnan(x).any() for x in data)
            
            # if not (contains_nans(sample["x"])
            #         or contains_nans(sample["y"])):
            flattened_batch.append((sample["shot_id"], 
                                    sample["window_index"], 
                                    sample["x"], 
                                    sample["y"]))

    flattened_batch = default_collate(flattened_batch) if (len(flattened_batch) > 0) else None

    if verbose:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info().rss / (1024**2)
        print(f"[Worker PID={proc.pid}] Memory={mem:.2f} MB")

        if flattened_batch is None:
            print("batch is None")
        else:
            print(
                f"The batch contains: {len(batch)} shots and {len(flattened_batch[0])} samples"
            )

            x = flattened_batch[2]
            y = flattened_batch[3]
            tensor_size = lambda t: t.nelement() * t.element_size() / 1024**2
            size = sum([tensor_size(t) for _,t in x.items()])
            size = size + sum([tensor_size(t) for _,t in y.items()])
            print(f"X and Y tensors memory={size:.2f} MB")

    return flattened_batch


def persistance_evaluation_loop(
    test_dataloader,
    feature_names,
    window_metrics,
    model = 'persistence'
):
    """
    Evaluate persistence model per shot/window and save incremental RMSEs to CSV.
    """
    
    # === Evaluation loop ===
    for batch_idx, batch in tqdm(enumerate(test_dataloader)):
        if batch is None:
            continue

        shot_id, window_id, x_test, y_test = batch

        # tqdm.write(f'-> Processing batch {batch_idx} which has {len(shot_id.unique())} shots and {len(shot_id)} elements.')
        print(f'-> Processing batch {batch_idx} which has {len(shot_id.unique())} shots and {len(shot_id)} elements.')

        # # Model prediction
        y_pred = {}
        if model == 'persistence':
            for key in x_test:
                last = x_test[key][...,-1].unsqueeze(-1)
                y_pred[key] = last.expand_as(y_test[key])
        elif model == 'mean':
            for key in y_test:
                signal_std = get_signals_metadata()
                std = signal_std[key]['std']
                y_pred[key] = ones_like(y_test[key]) * std
        else:
            print(f'Persistence pipeline: model {model} is not known.')


        # === Compute RMSEs per feature ===
        for feature_name in feature_names:
            y_t = (
                y_test[feature_name]
                .squeeze(1)
                .reshape(len(shot_id), -1)
                .numpy()
            )
            y_p = (
                y_pred[feature_name]
                .squeeze(1)
                .reshape(len(shot_id), -1)
                .numpy()
            )

            window_metrics.compute_and_append(y_t, y_p, shot_id, window_id, feature_name)

    print(f"✅ Evaluation done. RMSEs and MSEs saved (incrementally).")


def run_persistence_pipeline(task, pipeline_config):
    # -------------------------------------------------------------------
    # Load task config and modify it to keep auto-regressive data
    # -------------------------------------------------------------------
    config_task = get_task_config(task)

    if pipeline_config['model'] == 'persistence':
        ins = [f'{source}-{signal}' for source,signal in config_task['sources_and_signals']['input_name']]
        outs = [f'{source}-{signal}' for source,signal in config_task['sources_and_signals']['output_name']]
        signals = [signal for signal in outs if signal in ins]

        config_task['sources_and_signals']['input_name'] = [[source,signal] for source,signal in config_task['sources_and_signals']['input_name'] 
                    if f'{source}-{signal}' in signals]
        config_task['task_window_segmenter']['input_keys'] = config_task['sources_and_signals']['input_name']

        config_task['sources_and_signals']['output_name'] = [[source,signal] for source,signal in config_task['sources_and_signals']['output_name'] 
                    if f'{source}-{signal}' in signals]
        config_task['task_window_segmenter']['output_keys'] = config_task['sources_and_signals']['output_name']

        config_task['sources_and_signals']['actuator_name'] = None
        config_task['task_window_segmenter']['actuator_keys'] = None
    else:
        signals = [f'{source}-{signal}' for source,signal in config_task['sources_and_signals']['output_name']]

        config_task['sources_and_signals']['input_name'] = None
        config_task['task_window_segmenter']['input_keys'] = None
        config_task['sources_and_signals']['actuator_name'] = None
        config_task['task_window_segmenter']['actuator_keys'] = None


    # -------------------------------------------------------------------
    # Initialize task-specific metadata
    # -------------------------------------------------------------------
    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )

    if len(signals) > 0:
        # -------------------------------------------------------------------
        # Initialize MAST datasets
        # -------------------------------------------------------------------
        train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
            max_index = pipeline_config["subset_of_shots"]
        )
        print('Number of test shots: ', len(test_shots_))

        test_MAST_dataset = initialize_MAST_dataset( 
            config_task,
            test_shots_,
            local_flag = pipeline_config["local_flag"],
            use_std_scaling = True,
            return_incomplete_shots=True
        )

        test_dataset = initialize_model_dataset(
            test_MAST_dataset, 
            dict_task_metadata, 
            config_task, 
            PersistanceTransform(signals),
            test_mode = True
        )
        test_dataloader = DataLoader(
                dataset=test_dataset,
                collate_fn=MAST_collate_fn,
                **pipeline_config["dataloader_setting"]
        )

        # -------------------------------------------------------------------
        # Evaluation
        # -------------------------------------------------------------------
        window_metrics = WindowMetricsWriter(task, pipeline_config["output_dir"])
        persistance_evaluation_loop(test_dataloader, signals, window_metrics, model=pipeline_config["model"])
        compute_task_metrics(task, pipeline_config["output_dir"])
    else:
        print('No common signals between input and output - not possible to run persistance.')


if __name__ == "__main__":
    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    pipeline_config = {
        "subset_of_shots" : None,
        "local_flag" : True,
        "output_dir" : Path(__file__).parents[1]/'output/mean',
        "dataloader_setting" : {
            "batch_size" : 4,
            "num_workers": 4
        },
        "model" : "mean"
        # "model" : "persistence"
    }

    # -------------------------------------------------------------------
    # Looping over auto-regressive tasks
    # -------------------------------------------------------------------
    # persistence model tasks
    # From the persistence point of view, tasks 'task_4-1', 'task_4-2' are identical to 'task_3-2'
    # ar_tasks = ['task_2-1', 'task_3-1', 'task_3-2', 'task_4-4', 'task_4-5']

    # mean model tasks
    ar_tasks = ['task_2-1', 'task_2-2', 'task_2-3']
    for task in ar_tasks:
        print('---------------------------------------------')
        print('Running persistence pipeline for ', task)
        run_persistence_pipeline(task, pipeline_config)
        print('Finished persistence pipeline for ', task)
        print('---------------------------------------------\n\n')

    
    # compute_all_metrics(pipeline_config["output_dir"])
    print('DONE')


# sintr -A ukaea-ap002-cpu -p ukaea-amp -N1 -n1 --gres=gpu:1 -t 2:0:0
# sintr --gres=gpu:1 -A ukaea-ap002-gpu -p ukaea-amp -N1 -n1 -t 1:00:00

# sintr -A ukaea-ap002-cpu -p ukaea-amp -N1 -n1 --exclusive -t 2:0:0


# sintr -A ukaea-ap002-cpu -p ukaea-icl -N1 --ntasks=1 --cpus-per-task=32 -t 1:00:00

# task 3-2
            # "batch_size" : 24,
            # "num_workers": 6

# task 4-5
            # "batch_size" : 12,
            # "num_workers": 3

