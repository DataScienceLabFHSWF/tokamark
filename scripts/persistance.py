import os
import argparse
import pathlib as pl
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import numpy as np
import pandas as pd

import torch.multiprocessing as mp
from multiprocessing import cpu_count
from filelock import FileLock
import psutil

from MAST_benchmark.tasks import get_task_config, get_task_metadata
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_benchmark.data import initialize_MAST_dataset, initialize_model_dataset


class AutoAppendingDataFrame:
    def __init__(self, path: str, buffer_size: int = 1):
        """
        Auto-saving DataFrame that buffers rows and writes atomically after N batches.
        
        :param path: Path to the Parquet file.
        :param batch_size: Number of rows to buffer before saving.
        """
        self.path = pl.Path(path)
        self.buffer_size = buffer_size
        self.lock = FileLock(str(self.path) + ".lock")
        self.buffer = []
        self.columns = None

    def append(self, df_rows: pd.DataFrame):
        """Append rows to buffer and commit if threshold reached."""
        if self.columns is None:
            self.columns = list(df_rows.columns)

            # checking for column consistency
            if list(df_rows.columns) != self.columns:
                raise ValueError(f"Column mismatch: expected {self.columns}, got {list(df_rows.columns)}")

        self.buffer.append(df_rows)

        if len(self.buffer) >= self.buffer_size:
            self._commit()

    def _commit(self):
        """Commit buffered rows to disk atomically."""
        if not self.buffer:
            return
        # Merge buffer into main DataFrame
        df_new_data = pd.concat(self.buffer, ignore_index=True)
        self.buffer.clear()

        # Concurrent appending
        with self.lock:
            file_exists = self.path.exists() and self.path.stat().st_size > 0
            # If file doesn't exist or is empty, write header once
            df_new_data.to_csv(
                self.path,
                mode="a",                # append
                header=not file_exists,  # write header only on first write
                index=False
            )

    def flush(self):
        """Force commit of any buffered rows."""
        self._commit()

    def view(self) -> pd.DataFrame:
        """Return a copy of the current DataFrame."""
        return self.df.copy()


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


def persistance_evaluation_per_shot(
    test_dataloader,
    feature_names,
    output_writer
):
    """
    Evaluate persistence model per shot/window and save incremental RMSEs to CSV.
    """
    column_names = ["shot_id", "window_id", "feature_name", "RMSE"]
    
    # === Evaluation loop ===
    for batch_idx, batch in enumerate(test_dataloader):
        if batch is None:
            continue

        shot_id, window_id, x_test, y_test = batch

        # # Model prediction
        y_pred = {}
        for key in x_test:
            last = x_test[key][...,-1].unsqueeze(-1)
            y_pred[key] = last.expand_as(y_test[key])

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

            rmse_per_sample = np.sqrt(np.mean((y_t - y_p) ** 2, axis=1))
            # mse_per_sample = np.mean((y_t - y_p) ** 2, axis=1)

            data = np.column_stack((shot_id, 
                                    window_id, 
                                    [feature_name] * len(shot_id), 
                                    rmse_per_sample))
            df_eval = pd.DataFrame(data, columns=column_names)
            output_writer.append(df_eval)

        # # === Append to CSV ===
        # df_batch = pd.DataFrame(batch_rows)
        # df_batch.to_csv(csv_path, mode="a", header=False, index=False)

    print(f"✅ Evaluation done. RMSEs and MSEs saved (incrementally).")


if __name__ == "__main__":
    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    script_config = {
        "subset_of_shots" : 50,
        "local_flag" : True,
        "output_dir" : pl.Path(__file__).parent/'eval',
        "dataloader_setting" : {
            "batch_size" : 4,
            "num_workers": 0
        }
    }

    # -------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="task_4-4",
        help="Path to the task YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Load Task config
    config_task = get_task_config(args.task)

    # -------------------------------------------------------------------
    # Initialize task-specific metadata
    # -------------------------------------------------------------------
    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )

    # -------------------------------------------------------------------
    # Initialize MAST datasets
    # -------------------------------------------------------------------
    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index = script_config["subset_of_shots"]
    )

    test_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        test_shots_,
        local_flag = script_config["local_flag"],
        use_std_scaling = True,
        return_incomplete_shots=True
    )

    ins = [f'{source}-{signal}' for source,signal in config_task['sources_and_signals']['input_name']]
    outs = [f'{source}-{signal}' for source,signal in config_task['sources_and_signals']['output_name']]
    singnals = [signal for signal in outs if signal in ins]
    # TODO: check if signals list is empty

    test_dataset = initialize_model_dataset(
        test_MAST_dataset, dict_task_metadata, config_task, PersistanceTransform(singnals)
    )
    test_dataloader = DataLoader(
            dataset=test_dataset,
            collate_fn=MAST_collate_fn,
            **script_config["dataloader_setting"]
    )

    # -------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------

    # === Setup paths ===
    eval_file_path = script_config["output_dir"]\
        / config_task["task_name"]\
        / "rmse_per_sample.csv"
    eval_file_path.parent.mkdir(parents=True, exist_ok=True)
    if eval_file_path.exists():
        eval_file_path.unlink()
    writer = AutoAppendingDataFrame(eval_file_path)
    
    persistance_evaluation_per_shot(test_dataloader, singnals, writer)

    print('DONE')
    # cnn_save_traces_per_shot(
    #     test_dataloader, config_task, cnn_model, config_cnn, n_traces=10
    # )
