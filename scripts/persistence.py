"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
from pathlib import Path
from typing import Any, Sequence
from collections.abc import Mapping
from tqdm import tqdm
import numpy as np
from torch import ones_like, Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate  # noqa (access to protected method)
import torch.multiprocessing as mp
from multiprocessing import cpu_count
import psutil

from MAST_benchmark.tasks import get_task_config, get_task_metadata, get_signals_metadata
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_benchmark.data import initialize_MAST_dataset, initialize_model_dataset
from MAST_benchmark.evaluator import WindowMetricsWriter, compute_task_metrics, compute_all_metrics


# ======================================================================================================================
class PersistenceTransform:
    """
    Class for the Persistence Transform.

    Attributes
    ----------
    signals : list[str]
        List of source-signal items.

    Methods
    -------
    __call__(segment)
        Call method for the class to behave like a function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            signals: list[str]
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        signals : list[str]
            List of source-signal items.

        Returns
        -------
        None

        """

        self.signals = signals

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(
            self,
            segment: Mapping[str, Any]
    ) -> dict[str, Any]:
        """
        Call method for the class to behave like a function.

        Parameters
        ----------
        segment : Mapping[str, Any]
            Dictionary with metadata of target segment.

        Returns
        -------
        dict[str, Any]
            Signal data sample.

        """

        # ..............................................................................................................
        def get_signals_data(label: str) -> dict[str, Any]:
            """Get signal data given target label."""

            subset = {}
            for signal_name in self.signals:
                if signal_name in segment[label]:
                    subset[signal_name] = segment[label][signal_name]["values"]
            return subset

        # ..............................................................................................................

        return {
            "shot_id": segment["shot_id"],
            "window_index": segment["window_index"],
            "x": get_signals_data("input"),
            "y": get_signals_data("output")
        }

# ----------------------------------------------------------------------------------------------------------------------
def MAST_collate_fn(  # noqa N802
        batch: Sequence,
        verbose: bool = True
) -> list[Any]:
    """
    MAST collate function.

    Parameters
    ----------
    batch: Sequence
        Batch of data samples.
    verbose : bool
        If True, activate verbose mode.
        Optional. Default: True.

    Returns
    -------
    list[Any]
        Flattened batch.

    """

    # ..............................................................................................................
    def contains_nans(data: Any) -> bool:  # TODO: Should we keep this? [Mike]
        """Check if given data contains NaN values."""
        return any(np.isnan(x_).any() for x_ in data)

    # ..............................................................................................................
    def tensor_size(t: Tensor) -> float:
        """Get size of given tensor."""
        return t.nelement() * t.element_size() / 1024 ** 2

    # ..............................................................................................................

    # Flatten the batch of lists into a single list
    flattened_batch = []

    for shot in batch:
        for sample in shot:
            
            # if not (contains_nans(sample["x"])
            #         or contains_nans(sample["y"])):  # TODO: Should we keep this? [Mike]
            flattened_batch.append(
                (sample["shot_id"], sample["window_index"], sample["x"], sample["y"])
            )

    flattened_batch = default_collate(flattened_batch) if (len(flattened_batch) > 0) else None

    if verbose:
        proc = psutil.Process(pid=os.getpid())
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
            size = sum([tensor_size(t=t) for _, t in x.items()])
            size += sum([tensor_size(t=t) for _, t in y.items()])
            print(f"X and Y tensors memory={size:.2f} MB")

    return flattened_batch


# ----------------------------------------------------------------------------------------------------------------------
def persistence_evaluation_loop(
    test_dataloader: DataLoader,
    feature_names: list[str],
    window_metrics: WindowMetricsWriter,
    model: str = "persistence"
) -> None:
    """
    Evaluate persistence model per shot/window and save incremental RMSEs to CSV.

    Parameters
    ----------
    test_dataloader : DataLoader
        Test DataLoader instance.
    feature_names : list[str]
        List of feature (signal) names.
    window_metrics : WindowMetricsWriter
        Input WindowMetricsWriter instance.
    model : str
        Model type. Valid options are "persistence" and "mean".
        Optional. Default: "persistence".

    Returns
    -------
    None

    """
    
    # ..................................................................................................................
    # Evaluation loop
    # ..................................................................................................................

    for batch_idx, batch_ in tqdm(enumerate(test_dataloader)):
        if batch_ is None:
            print("WARNING: Empty batch skipped.")
            continue

        shot_ids, window_indices, x_test, y_test = batch_  # noqa (right number of values to unpack)

        # tqdm.write(
        #     f"-> Processing batch {batch_idx} which has {len(shot_id.unique())} shots and {len(shot_id)} elements."
        # )
        print(f"-> Processing batch {batch_idx} which has {len(shot_ids.unique())} shots and {len(shot_ids)} elements.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model prediction
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        y_pred = {}
        if model == "persistence":
            for key in x_test:
                last = x_test[key][..., -1].unsqueeze(-1)
                y_pred[key] = last.expand_as(other=y_test[key])
        elif model == "mean":
            signal_metadata = get_signals_metadata()
            for key in y_test:
                std = signal_metadata[key]["std"]
                y_pred[key] = ones_like(input=y_test[key]) * std  # FIXME: This should use mean, not std. So? [Mike]
        else:
            print(f"Persistence pipeline: model {model} is not known.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute RMSEs per feature
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for feature_name in feature_names:
            y_t = (
                y_test[feature_name]
                .squeeze(1)
                .reshape(len(shot_ids), -1)
                .numpy()
            )
            y_p = (
                y_pred[feature_name]
                .squeeze(1)
                .reshape(len(shot_ids), -1)
                .numpy()
            )

            window_metrics.compute_and_append(
                y_target=y_t,
                y_pred=y_p,
                shot_ids=shot_ids,
                window_indices=window_indices,
                feature_name=feature_name
            )

    print(f"\n✅ Evaluation done. RMSEs and MSEs saved (incrementally).")


# ----------------------------------------------------------------------------------------------------------------------
def run_persistence_pipeline(
        task: str,
        pipeline_config: Mapping[str, Any]
) -> None:
    """
    Parameters
    ----------
    task : str
        Target benchmark task.
    pipeline_config : Mapping[str, Any]
        Dictionary with pipeline configuration.

    Returns
    -------
    None

    """

    # ..................................................................................................................
    # Load task configuration and modify it to keep auto-regressive data
    # ..................................................................................................................

    config_task = get_task_config(task_name=task)

    if pipeline_config["model"] == "persistence":

        # Pipeline for "persistence" model

        # 1. Identify chosen signals
        ins = [f"{source}-{signal}" for source, signal in config_task["sources_and_signals"]["input_name"]]
        outs = [f"{source}-{signal}" for source, signal in config_task["sources_and_signals"]["output_name"]]
        chosen_signals = [signal for signal in outs if signal in ins]

        # 2. Update values of `input_name` field
        config_task["sources_and_signals"]["input_name"] = [
            [source, signal] for source, signal in config_task["sources_and_signals"]["input_name"]
            if f"{source}-{signal}" in chosen_signals
        ]

        # 3. Re-assign `input_keys` values with updated `input_name` values
        config_task["task_window_segmenter"]["input_keys"] = config_task["sources_and_signals"]["input_name"]

        # 4. Update values of `output_name` field
        config_task["sources_and_signals"]["output_name"] = [
            [source, signal] for source, signal in config_task["sources_and_signals"]["output_name"]
            if f"{source}-{signal}" in chosen_signals
        ]

        # 5. Re-assign `output_keys` values with updated `output_name` values
        config_task["task_window_segmenter"]["output_keys"] = config_task["sources_and_signals"]["output_name"]

        # 6. Set `actuator_name` and `actuator_keys` values to None
        config_task["sources_and_signals"]["actuator_name"] = None
        config_task["task_window_segmenter"]["actuator_keys"] = None

    else:

        # Pipeline for "mean" model

        # 1. Identify chosen signals
        chosen_signals = [f"{source}-{signal}" for source, signal in config_task["sources_and_signals"]["output_name"]]

        # 2. Set values of `input_name` field
        config_task["sources_and_signals"]["input_name"] = [["magnetics", "flux_loop_flux"]]  # FIXME: Why these two? [Mike]
        # TODO: Check if this should be `output_name` instead. [Mike]

        # 3. Re-assign `input_keys` values with updated `input_name` values
        config_task["task_window_segmenter"]["input_keys"] = config_task["sources_and_signals"]["input_name"]
        # TODO: Check if this should have `output_keys` and `output_name` instead. [Mike]

        # 3. Set `actuator_name` and `actuator_keys` values to None
        config_task["sources_and_signals"]["actuator_name"] = None
        config_task["task_window_segmenter"]["actuator_keys"] = None

    # ..................................................................................................................
    # Proceed if chosen signals are available
    # ..................................................................................................................

    if len(chosen_signals) > 0:

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize task-specific metadata
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dict_task_metadata = get_task_metadata(
            config_task=config_task,
            verbose=False
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize MAST datasets
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
            max_index=pipeline_config["max_shot_index"],
            shuffle=pipeline_config["shuffle"]
        )
        print(f"Number of test shots: {len(test_shots_)}")

        test_mast_dataset = initialize_MAST_dataset(
            config_task=config_task,
            shots_list=test_shots_,
            local_flag=pipeline_config["local_flag"],
            use_std_scaling=True,
            store_manager_settings=pipeline_config.get("store_manager_settings"),
            return_incomplete_shots=True
        )

        test_dataset = initialize_model_dataset(
            dataset=test_mast_dataset,
            dict_task_metadata=dict_task_metadata,
            config_task=config_task,
            model_specific_transform=PersistenceTransform(chosen_signals),
            test_mode=True
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            collate_fn=MAST_collate_fn,
            **pipeline_config["dataloader_setting"]
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        window_metrics = WindowMetricsWriter(
            task=task,
            output_dir=pipeline_config["output_dir"]
        )

        persistence_evaluation_loop(
            test_dataloader=test_dataloader,
            feature_names=chosen_signals,
            window_metrics=window_metrics,
            model=pipeline_config["model"]
        )

        compute_task_metrics(
            task=task,
            output_dir=pipeline_config["output_dir"]
        )

    else:
        print("No common signals between input and output - not possible to run persistence.")  # REMARK: Only case.


# ======================================================================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------------------------------------------------------------
    MEAN_MODE = False  # TODO: Re-run and check for both modes. [Rodrigo]
    DEMO_MODE = False  # TODO: Implement pipeline. [Rodrigo]

    if MEAN_MODE:

        # Model
        default_model = "mean"

        # Mean model tasks
        ar_tasks = ["task_2-1", "task_2-2", "task_2-3"]

    else:

        # Model
        default_model = "persistence"

        # Persistence model tasks
        # REMARK: From the persistence point of view, tasks "task_4-1", "task_4-2" are identical to "task_3-2".
        ar_tasks = ["task_2-1", "task_3-1", "task_3-2", "task_4-4", "task_4-5"]

    # ------------------------------------------------------------------------------------------------------------------

    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method(method="spawn", force=True)

    pipeline_config_ = {  # TODO: Perhaps use external configs (full and demo)? [Rodrigo]
        "max_shot_index": None,  # If None, all available shot IDs are used.
        "shuffle": False,
        "seed": 42,
        "local_flag": False,
        "store_manager_settings": {
            "base_local_zarr_path": "/mast/tokamark/v1"
        },
        "output_dir": Path(__file__).parents[1]/"output/mean",  # FIXME: Use correct directory. Also, should be <model>? [Rodrigo, Mike]
        "dataloader_setting": {
            "batch_size": 4,
            "num_workers": 0
        },
        "model": default_model,
        "all_metrics": False
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Looping over auto-regressive tasks
    # ------------------------------------------------------------------------------------------------------------------

    for task_ in ar_tasks:
        print("---------------------------------------------")
        print(f"Running persistence pipeline for {task_}...\n")
        run_persistence_pipeline(task=task_, pipeline_config=pipeline_config_)
        print(f"\nFinished persistence pipeline for {task_}.")
        print("---------------------------------------------\n\n")

    if pipeline_config_["all_metrics"]:
        print("\nComputing all metrics...")
        compute_all_metrics(output_dir=pipeline_config_["output_dir"])

    print("DONE")
