"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""
import time

start = time.perf_counter()

import torch 
import numpy as np
import argparse
from typing import Dict, Any, Sequence, Optional
from multiprocessing import cpu_count
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from MAST_benchmark.tools.utils import get_device
from MAST_benchmark.tools.utils import get_config_from_yaml
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_benchmark.tasks import get_task_metadata
from MAST_benchmark.data import (
    initialize_MAST_dataset, 
    initialize_TokaMark_dataset
)


# Set device
device = get_device()
# print(f"Using device: {device}\n")

# ======================================================================================================================
class ModelSpecificTransform:  # TEMPLATE
    """
    Model specific transform.

    Attributes
    ----------
    verbose : bool
        If True, activate verbose mode.

    Methods
    -------
    __call__(shot)
        Call method.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            verbose=False
    ) -> None:
        """
        Initialise class attributes.

        Parameters
        ----------
        verbose : bool
            If True, activate verbose mode.

        Returns
        -------
        None

        """

        # dictionary that persists across calls
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(
            self,
            shot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call method.

        Parameters
        ----------
        shot : Dict[str, Any]
            Target shot.

        Returns
        -------
        Dict[str, Any]
            Mapping with x and y values.

        """

        return {
            "x": (
                [data["values"] for var, data in shot["input"].items()]
                + [data["values"] for var, data in shot["actuator"].items()]
            ),
            "y": [data["values"] for var, data in shot["output"].items()],
        }


# ----------------------------------------------------------------------------------------------------------------------
def model_collate_fn(
        batch: Sequence,
        verbose: bool = False
) -> Optional[Any]:
    """
    Model collate function.

    Parameters
    ----------
    batch : Sequence
        Input batch.
    verbose : bool
        If True, activate verbose mode.

    Returns
    -------
    Optional[Any]
        Default collate function evaluated on flattened batch if feasible, None otherwise.

    """

    flattened_batch = [
        (item["shot_id"], item["window_index"], item["x"], item["y"])
        for item in batch
    ]

    if verbose:
        print(
            f"\nNumber of shots in a batch = {len(batch)}; number of samples (segments) = {len(flattened_batch)}"
        )
        if len(flattened_batch) == 0:
            print("batch is None")

    return default_collate(flattened_batch) if (len(flattened_batch) > 0) else None


# ======================================================================================================================
if __name__ == "__main__":

    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method(method="spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="task_1-1",
        help="The name of the task available in the benchmark",
    )
    parser.add_argument(
        "--config_model",
        type=str,
        # default="fairmast-data-preprocessing/scripts/config_model_test.yaml",
        default="config_model_test.yaml",

        help="Path to the model YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Note: instead of loading benchmark task, here we load a simple task from external file
    # config_task = get_config_from_yaml("fairmast-data-preprocessing/scripts/config_task_test.yaml")
    config_task = get_config_from_yaml("config_task_test.yaml")

    # Note: Uncomment the next 2 lines to use benchmark tasks
    # from MAST_benchmark.tasks import get_task_config
    # config_task = get_task_config(args.task)

    # Load CNN YAML config
    config_model = get_config_from_yaml(args.config_model)

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize datasets and metadata
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config_model["subset_of_shots"]
    )

    local_flag = config_model["local"]

    train_MAST_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=train_shots_[0:100],
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
        remove_outliers=True,
        verbose=True
    )
    val_MAST_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=val_shots_[0:100],
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
        remove_outliers=True,
        verbose=True
    )
    test_MAST_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=test_shots_[0:100],
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
        remove_outliers=True,
        verbose=True
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize task-specific metadata
    # ------------------------------------------------------------------------------------------------------------------

    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # EXAMPLE WITH MODEL SPECIFIC PIPELINE
    # ------------------------------------------------------------------------------------------------------------------

    model_specific_transform = ModelSpecificTransform()  # likely depends on dict_task_metadata

    train_model_dataset = initialize_TokaMark_dataset(
        dataset=train_MAST_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True,
        shuffle_windows = False,
        verbose=False
    )

    train_dataloader = DataLoader(
            dataset=train_model_dataset,
            collate_fn=model_collate_fn,
            **config_model["dataloader_setting"],
            pin_memory=True,
            # drop_last=True,
        )

    for batch_idx, batch_ in enumerate(train_dataloader):

        print(f"\nBatch {batch_idx}")
        # print(batch_)
        shot_id, window_index, x_train, y_train = batch_

        print("The list of shot ID is ", shot_id)
        print("The list of window Index is ", window_index)

        print("The x_train has been collated to shape (B, ..., T), ", [arr.shape for arr in x_train])
        # print("Mean x_train", [torch.nanmean(arr) for arr in x_train])
        # print("Std x_train", [np.nanstd(arr) for arr in x_train])

        print("The y_train has been collated to shape (B, ..., T), ", [arr.shape for arr in y_train])
        # print("Mean y_train", [torch.nanmean(arr) for arr in y_train])
        # print("Std y_train", [np.nanstd(arr) for arr in y_train])

    # print(x_train[0][0:10])
    # print("\n\n\n")
    # print(y_train[0][0:10])


end = time.perf_counter()

print(f"Elapsed time: {end - start:.4f} seconds")