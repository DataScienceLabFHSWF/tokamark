"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import time
import argparse
import json
from typing import Any, Sequence, Optional
from collections.abc import Mapping
from multiprocessing import cpu_count
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate  # noqa (access to protected method)

from MAST_tools.constants import (
    DEFAULT_CONFIG_MODEL_TEST_FILE,
    DEFAULT_CONFIG_MODEL_TEST_DEMO_FILE,
    DEFAULT_CONFIG_TASK_TEST_FILE,
    DEFAULT_BASE_LOCAL_ZARR_PATH
)
from MAST_benchmark.tools.utils import get_device, get_config_from_yaml
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_benchmark.tasks import get_task_metadata, get_task_config
from MAST_benchmark.data import initialize_MAST_dataset, initialize_TokaMark_dataset


# ------------------------------------------------------------------------------------------------------------------

DEMO_MODE = True

if DEMO_MODE:
    default_config_model_test_file = DEFAULT_CONFIG_MODEL_TEST_DEMO_FILE
else:
    default_config_model_test_file = DEFAULT_CONFIG_MODEL_TEST_FILE

# ------------------------------------------------------------------------------------------------------------------
# Preliminaries

start = time.perf_counter()

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

        # dictionary that persists across calls  # FIXME: Is this comment relevant? [Cecile]
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(
            self,
            shot: Mapping[str, Any]
    ) -> dict[str, Any]:
        """
        Call method.

        Parameters
        ----------
        shot : Dict[str, Any]
            Target shot.

        Returns
        -------
        dict[str, Any]
            Dictionary with "x" and "y" keys and values from `shot["input"] + shot["actuator"]` and `shot["output"]`
            items, respectively.

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
        Optional. Default: False.

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
        default="task_test",
        help="The name of the task available in the benchmark"
    )
    parser.add_argument(
        "--config_model",
        type=str,
        default=default_config_model_test_file,
        help="Path to the model YAML config file"
    )
    parser.add_argument(
        "--omit_std_scaling",
        action="store_true",
        help="Omit STD scaling. If not provided, it defaults to `omit_std_scaling = False`, which in turn results in "
             "use_std_scaling = False."
    )
    parser.add_argument(
        "--skip_incomplete_shots",
        action="store_true",
        help="Skip incomplete shots. If not provided, it defaults to `skip_incomplete_shots = False`, which in turn "
             "results in `return_incomplete_shots = True`."
    )
    parser.add_argument(
        "--keep_outliers",
        action="store_true",
        help="Keep outliers. If not provided, it defaults to `keep_outliers = False`, which in turn results in "
             "`remove_outliers = True`."
    )
    parser.add_argument(
        "--store_manager_settings",
        type=json.loads,
        help="User-defined store manager settings for the target MAST_dataset instance as defined in "
             "`src.MAST_tools.MAST_dataset.StoreManagerParameters`, passed as (keyword, value) pairs in a JSON object. "
             "It is useful to provide store manager settings different than the default ones defined in "
             "`src.MAST_tools.store_utils.MASTStorageManager.__init__`.",
        default='{\"base_local_zarr_path\":\"' + DEFAULT_BASE_LOCAL_ZARR_PATH + '\"}'
    )

    args, _ = parser.parse_known_args()

    # Trick to default boolean parameters to True
    args.use_std_scaling = not args.omit_std_scaling
    args.return_incomplete_shots = not args.skip_incomplete_shots
    args.remove_outliers = not args.keep_outliers

    if args.task == "task_test":
        # Instead of loading benchmark task, here we load a simple task from external file
        config_task = get_config_from_yaml(file_path=DEFAULT_CONFIG_TASK_TEST_FILE)
    else:
        # Otherwise, use the provided benchmark task
        config_task = get_task_config(task_name=args.task)

    # Load CNN YAML config
    config_model = get_config_from_yaml(file_path=args.config_model)

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize datasets and metadata
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config_model["max_shot_index"],
        shuffle=config_model["shuffle"],  # This defaults to False.
        seed=config_model["seed"]
    )

    local_flag = config_model["local"]

    train_MAST_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=train_shots_[0:100],  # FIXME: Why this 100? This is directly managed by "max_index" above. [Cecile]
        local_flag=local_flag,
        use_std_scaling=args.use_std_scaling,                   # It defaults to True
        return_incomplete_shots=args.return_incomplete_shots,   # It defaults to True
        remove_outliers=args.remove_outliers,                   # It defaults to True
        store_manager_settings=args.store_manager_settings,
        verbose=True
    )

    # val_MAST_dataset = initialize_MAST_dataset(  # FIXME: Unused instance [Cecile]
    #     config_task=config_task,
    #     shots_list=val_shots_[0:100],
    #     local_flag=local_flag,
    #     use_std_scaling=args.use_std_scaling,                   # It defaults to True
    #     return_incomplete_shots=args.return_incomplete_shots,   # It defaults to True
    #     remove_outliers=args.remove_outliers,                   # It defaults to True
    #     store_manager_settings=args.store_manager_settings,
    #     verbose=True
    # )

    # test_MAST_dataset = initialize_MAST_dataset(  # FIXME: Unused instance [Cecile]
    #     config_task=config_task,
    #     shots_list=test_shots_[0:100],
    #     local_flag=local_flag,
    #     use_std_scaling=args.use_std_scaling,                   # It defaults to True
    #     return_incomplete_shots=args.return_incomplete_shots,   # It defaults to True
    #     remove_outliers=args.remove_outliers,                   # It defaults to True
    #     store_manager_settings=args.store_manager_settings,
    #     verbose=True
    # )

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize task-specific metadata
    # ------------------------------------------------------------------------------------------------------------------

    dict_task_metadata = get_task_metadata(
        config_task=config_task,
        verbose=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # EXAMPLE WITH MODEL SPECIFIC PIPELINE
    # ------------------------------------------------------------------------------------------------------------------

    model_specific_transform = ModelSpecificTransform()  # Likely depends on dict_task_metadata

    train_model_dataset = initialize_TokaMark_dataset(
        dataset=train_MAST_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True,
        shuffle_windows=False,
        verbose=False
    )

    train_dataloader = DataLoader(
            dataset=train_model_dataset,
            collate_fn=model_collate_fn,
            **config_model["dataloader_setting"],
            pin_memory=True,
            # drop_last=True
        )

    # ..................................................................................................................
    # Evaluation loop
    # ..................................................................................................................

    for batch_idx, batch_ in enumerate(train_dataloader):

        print(f"\nBatch {batch_idx}")
        shot_id, window_index, x_train, y_train = batch_  # noqa (right number of values to unpack)

        print(f"The list of shot ID is {shot_id}")
        print(f"The list of window index is {window_index}")

        print("The x_train has been collated to shape (B, ..., T), ", [arr.shape for arr in x_train])
        # print("Mean x_train", [torch.nanmean(arr) for arr in x_train])
        # print("Std x_train", [np.nanstd(arr) for arr in x_train])

        print("The y_train has been collated to shape (B, ..., T), ", [arr.shape for arr in y_train])
        # print("Mean y_train", [torch.nanmean(arr) for arr in y_train])
        # print("Std y_train", [np.nanstd(arr) for arr in y_train])

        print("____________________________________________________\n")

    # print(x_train[0][0:10])
    # print("\n\n\n")
    # print(y_train[0][0:10])


end = time.perf_counter()

print("\n-----------------------------")
print(f"Elapsed time: {end - start:.4f} seconds")
