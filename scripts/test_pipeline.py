"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

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
    initialize_MAST_dataset, initialize_model_dataset
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
        for sublist in batch
        for item in sublist
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
        name_or_flags="--task",
        type=str,
        default="task_1-1",
        help="The name of the task available in the benchmark",
    )
    parser.add_argument(
        name_or_flags="--config_model",
        type=str,
        default="scripts/config_test.yaml",  # CHANGE HERE
        # default="/home/ir-zaya1/fusion/fairmast-data-preprocessing/scripts/config_test.yaml",
        help="Path to the model YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Note: instead of loading benchmark task, here we load a simple task from external file
    config_task = get_config_from_yaml("scripts/config_task_0-0.yaml")
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
        shots_list=train_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
        remove_outliers=True,
        verbose=True
    )
    val_MAST_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=val_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
        remove_outliers=True,
        verbose=True
    )
    test_MAST_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=test_shots_,
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

    # train_model_dataset = initialize_model_dataset(
    #     train_MAST_dataset, dict_metadata, config_task, None, verbose=True
    # )

    # # Set of 1-shot windows saved in <generator object TaskModelTransformWrapper.__getitem__ at 0x559637ec8200>

    # # You can transform them to list of list via:
    # list_list_shot_0 = [
    #     (
    #         item["shot_id"],
    #         item["window_index"],
    #         item["input"],
    #         item["actuator"],
    #         item["output"],
    #     )
    #     for item in train_model_dataset[0]
    # ]

    # shot_id, window_index, input, actuator, output = list_list_shot_0[0]
    # print("\n\n\nshot_id is ", shot_id)
    # print("window_id is ", window_index)
    # print("input")
    # print(input.keys())
    # print([input[var]["values"].shape for var in input])
    # print("actuator")
    # print(actuator.keys())
    # print([actuator[var]["values"].shape for var in actuator])
    # print("output")
    # print(output.keys())
    # print([output[var]["values"].shape for var in output])

    # # Or keep a list of dict with:
    # list_dict_shot_0 = [item for item in train_model_dataset[0]]
    # print(list_dict_shot_0[0].keys())
    # print("\n\n\nshot_id is ", list_dict_shot_0[0]["shot_id"])
    # print("window_id is ", list_dict_shot_0[0]["window_index"])
    # print("input, ", list_dict_shot_0[0]["input"].keys())
    # print("actuator, ", list_dict_shot_0[0]["actuator"].keys())
    # print("output, ", list_dict_shot_0[0]["output"].keys())

    # ------------------------------------------------------------------------------------------------------------------
    # EXAMPLE WITH MODEL SPECIFIC PIPELINE
    # ------------------------------------------------------------------------------------------------------------------

    model_specific_transform = ModelSpecificTransform()  # likely depends on dict_task_metadata

    train_model_dataset = initialize_model_dataset(
        dataset=train_MAST_dataset,
        dict_task_metadata=dict_task_metadata,
        config_task=config_task,
        model_specific_transform=model_specific_transform,
        test_mode=True,
        verbose=False
    )
    val_model_dataset = initialize_model_dataset(
        dataset=val_MAST_dataset,
        dict_task_metadata=dict_task_metadata,
        config_task=config_task,
        model_specific_transform=model_specific_transform,
        test_mode=True,
        verbose=False
    )
    test_model_dataset = initialize_model_dataset(
        dataset=test_MAST_dataset,
        dict_task_metadata=dict_task_metadata,
        config_task=config_task,
        model_specific_transform=model_specific_transform,
        test_mode=True,
        verbose=False
    )

    train_dataloader = DataLoader(
            dataset=train_model_dataset,
            collate_fn=model_collate_fn,
            **config_model['dataloader_setting']
        )

    for batch_idx, batch_ in enumerate(train_dataloader):

        print(f"\nBatch {batch_idx}")
        # print(batch_)
        shot_id, window_index, x_train, y_train = batch_  # FIXME: Warning about more values to unpack.

        print("The list of shot ID is an object of shape, ", shot_id.shape)
        print("The list of window Index is an object of shape, ", window_index.shape)
        print("The x_train has been collated to shape (B, ..., T), ", [arr.shape for arr in x_train])
        # print("Mean x_train", [torch.nanmean(arr) for arr in x_train])
        # print("Std x_train", [np.nanstd(arr) for arr in x_train])
        print("The y_train has been collated to shape (B, ..., T), ", [arr.shape for arr in y_train])
        # print("Mean y_train", [torch.nanmean(arr) for arr in y_train])
        # print("Std y_train", [np.nanstd(arr) for arr in y_train])

    # print(x_train[0][0:10])
    # print("\n\n\n")
    # print(y_train[0][0:10])
