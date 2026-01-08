import argparse
import yaml
from multiprocessing import cpu_count
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from typing import Dict, Any

from scripts.globals import REPO_ROOT

from scripts.pipeline_tools.utils import get_device
from scripts.pipeline_tools.utils import get_train_test_val_shots

from scripts.pipeline_tools.get_task_metadata import (
    get_task_metadata
)

from scripts.pipeline_tools.initialize_MAST_dataset import (
    initialize_MAST_dataset
)

from scripts.pipeline_tools.initialize_model_dataset import (
    initialize_model_dataset,
)

# Set device
device = get_device()
# print(f"Using device: {device}\n")


# ----------------------------------------------------------------------------------------------------------------------
class ModelSpecificTransform:  # TEMPLATE
    def __init__(self, verbose=False):
        # dictionary that persists across calls
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "x": (
                [data["values"] for var, data in shot["input"].items()]
                + [data["values"] for var, data in shot["actuator"].items()]
            ),
            "y": [data["values"] for var, data in shot["output"].items()],
        }

# ----------------------------------------------------------------------------------------------------------------------
def model_collate_fn(batch, verbose=True):
    flattened_batch = [
        (item["shot_id"], item["window_index"], item["x"], item["y"])
        for sublist in batch
        for item in sublist
        # if not (
        #     any(np.isnan(np.array(x)).any() for x in item['x']) or
        #     any(np.isnan(np.array(y)).any() for y in item['y'])
        #     )
    ]
    if verbose:
        print(
            f"\nNumber of samples from batch = {len(batch)} shots is N = {len(flattened_batch)}"
        )
        if len(flattened_batch) == 0:
            print("batch is None")
    return default_collate(flattened_batch) if (len(flattened_batch) > 0) else None


if __name__ == "__main__":
    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    # -------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_task",
        type=str,
        default="/configs_task/config_task_0-0.yaml",
        help="Path to the task YAML config file",
    )
    parser.add_argument(
        "--config_model",
        type=str,
        default="/scripts/pipelines/config_test.yaml",  # CHANGE HERE
        help="Path to the model YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Load Task YAML config
    with open(REPO_ROOT + args.config_task, "r") as f:
        config_task = yaml.safe_load(f)

    # Load CNN YAML config
    with open(REPO_ROOT + args.config_model, "r") as f:
        config_model = yaml.safe_load(f)

    # -------------------------------------------------------------------
    # Initialize datasets and metadata
    # -------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config_task["subset_of_shots"]
    )

    train_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        train_shots_,
        use_std_scaling = False,
        return_incomplete_shots=True
    )
    val_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        val_shots_,
        use_std_scaling = False,
        return_incomplete_shots=True
    )
    test_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        test_shots_,
        use_std_scaling = False,
        return_incomplete_shots=True
    )

    # -------------------------------------------------------------------
    # Initialize task-specific metadata
    # -------------------------------------------------------------------

    dict_metadata = get_task_metadata(
        config_task,
        verbose=False
    )

    # -------------------------------------------------------------------
    # EXAMPLE WITH MODEL SPECIFIC PIPELINE
    # -------------------------------------------------------------------

    # train_model_dataset = initialize_model_dataset(
    #     train_MAST_dataset, dict_metadata, config_task, None, verbose=True
    # )


    # # The set of windows of 1 shot is saved in <generator object TaskModelTransformWrapper.__getitem__ at 0x559637ec8200>

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

    # -------------------------------------------------------------------
    # EXAMPLE WITH MODEL SPECIFIC PIPELINE
    # -------------------------------------------------------------------

    model_specific_transform = ModelSpecificTransform() # CHANGE HERE

    train_model_dataset = initialize_model_dataset(
        train_MAST_dataset,
        dict_metadata,
        config_task,
        model_specific_transform,
        verbose = False)
    val_model_dataset = initialize_model_dataset(
        val_MAST_dataset,
        dict_metadata,
        config_task,
        model_specific_transform,
        verbose = False)
    test_model_dataset = initialize_model_dataset(
        test_MAST_dataset,
        dict_metadata,
        config_task,
        model_specific_transform,
        verbose = False)

    train_dataloader = DataLoader(
            dataset=train_model_dataset,
            collate_fn=model_collate_fn,
            **config_model['dataloader_setting']
        )

    for batch_idx, batch in enumerate(train_dataloader):

        print(f"\nBatch {batch_idx}")
        # print(batch)
        shot_id, window_index, x_train, y_train = batch

        print("The list of shot ID is an object of shape, ", shot_id.shape)
        print("The list of window Index is an object of shape, ", window_index.shape)
        print("The x_train has been collated to shape (B, ..., T), ", [arr.shape for arr in x_train])
        print("The y_train has been collated to shape (B, ..., T), ", [arr.shape for arr in y_train])

    print(x_train[0][0:10])
    print("\n\n\n")
    print(y_train[0][0:10])
