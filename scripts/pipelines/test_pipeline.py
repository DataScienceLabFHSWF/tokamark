import argparse
import yaml
from multiprocessing import cpu_count
import torch.multiprocessing as mp

from torch.utils.data._utils.collate import default_collate
import numpy as np
from typing import Dict, Any

# -------------------------------------------------------------------
# Repo-specific imports
# -------------------------------------------------------------------
from globals import REPO_ROOT
from scripts.pipelines.utils.device_utils import get_device

from scripts.pipelines.utils.utils import ( 
    initialize_model_datasets,
    initialize_dataloaders, 
)

from scripts.pipelines.utils.preprocessing_utils import initialize_datasets_and_metadata_for_task

# Set device
device = get_device()
# print(f"Using device: {device}\n")

class ModelSpecificTransform: # TEMPLATE

    def __init__(self, verbose=False):
        # dictionary that persists across calls
        self.verbose = verbose
 
    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        
        return {
            'x': ( [ data["values"] for var, data in shot["input"].items() ] 
                    + [ data["values"] for var, data in shot["actuator"].items() ]
                 ),
            'y': [ data["values"] for var, data in shot["output"].items() ]
            }

def model_collate_fn(batch, verbose=True):
    flattened_batch = [ (item['shot_id'], item['window_index'], item['x'], item['y'])
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
        if (len(flattened_batch) == 0):
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
        default="/scripts/pipelines/configs/configs_task/config_task_0-0.yaml",
        help="Path to the task YAML config file",
    )
    parser.add_argument(
        "--config_model",
        type=str,
        default="/scripts/pipelines/configs/configs_cnn/config_cnn_reconstruction.yaml", # CHANGE HERE
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
    datasets_train_val_test, dict_metadata = initialize_datasets_and_metadata_for_task(config_task) 

    # -------------------------------------------------------------------
    # MODEL SPECIFIC pipeline
    # -------------------------------------------------------------------

    model_specific_transform = ModelSpecificTransform() # CHANGE HERE
    
    datasets_model = initialize_model_datasets(
        datasets_train_val_test,
        dict_metadata, 
        config_task,
        model_specific_transform,
        verbose = True)
    
    dataloaders_model = initialize_dataloaders( datasets_model,
                                                model_collate_fn,
                                                **config_model['dataloader_setting'])

    # -------------------------------------------------------------------
    # MODEL SPECIFIC training
    # -------------------------------------------------------------------

    train_dataloader = dataloaders_model["train"]
    for batch_idx, batch in enumerate(train_dataloader):
        
        print(f"\nBatch {batch_idx}")
        # print(batch)
        shot_id, window_index, x_train, y_train = batch

        print("The list of shot ID is an object of shape, ", shot_id.shape)
        print("The list of window Index is an object of shape, ", window_index.shape)
        print("The x_train has been collated to shape (B, ..., T), ", [arr.shape for arr in x_train])
        print("The y_train has been collated to shape (B, ..., T), ", [arr.shape for arr in y_train])

        # print(x_train[0][0:10])
        # print("\n\n\n")
        # print(y_train[0][0:10])