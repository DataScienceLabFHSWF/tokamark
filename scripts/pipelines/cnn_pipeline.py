import argparse
import yaml
from multiprocessing import cpu_count
import torch.multiprocessing as mp

# -------------------------------------------------------------------
# Repo-specific imports
# -------------------------------------------------------------------
from globals import REPO_ROOT
from pipelines.utils.device_utils import get_device
from pipelines.utils.preprocessing_utils import initialize_datasets_and_metadata_for_task
from pipelines.utils.cnn_utils import (
    initialize_cnn_dataloaders_and_models,
    loop_for_cnn_training,
)

# Set device
device = get_device()
print(f"Using device: {device}\n")


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
        default="/pipelines/configs/configs_task/config_task_0.yaml",
        help="Path to the task YAML config file",
    )
    parser.add_argument(
        "--config_cnn",
        type=str,
        default="/pipelines/configs/configs_cnn/config_cnn_test.yaml",
        help="Path to the model YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Load Task YAML config
    with open(REPO_ROOT + args.config_task, "r") as f:
        config_task = yaml.safe_load(f)
    print("Task configuration:")
    print(config_task)

    # Load CNN YAML config
    with open(REPO_ROOT + args.config_cnn, "r") as f:
        config_cnn = yaml.safe_load(f)
    print("Model configuration:")
    print(config_cnn)

    # -------------------------------------------------------------------
    # Initialize datasets and metadata
    # -------------------------------------------------------------------
    datasets_train_val_test, dict_metadata = initialize_datasets_and_metadata_for_task(config_task) 

    # -------------------------------------------------------------------
    # CNN pipeline
    # -------------------------------------------------------------------
    dataloaders_cnn, cnn_model = initialize_cnn_dataloaders_and_models(
        datasets_train_val_test,
        dict_metadata,
        config_cnn,
        verbose=True,
    )

    first_batch = next(iter(dataloaders_cnn["train"]))
    print("First batch from train dataloader:")
    print(first_batch)
    print("CNN model architecture:")
    print(cnn_model)

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    best_model_state, early_stop = loop_for_cnn_training(
        base_cnn_model=cnn_model,
        train_dataloader=dataloaders_cnn["train"],
        val_dataloader=dataloaders_cnn["val"],
        **config_cnn["training_args"],
        output_dir=REPO_ROOT + config_cnn["paths"]["data_output_directory"],
        verbose=True,
    )

    # -------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
    # CNN Evaluation PER SHOT
    # ------------------------------------------------------------------------------------------------------------------

    test_dataset = datasets_train_val_test["test"]
    best_model_path = OUTPUT_FOLDER + "best_model.pt"
    # Restore best model weights
    cnn_model.load_state_dict(torch.load(best_model_path, map_location=device))
    cnn_model.eval()

    # Evaluation per shot

    if RUN_EVALUATION:
        cnn_evaluation_per_shot(cnn_model, 
                                test_shots_, 
                                LOCAL_FLAG,
                                source_signal_list,
                                signal_transform_map,
                                shot_transform,
                                order_var_for_inv_std,
                                dict_mean,
                                dict_std,
                                OUTPUT_FOLDER)

    # ------------------------------------------------------------------------------------------------------------------
