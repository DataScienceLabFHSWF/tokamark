import os
import sys
import yaml
import argparse
import shutil

import pickle
import torch
import torch.multiprocessing as mp
from multiprocessing import cpu_count


# ----------------------------------------------------------------------------------------------------------------------
# Repo-specific imports

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)  # noqa: E402

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# print(f"REPO_ROOT: {REPO_ROOT}")

from torch.utils.data import DataLoader
from scripts.MAST_tools.MAST_dataset import MastDataset

from scripts.pipelines.utils.utils import (
    get_train_test_val_shots,
    initialize_datasets,
    initialize_dataloaders,
    ComposeTransforms,
)
from scripts.pipelines.utils.cnn_utils import (
    build_cnn_signal_transform_map, 
    build_cnn_shot_transform_map,
    create_cnn_architecture,
    MultiOutputMSELoss,
    loop_for_cnn_training,
    cnn_evaluation_per_shot,
    flatten_then_collate,
    get_cnn_order_scaling,
)


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    # device = torch.device("mps")
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ======================================================================================================================
if __name__ == "__main__":
    print(f"\nNumber of available cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # GENERAL SETTINGS
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Data Augmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="/scripts/pipelines/configs/config_cnn_test.yaml",
        help="Path to the config YAML file",
    )
    args, unknown = parser.parse_known_args()

    # Load parameters from YAML configuration
    with open(REPO_ROOT + args.config, "r") as f:
        parameters = yaml.safe_load(f)
    print(parameters)  # optional, to verify contents

    # ..................................................................................................................
    # Parameters Setting

    # General parameters
    LOCAL_FLAG = parameters["local"]
    SUBSET_OF_SHOTS = parameters["subset_of_shots"]
    OUTPUT_FOLDER = REPO_ROOT + parameters["paths"]["data_output_directory"]
    RUN_EVALUATION = parameters["run_evaluation"]

    # Copy the config file into the output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    config_src = REPO_ROOT + args.config
    config_dst = os.path.join(OUTPUT_FOLDER, os.path.basename(config_src))
    # shutil.copy2(config_src, config_dst)

    # Data
    source_signal_list = (
        parameters["input"]["data_names"] 
            + parameters["input"].get("exog_names", [])
            + parameters["input"]["target_names"]
    ) 
    source_signal_list = [s for i, s in enumerate(source_signal_list) if s not in source_signal_list[:i]] # Avoid repetition

    # Preprocessing parameters
    ref_freq = parameters["ref_freq"]
    parameters_window_segmenter = parameters["window_segmenter_setting"]
    
    parameters_window_segmenter["x_keys"] = [
        f"{source}-{signal}" for source, signal in parameters_window_segmenter["x_keys"]
    ]
    
    parameters_window_segmenter["exog_keys"] = parameters["input"].get("exog_names", [])
    if parameters_window_segmenter["exog_keys"] != []:
        parameters_window_segmenter["exog_keys"] = [
            f"{source}-{signal}" for source, signal in parameters_window_segmenter["exog_keys"]
            ]
    print(parameters_window_segmenter["exog_keys"])

    parameters_window_segmenter["y_keys"] = [
        f"{source}-{signal}" for source, signal in parameters_window_segmenter["y_keys"]
    ]
    
    # Model Architecture parameters
    cnn_args = parameters["cnn_model"]

    # Dataloader paraneters
    dataloader_setting = parameters["dataloader_setting"]

    # Training parameters
    training_args = parameters["training"]
    # training_args['loss_criterion'] = torch.nn.MSELoss()

    training_args["loss_criterion"] = MultiOutputMSELoss()

    # ------------------------------------------------------------------------------------------------------------------
    # PRELIMINARY TASKS
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    # Preprocessing pipeline

    # Create sets of shot IDs for training, validation and testing
    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=SUBSET_OF_SHOTS
    )

    # Fit mean and std for signal transformation
    with open(
        REPO_ROOT + parameters["standardscaling_setting"]["mean_path"], "rb"
    ) as f:
        dict_mean = pickle.load(f)
    with open(REPO_ROOT + parameters["standardscaling_setting"]["std_path"], "rb") as f:
        dict_std = pickle.load(f)

    # ..................................................................................................................
    # For CNN pipeline

    # For training preprocessing
    signal_transform_map = build_cnn_signal_transform_map(source_signal_list, dict_mean, dict_std, ref_freq)
    shot_transform = build_cnn_shot_transform_map(parameters_window_segmenter)

    # For unstandardscaling
    shot_transform_without_CNN = build_cnn_shot_transform_map(parameters_window_segmenter, remove_CNN_transform=True)

    # Get unstandardscaling order
    order_var_for_inv_std = get_cnn_order_scaling(
        LOCAL_FLAG,
        train_shots_,
        source_signal_list,
        signal_transform_map,
        shot_transform_without_CNN,
    )

    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots_, "val": val_shots_, "test": test_shots_},
        sig_tran_map=signal_transform_map,
        shot_tran=shot_transform,
        local_flag=LOCAL_FLAG,
        verbose=True,
    )

    # Prepare dataloaders
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=flatten_then_collate,
        **dataloader_setting,
        verbose=True,
    )
    train_dataloader = dataloaders_train_val_test["train"]
    val_dataloader = dataloaders_train_val_test["val"]
    # test_dataloader = dataloaders_train_val_test["test"]

    # Create CNN architecture
    cnn_model = create_cnn_architecture(
        train_dataloader_=train_dataloader, **cnn_args, verbose=True
    )

    # ------------------------------------------------------------------------------------------------------------------
    # CNN Training
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    # Training loop

    best_model_state, early_stop = loop_for_cnn_training(
        base_cnn_model=cnn_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **training_args,
        output_dir=OUTPUT_FOLDER,
        verbose=True,
    )

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
