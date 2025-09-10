import os
import sys
import yaml
import argparse

import pickle
import torch
import torch.multiprocessing as mp
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------------------------------------------------
# Repo-specific imports

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(),
    "..", ".."
))  # noqa: E402

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# print(f"REPO_ROOT: {REPO_ROOT}")

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.utils.utils import (
    get_train_test_val_shots, 
    initialize_datasets, 
    initialize_dataloaders,
    ComposeTransforms
)
from scripts.pipelines.utils.cnn_utils import (
    flatten_then_collate,
    plot_shot,
    plot_shot_gif
)
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform
)
from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform
)
from scripts.pipelines.transforms.signal_level_transforms.reshape_lcfs_transform import (
ReshapeLcfsTransform
)
from scripts.pipelines.transforms.shot_level_transforms.truncation_transform import (
    TruncationTransform
)
from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform
)
from scripts.pipelines.transforms.signal_level_transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform
)
from scripts.pipelines.transforms.shot_level_transforms.drop_sample_with_nans import (
    DropSampleWithNans
)
from scripts.pipelines.transforms.shot_level_transforms.cnn_transform import CNNTransform

from cnn_pipeline import create_cnn_architecture


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    device = torch.device("mps")
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
        default= REPO_ROOT + "/scripts/pipelines/configs/config_cnn_test.yaml",
        help="Path to the config YAML file"
    )
    args, unknown = parser.parse_known_args()

    # Load parameters from YAML configuration
    with open(args.config, "r") as f:
        parameters = yaml.safe_load(f)
    print(parameters)  # optional, to verify contents

    # ..................................................................................................................
    # Parameters Setting

    # General parameters
    LOCAL_FLAG = parameters["local"]  
    SUBSET_OF_SHOTS = parameters["subset_of_shots"] 
    OUTPUT_FOLDER = parameters["paths"]["data_output_directory"]
    RUN_EVALUATION = parameters["run_evaluation"]
    
    # Data
    source_signal_list = parameters["input"]["data_names"] + parameters["input"]["target_names"] 

    # Preprocessing parameters
    ref_freq = parameters["ref_freq"]
    parameters_window_segementer = parameters["window_segmenter_setting"]
    parameters_window_segementer["x_keys"] = [ f"{source}-{signal}" for source, signal 
                                              in parameters_window_segementer["x_keys"] ]
    parameters_window_segementer["y_keys"] = [ f"{source}-{signal}" for source, signal 
                                              in parameters_window_segementer["y_keys"] ]
    # Model Architecture parameters
    cnn_args = parameters["cnn_model"]

    # Dataloader paraneters
    dataloader_setting = parameters["dataloader_setting"]


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
    with open(parameters["standardscaling_setting"]["mean_path"], "rb") as f:
        dict_mean = pickle.load(f)
    with open(parameters["standardscaling_setting"]["std_path"], "rb") as f:
        dict_std = pickle.load(f)

    # Get the user-defined composite signal transform map
    signal_transform_map = {var: ComposeTransforms([
        StdScalingTransform(dict_mean[var], dict_std[var]),
        SamplingToReferenceTimeTransform(ref_freq),
    ])
        for var in [f'{source}-{signal}' for source, signal in source_signal_list]
    }
    
    for var in ['magnetics-flux_loop_flux', 'magnetics-b_field_pol_probe_ccbv_field',
                'magnetics-b_field_pol_probe_obr_field', 'magnetics-b_field_pol_probe_obv_field', 
                'magnetics-b_field_tor_probe_saddle_voltage']:
        
        signal_transform_map[var] = ComposeTransforms([
            FillProfileWithZerosTransform(),
            StdScalingTransform(dict_mean[var], dict_std[var]), # nan on one channel in std
            # FillProfileWithZerosTransform(),
            SamplingToReferenceTimeTransform(ref_freq),
        ])
    
    for var in ['equilibrium-lcfs_r', 'equilibrium-lcfs_z']:
        
        signal_transform_map[var] = ComposeTransforms([
            ReshapeLcfsTransform(),
            StdScalingTransform(dict_mean[var], dict_std[var]),
            SamplingToReferenceTimeTransform(ref_freq),
        ])

    # ..................................................................................................................
    # For CNN pipeline

    shot_transform = ComposeTransforms([  # shape-consistent transform
        TruncationTransform(),
        WindowSegmenterTransform(**parameters_window_segementer),  # shape-modifying transform
        DropSampleWithNans(),
        CNNTransform()  # shape-modifying transform
        ])

    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots_, "val": val_shots_, "test": test_shots_},
        sig_tran_map=signal_transform_map,
        shot_tran=shot_transform,
        local_flag=LOCAL_FLAG,
        verbose=True
    )

    # Prepare dataloaders
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=flatten_then_collate,
        **dataloader_setting,
        verbose=True
    )
    test_dataloader = dataloaders_train_val_test["test"]

    # Create CNN architecture
    cnn_model = create_cnn_architecture(
        train_dataloader_=test_dataloader,
        **cnn_args,
        verbose=True
    )
    
    # ------------------------------------------------------------------------------------------------------------------
    # CNN Visualization
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    # Load trained model

    # Load saved state
    best_model_path = OUTPUT_FOLDER + "best_model.pt"
    cnn_model.load_state_dict(torch.load(best_model_path, 
                                        map_location=torch.device(device)))
    # Optional: move to device
    cnn_model.to(device)

    # ..................................................................................................................
    # Fixed visualization

    os.makedirs("shot_images", exist_ok=True)

    cnn_shot_dataset = MastDataset(
                            local=True,
                            shots_list=test_shots_[0:10],
                            source_signal_list=source_signal_list,
                            signal_level_transform_map=signal_transform_map,
                            shot_level_transform=shot_transform
                        )
    shot_dataloader = DataLoader(
            cnn_shot_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn = flatten_then_collate
        )

    cnn_model.eval()  # Set model to eval mode
    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    y_preds = []
    y_trues = []

    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(shot_dataloader):
            
            # Convert inputs to tensors with batch dim = 1
            x_test = [arr.to(torch.float32).to(device) for arr in x_test]
            y_test = [arr.to(torch.float32).to(device) for arr in y_test]  
            outputs = cnn_model(*x_test)
            
            # pred = outputs[0].cpu().squeeze(0).numpy()
            pred = [arr.cpu().numpy() for arr in outputs]
            # true = y_test[0].cpu().squeeze(0).numpy()
            true = [arr.cpu().numpy() for arr in y_test]
            y_preds.append(pred) 
            y_trues.append(true)

            # Save the image
            plot_shot(pred, true, i, ref_freq, out_dir = OUTPUT_FOLDER )
            
            # Save the gif
            plot_shot_gif(pred, true, i, ref_freq, out_dir=OUTPUT_FOLDER)
    
