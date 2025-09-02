import os
import sys
import csv

import yaml
import argparse

import pickle
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from multiprocessing import cpu_count


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
from scripts.pipelines.utils.utils import read_data_split_csv, flatten_then_collate
from scripts.pipelines.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.pipelines.preprocessing.standardscaling_preprocessing import get_mean_shot, get_std_shot
from scripts.pipelines.utils.utils import ComposeTransforms
from utils.cnn_utils import get_train_test_val_shots, initialize_datasets, initialize_dataloaders

from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform
)
from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform
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

from scripts.pipelines.models.cnn_model import MultiBranchCNNModel


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ----------------------------------------------------------------------------------------------------------------------
def create_cnn_architecture(
        train_dataloader_,
        D,
        verbose=False
):
    print(D)
    if verbose:
        print("\n\n----------MODEL INITIALIZATION----------\n")

    input_shapes = [arr.shape for arr in train_dataloader_.dataset[0][0][0]]
    if verbose:
        print(f"input_shapes: {input_shapes}")

    output_shape = [arr.shape for arr in train_dataloader_.dataset[0][0][1]]
    if verbose:
        print(f"output_shape: {output_shape}")

    return MultiBranchCNNModel(input_shapes, output_shape, D).to(device)


# ----------------------------------------------------------------------------------------------------------------------
def loop_for_cnn_training(
        base_cnn_model,
        train_dataloader, 
        val_dataloader,
        lr,
        max_epochs,
        loss_criterion,
        patience,
        output_dir,
        verbose=True

):

    if verbose:
        print('\n\n----------CNN TRAINING----------\n')

    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"Output folder to save trained model: {output_dir}")

    optimizer = torch.optim.Adam(base_cnn_model.parameters(), lr=lr)

    best_model_state_ = None
    best_val_loss = float('inf')
    early_stop_ = False
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        base_cnn_model.train()
        running_loss = 0.0
        num_batches = 0

        if verbose:
            print(f'\nEpoch {epoch+1}\n')

        for batch_idx, (x_train, y_train) in enumerate(train_dataloader):

            x_train = [arr.to(torch.float32).to(device) for arr in x_train]
            y_train = y_train[0].to(torch.float32).to(device)

            actual_batch_size = len(y_train)
            if verbose:
                # print(y_train.shape)
                print(f'Batch {batch_idx} size is {actual_batch_size}')

            outputs_ = base_cnn_model(*x_train).squeeze()
            loss_ = loss_criterion(outputs_, y_train)
            if verbose:
                # print(f"outputs' shape: {outputs_.shape}")
                print(f'Batch loss: {loss_}')

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

            running_loss += loss_.item() * actual_batch_size
            num_batches += actual_batch_size

        avg_loss = running_loss / num_batches

        if verbose:
            print(f"Epoch [{epoch+1}/{max_epochs}], Average Loss: {avg_loss:.4f}")

        # Validation phase & Early stopping check

        base_cnn_model.eval()
        val_running_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_val, y_val in val_dataloader:
                x_val = [arr.to(torch.float32).to(device) for arr in x_val]
                y_val = y_val[0].to(torch.float32).to(device)

                val_outputs = base_cnn_model(*x_val).squeeze()
                val_loss = loss_criterion(val_outputs, y_val)

                actual_batch_size = len(y_val)

                val_running_loss += val_loss.item() * actual_batch_size
                val_batches += actual_batch_size

        avg_val_loss = val_running_loss / val_batches
        
        if verbose:
            print(f"Epoch [{epoch+1}/{max_epochs}], Average Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state_ = base_cnn_model.state_dict()

            # Save best model state
            torch.save(best_model_state_, output_dir + "best_model.pt")

        else:
            epochs_no_improve += 1
            if verbose:
                print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                early_stop_ = True
                if verbose:
                    print("Early stopping triggered.")
                break

    return best_model_state_, early_stop_


# ----------------------------------------------------------------------------------------------------------------------
def cnn_evaluation_per_shot(
        cnn_model,
        test_dataloader,
    ):
        cnn_model.eval()

        test_loss_per_var = torch.tensor(
            [0 for i in range(test_dataloader.dataset[0][0][1][0].shape[0])]
        ).to(device)
    
        test_batches = 0

        with torch.no_grad():  # Disable gradient calculation for efficiency
            for x_test, y_test in test_dataloader:
                x_test = [arr.to(torch.float32).to(device) for arr in x_test]
                y_test = y_test[0].to(torch.float32).to(device)

                outputs = cnn_model(*x_test).squeeze()

                loss_per_var = torch.nn.MSELoss(reduction='none')(outputs, y_test).mean(dim=0)
                test_loss_per_var = test_loss_per_var + loss_per_var

                test_batches += len(y_test)

        avg_test_loss = test_loss_per_var / test_batches
        print(f"Test Loss: {avg_test_loss}")

        return(avg_test_loss)


# ======================================================================================================================
if __name__ == "__main__":

    print(f"\nNumber of available cores: {cpu_count()}\n")
    # mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # GENERAL SETTINGS
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Data Augmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/ir-rous1/hncdi-fusion-plasma/fairmast-data-preprocessing/scripts/pipelines/configs/config_cnn_test.yaml",
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

    # Training parameters
    training_args = parameters["training"]
    training_args['loss_criterion'] = torch.nn.MSELoss()


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
        FillProfileWithZerosTransform(),
        StdScalingTransform(dict_mean[var], dict_std[var]),
        SamplingToReferenceTimeTransform(ref_freq),
    ])
        for var in [f'{source}-{signal}' for source, signal in source_signal_list]
    }

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
    train_dataloader = dataloaders_train_val_test["train"]
    val_dataloader = dataloaders_train_val_test["val"]
    # test_dataloader = dataloaders_train_val_test["test"]

    # Create CNN architecture
    cnn_model = create_cnn_architecture(
        train_dataloader_=train_dataloader,
        **cnn_args,
        verbose=True
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
        verbose=True
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

        with open(OUTPUT_FOLDER + 'test_loss_per_var.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['shot_id', f"MSE for {parameters_window_segementer['y_keys']}"])  # Header

            for shot_id in test_shots_:
                print(f'Evaluating shot {shot_id}')
            
                test_shot_dataset = MastDataset(
                    local=LOCAL_FLAG,
                    shots_list=[shot_id],
                    source_signal_list=source_signal_list,
                    signal_level_transform_map=signal_transform_map,
                    shot_level_transform=shot_transform
                )

                test_shot_dataloader = DataLoader(
                    dataset=test_shot_dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=flatten_then_collate
                )
            
                if len(test_shot_dataloader.dataset[0]) > 0:  # I.e. if this is a valid shot with windows

                    avg_test_loss = cnn_evaluation_per_shot(
                                        cnn_model,
                                        test_shot_dataloader,
                                    )
                    print(f"Test Loss: {avg_test_loss}")

                    writer.writerow([shot_id, avg_test_loss.cpu().tolist()])
                    f.flush()
                
                else:
                    print(f"Shot {shot_id} not run properly, likely empty slice")
                    continue


    # ------------------------------------------------------------------------------------------------------------------