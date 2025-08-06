import os
import sys
import csv

import pickle
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in globals()
                                         else os.getcwd(), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# print(f"REPO_ROOT: {REPO_ROOT}")

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.main_pipeline.utils.utils import read_data_split_csv, flatten_then_collate
from scripts.main_pipeline.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.main_pipeline.preprocessing.standardscaling_preprocessing import get_mean_shot, get_std_shot
from scripts.main_pipeline.utils.utils import ComposeTransforms

from scripts.main_pipeline.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import(
    StdScalingTransform
)
from scripts.main_pipeline.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform
)
from scripts.main_pipeline.transforms.shot_level_transforms.truncation_transform import (
    TruncationTransform
)
from scripts.main_pipeline.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform
)
from scripts.main_pipeline.transforms.signal_level_transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform
)
from scripts.main_pipeline.transforms.shot_level_transforms.drop_sample_with_nans import (
    DropSampleWithNans
)
from scripts.main_pipeline.transforms.shot_level_transforms.cnn_transform import CNNTransform

from scripts.main_pipeline.models.cnn_model import MultiBranchCNNModel


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ----------------------------------------------------------------------------------------------------------------------
def get_train_test_val_shots(max_index=None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index:
        train_sh = train_sh[0:max_index]
        val_sh = val_sh[0:max_index]
        test_sh = test_sh[0:max_index]

    return train_sh, test_sh, val_sh


# ----------------------------------------------------------------------------------------------------------------------
def fit_mean_and_std_for_signal_transform(output_sub_dir, verbose=False):

    if verbose:
        print('\n\n----------TRANSFORM FITTING----------\n')

    preprocessing_train_dataset = MastDataset(
        local=LOCAL_FLAG,
        shots_list=yamane_sampled_shot_list(train_shots, error=0.05),
        source_signal_list=source_signal_list,
        signal_level_transform_map=None,
        shot_level_transform=None
    )

    if verbose:
        print(f"len(preprocessing_train_dataset): {len(preprocessing_train_dataset)}")

    dict_mean_ = get_mean_shot(preprocessing_train_dataset)
    if verbose: 
        print(f"dict_mean_ is {dict_mean_}")
    dict_std_ = get_std_shot(preprocessing_train_dataset)
    if verbose: 
        print(f"dict_std_ is {dict_std_}")

    # Save dict_mean and dict_std used!

    output_dir_ = os.path.join("output", output_sub_dir)
    os.makedirs(output_dir_, exist_ok=True)
    if verbose:
        print(f"Output folder to save fitted mean and std dicts: {output_dir_}")

    with open(output_dir_ + 'dict_mean_shot.pkl', 'wb') as f_:
        pickle.dump(dict_mean_, f_)
    with open(output_dir_ + 'dict_std_shot.pkl', 'wb') as f_:
        pickle.dump(dict_std_, f_)

    return dict_mean_, dict_std_


# ----------------------------------------------------------------------------------------------------------------------
def initialize_datasets(
        sources_and_signals,
        shots,
        sig_tran_map,
        shot_tran,
        local_flag=False,
        verbose=False

):

    datasets_ = {"train": None, "val": None, "test": None}

    # ..................................................................................................................
    # Train

    if shots["train"]:
        datasets_["train"] = MastDataset(
            local=local_flag,
            shots_list=shots["train"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran
        )
        if verbose:
            print(f"len(mast_train_dataset): {len(datasets_['train'])}")

    # ..................................................................................................................
    # Val

    if shots["val"]:
        datasets_["val"] = MastDataset(
            local=local_flag,
            shots_list=shots["val"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran
        )
        if verbose:
            print(f"len(val_dataset): {len(datasets_['val'])}")

    # ..................................................................................................................
    # Test

    if shots["test"]:
        datasets_["test"] = MastDataset(
            local=local_flag,
            shots_list=shots["test"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran
        )
        if verbose:
            print(f"len(test_dataset): {len(datasets_['test'])}")

    # ..................................................................................................................
    # Return

    return datasets_


# ----------------------------------------------------------------------------------------------------------------------
def initialize_dataloaders(
        datasets,
        collate_function,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=False,
        verbose=False
):

    dataloaders_ = {"train": None, "val": None, "test": None}

    if verbose:
        print('\n\n----------DATASET & DATALOADER INITIALIZATION----------\n')

    # ..................................................................................................................
    # Train

    if datasets["train"]:
        dataloaders_["train"] = DataLoader(
            dataset=datasets["train"],
            batch_size=batch_size,
            # batch_size=len(datasets['train']),
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function
        )

    # ..................................................................................................................
    # Val

    if datasets["val"]:
        dataloaders_["val"] = DataLoader(
            dataset=datasets["val"],
            batch_size=batch_size,
            # batch_size=len(datasets["val"]),
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function
        )

    # ..................................................................................................................
    # Test

    if datasets["test"]:
        dataloaders_["test"] = DataLoader(
            dataset=datasets["test"],
            batch_size=batch_size,
            # batch_size=len(datasets["test"]),
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function
        )

    # ..................................................................................................................
    # Return

    return dataloaders_


# ----------------------------------------------------------------------------------------------------------------------
def create_cnn_architecture(
        train_dataloader_,
        verbose=False
):

    if verbose:
        print("\n\n----------MODEL INITIALIZATION----------\n")

    input_shapes = [arr.shape for arr in train_dataloader_.dataset[0][0][0]]
    if verbose:
        print(f"input_shapes: {input_shapes}")

    output_shape = [arr.shape for arr in train_dataloader_.dataset[0][0][1]]
    if verbose:
        print(f"output_shape: {output_shape}")

    return MultiBranchCNNModel(input_shapes, output_shape).to(device)


# ----------------------------------------------------------------------------------------------------------------------
def loop_for_cnn_training(
        base_cnn_model,
        lr,
        best_val_loss,
        loss_criterion,
        patience,
        output_sub_dir,
        verbose=False

):

    if verbose:
        print('\n\n----------CNN TRAINING----------\n')

    output_dir_ = os.path.join("output", output_sub_dir)
    os.makedirs(output_dir_, exist_ok=True)
    if verbose:
        print(f"Output folder to save trained model: {output_dir_}")

    optimizer = torch.optim.Adam(base_cnn_model.parameters(), lr=lr)

    best_model_state_ = None
    early_stop_ = False
    epochs_no_improve = 0
    for epoch in range(MAX_EPOCHS):
        base_cnn_model.train()
        running_loss = 0.0
        num_batches = 0

        if verbose:
            print(f'\nEpoch {epoch+1}\n')

        for batch_idx, (x_train, y_train) in enumerate(train_dataloader):

            x_train = [arr.to(torch.float32).to(device) for arr in x_train]
            # print([arr.shape for arr in x_train])
            # print(y_train.min().item(), y_train.max().item())

            y_train = y_train[0].to(torch.float32).to(device)
            if verbose:
                # print(y_train.shape)
                print(f'Batch {batch_idx} size is {len(y_train)}')

            outputs_ = base_cnn_model(*x_train).squeeze()
            loss_ = loss_criterion(outputs_, y_train)
            if verbose:
                # print(f"outputs' shape: {outputs_.shape}")
                print(f'Batch loss: {loss_}')

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

            running_loss += loss_.item()
            num_batches += len(y_train)
            if verbose:
                print(f'Actual num_batches is {num_batches}')

        avg_loss = running_loss / num_batches
        # print(f"Epoch [{epoch+1}/{MAX_EPOCHS}], Average Loss: {avg_loss:.4f}")

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

                val_running_loss += val_loss.item()
                val_batches += len(y_val)

        avg_val_loss = val_running_loss / val_batches
        if verbose:
            print(f"Epoch [{epoch+1}/{MAX_EPOCHS}], Average Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state_ = base_cnn_model.state_dict()

            # Save best model state
            torch.save(best_model_state_, output_dir_ + "best_model.pt")

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


# ======================================================================================================================
if __name__ == "__main__":

    print(f"\nNumber of available cores: {cpu_count()}\n")

    # ------------------------------------------------------------------------------------------------------------------
    # GENERAL SETTINGS
    # ------------------------------------------------------------------------------------------------------------------

    LOCAL_FLAG = False
    mp.set_start_method("spawn", force=True)

    # ..................................................................................................................
    # For common pipeline

    SUBSET_OF_SHOTS = 4  # <- This can be None for the entire dataset, or a small integer.
    OUTPUT_SUB_FOLDER = 'cnn_output/'  # <- Sub-folder within /output/
    BATCH_SIZE = 2  # 500
    NUM_WORKERS = 2  # 64
    MAX_EPOCHS = 3  # 500

    REF_FREQ = 0.005

    source_signal_list = [
        ('magnetics', 'flux_loop_flux'),
        ('magnetics', 'b_field_pol_probe_ccbv_field'),
        ('magnetics', 'b_field_pol_probe_obr_field'),
        ('magnetics', 'b_field_pol_probe_obv_field'),
        ('pf_active', 'solenoid_current'),
        ('pf_active', 'coil_voltage'),
        ('pf_active', 'coil_current'),
        ('pulse_schedule', 'i_plasma'),
        ('summary', 'power_nbi'),
        ('equilibrium', 'psi'),
        ('equilibrium', 'elongation'),
        ('equilibrium', 'elongation_axis'),
        ('equilibrium', 'triangularity_upper'),
        ('equilibrium', 'triangularity_lower'),
        ('equilibrium', 'minor_radius'),
        ('equilibrium', 'magnetic_axis_r'),
        ('equilibrium', 'magnetic_axis_z')
    ]

    # ..................................................................................................................
    # For CNN pipeline

    PARAMETERS_WINDOWS_SEGMENTER = {
        'x_keys': [
            # 'magnetics-flux_loop_flux',
            # 'magnetics-b_field_pol_probe_ccbv_field',
            # 'magnetics-b_field_pol_probe_obr_field',
            # 'magnetics-b_field_pol_probe_obv_field',
            # 'pf_active-solenoid_current',
            # 'pf_active-coil_voltage',
            # 'pf_active-coil_current',
            # 'pulse_schedule-i_plasma',
            # 'summary-power_nbi',
            'equilibrium-psi'
        ],
        'y_keys': [
            'equilibrium-elongation',
            # 'equilibrium-elongation_axis',
            # 'equilibrium-triangularity_upper',
            # 'equilibrium-triangularity_lower',
            # 'equilibrium-minor_radius',
            # 'equilibrium-magnetic_axis_r',
            # 'equilibrium-magnetic_axis_z',
        ],
        'x_window_sec': 0,
        'y_window_sec': 0,
        'dt_sec': 0.025,
        'stride_sec': None,
        'stride_unitary': True,
        'min_samples_per_window': 1,
        'verbose': False,
    }

    # ..................................................................................................................
    # For CNN training

    LOSS_CRITERION = torch.nn.MSELoss()
    LEARNING_RATE = 1e-4
    BEST_VALUE_LOSS = float('inf')
    PATIENCE = 5  # <- Maximum number of admissible epochs without improvement
    RUN_EVALUATION = True

    # ------------------------------------------------------------------------------------------------------------------
    # PRELIMINARY TASKS
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    # For common pipeline

    # Create sets of shot IDs for training, validation and testing
    train_shots, test_shots, val_shots = get_train_test_val_shots(
        max_index=SUBSET_OF_SHOTS
    )

    # Fit mean and std for signal transformation
    dict_mean, dict_std = fit_mean_and_std_for_signal_transform(
        output_sub_dir=OUTPUT_SUB_FOLDER,
        verbose=True
    )

    # Get the user-defined composite signal transform map
    signal_transform_map = {var: ComposeTransforms([
        # ForwardFillImputerTransform(),
        # SamplewiseNormalizeTransform(),
        FillProfileWithZerosTransform(),
        StdScalingTransform(dict_mean[var], dict_std[var]),
        # FillWithZerosImputerTransform(),
        SamplingToReferenceTimeTransform(REF_FREQ),
    ])
        for var in [f'{source}-{signal}' for source, signal in source_signal_list]
    }
    signal_transform_map['equilibrium-psi']= ComposeTransforms([
        # ForwardFillImputerTransform(),
        # SamplewiseNormalizeTransform(),
        # FillProfileWithZerosTransform(),
        StdScalingTransform(dict_mean['equilibrium-psi'], dict_std['equilibrium-psi']),
        # FillWithZerosImputerTransform(),
        SamplingToReferenceTimeTransform(REF_FREQ),
    ])

    # ..................................................................................................................
    # For CNN pipeline

    shot_transform = ComposeTransforms([  # shape-consistent transform
        TruncationTransform(),
        WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),  # shape-modifying transform
        DropSampleWithNans(),
        CNNTransform()  # shape-modifying transform
        ])

    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": test_shots},
        sig_tran_map=signal_transform_map,
        shot_tran=shot_transform,
        local_flag=LOCAL_FLAG,
        verbose=True
    )

    # Prepare dataloaders
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=flatten_then_collate,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        verbose=True
    )
    train_dataloader = dataloaders_train_val_test["train"]
    val_dataloader = dataloaders_train_val_test["val"]
    # test_dataloader = dataloaders_train_val_test["test"]

    # Create CNN architecture
    cnn_model = create_cnn_architecture(
        train_dataloader_=train_dataloader,
        verbose=True
    )
    
    # ------------------------------------------------------------------------------------------------------------------
    # CNN Training
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    # Training loop

    best_model_state, early_stop = loop_for_cnn_training(
        base_cnn_model=cnn_model,
        lr=LEARNING_RATE,
        best_val_loss=BEST_VALUE_LOSS,
        loss_criterion=LOSS_CRITERION,
        patience=PATIENCE,
        output_sub_dir=OUTPUT_SUB_FOLDER,
        verbose=True
    )

    # ..................................................................................................................
    # Evaluation

    output_dir = os.path.join("output", OUTPUT_SUB_FOLDER)

    if RUN_EVALUATION:

        if best_model_state is not None:

            # Restore best model weights
            cnn_model.load_state_dict(best_model_state)

            # Set the model to evaluation mode
            cnn_model.eval()

            # Evaluation per shot

            with open(output_dir + 'test_loss_per_var.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['shot_id', f"MSE for {PARAMETERS_WINDOWS_SEGMENTER['y_keys']}"])  # Header

                for shot_id in test_shots:
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
                        num_workers=16,
                        shuffle=True,
                        drop_last=False,
                        collate_fn=flatten_then_collate
                    )
                
                    if len(test_shot_dataloader.dataset[0]) > 0:  # I.e. if this is a valid shot with windows

                        test_loss_per_var = torch.tensor(
                            [0 for i in range(test_shot_dataloader.dataset[0][0][1][0].shape[0])]
                        ).to(device)
                    
                        test_batches = 0

                        with torch.no_grad():  # Disable gradient calculation for efficiency
                            for x_test, y_test in test_shot_dataloader:
                                x_test = [arr.to(torch.float32).to(device) for arr in x_test]
                                y_test = y_test[0].to(torch.float32).to(device)

                                outputs = cnn_model(*x_test).squeeze()

                                loss_per_var = torch.nn.MSELoss(reduction='none')(outputs, y_test).mean(dim=0)
                                test_loss_per_var = test_loss_per_var + loss_per_var

                                test_batches += len(y_test)

                        avg_test_loss = test_loss_per_var / test_batches
                        print(f"Test Loss: {avg_test_loss}")

                        writer.writerow([shot_id, avg_test_loss.cpu().tolist()])
                        f.flush()
                    
                    else:
                        print(f"Shot {shot_id} not run properly, likely empty slice")
                        continue

    # ------------------------------------------------------------------------------------------------------------------
