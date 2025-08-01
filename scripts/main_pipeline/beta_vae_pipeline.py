import os
import sys
import pickle
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from collections import defaultdict

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print(f"REPO_ROOT: {REPO_ROOT}")

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.main_pipeline.utils.utils import read_data_split_csv
from scripts.main_pipeline.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.main_pipeline.preprocessing.standardscaling_preprocessing import (
    get_mean_shot,
    get_std_shot,
)
from scripts.main_pipeline.utils.utils import ComposeTransforms
from scripts.main_pipeline.transforms.signal_level_transforms.fill_with_zeros_imputer_transform import (
    FillWithZerosImputerTransform,
)
from scripts.main_pipeline.transforms.signal_level_transforms.forward_fill_imputer_transform import (
    ForwardFillImputerTransform,
)
from scripts.main_pipeline.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform,
)
from scripts.main_pipeline.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform,
)
from scripts.main_pipeline.transforms.shot_level_transforms.truncation_transform import (
    TruncationTransform,
)
from scripts.main_pipeline.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)
from scripts.main_pipeline.transforms.shot_level_transforms.beta_vae_transform import (
    BetaVAETransform,
)
from scripts.main_pipeline.models.beta_vae_model import BetaVAE
from multiprocessing import cpu_count

print(f"\nNumber of Cores: {cpu_count()}\n")

# Determine device to train on
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_train_test_val_shots(max_index=None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index:
        train_sh = train_sh[0:max_index]
        val_sh = val_sh[0:max_index]
        test_sh = test_sh[0:max_index]

    return train_sh, test_sh, val_sh


def fit_mean_and_std_for_signal_transform(
    output_sub_dir, verbose=False, use_existing=False
):
    """
    Fit or load mean and std for signal transformation.

    Args:
        output_sub_dir: Directory to save/load fitted parameters
        verbose: Print verbose output
        use_existing: If True, try to load existing fitted parameters instead of re-fitting
    """
    output_dir = os.path.join("output", output_sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    mean_path = os.path.join(output_dir, "dict_mean_shot.pkl")
    std_path = os.path.join(output_dir, "dict_std_shot.pkl")

    # Try to load existing files if requested
    if use_existing and os.path.exists(mean_path) and os.path.exists(std_path):
        if verbose:
            print("\n\n----------LOADING EXISTING FITTED PARAMETERS----------\n")
            print(f"Loading fitted parameters from: {output_dir}")

        try:
            with open(mean_path, "rb") as f:
                dict_mean_ = pickle.load(f)
            with open(std_path, "rb") as f:
                dict_std_ = pickle.load(f)

            if verbose:
                print(f"Successfully loaded mean and std dictionaries")
                print(f"Mean dict keys: {list(dict_mean_.keys())}")
                print(f"Std dict keys: {list(dict_std_.keys())}")

            return dict_mean_, dict_std_

        except Exception as e:
            if verbose:
                print(f"Error loading existing fitted parameters: {e}")
                print("Falling back to re-fitting")

    if verbose:
        print("\n\n----------TRANSFORM FITTING----------\n")

    preprocessing_train_dataset = MastDataset(
        local=LOCAL_FLAG,
        shots_list=yamane_sampled_shot_list(train_shots, error=0.05),
        source_signal_list=source_signal_list,
        signal_level_transform_map=None,
        shot_level_transform=None,
    )

    if verbose:
        print(f"len(preprocessing_train_dataset): {len(preprocessing_train_dataset)}")

    dict_mean_ = get_mean_shot(preprocessing_train_dataset)
    dict_std_ = get_std_shot(preprocessing_train_dataset)

    # Save fitted parameters
    if verbose:
        print(f"Output folder to save fitted mean and std dicts: {output_dir}")

    with open(mean_path, "wb") as f:
        pickle.dump(dict_mean_, f)
    with open(std_path, "wb") as f:
        pickle.dump(dict_std_, f)

    return dict_mean_, dict_std_


def beta_vae_collate_fn(batch):
    """Custom collate function for β-VAE training"""
    print(f"Collating β-VAE batch of size {len(batch)}")

    # Flatten the batch of lists into a single list
    flattened_batch = [item for sublist in batch for item in sublist]
    print(
        f"Number of signal segments from batch = {len(batch)} shots is N = {len(flattened_batch)}"
    )

    # Group by signal name
    signal_groups = defaultdict(list)
    for item in flattened_batch:
        signal_groups[item["signal_name"]].append(item["data"])

    # Convert to tensors for each signal group
    batched_signals = {}
    for signal_name, data_list in signal_groups.items():
        try:
            batched_signals[signal_name] = torch.stack(
                [torch.from_numpy(data) for data in data_list]
            )
        except Exception as e:
            print(f"Error batching signal {signal_name}: {e}")
            continue

    return batched_signals


def initialize_datasets(
    sources_and_signals, shots, sig_tran_map, shot_tran, local_flag=False, verbose=False
):
    datasets_ = {"train": None, "val": None, "test": None}

    # Train
    if shots["train"]:
        datasets_["train"] = MastDataset(
            local=local_flag,
            shots_list=shots["train"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
        )
        if verbose:
            print(f"len(mast_train_dataset): {len(datasets_['train'])}")

    # Val
    if shots["val"]:
        datasets_["val"] = MastDataset(
            local=local_flag,
            shots_list=shots["val"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
        )
        if verbose:
            print(f"len(val_dataset): {len(datasets_['val'])}")

    # Test
    if shots["test"]:
        datasets_["test"] = MastDataset(
            local=local_flag,
            shots_list=shots["test"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
        )
        if verbose:
            print(f"len(test_dataset): {len(datasets_['test'])}")

    return datasets_


def initialize_dataloaders(
    datasets,
    collate_function,
    batch_size,
    num_workers,
    shuffle=True,
    drop_last=False,
    verbose=False,
):
    dataloaders_ = {"train": None, "val": None, "test": None}

    if verbose:
        print("\n\n----------DATASET & DATALOADER INITIALIZATION----------\n")

    # Train
    if datasets["train"]:
        dataloaders_["train"] = DataLoader(
            dataset=datasets["train"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
        )

    # Val
    if datasets["val"]:
        dataloaders_["val"] = DataLoader(
            dataset=datasets["val"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
        )

    # Test
    if datasets["test"]:
        dataloaders_["test"] = DataLoader(
            dataset=datasets["test"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
        )

    return dataloaders_


def create_beta_vae_models(train_dataloader_, verbose=False):
    """Create β-VAE models for each signal type"""
    if verbose:
        print("\n\n----------β-VAE MODEL INITIALIZATION----------\n")

    # Get sample batch to determine signal shapes
    sample_batch = next(iter(train_dataloader_))

    models = {}
    for signal_name, signal_data in sample_batch.items():
        input_length = signal_data.shape[-1]  # Last dimension is time

        if verbose:
            print(
                f"Signal: {signal_name}, Shape: {signal_data.shape}, Input length: {input_length}"
            )

        model = BetaVAE(input_length=input_length, latent_dim=LATENT_DIM, beta=BETA).to(
            device
        )

        models[signal_name] = model

        if verbose:
            print(f"Created BetaVAE for {signal_name}")

    return models


def train_beta_vae_models(
    models, train_dataloader, val_dataloader, output_sub_dir, verbose=False
):
    """Train β-VAE models for each signal"""
    if verbose:
        print("\n\n----------β-VAE TRAINING----------\n")

    output_dir = os.path.join("output", output_sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create optimizers for each model
    optimizers = {}
    for signal_name, model in models.items():
        optimizers[signal_name] = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training tracking
    best_losses = {signal_name: float("inf") for signal_name in models.keys()}
    best_model_states = {}

    for epoch in range(MAX_EPOCHS):
        if verbose:
            print(f"\nEpoch {epoch+1}\n")

        # Training phase
        for signal_name, model in models.items():
            model.train()

        train_losses = defaultdict(float)
        train_counts = defaultdict(int)

        for batch_idx, batch_signals in enumerate(train_dataloader):
            if verbose and batch_idx == 0:
                print(f"Available signals in batch: {list(batch_signals.keys())}")

            for signal_name, signal_data in batch_signals.items():
                if signal_name not in models:
                    continue

                model = models[signal_name]
                optimizer = optimizers[signal_name]

                # Prepare data
                if signal_data.dim() == 2:  # (batch, time)
                    x = signal_data.to(device)
                else:  # (batch, channels, time) -> flatten channels
                    x = signal_data.view(-1, signal_data.size(-1)).to(device)

                # Forward pass
                reconstruction, mu, logvar, z = model(x)
                total_loss, recon_loss, kl_loss = model.loss_function(
                    reconstruction, x, mu, logvar
                )

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_losses[signal_name] += total_loss.item()
                train_counts[signal_name] += x.size(0)

        # Validation phase
        val_losses = defaultdict(float)
        val_counts = defaultdict(int)

        for signal_name, model in models.items():
            model.eval()

        with torch.no_grad():
            for batch_signals in val_dataloader:
                for signal_name, signal_data in batch_signals.items():
                    if signal_name not in models:
                        continue

                    model = models[signal_name]

                    # Prepare data
                    if signal_data.dim() == 2:
                        x = signal_data.to(device)
                    else:
                        x = signal_data.view(-1, signal_data.size(-1)).to(device)

                    # Forward pass
                    reconstruction, mu, logvar, z = model(x)
                    total_loss, recon_loss, kl_loss = model.loss_function(
                        reconstruction, x, mu, logvar
                    )

                    val_losses[signal_name] += total_loss.item()
                    val_counts[signal_name] += x.size(0)

        # Print epoch results and save best models
        for signal_name in models.keys():
            if train_counts[signal_name] > 0:
                avg_train_loss = train_losses[signal_name] / train_counts[signal_name]
                avg_val_loss = (
                    val_losses[signal_name] / val_counts[signal_name]
                    if val_counts[signal_name] > 0
                    else float("inf")
                )

                if verbose:
                    print(
                        f"Signal {signal_name:30s} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                    )

                # Save best model
                if avg_val_loss < best_losses[signal_name]:
                    best_losses[signal_name] = avg_val_loss
                    best_model_states[signal_name] = models[signal_name].state_dict()

                    # Save best model state
                    model_path = os.path.join(
                        output_dir, f"best_beta_vae_{signal_name.replace('/', '_')}.pt"
                    )
                    torch.save(best_model_states[signal_name], model_path)

    return best_model_states


if __name__ == "__main__":

    LOCAL_FLAG = False
    mp.set_start_method("spawn", force=True)

    # For common pipeline
    SUBSET_OF_SHOTS = 25  # Can be None for entire dataset
    OUTPUT_SUB_FOLDER = "beta_vae_output/"
    BATCH_SIZE = 5
    NUM_WORKERS = 0
    MAX_EPOCHS = 100
    REF_FREQ = 0.001
    USE_EXISTING_FITTED_PARAMS = True  # Set to True to load existing fitted parameters

    # For β-VAE specific settings
    LATENT_DIM = 32  # Dimensionality of latent space
    BETA = 1.0  # β parameter for KL divergence weighting
    LEARNING_RATE = 1e-3

    source_signal_list = [
        ("magnetics", "flux_loop_flux"),
        # ('magnetics', 'b_field_pol_probe_ccbv_field'),
        # ('pf_active', 'solenoid_current'),
        # ('pf_active', 'coil_voltage'),
        # ('pulse_schedule', 'i_plasma'),
        # ('summary', 'power_nbi'),
        # ('equilibrium', 'elongation'),
        # ('equilibrium', 'minor_radius'),
    ]

    # Parameters for window segmentation (no x/y split for VAE)
    PARAMETERS_WINDOWS_SEGMENTER = {
        "x_keys": [f"{source}-{signal}" for source, signal in source_signal_list],
        "y_keys": [
            f"{source}-{signal}" for source, signal in source_signal_list
        ],  # Same as x for VAE
        "x_window_sec": 0.1,  # 100ms windows
        "y_window_sec": 0.1,
        "dt_sec": 0.0,  # No delay needed
        "stride_sec": None,
        "stride_unitary": True,
        "min_samples_per_window": 1,
        "verbose": False,
    }

    # Create sets of shot IDs for training, validation and testing
    train_shots, test_shots, val_shots = get_train_test_val_shots(
        max_index=SUBSET_OF_SHOTS
    )

    # Fit mean and std for signal transformation
    dict_mean, dict_std = fit_mean_and_std_for_signal_transform(
        output_sub_dir=OUTPUT_SUB_FOLDER,
        verbose=True,
        use_existing=USE_EXISTING_FITTED_PARAMS,
    )

    # Get the signal transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [
                ForwardFillImputerTransform(),
                StdScalingTransform(dict_mean[var], dict_std[var]),
                FillWithZerosImputerTransform(),
                SamplingToReferenceTimeTransform(REF_FREQ),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Shot-level transform for β-VAE
    shot_transform = ComposeTransforms(
        [
            TruncationTransform(),
            WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
            BetaVAETransform(),
        ]
    )

    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": test_shots},
        sig_tran_map=signal_transform_map,
        shot_tran=shot_transform,
        local_flag=LOCAL_FLAG,
        verbose=True,
    )

    # Prepare dataloaders
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=beta_vae_collate_fn,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        verbose=True,
    )
    train_dataloader = dataloaders_train_val_test["train"]
    val_dataloader = dataloaders_train_val_test["val"]
    test_dataloader = dataloaders_train_val_test["test"]

    # Create β-VAE models
    beta_vae_models = create_beta_vae_models(
        train_dataloader_=train_dataloader, verbose=True
    )

    best_model_states = train_beta_vae_models(
        models=beta_vae_models,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_sub_dir=OUTPUT_SUB_FOLDER,
        verbose=True,
    )

    print("\n\n----------TRAINING COMPLETE----------")
    print(f"Trained β-VAE models for {len(best_model_states)} signals")
    print(f"Models saved in: output/{OUTPUT_SUB_FOLDER}")
