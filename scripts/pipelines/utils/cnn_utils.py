import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v3 as iio
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from torch.utils.data._utils.collate import default_collate

from typing import Dict, List
from torch.utils.data import DataLoader

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..", "..", "..",
    )
)  # noqa: E402
print(REPO_ROOT) # this adds /rds/project/rds-mOlK9qn0PlQ/ir-rous1/hncdi-fusion-plasma/fairmast-data-preprocessing
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# print(f"REPO_ROOT: {REPO_ROOT}")

from scripts.pipelines.utils.utils import (
    seed_worker, 
    make_data_generator,
    get_train_test_val_shots,
    initialize_datasets,
    initialize_dataloaders,
    ModelTransformWrapper,
)

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.utils.utils import (
    ComposeTransforms,
)
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.reshape_lcfs_transform import (
    ReshapeLcfsTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.truncation_transform import (
    TruncationTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.timestamp_window_segmenter_transform import (
    TimestampWindowSegmenterTransform,
)

from scripts.pipelines.transforms.shot_level_transforms.truncate_windows_transform import (
    WindowTruncationTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_thomson_with_zeros_imputer_transform import (
FillThomsonWithZerosTransform
)
from scripts.pipelines.transforms.shot_level_transforms.drop_sample_with_nans import (
    DropSampleWithNans,
)
from scripts.pipelines.transforms.shot_level_transforms.cnn_transform import (
    CNNTransform,
)
# from scripts.pipelines.transforms.shot_level_transforms.time_cnn_transform import (
#     TimeCNNTransform,
# )
from scripts.pipelines.models.cnn_model import MultiBranchCNNModel
# from scripts.pipelines.models.time_cnn_model_update import MultiBranchTimeCNNModel

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


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ----------------------------------------------------------------------------------------------------------------------
# CNN PREPROCESSING 
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# def build_cnn_signal_transform_map(
#     source_signal_list: List[tuple],
#     dict_mean: Dict[str, float],
#     dict_std: Dict[str, float],
#     ref_freq: float
# ):
#     """Builds the signal transform map for each variable."""

#     # Define base signal_transform_map
#     print(source_signal_list)
#     print('before signal_transform_map')
#     signal_transform_map = {
#         var: ComposeTransforms(
#             [
#                 StdScalingTransform(dict_mean[var], dict_std[var]),
#                 # SamplingToReferenceTimeTransform(ref_freq),
#             ]
#         )
#         for var in [f"{source}-{signal}" for source, signal in source_signal_list]
#     }

#     # Specific case of profiles with Nans in full channel
#     for var in [
#         "magnetics-flux_loop_flux",
#         "magnetics-b_field_pol_probe_ccbv_field",
#         "magnetics-b_field_pol_probe_obr_field",
#         "magnetics-b_field_pol_probe_obv_field",
#         "magnetics-b_field_tor_probe_saddle_voltage",
#     ]:
#         signal_transform_map[var] = ComposeTransforms(
#             [
#                 FillProfileWithZerosTransform(),
#                 StdScalingTransform(dict_mean[var], dict_std[var]),
#                 # SamplingToReferenceTimeTransform(ref_freq),
#             ]
#         )

#     # Specific case of reformating LCFS
#     for var in ["equilibrium-lcfs_r", "equilibrium-lcfs_z"]:
#         signal_transform_map[var] = ComposeTransforms(
#             [
#                 ReshapeLcfsTransform(),
#                 StdScalingTransform(dict_mean[var], dict_std[var]),
#                 # SamplingToReferenceTimeTransform(ref_freq),
#             ]
#         )

#     # Specific filling with zeros for shomson scattering
#     for var in ["thomson_scattering-t_e", "thomson_scattering-n_e"]:
#         signal_transform_map[var] = ComposeTransforms(
#             [
#                 StdScalingTransform(dict_mean[var], dict_std[var]),
#                 FillThomsonWithZerosTransform(),
#                 # SamplingToReferenceTimeTransform(ref_freq),
#             ]
#         )

#     return signal_transform_map



def initialize_cnn_datasets(datasets_train_val_test, 
                            dict_metadata: Dict,
                            parameters: Dict[str, float],):

    cnn_specific_transform = ComposeTransforms([  
        TimestampWindowSegmenterTransform(
            dict_metadata,
            **parameters["window_segmenter_setting"], 
        ), 
        # DropSampleWithNans(verbose=True),
        CNNTransform() ,
    ])

    datasets_ = {"train": None, "val": None, "test": None}

    # ..................................................................................................................
    # Train
    datasets_["train"] = ModelTransformWrapper(datasets_train_val_test["train"], 
                                              cnn_specific_transform)
    
    # ..................................................................................................................
    # Val
    datasets_["val"] = ModelTransformWrapper(datasets_train_val_test["val"], 
                                              cnn_specific_transform)

    # ..................................................................................................................
    # Test
    datasets_["test"] = ModelTransformWrapper(datasets_train_val_test["test"], 
                                              cnn_specific_transform)
    # ..................................................................................................................
    # Return
    return datasets_

    
def initialize_cnn_dataloaders_and_models(datasets_train_val_test, 
                            dict_metadata: Dict,
                            parameters: Dict[str, float],
                            verbose=False,
                            seed: int | None = None,
                            pin_memory: bool | None = None,
                            ):
    
    if verbose:
        print('\n\n---------- CNN DATASET & DATALOADER INITIALIZATION----------\n')
    
    cnn_datasets = initialize_cnn_datasets(datasets_train_val_test, 
                                           dict_metadata,
                                           parameters)
                                           

    cnn_dataloaders_ = {"train": None, "val": None, "test": None}

    # ▶ Prepare reproducible seeding parts for DataLoader
    worker_fn = None
    generator = None
    if seed is not None:
        worker_fn = seed_worker
        generator = make_data_generator(seed)

    # sensible default for pin_memory
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # ..................................................................................................................
    # Train
    cnn_dataloaders_["train"] = DataLoader(
        dataset=cnn_datasets["train"],
        **parameters["dataloader_setting"],
        shuffle=True,
        drop_last=False,
        collate_fn=cnn_training_collate_fn,
        worker_init_fn=worker_fn,  # ▶
        generator=generator,  # ▶ controls shuffle order deterministically
        pin_memory=pin_memory,
    )

    # ..................................................................................................................
    # Val
    cnn_dataloaders_["val"] = DataLoader(
        dataset=cnn_datasets["val"],
        **parameters["dataloader_setting"],
        shuffle=True,
        drop_last=False,
        collate_fn=cnn_training_collate_fn,
        worker_init_fn=worker_fn,  # ▶ ensures worker RNG is fixed
        generator=generator,  # ▶ reproducible order if shuffle=True
        pin_memory=pin_memory,
    )

    # ..................................................................................................................
    # Test
    cnn_dataloaders_["test"] = DataLoader(
        dataset=cnn_datasets["test"],
        **parameters["dataloader_setting"],
        shuffle=False,
        drop_last=False,
        collate_fn=cnn_inference_collate_fn,
        worker_init_fn=worker_fn,
        generator=generator,
        pin_memory=pin_memory,
    )

    # ..................................................................................................................
    # Model
    cnn_model = create_cnn_architecture(cnn_dataloaders_["train"], 
                                         **parameters['cnn_settings'],
                                         verbose=False)

    return cnn_dataloaders_, cnn_model



# ----------------------------------------------------------------------------------------------------------------------
# def build_cnn_shot_transform_map(
#     dict_metadata: Dict,
#     parameters_window_segmenter: Dict[str, float],
#     # remove_CNN_transform: bool = False,
# ):
#     """Builds the shot transform map for all variable."""


#     # if remove_CNN_transform:
#     #     shot_transform = ComposeTransforms([  
#     #         # TruncationTransform(),
#     #         SampleSegmenterTransform(
#     #             **parameters_window_segmenter
#     #         ), 
#     #         # DropSampleWithNans(verbose=True),
#     #         ])
#     # else:
#     shot_transform = ComposeTransforms([  
#         # TruncationTransform(),
#         TimestampWindowSegmenterTransform(
#             dict_metadata,
#             **parameters_window_segmenter, 
#         ), 
#         # DropSampleWithNans(verbose=True),
#         CNNTransform() ,
#     ])

#     return shot_transform

# ----------------------------------------------------------------------------------------------------------------------
# COLLATE FUNCTION
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def cnn_training_collate_fn(batch):
    print(f"Collating batch of size {len(batch)}")

    # Flatten the batch of lists into a single list
    flattened_batch = [ (item['shot_id'], item['window_index'], item['x'], item['y'])
                       for sublist in batch
                       for item in sublist
                       if not (
                           any(np.isnan(np.array(x)).any() for x in item['x']) or
                           any(np.isnan(np.array(y)).any() for y in item['y']) 
                           )
                       ]
    
    print(
        f"Number of samples from batch = {len(batch)} shots is N = {len(flattened_batch)}"
    )

    return default_collate(flattened_batch) if (len(flattened_batch) > 0) else None

def cnn_inference_collate_fn(batch):
    print(f"Collating batch of size {len(batch)}")

    # Flatten the batch of lists into a single list
    flattened_batch = []
    flattened_batch = [(item['x'], item['y']) for sublist in batch for item in sublist]
    
    print(
        f"Number of samples from batch = {len(batch)} shots is N = {len(flattened_batch)}"
    )

    # Use the default collate function
    return default_collate(flattened_batch) if (len(flattened_batch) > 0) else None


# ----------------------------------------------------------------------------------------------------------------------
# CNN TRAINING
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def create_cnn_architecture(dataloader_, D, verbose=False):
    print(D)
    if verbose:
        print("\n\n----------MODEL INITIALIZATION----------\n")
    for l in range(len(dataloader_.dataset)):
        try:
            input_shapes = [arr.shape for arr in dataloader_.dataset[l][0]['x']]
            output_shape = [arr.shape for arr in dataloader_.dataset[l][0]['y']]
            if verbose:
                print(f"input_shapes: {input_shapes}")
                print(f"output_shape: {output_shape}")
            break
        except Exception as e:
            print(f"Skipping {dataloader_.dataset.get_shot_id(l)} because shot not trainable: {e}")
            continue
            
    return MultiBranchCNNModel(input_shapes, output_shape, D).to(device)

# def create_time_cnn_architecture(train_dataloader_, D, verbose=False):
#     print(D)
#     if verbose:
#         print("\n\n----------MODEL INITIALIZATION----------\n")
#     for l in range(len(train_dataloader_.dataset)):
#         try:
#             input_shapes = [arr.shape for arr in train_dataloader_.dataset[l][0][0]]
#             output_shape = [arr.shape for arr in train_dataloader_.dataset[l][0][1]]
#             if verbose:
#                 print(f"input_shapes: {input_shapes}")
#                 print(f"output_shape: {output_shape}")
#             break
#         except Exception as e:
#             continue
#             # print(f"Skipping {l} because shot not trainable: {e}")

#     return MultiBranchTimeCNNModel(input_shapes, output_shape, D).to(device)

# ----------------------------------------------------------------------------------------------------------------------
class MultiOutputMSELoss(nn.Module):
    def __init__(self, reduction="mean", weights=None):
        super().__init__()
        self.reduction = reduction
        self.weights = weights  # e.g. [1.0, 0.5, 0.1, 2.0]

    def forward(self, y_preds, y_trues):
        assert len(y_preds) == len(y_trues), "Mismatch in number of outputs"
        losses = []
        for i, (yp, yt) in enumerate(zip(y_preds, y_trues)):
            assert yp.shape == yt.shape, (
                f"Shape mismatch at output {i}: {yp.shape} vs {yt.shape}"
            )
            l = F.mse_loss(yp, yt, reduction=self.reduction)
            if self.weights is not None:
                l = self.weights[i] * l
            losses.append(l)
        return sum(losses)

# ----------------------------------------------------------------------------------------------------------------------
def loop_for_cnn_training(
    base_cnn_model,
    train_dataloader,
    val_dataloader,
    lr,
    max_epochs,
    patience,
    output_dir,
    verbose=True,
):
    if verbose:
        print("\n\n----------CNN TRAINING----------\n")

    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Output folder to save trained model: {output_dir}")

    loss_criterion = MultiOutputMSELoss()
    optimizer = torch.optim.Adam(base_cnn_model.parameters(), lr=lr)

    best_model_state_ = None
    best_val_loss = float("inf")
    early_stop_ = False
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        base_cnn_model.train()
        running_loss = 0.0
        num_batches = 0

        if verbose:
            print(f"\nEpoch {epoch + 1}\n")

        for batch_idx, (shot_id, window_id, x_train, y_train) in enumerate(train_dataloader):

            x_train = [arr.to(torch.float32).to(device) for arr in x_train]
            # y_train = y_train[0].to(torch.float32).to(device)
            y_train = [arr.to(torch.float32).to(device) for arr in y_train]

            actual_batch_size = y_train[0].shape[0]
            if verbose:
                # print(y_train.shape)
                print(f"Batch {batch_idx} size is {actual_batch_size}")

            # outputs_ = base_cnn_model(*x_train).squeeze()
            outputs_ = base_cnn_model(*x_train)

            loss_ = loss_criterion(outputs_, y_train)
            if verbose:
                # print(f"outputs' shape: {outputs_.shape}")
                print(f"Batch loss: {loss_}")

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

            running_loss += loss_.item() * actual_batch_size
            num_batches += actual_batch_size

        avg_loss = running_loss / num_batches

        if verbose:
            print(f"Epoch [{epoch + 1}/{max_epochs}], Average Loss: {avg_loss:.4f}")

        # Validation phase & Early stopping check

        base_cnn_model.eval()
        val_running_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for (shot_id, window_id, x_val, y_val) in val_dataloader:
                x_val = [arr.to(torch.float32).to(device) for arr in x_val]
                # y_val = y_val[0].to(torch.float32).to(device)
                y_val = [arr.to(torch.float32).to(device) for arr in y_val]

                # val_outputs = base_cnn_model(*x_val).squeeze()
                val_outputs = base_cnn_model(*x_val)

                val_loss = loss_criterion(val_outputs, y_val)

                actual_batch_size = y_val[0].shape[0]

                val_running_loss += val_loss.item() * actual_batch_size
                val_batches += actual_batch_size

        avg_val_loss = val_running_loss / val_batches

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{max_epochs}], Average Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            )

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
def cnn_evaluation_per_shot(cnn_model, 
                            test_shots_, 
                            LOCAL_FLAG,
                            source_signal_list,
                            signal_transform_map,
                            shot_transform,
                            order_var_for_inv_std,
                            dict_mean,
                            dict_std,
                            OUTPUT_FOLDER):
    cnn_model.eval()

    with open(OUTPUT_FOLDER + "test_loss_per_var.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shot_id", "n_windows"] + order_var_for_inv_std)  # Header

        for shot_id in test_shots_:
            print(f"Evaluating shot {shot_id}")

            test_shot_dataset = MastDataset(
                local=LOCAL_FLAG,
                shots_list=[shot_id],
                source_signal_list=source_signal_list,
                signal_level_transform_map=signal_transform_map,
                shot_level_transform=shot_transform,
            )

            test_shot_dataloader = DataLoader(
                dataset=test_shot_dataset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                drop_last=False,
                collate_fn=flatten_then_collate,
            )

            if (
                len(test_shot_dataloader.dataset[0]) > 0
            ):  # i.e. if this is a valid shot with windows
                with torch.no_grad():  # Disable gradient calculation for efficiency
                    shot_id, window_id, x_test, y_test = next(iter(test_shot_dataloader))
                    x_test = [arr.to(torch.float32).to(device) for arr in x_test]

                    y_test = flatten_blocks(y_test)
                    y_test = inverse_standardize(
                        y_test, order_var_for_inv_std, dict_mean, dict_std
                    )
                    y_test = [
                        torch.from_numpy(arr).float().to(device) for arr in y_test
                    ]

                    outputs_ = flatten_blocks(cnn_model(*x_test))
                    outputs_ = inverse_standardize(
                        outputs_, order_var_for_inv_std, dict_mean, dict_std
                    )
                    outputs_ = [
                        torch.from_numpy(arr).float().to(device) for arr in outputs_
                    ]

                    print([arr.shape for arr in outputs_])

                    # avg_test_loss = [
                    #     torch.nn.MSELoss(reduction="mean")(pred, true)
                    #     .mean(dim=0)
                    #     .item()
                    #     for pred, true in zip(outputs_, y_test)
                    # ]

                    rmse_per_batch = []
                    for pred, true in zip(outputs_, y_test):
                        # all dims except first
                        dims = tuple(range(1, pred.ndim))
                        rmse = torch.sqrt(torch.mean((pred - true) ** 2, dim=dims))  # [batch]
                        rmse_mean = rmse.mean().item()  # scalar RMSE
                        rmse_per_batch.append(rmse_mean)

                    print("RMSE per var:", rmse_per_batch)

                writer.writerow([shot_id, len(x_test[0])] + rmse_per_batch)
                f.flush()

            else:
                print(f"Shot {shot_id} not run properly, likely empty slice")
                writer.writerow([shot_id, None] + [None] * len(order_var_for_inv_std))
                f.flush()
                continue
    

# ----------------------------------------------------------------------------------------------------------------------
# INVERSE STDSCALING
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def flatten_blocks(list_y):
    """
    Flattens each block of predictions into individual series.

    """
    new_list_y = []
    for block in list_y:
        if isinstance(block, torch.Tensor):
            block = block.detach().cpu().numpy()  # detach, move to CPU, convert to numpy
        else:
            block = np.asarray(block)
            
        if block.ndim == 1:
            new_list_y.append(block)
        else:
            new_list_y += [
                np.squeeze(s, axis=1) for s in np.split(block, block.shape[1], axis=1)
            ]
    return new_list_y


# ----------------------------------------------------------------------------------------------------------------------
def get_cnn_order_scaling(
    LOCAL_FLAG,
    test_shots_,
    source_signal_list,
    signal_transform_map,
    shot_transform_without_CNN,
    transform_temp = CNNTransform(),
    verbose = True
):
    dataset_temp = MastDataset(
        local=LOCAL_FLAG,
        shots_list=test_shots_,
        source_signal_list=source_signal_list,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform=shot_transform_without_CNN,
    )

    for l in range(len(dataset_temp)):
        data_temp = dataset_temp[l]
        data_temp = transform_temp(data_temp)
        try:
            order_var_for_inv_std = [
                item for arr in flatten_blocks(transform_temp.var_groups["y"]) for item in arr
            ]
            return order_var_for_inv_std
        except Exception as e:
            continue
            # print(f"Skipping {l} because shot not trainable: {e}")



# ----------------------------------------------------------------------------------------------------------------------
def inverse_standardize(flat_data, order_vars, dict_mean, dict_std):
    """
    Inverse standardize a list of flattened arrays using provided means and stds.

    Args:
        flat_data (list of np.ndarray): Flattened predicted/true arrays.
        order_vars (list): Variable names, aligned with flat_data.
        dict_mean (dict): Mapping var -> mean.
        dict_std (dict): Mapping var -> std.

    Returns:
        list of np.ndarray: Inverse standardized arrays.
    """
    new_flat = []
    for var, data in zip(order_vars, flat_data):
        # print(var)
        mean = dict_mean[var]
        std = dict_std[var]

        # prevent division by zero (replace 0 with 1)
        std_safe = np.where(std == 0, 1.0, std)

        # inverse transform
        new_data = data.T * std_safe[..., None] + mean[..., None]
        new_flat.append(np.squeeze(new_data.T))

    return new_flat


# ----------------------------------------------------------------------------------------------------------------------
# VISUALISATION
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def plot_shot_gif(
    flat_preds,
    flat_trues,
    order_var_list,
    avg_test_loss,
    shot_idx,
    ref_freq,
    out_dir="shot_gifs",
    fps=8,
    cleanup=True,
):
    """
    Create an animated GIF showing predictions vs ground truth over time for a given shot.

    Parameters
    ----------
    list_y_pred, list_y_true : list of arrays
        Each array shape (T, N) or (T,)
    shot_idx : int or str
        Identifier for the shot (used in filenames)
    ref_freq : float
        Time step in ms
    out_dir : str
        Directory to store frames and GIF
    fps : int
        Frames per second of the animation
    cleanup : bool
        If True, delete individual frame PNGs after creating the GIF
    """
    out_folder = Path(out_dir) / f"shot_gifs"
    frame_dir = out_folder / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    n_outputs = len(flat_preds)
    T = min(y.shape[0] for y in flat_preds)  # number of time steps (shortest sequence)

    # Determine global min/max for 2D profiles
    # profile_min, profile_max = None, None
    max_list = [ max(v_pred, v_true) for v_pred, v_true in zip([arr.max() for arr in flat_preds], [arr.max() for arr in flat_trues]) ]
    min_list = [ min(v_pred, v_true) for v_pred, v_true in zip([arr.min() for arr in flat_preds], [arr.min() for arr in flat_trues]) ]
    # for y_pred, y_true in zip(flat_preds, flat_trues):
    #     if y_pred.ndim == 2:
    #         vmin = min(y_true.min(), y_pred.min())
    #         vmax = max(y_true.max(), y_pred.max())
    #         if profile_min is None or vmin < profile_min:
    #             profile_min = vmin
    #         if profile_max is None or vmax > profile_max:
    #             profile_max = vmax

    # Row height ratios for subplot layout
    row_heights = [0.5 if y.ndim == 1 else 1.0 for y in flat_preds]
    fig_height = sum(row_heights) * 4

    frame_paths = []

    # Loop over time steps
    for t in range(1, T + 1):
        fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(
            n_outputs, 3, width_ratios=[1, 1, 1], height_ratios=row_heights, hspace=0.4
        )

        viz_max_t = T * ref_freq
        time_ms = np.arange(t) * ref_freq

        for j, (var, y_pred, y_true, rmse, var_min, var_max) in enumerate(
            zip(order_var_list, flat_preds, flat_trues, avg_test_loss, min_list, max_list)
        ):
            viz_min_y = min(y_true.min(), y_pred.min()) - abs(
                min(y_true.min(), y_pred.min()) * 0.05
            )
            viz_max_y = max(y_true.max(), y_pred.max()) + abs(
                max(y_true.max(), y_pred.max()) * 0.05
            )

            yp = y_pred[:t]
            yt = y_true[:t]

            ax_gt = fig.add_subplot(gs[j, 0])
            ax_pred = fig.add_subplot(gs[j, 1])
            ax_diff = fig.add_subplot(gs[j, 2])

            if yp.ndim == 1:
                # 1D time series
                ax_gt.plot(time_ms, yt, lw=2, color="blue")
                ax_gt.set_title(f"{var} - Ground Truth")
                ax_gt.set_xlim(0, viz_max_t)
                ax_gt.set_ylim(viz_min_y, viz_max_y)
                # ax_gt.set_xlabel("Time (ms)")
                # ax_gt.set_ylabel("Value")

                ax_pred.plot(time_ms, yp, lw=2, color="orange")
                ax_pred.set_title(f"{var} - Prediction RMSE {round(rmse, 3)}")
                ax_pred.set_xlim(0, viz_max_t)
                ax_pred.set_ylim(viz_min_y, viz_max_y)
                # ax_pred.set_xlabel("Time (ms)")
                # ax_pred.set_ylabel("Value")

            elif yp.ndim == 2:
                # 2D profile
                D = yp.shape[1]

                ax_gt.imshow(
                    yt.T,
                    aspect="auto",
                    origin="lower",
                    extent=[time_ms[0], time_ms[-1], 0, D],
                    cmap="viridis",
                    vmin=var_min,
                    vmax=var_max,
                )
                ax_gt.set_title(f"{var} - Ground Truth")
                ax_gt.set_xlim(0, viz_max_t)
                # ax_gt.set_xlabel("Time (ms)")
                # ax_gt.set_ylabel("Profile index")

                ax_pred.imshow(
                    yp.T,
                    aspect="auto",
                    origin="lower",
                    extent=[time_ms[0], time_ms[-1], 0, D],
                    cmap="viridis",
                    vmin=var_min,
                    vmax=var_max,
                )
                ax_pred.set_title(f"{var} - Prediction RMSE {round(rmse, 3)}")
                ax_pred.set_xlim(0, viz_max_t)
                # ax_pred.set_xlabel("Time (ms)")
                # ax_pred.set_ylabel("Profile index")

                # Plot difference in grayscale
                ax_diff.imshow(
                    (yp-yt).T,
                    aspect="auto",
                    origin="lower",
                    extent=[time_ms[0], time_ms[-1], 0, D],
                    # cmap="gray",         # 👈 grayscale colormap
                    # vmin=-abs(diff).max(),
                    # vmax=abs(diff).max(),  # symmetric limits for positive/negative differences
                )

                ax_diff.set_title(f"{var} - Difference")
                ax_diff.set_xlim(0, viz_max_t)

            elif yp.ndim == 3:
                # 3D image

                img_gt = yt[-1]
                # print(img)
                ax_gt.imshow(
                    img_gt.T,
                    # aspect="auto",
                    cmap="viridis",
                    # vmin=var_min,
                    # vmax=var_max,
                )
                ax_gt.set_title(f"{var} - Ground Truth")
                ax_gt.axis("off")

                img_pred = yp[-1]
                # print(img)
                ax_pred.imshow(
                    img_pred.T,
                    # aspect="auto",
                    cmap="viridis",
                    # vmin=var_min,
                    # vmax=var_max,
                )
                # ax_pred.set_title(f"{var} - Prediction RMSE {round(rmse, 3)}")
                # --- Dynamically compute RMSE ---
                # Convert to torch tensors (and ensure they’re on same device)
                gt_tensor = torch.as_tensor(img_gt, dtype=torch.float32)
                pred_tensor = torch.as_tensor(img_pred, dtype=torch.float32)
                computed_rmse = torch.sqrt(torch.mean((pred_tensor - gt_tensor) ** 2)).item()

                ax_pred.set_title(f"{var} - Prediction RMSE {computed_rmse:.3e}")
                ax_pred.axis("off")

                img_diff = yp[-1] - yt[-1]
                # Compute limits for symmetric color scale
                # vmax = abs(img_diff).max()
                # vmax = abs(img_diff).max()
                # print(vmax)
                vmax = 0.025
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                # norm = mcolors.TwoSlopeNorm(vcenter=0)
                # print(img)
                ax_diff.imshow(
                    img_diff.T,
                    # aspect="auto",
                    cmap="coolwarm",
                    norm=norm,
                    # vmin=var_min,
                    # vmax=var_max,
                )
                ax_diff.set_title(f"{var} - Difference")
                ax_diff.axis("off")

            else:
                raise ValueError(
                    f"Unsupported shape {yp.shape} (expected (T,) or (T,D))"
                )

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        frame_path = frame_dir / f"frame_{t:04d}.png"
        plt.savefig(frame_path, dpi=150, 
                    bbox_inches="tight")
        plt.close(fig)

        frame_paths.append(frame_path)

    # Build GIF
    gif_path = out_folder / f"shot_{shot_idx}.gif"
    # Duration per frame in secondst    frames = [iio.imread(frame) for frame in frame_paths]
    frames = [iio.imread(frame) for frame in frame_paths] + [
        iio.imread(frame_paths[-1])
    ]
    durations = [1 / fps] * (len(frames) - 2) + [5] + [1 / fps]  # last frame = 5 sec
    iio.imwrite(
        gif_path,
        frames,
        format="GIF",
        duration=durations,
        # loop=0
    )

    # durations = [1 / fps] * (len(frame_paths) - 1) + [5]  # last frame = 5 sec
    # with iio.get_writer(gif_path, mode="I", loop=0) as writer:
    #     for frame, dur in zip(frame_paths, durations):
    #         writer.append_data(iio.imread(frame), duration=dur)

    # Optional cleanup
    if cleanup:
        for frame in frame_paths:
            frame.unlink()

    print(f"Saved GIF: {gif_path}")


# ----------------------------------------------------------------------------------------------------------------------
def plot_shot_image(flat_preds, flat_trues, order_var_list, avg_test_loss, shot_idx, ref_freq, out_dir="shot_images"):
    """
    Create a static image comparing predictions vs ground truth for a given shot.

    Parameters
    ----------
    flat_preds, flat_trues : list of arrays
        Each array shape (T,), (T, D), or (T, H, W)
    order_var_list : list of str
        Variable names
    avg_test_loss : list of float
        MSE values for each variable
    shot_idx : int or str
        Identifier for the shot (used in filenames)
    ref_freq : float
        Time step in ms
    out_dir : str
        Directory to save the output image
    """

    # out_folder = Path(out_dir)
    out_folder = Path(out_dir) / f"shot_images"
    out_folder.mkdir(parents=True, exist_ok=True)

    n_outputs = len(flat_preds)
    # row_heights = [0.5 if y.ndim == 1 else 1.0 for y in flat_preds]
    row_heights = [1 if y.ndim == 1 else 1.0 for y in flat_preds]
    fig_height = sum(row_heights) * 4

    fig, axes = plt.subplots(
        n_outputs, 2, figsize=(18, fig_height),
        gridspec_kw={'width_ratios': [1, 1], 'height_ratios': row_heights}
    )

    if n_outputs == 1:
        axes = np.array([axes])  # keep indexing consistent
    if axes.ndim == 1:
        axes = axes[:, None]

    viz_max_t = min(y.shape[0] for y in flat_preds) * ref_freq
    time_ms = np.arange(min(y.shape[0] for y in flat_preds)) * ref_freq

    for j, (var, y_pred, y_true, rmse) in enumerate(zip(order_var_list, flat_preds, flat_trues, avg_test_loss)):
        if y_pred.ndim == 1:
            # --- 1D: plot both curves on the same subplot ---
            ax = axes[j, 0]
            ax.plot(time_ms, y_true, lw=2, color="blue", label="Ground Truth")
            ax.plot(time_ms, y_pred, lw=2, color="cyan", label="Prediction")
            ax.set_xlim(0, viz_max_t)
            ax.set_title(f"{var} (RMSE={round(rmse,3)})")
            ax.legend()
            axes[j, 1].axis("off")  # right panel empty

        elif y_pred.ndim == 2:
            # --- 2D: keep ground truth vs prediction side by side ---
            D = y_pred.shape[1]
            vmin = min(y_true.min(), y_pred.min())
            vmax = max(y_true.max(), y_pred.max())

            axes[j, 0].imshow(
                y_true.T, aspect="auto", origin="lower",
                extent=[time_ms[0], time_ms[-1], 0, D],
                cmap="viridis", vmin=vmin, vmax=vmax
            )
            axes[j, 0].set_title(f"{var} - Ground Truth")

            axes[j, 1].imshow(
                y_pred.T, aspect="auto", origin="lower",
                extent=[time_ms[0], time_ms[-1], 0, D],
                cmap="viridis", vmin=vmin, vmax=vmax
            )
            axes[j, 1].set_title(f"{var} - Prediction (RMSE={round(rmse,3)})")

        elif y_pred.ndim == 3:
            # --- 3D: just black images ---
            black_img = np.zeros_like(y_true[-1])

            axes[j, 0].imshow(black_img, cmap="gray")
            axes[j, 0].set_title(f"{var} - Ground Truth (not shown)")

            axes[j, 1].imshow(black_img, cmap="gray")
            axes[j, 1].set_title(f"{var} - Prediction (not shown)")

        else:
            raise ValueError(f"Unsupported shape {y_pred.shape}")

    plt.tight_layout()
    out_path = out_folder / f"shot_{shot_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved image: {out_path}")

