import os
import sys
import csv

import pickle
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from IPython.display import Image, display


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
def get_train_test_val_shots(max_index=None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index:
        train_sh = train_sh[0:max_index]
        val_sh = val_sh[0:max_index]
        test_sh = test_sh[0:max_index]

    return train_sh, test_sh, val_sh


# ----------------------------------------------------------------------------------------------------------------------
def fit_mean_and_std_for_signal_transform(output_sub_dir, train_shots, source_signal_list, verbose=False, local=False):

    if verbose:
        print('\n\n----------TRANSFORM FITTING----------\n')

    preprocessing_train_dataset = MastDataset(
        local=local,
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


# ------------------------------------------------------------------------------------------------------------------
# VISUALISATION
# ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# def plot_shot(y_pred, y_true, shot_idx, ref_freq, out_dir="shot_images"):

#     out_folder = f"{out_dir}shot_images/"
#     Path(out_folder).mkdir(parents=True, exist_ok=True)

#     num_samples, num_outputs = y_pred.shape
#     fig, axes = plt.subplots(num_outputs, 1, figsize=(10, 3 * num_outputs), sharex=True)
#     fig.suptitle(f"Shot {shot_idx}: Model Predictions vs Ground Truth", fontsize=16)

#     if num_outputs == 1:
#         axes = [axes]

#     # Create ticks based on time (ms)
#     step = max(1, num_samples // 20)  # aim for ~20 ticks
#     ticks = np.arange(0, num_samples, step)
#     labels = (ref_freq * ticks * 1000).astype(int)  # ms

#     for k in range(num_outputs):
#         axes[k].plot(y_true[:, k], label='Ground Truth', lw=2)
#         axes[k].plot(y_pred[:, k], label='Prediction', linestyle='--', lw=2)
#         axes[k].set_title(f'Output {k}')
#         axes[k].set_ylabel("Value")
#         axes[k].set_xticks(ticks, labels)
#         axes[k].legend()

#     axes[-1].set_xlabel("Time (ms)")

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     print(f"{out_folder}shot_{shot_idx}.png")
#     plt.savefig(f"{out_folder}shot_{shot_idx}.png", dpi=300, bbox_inches="tight")
#     plt.close(fig)  # close to avoid memory issues

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec

def plot_shot(list_y_pred, list_y_true, shot_idx, ref_freq, out_dir="shot_images"):
    """
    Plot model predictions vs ground truth for 1D time series and 2D profiles,
    all outputs on a single image with subplots. Ground Truth on left, Prediction on right.
    1D time series rows are half the height of 2D profile rows. No colorbars.

    Parameters
    ----------
    list_y_pred, list_y_true : list of arrays
        Each array shape (T, N) or (T,)
    shot_idx : int or str
    ref_freq : float
        Time step in ms
    out_dir : str
    """
    out_folder = Path(out_dir) / "shot_images"
    out_folder.mkdir(parents=True, exist_ok=True)

    # Flatten each block into individual series
    new_y_pred = []
    for block in list_y_pred:
        block = np.asarray(block)
        if block.ndim == 1:
            new_y_pred.append(block)
        else:
            new_y_pred += [np.squeeze(s, axis=1)
                           for s in np.split(block, block.shape[1], axis=1)]

    new_y_true = []
    for block in list_y_true:
        block = np.asarray(block)
        if block.ndim == 1:
            new_y_true.append(block)
        else:
            new_y_true += [np.squeeze(s, axis=1)
                           for s in np.split(block, block.shape[1], axis=1)]

    n_outputs = len(new_y_pred)
    
    # Determine global min/max for 2D profiles
    profile_min, profile_max = None, None
    for y_pred, y_true in zip(new_y_pred, new_y_true):
        if y_pred.ndim == 2:
            vmin = min(y_true.min(), y_pred.min())
            vmax = max(y_true.max(), y_pred.max())
            if profile_min is None or vmin < profile_min:
                profile_min = vmin
            if profile_max is None or vmax > profile_max:
                profile_max = vmax

    # Determine row height ratios
    row_heights = [0.5 if y.ndim == 1 else 1.0 for y in new_y_pred]
    fig_height = sum(row_heights) * 4
    fig = plt.figure(figsize=(18, fig_height))
    # fig.suptitle(f"Shot {shot_idx}: Model Predictions vs Ground Truth", fontsize=16)

    gs = gridspec.GridSpec(n_outputs, 2, width_ratios=[1,1], height_ratios=row_heights, hspace=0.4)

    for j, (y_pred, y_true) in enumerate(zip(new_y_pred, new_y_true)):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: pred {y_pred.shape}, true {y_true.shape}")

        T = y_pred.shape[0]
        time_ms = np.arange(T) * ref_freq

        ax_gt = fig.add_subplot(gs[j, 0])
        ax_pred = fig.add_subplot(gs[j, 1])

        if y_pred.ndim == 1:
            # 1D time series
            ax_gt.plot(time_ms, y_true, lw=2, color='blue')
            ax_gt.set_title(f"Output {j} - Ground Truth")
            ax_gt.set_xlabel("Time (ms)")
            ax_gt.set_ylabel("Value")

            ax_pred.plot(time_ms, y_pred, lw=2, color='orange')
            ax_pred.set_title(f"Output {j} - Prediction")
            ax_pred.set_xlabel("Time (ms)")
            ax_pred.set_ylabel("Value")

        elif y_pred.ndim == 2:
            # 2D profile
            D = y_pred.shape[1]

            ax_gt.imshow(
                y_true.T, aspect="auto", origin="lower",
                extent=[time_ms[0], time_ms[-1], 0, D],
                cmap="viridis", vmin=profile_min, vmax=profile_max
            )
            ax_gt.set_title(f"Output {j} - Ground Truth")
            ax_gt.set_xlabel("Time (ms)")
            ax_gt.set_ylabel("Profile index")

            ax_pred.imshow(
                y_pred.T, aspect="auto", origin="lower",
                extent=[time_ms[0], time_ms[-1], 0, D],
                cmap="viridis", vmin=profile_min, vmax=profile_max
            )
            ax_pred.set_title(f"Output {j} - Prediction")
            ax_pred.set_xlabel("Time (ms)")
            ax_pred.set_ylabel("Profile index")

        else:
            raise ValueError(f"Unsupported shape {y_pred.shape} (expected (T,) or (T,D))")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_folder / f"shot_{shot_idx}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------------------------------------------------
# def plot_shot_gif(y_pred, y_true, shot_idx, ref_freq, delta_pred, y_keys=None, out_dir="shot_gifs"):
#     """
#     Create an animated GIF of predictions vs ground truth for a single shot.

#     Args:
#         y_pred (np.ndarray): Predictions, shape (num_samples, num_outputs)
#         y_true (np.ndarray): Ground truth, shape (num_samples, num_outputs)
#         shot_idx (int): Index of the shot for file naming
#         ref_freq (float): Reference frequency (Hz)
#         delta_pred (float): Prediction delay in seconds
#         y_keys (list[str]): Labels for outputs (optional)
#         out_dir (str): Directory to save GIFs
#     """
#     out_folder = f"{out_dir}shot_gifs/"
#     Path(out_folder).mkdir(parents=True, exist_ok=True)

#     num_samples, num_outputs = y_pred.shape
#     delta_pred_samples = int(delta_pred / ref_freq)

#     # Compute min/max per output dimension for consistent y-scaling
#     y_min = np.minimum(y_pred.min(axis=0), y_true.min(axis=0))
#     y_max = np.maximum(y_pred.max(axis=0), y_true.max(axis=0))

#     frames = []
#     for end_idx in range(1, num_samples + 1):
#         fig, axes = plt.subplots(num_outputs, 1, figsize=(10, 3 * num_outputs), sharex=True)
#         fig.suptitle(f"Shot {shot_idx}: Model Predictions vs Ground Truth", fontsize=14)

#         if num_outputs == 1:
#             axes = [axes]

#         for k in range(num_outputs):
#             pred_x = np.arange(end_idx) - delta_pred_samples
#             valid_mask = pred_x >= 0
#             pred_y = y_true[pred_x[valid_mask], k]

#             axes[k].plot(pred_x[valid_mask], pred_y, label="Ground Truth", linestyle='-', color='blue')
#             axes[k].plot(range(end_idx), y_pred[:end_idx, k], label=f"Prediction (Δ={delta_pred:.3f}s)", linestyle='--', color='orange')

#             # Fixed y-axis limits
#             axes[k].set_ylim(y_min[k] - 0.05 * abs(y_min[k]), y_max[k] + 0.05 * abs(y_max[k]))

#             # X-axis ticks (time in ms)
#             step = max(1, num_samples // 20)
#             ticks = np.arange(0, num_samples, step)
#             labels = (ref_freq * ticks * 1000).astype(int)
#             axes[k].set_xticks(ticks, labels)

#             ylabel = y_keys[k] if y_keys is not None else f"Output {k}"
#             axes[k].set_ylabel(ylabel)
#             axes[k].legend()

#         axes[-1].set_xlabel("Time (ms)")

#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#         # Convert to RGBA buffer
#         fig.canvas.draw()
#         frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#         frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
#         frames.append(frame[:, :, :3])  # drop alpha channel

#         plt.close(fig)

#     # Save GIF
#     gif_path = f"{out_folder}shot_{shot_idx:03d}.gif"
#     imageio.mimsave(gif_path, frames, format="GIF", duration=0.10, loop=0)

#     return gif_path

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# from pathlib import Path
# import imageio


# def plot_shot_gif(list_y_pred, list_y_true, shot_idx, ref_freq, delta_pred, y_keys, out_dir="shot_gifs"):
#     """
#     Create a GIF animation comparing model predictions vs ground truth
#     over time, frame by frame.

#     Parameters
#     ----------
#     list_y_pred, list_y_true : list of arrays
#         Each array shape (T, N) or (T,)
#     shot_idx : int or str
#     ref_freq : float
#         Time step in ms
#     out_dir : str
#     """
#     out_folder = Path(out_dir) / "shot_gifs"
#     out_folder.mkdir(parents=True, exist_ok=True)

#     # Flatten each block into individual series
#     new_y_pred = []
#     for block in list_y_pred:
#         block = np.asarray(block)
#         if block.ndim == 1:
#             new_y_pred.append(block)
#         else:
#             new_y_pred += [np.squeeze(s, axis=1)
#                            for s in np.split(block, block.shape[1], axis=1)]

#     new_y_true = []
#     for block in list_y_true:
#         block = np.asarray(block)
#         if block.ndim == 1:
#             new_y_true.append(block)
#         else:
#             new_y_true += [np.squeeze(s, axis=1)
#                            for s in np.split(block, block.shape[1], axis=1)]

#     n_outputs = len(new_y_pred)

#     # Determine global min/max for 2D profiles
#     profile_min, profile_max = None, None
#     for y_pred, y_true in zip(new_y_pred, new_y_true):
#         if y_pred.ndim == 2:
#             vmin = min(y_true.min(), y_pred.min())
#             vmax = max(y_true.max(), y_pred.max())
#             if profile_min is None or vmin < profile_min:
#                 profile_min = vmin
#             if profile_max is None or vmax > profile_max:
#                 profile_max = vmax

#     # Row heights
#     row_heights = [0.5 if y.ndim == 1 else 1.0 for y in new_y_pred]
#     fig_height = sum(row_heights) * 4

#     T = new_y_pred[0].shape[0]  # total time steps
#     time_ms = np.arange(T) * ref_freq

#     frames = []

#     for t in range(T):
#         fig = plt.figure(figsize=(18, fig_height))
#         gs = gridspec.GridSpec(n_outputs, 2, width_ratios=[1, 1], height_ratios=row_heights, hspace=0.4)

#         for j, (y_pred, y_true) in enumerate(zip(new_y_pred, new_y_true)):
#             ax_gt = fig.add_subplot(gs[j, 0])
#             ax_pred = fig.add_subplot(gs[j, 1])

#             if y_pred.ndim == 1:
#                 # Plot up to time t
#                 ax_gt.plot(time_ms[:t+1], y_true[:t+1], lw=2, color='blue')
#                 ax_pred.plot(time_ms[:t+1], y_pred[:t+1], lw=2, color='orange')

#                 ax_gt.set_title(f"Output {j} - Ground Truth")
#                 ax_pred.set_title(f"Output {j} - Prediction")
#                 ax_gt.set_xlabel("Time (ms)")
#                 ax_pred.set_xlabel("Time (ms)")
#                 ax_gt.set_ylabel("Value")
#                 ax_pred.set_ylabel("Value")

#             elif y_pred.ndim == 2:
#                 D = y_pred.shape[1]
#                 ax_gt.imshow(
#                     y_true[:t+1].T, aspect="auto", origin="lower",
#                     extent=[time_ms[0], time_ms[t], 0, D],
#                     cmap="viridis", vmin=profile_min, vmax=profile_max
#                 )
#                 ax_pred.imshow(
#                     y_pred[:t+1].T, aspect="auto", origin="lower",
#                     extent=[time_ms[0], time_ms[t], 0, D],
#                     cmap="viridis", vmin=profile_min, vmax=profile_max
#                 )

#                 ax_gt.set_title(f"Output {j} - Ground Truth")
#                 ax_pred.set_title(f"Output {j} - Prediction")
#                 ax_gt.set_xlabel("Time (ms)")
#                 ax_pred.set_xlabel("Time (ms)")
#                 ax_gt.set_ylabel("Profile index")
#                 ax_pred.set_ylabel("Profile index")

#         plt.tight_layout()
#         # Save frame to memory (not disk)
#         fig.canvas.draw()
#         # w, h = fig.canvas.get_width_height()
#         image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
#         # image = image.reshape(h, w, 3)
#         frames.append(image)
#         plt.close(fig)

#     # Save gif
#     out_path = out_folder / f"shot_{shot_idx}.gif"
#     imageio.mimsave(out_path, frames, fps=10)  # adjust fps if needed
#     print(f"Saved GIF: {out_path}")


# import imageio
# from pathlib import Path
# import numpy as np

# def plot_shot_gif(list_y_pred, list_y_true, shot_idx, ref_freq, out_dir="shot_images"):
#     """
#     Create an animated GIF from model predictions vs ground truth using `plot_shot`.

#     Parameters
#     ----------
#     list_y_pred, list_y_true : list of arrays
#         Each array shape (T, N) or (T,)
#     ref_freq : float
#         Time step in ms
#     out_dir : str
#     gif_name : str
#         Name of the output GIF file
#     """
#     out_folder = Path(out_dir) / "gif_frames"
#     out_folder.mkdir(parents=True, exist_ok=True)

#     T = min(y.shape[0] for y in list_y_pred)  # total frames (shortest output)
#     frame_paths = []

#     for t in range(1, T + 1):
#         # Slice all outputs up to current time step
#         sliced_preds = [y[:t] for y in list_y_pred]
#         sliced_trues = [y[:t] for y in list_y_true]

#         # Save frame using your existing function
#         frame_file = out_folder / f"frame_{t:04d}.png"
#         plot_shot(sliced_preds, sliced_trues, shot_idx=t, ref_freq=ref_freq, out_dir=out_folder)
#         frame_paths.append(str(frame_file))

#     # Create GIF
#     gif_path = out_folder / f"shot_{shot_idx}.gif"
#     with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
#         for fp in frame_paths:
#             image = imageio.imread(fp)
#             writer.append_data(image)

#     print(f"Saved GIF: {gif_path}")





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import imageio

def plot_shot_gif(list_y_pred, list_y_true, shot_idx, ref_freq, out_dir="shot_gifs", fps=10, cleanup=True):
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

    # Flatten outputs into individual series (same logic as in plot_shot)
    def flatten_outputs(list_arr):
        new_list = []
        for block in list_arr:
            block = np.asarray(block)
            if block.ndim == 1:
                new_list.append(block)
            else:
                new_list += [np.squeeze(s, axis=1)
                             for s in np.split(block, block.shape[1], axis=1)]
        return new_list

    flat_preds = flatten_outputs(list_y_pred)
    flat_trues = flatten_outputs(list_y_true)

    n_outputs = len(flat_preds)
    T = min(y.shape[0] for y in flat_preds)  # number of time steps (shortest sequence)

    # Determine global min/max for 2D profiles
    profile_min, profile_max = None, None
    for y_pred, y_true in zip(flat_preds, flat_trues):
        if y_pred.ndim == 2:
            vmin = min(y_true.min(), y_pred.min())
            vmax = max(y_true.max(), y_pred.max())
            if profile_min is None or vmin < profile_min:
                profile_min = vmin
            if profile_max is None or vmax > profile_max:
                profile_max = vmax

    # Row height ratios for subplot layout
    row_heights = [0.5 if y.ndim == 1 else 1.0 for y in flat_preds]
    fig_height = sum(row_heights) * 4

    frame_paths = []

    # Loop over time steps
    for t in range(1, T + 1):
        fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(
            n_outputs, 2,
            width_ratios=[1, 1],
            height_ratios=row_heights,
            hspace=0.4
        )

        time_ms = np.arange(t) * ref_freq

        for j, (y_pred, y_true) in enumerate(zip(flat_preds, flat_trues)):
            yp = y_pred[:t]
            yt = y_true[:t]

            ax_gt = fig.add_subplot(gs[j, 0])
            ax_pred = fig.add_subplot(gs[j, 1])

            if yp.ndim == 1:
                # 1D time series
                ax_gt.plot(time_ms, yt, lw=2, color="blue")
                ax_gt.set_title(f"Output {j} - Ground Truth")
                ax_gt.set_xlabel("Time (ms)")
                ax_gt.set_ylabel("Value")

                ax_pred.plot(time_ms, yp, lw=2, color="orange")
                ax_pred.set_title(f"Output {j} - Prediction")
                ax_pred.set_xlabel("Time (ms)")
                ax_pred.set_ylabel("Value")

            elif yp.ndim == 2:
                # 2D profile
                D = yp.shape[1]

                ax_gt.imshow(
                    yt.T, aspect="auto", origin="lower",
                    extent=[time_ms[0], time_ms[-1], 0, D],
                    cmap="viridis", vmin=profile_min, vmax=profile_max
                )
                ax_gt.set_title(f"Output {j} - Ground Truth")
                ax_gt.set_xlabel("Time (ms)")
                ax_gt.set_ylabel("Profile index")

                ax_pred.imshow(
                    yp.T, aspect="auto", origin="lower",
                    extent=[time_ms[0], time_ms[-1], 0, D],
                    cmap="viridis", vmin=profile_min, vmax=profile_max
                )
                ax_pred.set_title(f"Output {j} - Prediction")
                ax_pred.set_xlabel("Time (ms)")
                ax_pred.set_ylabel("Profile index")

            else:
                raise ValueError(f"Unsupported shape {yp.shape} (expected (T,) or (T,D))")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        frame_path = frame_dir / f"frame_{t:04d}.png"
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        frame_paths.append(frame_path)

    # Build GIF
    gif_path = out_folder / f"shot_{shot_idx}.gif"
    with imageio.get_writer(gif_path, mode="I", fps=fps, loop=0) as writer:
        for frame in frame_paths:
            writer.append_data(imageio.imread(frame))

    # Optional cleanup
    if cleanup:
        for frame in frame_paths:
            frame.unlink()

    print(f"Saved GIF: {gif_path}")
