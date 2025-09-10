import os
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import matplotlib.gridspec as gridspec
from torch.utils.data._utils.collate import default_collate 



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


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ------------------------------------------------------------------------------------------------------------------
# COLLATE FUNCTION
# ------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
def flatten_then_collate(batch):

    print(f"Collating batch of size {len(batch)}")
    
    # Flatten the batch of lists into a single list
    flattened_batch = []
    if isinstance(batch[0], list):
        flattened_batch = [item for sublist in batch for item in sublist]
        print(f'Number of samples from batch = {len(batch)} shots is N = {len(flattened_batch)}')

    # Use the default collate function
    return default_collate(flattened_batch) if (len(flattened_batch) > 0) else None


# ------------------------------------------------------------------------------------------------------------------
# VISUALISATION
# ------------------------------------------------------------------------------------------------------------------

import numpy as np

def flatten_blocks(list_y):
    """
    Flattens each block of predictions into individual series.

    """
    new_list_y = []
    for block in list_y:
        block = np.asarray(block)
        if block.ndim == 1:
            new_list_y.append(block)
        else:
            new_list_y += [
                np.squeeze(s, axis=1)
                for s in np.split(block, block.shape[1], axis=1)
            ]
    return new_list_y


# ------------------------------------------------------------------------------------------------------------------
def plot_shot(new_y_pred, new_y_true, shot_idx, ref_freq, out_dir="shot_images"):
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


# ------------------------------------------------------------------------------------------------------------------
def plot_shot_gif(flat_preds, flat_trues, shot_idx, ref_freq, out_dir="shot_gifs", fps=10, cleanup=True):
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
            
            elif yp.ndim == 3:
                # 3D image
                
                img = yt[-1]
                # print(img)
                ax_gt.imshow(
                    img, aspect="auto",
                    cmap="viridis",
                )
                ax_gt.set_title(f"Output {j} - Ground Truth")

                img = yp[-1]
                # print(img)
                ax_pred.imshow(
                    img, aspect="auto",
                    cmap="viridis",
                )
                ax_pred.set_title(f"Output {j} - Prediction")

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
