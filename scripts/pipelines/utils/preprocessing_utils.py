import os
import sys
import torch
import pickle
import numpy as np

from typing import Dict, List

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

# from scripts.MAST_tools.MAST_dataset import MastDataset

from scripts.pipelines.utils.utils import (
    get_train_test_val_shots,
    initialize_datasets,
    ComposeTransforms,
)
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.reshape_lcfs_transform import (
    ReshapeLcfsTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.rolling_segmenter_transform import (
    RollingSegmenterTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_thomson_with_zeros_imputer_transform import (
FillThomsonWithZerosTransform
)

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
# COMMON PREPROCESSING 
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def build_common_signal_transform_map(
    source_signal_list: List[tuple],
    dict_mean: Dict[str, float],
    dict_std: Dict[str, float],
):
    """Builds the signal transform map for each variable."""

    # Define base signal_transform_map
    print(source_signal_list)
    print('before signal_transform_map')
    signal_transform_map = {
        var: ComposeTransforms(
            [
                StdScalingTransform(dict_mean[var], dict_std[var]),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Specific case of profiles with Nans in full channel
    for var in [
        "magnetics-flux_loop_flux",
        "magnetics-b_field_pol_probe_ccbv_field",
        "magnetics-b_field_pol_probe_obr_field",
        "magnetics-b_field_pol_probe_obv_field",
        "magnetics-b_field_tor_probe_saddle_voltage",
    ]:
        signal_transform_map[var] = ComposeTransforms(
            [
                StdScalingTransform(dict_mean[var], dict_std[var]),
                FillProfileWithZerosTransform(),
            ]
        )

    # Specific case of reformating LCFS
    for var in ["equilibrium-lcfs_r", "equilibrium-lcfs_z"]:
        signal_transform_map[var] = ComposeTransforms(
            [
                ReshapeLcfsTransform(),
                StdScalingTransform(dict_mean[var], dict_std[var]),
            ]
        )

    # Specific filling with zeros for shomson scattering
    for var in ["thomson_scattering-t_e", "thomson_scattering-n_e"]:
        signal_transform_map[var] = ComposeTransforms(
            [
                StdScalingTransform(dict_mean[var], dict_std[var]),
                FillThomsonWithZerosTransform(),
            ]
        )

    return signal_transform_map



# ----------------------------------------------------------------------------------------------------------------------
def build_common_shot_transform_map(
    rolling_window_segmenter: Dict[str, float]
):
    """Builds the shot transform map for all variable."""

    shot_transform = ComposeTransforms([  
        RollingSegmenterTransform(
            **rolling_window_segmenter
        ), 
    ])

    return shot_transform

# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

def get_metadata(dataset, max_samples=100, verbose=True):
    """
    Find the first sample where each signal has a non-empty time array,
    then compute dt (median time step) and shape information.
    """
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            raise ValueError("❌ No valid sample found within limit.")
        
        # Check that each signal has a non-empty time array
        valid = all(len(signal.get("time", [])) > 1 for signal in sample.values())
        if not valid:
            continue  # Skip invalid sample

        # Found a valid sample
        info = {}
        for key, signal in sample.items():
            time = np.array(signal["time"])
            values = np.array(signal["values"])

            # Compute median dt
            dt = np.median(np.diff(time)) if len(time) > 1 else None

            info[key] = {
                "dt": dt,
                "values_shape": values.shape[:-1],  # exclude time dimension if last axis is time
            }

        if verbose:
            print(f"✅ Using sample #{i} as valid reference")
            for key, val in info.items():
                print(f"\nSignal: {key}")
                if val["dt"] is not None:
                    print(f"  dt: {val['dt']:.5f} s")
                else:
                    print(f"  dt: None")
                print(f"  Values shape: {val['values_shape']}")

        return info

    # If loop finishes without finding a valid sample:
    raise ValueError("❌ No valid sample found in dataset.")


# ----------------------------------------------------------------------------------------------------------------------
def initialize_datasets_and_metadata_for_task(
        config_task
):
    # ..................................................................................................................
    # Get shot id
    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config_task["subset_of_shots"]
    )

    # ..................................................................................................................
    # Get unique source-signal
    source_signal_list = (
        config_task["sources_and_signals"].get("x_past", [])
            + config_task["sources_and_signals"].get("x_future", [])
            + config_task["sources_and_signals"].get("y_past", [])
            + config_task["sources_and_signals"].get("y_future", [])
    ) 
    source_signal_list = [s for i, s in enumerate(source_signal_list) if s not in source_signal_list[:i]] # Unicity

    # ..................................................................................................................
    # Build signal transform map
    with open(
        REPO_ROOT + config_task["standardscaling_setting"]["mean_path"], "rb"
    ) as f:
        dict_mean = pickle.load(f)
    with open(REPO_ROOT + config_task["standardscaling_setting"]["std_path"], "rb") as f:
        dict_std = pickle.load(f)
    signal_transform_map = build_common_signal_transform_map(source_signal_list, 
                                                             dict_mean, 
                                                             dict_std)

    # ..................................................................................................................
    # Get metadata
    datasets_for_metadata = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots_, "val": val_shots_, "test": test_shots_},
        sig_tran_map=signal_transform_map,
        shot_tran=None,
        local_flag=config_task["local"],
        verbose=True,
    )
    dict_metadata = get_metadata(datasets_for_metadata["train"])

    # ..................................................................................................................
    # Build shot transform to cut accessible objects for task
    shot_transform = build_common_shot_transform_map(config_task["rolling_window_segmenter_setting"])

    # ..................................................................................................................
    # Get datasets 
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots_, "val": val_shots_, "test": test_shots_},
        sig_tran_map=signal_transform_map,
        shot_tran=shot_transform,
        local_flag=config_task["local"],
        verbose=True,
    )

    return datasets_train_val_test, dict_metadata