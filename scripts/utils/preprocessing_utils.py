import os
import sys
import pickle
import numpy as np

from typing import Dict, List

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
        "..",
    )
)  # noqa: E402
# print(REPO_ROOT) # this adds /rds/project/rds-mOlK9qn0PlQ/ir-rous1/hncdi-fusion-plasma/fairmast-data-preprocessing
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.pipelines.utils.utils import (
    get_train_test_val_shots,
    initialize_datasets,
    ComposeTransforms,
)
from scripts.transforms.stdscale_transform import (
    StdScalingTransform,
)
from scripts.transforms.reshape_lcfs_transform import (
    ReshapeLcfsTransform,
)
from scripts.transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform,
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
        "thomson_scattering-t_e", 
        "thomson_scattering-n_e"
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

    return signal_transform_map


# ----------------------------------------------------------------------------------------------------------------------
def get_metadata(
    dataset, config_task, dict_mean, dict_std, max_samples=100, verbose=True
):
    """
    Find the first sample where each signal has a non-empty time array,
    then compute dt (median time step) and shape information.

    Minimal patch:
    - return structure is now split by role:
        {
          "sec_stride": float,
          "input":    {...},
          "actuator": {...},
          "output":   {...},
        }
    """

    input_keys = [
        f"{source}-{signal}"
        for source, signal in (config_task["task_window_segmenter"]["input_keys"] or [])
    ]
    input_length = config_task["task_window_segmenter"]["input_length"]

    actuator_keys = [
        f"{source}-{signal}"
        for source, signal in (
            config_task["task_window_segmenter"]["actuator_keys"] or []
        )
    ]
    delta = config_task["task_window_segmenter"]["delta"]

    output_keys = [
        f"{source}-{signal}"
        for source, signal in (
            config_task["task_window_segmenter"]["output_keys"] or []
        )
    ]
    output_length = config_task["task_window_segmenter"]["output_length"]

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
            dt = round(np.median(np.diff(time)), 6) if len(time) > 1 else None

            info[key] = {
                "dt": dt,
                "values_shape": values.shape[:-1],  # exclude time dimension
                "mean": dict_mean[key],
                "std": dict_std[key],
            }

        # Get stride from all dt --> this is now a top block, it's common to all
        # the signals
        sec_stride = min(
            [info[key]["dt"] for key in output_keys]
        )  # min for training, max for test is enviseagable

        # --- plit into role-scoped dicts (avoid overwriting) ---
        out = {"sec_stride": sec_stride, "input": {}, "actuator": {}, "output": {}}

        # input
        for key in input_keys:
            dt = info[key]["dt"]
            sec_length = input_length
            out["input"][key] = dict(info[key])
            out["input"][key]["sec_length"] = sec_length
            out["input"][key]["ts_length"] = int(np.round(sec_length / dt))
            out["input"][key]["ts_stride"] = int(np.round(sec_stride / dt))

        # actuator
        for key in actuator_keys:
            dt = info[key]["dt"]
            sec_length = input_length + delta + output_length
            out["actuator"][key] = dict(info[key])
            out["actuator"][key]["sec_length"] = sec_length
            out["actuator"][key]["ts_length"] = int(np.round(sec_length / dt))
            out["actuator"][key]["ts_stride"] = int(np.round(sec_stride / dt))

        # output
        for key in output_keys:
            dt = info[key]["dt"]
            sec_length = output_length
            out["output"][key] = dict(info[key])
            out["output"][key]["sec_length"] = sec_length
            out["output"][key]["ts_length"] = int(np.round(sec_length / dt))
            out["output"][key]["ts_stride"] = int(np.round(sec_stride / dt))
        # ---------------------------------------------------------------------

        if verbose:
            print(f"✅ Using sample #{i} as valid reference")
            for role in ("input", "actuator", "output"):
                for key, val in out[role].items():
                    print(f"\nSignal: {key}")
                    if val["dt"] is not None:
                        print(f"  dt: {val['dt']:.5f} s")
                    else:
                        print("  dt: None")
                    print(f"  Values shape: {val['values_shape']}")

        return out

    # If loop finishes without finding a valid sample:
    raise ValueError("❌ No valid sample found in dataset.")


# ----------------------------------------------------------------------------------------------------------------------
def initialize_datasets_and_metadata_for_task(config_task):
    # ..................................................................................................................
    # Get shot id
    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config_task["subset_of_shots"]
    )

    # ..................................................................................................................
    # Get unique source-signal
    source_signal_list = (
        (config_task["sources_and_signals"].get("input_name") or [])
        + (config_task["sources_and_signals"].get("actuator_name") or [])
        + (config_task["sources_and_signals"].get("output_name") or [])
    )

    source_signal_list = [
        s for i, s in enumerate(source_signal_list) if s not in source_signal_list[:i]
    ]  # Unicity

    # ..................................................................................................................
    # Build signal transform map
    with open(
        REPO_ROOT + config_task["standardscaling_setting"]["mean_path"], "rb"
    ) as f:
        dict_mean = pickle.load(f)
    with open(
        REPO_ROOT + config_task["standardscaling_setting"]["std_path"], "rb"
    ) as f:
        dict_std = pickle.load(f)
    signal_transform_map = build_common_signal_transform_map(
        source_signal_list, dict_mean, dict_std
    )

    # ..................................................................................................................
    # Get metadata
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots_, "val": val_shots_, "test": test_shots_},
        sig_tran_map=signal_transform_map,
        shot_tran=None,
        local_flag=config_task["local"],
        verbose=False,
    )
    dict_metadata = get_metadata(
        datasets_train_val_test["train"],
        config_task,
        dict_mean,
        dict_std,
        verbose=False,
    )

    return datasets_train_val_test, dict_metadata
