from typing import Optional, Mapping, Any
import os
import pickle
import numpy as np

from MAST_tools.MAST_dataset import MastDataset
from MAST_benchmark.tools.Task_Model_Wrapper import TaskModelTransformWrapper
from MAST_benchmark.tools.path import METADATA_DIR
from MAST_benchmark.tools.data_split import (
    get_train_test_val_shots
)
from MAST_benchmark.tools.MAST_composite_transform import (
    build_common_signal_transform_map,
)


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


def initialize_MAST_datasets(
    sources_and_signals,
    shots,
    sig_tran_map,
    shot_tran,
    local_flag=False,
    return_incomplete_shots=True,
    verbose=False,
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
            shot_level_transform=shot_tran,
            return_incomplete_shots=return_incomplete_shots,
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
            shot_level_transform=shot_tran,
            return_incomplete_shots=return_incomplete_shots,
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
            shot_level_transform=shot_tran,
            return_incomplete_shots=return_incomplete_shots,
        )
        if verbose:
            print(f"len(test_dataset): {len(datasets_['test'])}")

    # ..................................................................................................................
    # Return

    return datasets_


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
    mean_file_path = os.path.join(METADATA_DIR, 
                                  config_task["standardscaling_setting"]["mean_path"])
    with open(mean_file_path, "rb") as f:
        dict_mean = pickle.load(f)

    std_file_path = os.path.join(METADATA_DIR,
                                 config_task["standardscaling_setting"]["std_path"])        
    with open(std_file_path, "rb") as f:
        dict_std = pickle.load(f)

    signal_transform_map = build_common_signal_transform_map(
        source_signal_list, dict_mean, dict_std
    )

    # ..................................................................................................................
    # Get metadata
    datasets_train_val_test = initialize_MAST_datasets(
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


# ----------------------------------------------------------------------------------------------------------------------
def initialize_model_dataset(
    dataset: Optional[MastDataset],
    dict_metadata: Mapping[str, Any],
    config_task: Mapping[str, Any],
    model_specific_transform=None,
    *,
    verbose: bool = False,
) -> Optional[Any]:
    """
    Wrap a single baseline shot-level dataset with TaskModelTransformWrapper.

    This is a single-split version of initialize_model_datasets(), intended to
    be called explicitly for each split (train/val/test) from entrypoint scripts.

    Parameters
    ----------
    dataset:
        Baseline shot-level dataset (e.g., MastDataset) for one split, or None.
    dict_metadata:
        Metadata dictionary produced by the baseline pipeline (dt, shapes, etc.).
    config_task:
        Task configuration dict containing `task_window_segmenter` (keys, lengths, delta).
    model_specific_transform:
        Optional model-specific transform chain applied per window.
    verbose:
        If True, enables verbose prints in the wrapper.

    Returns
    -------
    TaskModelTransformWrapper | None
        Wrapped dataset, or None if input dataset is None.
    """
    if dataset is None:
        return None

    return TaskModelTransformWrapper(
        dataset,
        dict_metadata,
        config_task,
        model_specific_transform,
        verbose=verbose,
    )