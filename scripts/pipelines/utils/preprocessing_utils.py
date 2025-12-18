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
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.reshape_lcfs_transform import (
    ReshapeLcfsTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_thomson_with_zeros_imputer_transform import (
    FillThomsonWithZerosTransform,
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
def get_metadata(
    dataset,
    config_task,
    dict_mean,
    dict_std,
    max_samples=100,
    verbose=True,
):
    """
    Find the first valid sample (all signals have time arrays), then compute:
      - dt (median time step)
      - values_shape (all dims except time axis)
      - mean/std (from dict_mean/dict_std)
      - role-specific sec_length/ts_length
      - per-signal ts_stride (derived from global sec_stride)

    Returns
    -------
    dict_metadata = {
        "sec_stride": float,
        "input":    { key: {...}, ... },
        "actuator": { key: {...}, ... },
        "output":   { key: {...}, ... },
    }

    where key is "source-signal".
    """

    seg = config_task["task_window_segmenter"]

    input_keys = [f"{src}-{sig}" for src, sig in (seg.get("input_keys") or [])]
    actuator_keys = [f"{src}-{sig}" for src, sig in (seg.get("actuator_keys") or [])]
    output_keys = [f"{src}-{sig}" for src, sig in (seg.get("output_keys") or [])]

    input_length = float(seg["input_length"])
    output_length = float(seg["output_length"])
    delta = float(seg.get("delta", 0.0))

    def _role_sec_length(role: str) -> float:
        if role == "input":
            return input_length
        if role == "actuator":
            return input_length + delta + output_length
        if role == "output":
            return output_length
        raise ValueError(f"Unknown role: {role!r}")

    for i, sample in enumerate(dataset):
        if i >= max_samples:
            raise ValueError("❌ No valid sample found within limit.")

        # valid sample = all signals have a non-empty time array
        valid = all(len(signal.get("time", [])) > 1 for signal in sample.values())
        if not valid:
            continue

        # Base per-key info (dt/shape/mean/std), independent of role lengths
        base_info = {}
        for key, signal in sample.items():
            time = np.asarray(signal["time"])
            values = np.asarray(signal["values"])

            dt = round(float(np.median(np.diff(time))), 6) if len(time) > 1 else None

            if key not in dict_mean or key not in dict_std:
                raise KeyError(f"Missing mean/std for {key!r}")

            base_info[key] = {
                "dt": dt,
                "values_shape": values.shape[:-1],  # exclude time axis
                "mean": dict_mean[key],
                "std": dict_std[key],
            }

        # Global stride in seconds = min dt among outputs
        out_dts = [
            base_info[k]["dt"]
            for k in output_keys
            if k in base_info and base_info[k]["dt"] is not None
        ]
        if not out_dts:
            raise ValueError("❌ Cannot compute sec_stride: no valid output dt found.")
        sec_stride = float(min(out_dts))

        dict_metadata = {
            "sec_stride": sec_stride,
            "input": {},
            "actuator": {},
            "output": {},
        }

        for role, keys in (
            ("input", input_keys),
            ("actuator", actuator_keys),
            ("output", output_keys),
        ):
            sec_len = _role_sec_length(role)

            for key in keys:
                if key not in base_info:
                    raise KeyError(
                        f"Signal {key!r} from config not found in sample keys."
                    )

                dt = base_info[key]["dt"]
                if dt is None or dt <= 0:
                    raise ValueError(f"Invalid dt for {key!r}: {dt}")

                entry = dict(base_info[key])  # shallow copy is enough here
                entry["sec_length"] = sec_len
                entry["ts_length"] = int(np.round(sec_len / dt))
                entry["ts_stride"] = int(np.round(sec_stride / dt))

                dict_metadata[role][key] = entry

        if verbose:
            print(f"✅ Using sample #{i} as valid reference")
            print(f"Global sec_stride: {sec_stride:.6f} s")
            for role in ("input", "actuator", "output"):
                print(f"\n[{role}]")
                for key, val in dict_metadata[role].items():
                    print(f"  {key}")
                    print(f"    dt: {val['dt']:.6f} s")
                    print(f"    values_shape: {val['values_shape']}")
                    print(f"    sec_length: {val['sec_length']}")
                    print(f"    ts_length: {val['ts_length']}")
                    print(f"    ts_stride: {val['ts_stride']}")

        return dict_metadata

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
