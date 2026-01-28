"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
import yaml
import numpy as np
from typing import Any, Mapping

from MAST_benchmark.tools.path import TASKS_CONFIGS_DIR
from MAST_benchmark.tools.path import METADATA_DIR
from MAST_benchmark.tools.utils import get_config_from_yaml


tasks_configs_map = {
    "task_1-1": "group_1_reconstruction/task_1-1.yaml",
    "task_1-2": "group_1_reconstruction/task_1-2.yaml",
    "task_1-3": "group_1_reconstruction/task_1-3.yaml",
    "task_2-1": "group_2_magnetics_dynamics/task_2-1.yaml",
    "task_2-2": "group_2_magnetics_dynamics/task_2-2.yaml",
    "task_2-3": "group_2_magnetics_dynamics/task_2-3.yaml",
    "task_3-1": "group_3_profiles_dynamics/task_3-1.yaml",
    "task_3-2": "group_3_profiles_dynamics/task_3-2.yaml",
    "task_3-3": "group_3_profiles_dynamics/task_3-3.yaml",
    "task_4-1": "group_4_mhd_activity/task_4-1.yaml",
    "task_4-2": "group_4_mhd_activity/task_4-2.yaml",
    "task_4-3": "group_4_mhd_activity/task_4-3.yaml",
    "task_4-4": "group_4_mhd_activity/task_4-4.yaml",
    "task_4-5": "group_4_mhd_activity/task_4-5.yaml"
}


# ----------------------------------------------------------------------------------------------------------------------
def get_task_config(
        task_name: str
) -> Any:
    """
    Get task configuration by task name.

    Parameters
    ----------
    task_name : str
        Name of the target task.

    Returns
    -------
    Any
        Configuration from YAML file.

    """

    task_path = tasks_configs_map[task_name]
    file_path = os.path.join(TASKS_CONFIGS_DIR, task_path)

    return get_config_from_yaml(file_path)


# ----------------------------------------------------------------------------------------------------------------------
def get_task_metadata(
    config_task: Mapping,
    verbose: bool = False
) -> Mapping:
    """
    Get task metadata for target configuration task.

    Parameters
    ----------
    config_task : Mapping
        Dictionary with task configuration.
    verbose : bool
        If True, activate verbose mode.

    Returns
    -------
    Mapping
        Output mapping with task metadata.

    """
    
    # ..................................................................................................................
    # Import data metadata
    metadata_path = os.path.join(METADATA_DIR, 'dict_stats_metadata.yaml')
    with open(metadata_path, "r") as f:  # 'rb' = read binary
        dict_stats_metadata = yaml.safe_load(f)

    # ..................................................................................................................
    # Import task specific roles

    # ---------------------------------------------------------------------
    # input
    input_keys = [
        f"{source}-{signal}"
        for source, signal in (config_task["task_window_segmenter"]["input_keys"] or [])
    ]
    # ---------------------------------------------------------------------
    # actuator
    actuator_keys = [
        f"{source}-{signal}"
        for source, signal in (
            config_task["task_window_segmenter"]["actuator_keys"] or []
        )
    ]
    # ---------------------------------------------------------------------
    # output
    output_keys = [
        f"{source}-{signal}"
        for source, signal in (
            config_task["task_window_segmenter"]["output_keys"] or []
        )
    ]

    # ..................................................................................................................
    # Import task specific values

    input_length = config_task["task_window_segmenter"]["input_length"]
    output_length = config_task["task_window_segmenter"]["output_length"]
    delta = config_task["task_window_segmenter"]["delta"]

    # Get stride from config file
    sec_stride = config_task["stride_window"]  # min([dict_metadata[key]["dt"] for key in output_keys])

    # --- plit into role-scoped dicts (avoid overwriting) ---
    out = {"sec_stride": sec_stride, "input": {}, "actuator": {}, "output": {}}

    # ..................................................................................................................
    # Get informations

    # ---------------------------------------------------------------------
    # input
    for key in input_keys:
        dt = dict_stats_metadata[key]["dt"]
        sec_length = config_task["task_window_segmenter"]["input_length"]
        out["input"][key] = dict(dict_stats_metadata[key])
        out["input"][key]["dt"] = dt
        out["input"][key]["values_shape"] = dict_stats_metadata[key]["values_shape"]
        out["input"][key]["mean"] = dict_stats_metadata[key]["mean"]
        out["input"][key]["std"] = dict_stats_metadata[key]["std"]
        out["input"][key]["sec_length"] = sec_length
        out["input"][key]["ts_length"] = int(np.round(sec_length / dt))
        out["input"][key]["ts_stride"] = int(np.round(sec_stride / dt))

    # ---------------------------------------------------------------------
    # actuator
    for key in actuator_keys:
        dt = dict_stats_metadata[key]["dt"]
        sec_length = input_length + delta + output_length
        out["actuator"][key] = dict(dict_stats_metadata[key])
        out["actuator"][key]["dt"] = dt
        out["actuator"][key]["values_shape"] = dict_stats_metadata[key]["values_shape"]
        out["actuator"][key]["mean"] = dict_stats_metadata[key]["mean"]
        out["actuator"][key]["std"] = dict_stats_metadata[key]["std"]
        out["actuator"][key]["sec_length"] = sec_length
        out["actuator"][key]["ts_length"] = int(np.round(sec_length / dt))
        out["actuator"][key]["ts_stride"] = int(np.round(sec_stride / dt))

    # ---------------------------------------------------------------------
    # output
    for key in output_keys:
        dt = dict_stats_metadata[key]["dt"]
        sec_length = output_length
        out["output"][key] = dict(dict_stats_metadata[key])
        out["output"][key]["dt"] = dt
        out["output"][key]["values_shape"] = dict_stats_metadata[key]["values_shape"]
        out["output"][key]["mean"] = dict_stats_metadata[key]["mean"]
        out["output"][key]["std"] = dict_stats_metadata[key]["std"]
        out["output"][key]["sec_length"] = sec_length
        out["output"][key]["ts_length"] = int(np.round(sec_length / dt))
        out["output"][key]["ts_stride"] = int(np.round(sec_stride / dt))
    
    if verbose:
        for role in ("input", "actuator", "output"):
            for key, val in out[role].items():
                print(f"\nSignal: {key}")
                if val["dt"] is not None:
                    print(f"  dt: {val['dt']:.5f} s")
                else:
                    print("  dt: None")
                print(f"  Values shape: {val['values_shape']}")

    return out
