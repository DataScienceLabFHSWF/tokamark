"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import argparse
import yaml
import json
import numpy as np
from typing import Any
from multiprocessing import cpu_count
import torch.multiprocessing as mp

from MAST_benchmark.data import initialize_MAST_dataset
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_tools.constants import (
    DEFAULT_CONFIG_GET_METADATA_FILE,
    DEFAULT_CONFIG_GET_METADATA_DEMO_FILE,
    DEFAULT_SIGNALS_MEAN_STD_TRAIN_FILE,
    DEFAULT_SIGNALS_STATS_FILE,
    DEFAULT_BASE_LOCAL_ZARR_PATH
)


# ----------------------------------------------------------------------------------------------------------------------

DEMO_MODE = True  # TODO: Check why demo results are equal to full results. [Rodrigo, Cecile]

if DEMO_MODE:
    default_config = DEFAULT_CONFIG_GET_METADATA_DEMO_FILE
    default_max_samples = 2
    default_signals_stats_file_path = DEFAULT_SIGNALS_STATS_FILE.replace(".yaml", "_DEMO.yaml")
else:
    default_config = DEFAULT_CONFIG_GET_METADATA_FILE
    default_max_samples = 100
    default_signals_stats_file_path = DEFAULT_SIGNALS_STATS_FILE


# ----------------------------------------------------------------------------------------------------------------------
def to_python(
        obj: Any
) -> Any:
    """
    Turn a numpy-based object into a numpy-free object.

    Parameters
    ----------
    obj : Any
        An arbitrary target object.

    Returns
    -------
    Any
        A numpy-free version of the input `obj` object.

    """

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)  # FIXME: This should have the same logic as the list case. [Cecile]
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


# ======================================================================================================================
if __name__ == "__main__":

    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method(method="spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to the config_get_metadata YAML file."
    )
    parser.add_argument(
        "--signals_mean_std_train_file_path",
        type=str,
        default=DEFAULT_SIGNALS_MEAN_STD_TRAIN_FILE,
        help="Path to the dict_signals_mean_std_train YAML file."
    )
    parser.add_argument(
        "--signals_stats_file_path",
        type=str,
        default=default_signals_stats_file_path,
        help="Path to the YAML file where signals statistics will be saved."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=default_max_samples,
        help="Maximum number of samples."
    )
    parser.add_argument(
        "--use_std_scaling",
        action="store_true",
        help="Activate STD scaling. If not provided, it defaults to `use_std_scaling = False`."
    )
    parser.add_argument(
        "--skip_incomplete_shots",
        action="store_true",
        help="Skip incomplete shots. If not provided, it defaults to `skip_incomplete_shots = False`, which in turn "
             "results in `return_incomplete_shots = True`."
    )
    parser.add_argument(
        "--keep_outliers",
        action="store_true",
        help="Keep outliers. If not provided, it defaults to `keep_outliers = False`, which in turn results in "
             "`remove_outliers = True`."
    )
    parser.add_argument(
        "--store_manager_settings",
        type=json.loads,
        help="User-defined store manager settings for the target MAST_dataset instance as defined in "
             "`src.MAST_tools.MAST_dataset.StoreManagerParameters`, passed as (keyword, value) pairs in a JSON object. "
             "It is useful to provide store manager settings different than the default ones defined in "
             "`src.MAST_tools.store_utils.MASTStorageManager.__init__`.",
        default='{"base_local_zarr_path":"' + DEFAULT_BASE_LOCAL_ZARR_PATH + '"}',
    )

    args, _ = parser.parse_known_args()

    # Trick to default boolean parameters to True
    args.return_incomplete_shots = not args.skip_incomplete_shots
    args.remove_outliers = not args.keep_outliers

    # Load Task YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------------------------------------------------------
    # Specific Data Preprocessing for LCFS profiles
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config["max_shot_index"],
        shuffle=config["shuffle"],
        seed=config["seed"]
    )

    local_flag = config["local"]

    # ------------------------------------------------------------------------------------------------------------------
    # Load dict_signals_mean_std_train.yaml
    # ------------------------------------------------------------------------------------------------------------------

    with open(args.signals_mean_std_train_file_path) as f:
        dict_mean_std = yaml.safe_load(f)

    # ------------------------------------------------------------------------------------------------------------------
    # Create unstandardized train dataset
    # ------------------------------------------------------------------------------------------------------------------

    preprocessing_train_dataset = initialize_MAST_dataset( 
        config_task=config,
        shots_list=train_shots_,
        local_flag=local_flag,
        use_std_scaling=args.use_std_scaling,                   # It defaults to False
        return_incomplete_shots=args.return_incomplete_shots,   # It defaults to True
        remove_outliers=args.remove_outliers,                   # It defaults to True
        store_manager_settings=args.store_manager_settings
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------------------------------------------------------

    dict_metadata = {}

    # Get expected variable count
    first_sample = next(iter(preprocessing_train_dataset))
    target_vars = set(first_sample.keys())

    for i, sample in enumerate(preprocessing_train_dataset):  # noqa (type check)
        # print(i)

        if i >= args.max_samples:
            break  # Stop after limit, but keep what we collected

        for var, signal in sample.items():

            # Only compute once per variable
            if var in dict_metadata:
                continue

            time = np.array(signal.get("time", []))
            values = np.array(signal.get("values", []))

            # Skip invalid signal
            if len(time) == 0 or len(values) == 0:
                print("Skipping var", var)
                continue
            
            # print("\nSaving var", var)

            # Compute median dt
            dt = round(np.median(np.diff(time)), 6)

            dict_metadata[var] = {
                "dt": dt,
                "values_shape": values.shape[:-1],  # Exclude time dimension
                "mean": dict_mean_std[var]["mean"]["no_outliers_z6"], 
                "std": dict_mean_std[var]["std"]["no_outliers_z6"],
            }
        
        # Stop once all variables are filled
        if set(dict_metadata.keys()) == target_vars:
            print("dict_metadata fully filled. Stopping.")
            break

    # ------------------------------------------------------------------------------------------------------------------
    # Optional safety check
    # ------------------------------------------------------------------------------------------------------------------

    if len(dict_metadata) == 0:
        raise ValueError("No valid signals found within limit.")

    # ------------------------------------------------------------------------------------------------------------------
    # Save and clean metadata
    # ------------------------------------------------------------------------------------------------------------------

    clean_dict = to_python(obj=dict_metadata)
    with open(args.signals_stats_file_path, "w") as f_:
        yaml.dump(clean_dict, f_, sort_keys=False)
